#!/usr/bin/env python3
"""
asr.py - hearing what the caller said.

Vosk, constrained to a grammar built from the contact list. That constraint is
the whole trick: against a closed vocabulary of twenty names plus digits plus
yes/no, a small model on a Pi 4 gets close to exact. Turned loose on open
English it does not, and it is slower.

Design rules carried over from contacts.py:

  * A wrong answer is worse than no answer. Every result carries a status and
    a confidence, and low-confidence results are reported as such rather than
    passed upward as if they were heard clearly.
  * '[unk]' stays in the grammar. Without it Vosk forces unrecognised speech
    onto the nearest allowed phrase, which is how a cough becomes a phone call.
  * The yes/no step gets its own two-word grammar. Confirmation is the last
    guard before dialling, so it is worth a separate, tighter recogniser.

Standalone use:

    python src/asr.py selftest              # no model, no mic, no network
    python src/asr.py grammar               # print the grammar from contacts
    python src/asr.py check                 # names the model cannot hear
    python src/asr.py file sample.wav       # transcribe a recording
    python src/asr.py listen                # live from the mic
    python src/asr.py confirm               # live yes/no test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DEFAULT_MODEL = "vosk-model-small-en-us-0.15"

SAMPLE_RATE = 16000          # Vosk small models want 16 kHz mono
BLOCK = 4000                 # frames per read, 250 ms
DEFAULT_TIMEOUT = 8.0        # give up listening after this
SILENCE_TAIL = 0.8           # end the utterance after this much quiet
MIN_CONFIDENCE = 0.70        # below this, treat it as not heard

try:
    from contacts import ContactBook, NO_WORDS, YES_WORDS, normalise
except ImportError:  # contacts.py not beside us; keep asr usable anyway
    ContactBook = None  # type: ignore[assignment]
    YES_WORDS = {"yes", "yeah", "yep", "correct", "right", "affirmative"}
    NO_WORDS = {"no", "nope", "wrong", "incorrect", "negative"}

    def normalise(text: str) -> str:  # type: ignore[misc]
        import re
        return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()

CONFIRM_GRAMMAR = json.dumps(sorted(YES_WORDS | NO_WORDS) + ["[unk]"])


# --------------------------------------------------------------------------
# result type
# --------------------------------------------------------------------------
@dataclass
class ASRResult:
    """What was heard.

    status is one of:
        ok              - clear speech above the confidence floor
        low_confidence  - words came back, but not reliably enough to act on
        unknown         - recognised as out-of-grammar speech
        silence         - the caller said nothing
        timeout         - the listening window expired mid-attempt
    """
    status: str
    text: str = ""
    confidence: float = 0.0
    words: list[dict] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def __str__(self) -> str:
        return f"{self.status} ({self.confidence:.2f}) {self.text!r}"


def classify_yes_no(text: str) -> Optional[bool]:
    """True for yes, False for no, None if it was neither.

    Checks word membership rather than equality, so "yes that's right" and
    "no wrong number" both land correctly.
    """
    words = set(normalise(text).split())
    yes = bool(words & {normalise(w) for w in YES_WORDS})
    no = bool(words & {normalise(w) for w in NO_WORDS})
    if yes and not no:
        return True
    if no and not yes:
        return False
    return None


def mean_confidence(result: dict) -> float:
    """Average per-word confidence from a Vosk result."""
    words = result.get("result") or []
    confs = [w.get("conf", 0.0) for w in words if "conf" in w]
    if not confs:
        return 1.0 if result.get("text") else 0.0
    return float(sum(confs) / len(confs))


# --------------------------------------------------------------------------
# audio helpers
# --------------------------------------------------------------------------
def _lowpass_kernel(rate: int, cutoff: float, taps: int = 101) -> np.ndarray:
    n = np.arange(taps) - (taps - 1) / 2.0
    fc = cutoff / rate
    h = np.sinc(2 * fc * n) * 2 * fc * np.hamming(taps)
    return h / np.sum(h)


def resample_to_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    """Downsample int16 audio to 16 kHz.

    Needed because USB sound cards usually open at 44.1 or 48 kHz and Vosk's
    small models only accept 16 kHz. Low-passes first so the downsample does
    not fold high frequencies back into the speech band as aliasing.
    """
    if rate == SAMPLE_RATE or samples.size == 0:
        return samples.astype(np.int16)
    audio = samples.astype(np.float64)
    if rate > SAMPLE_RATE:
        kernel = _lowpass_kernel(rate, SAMPLE_RATE / 2.0 * 0.9)
        audio = np.convolve(audio, kernel, mode="same")
    duration = audio.size / rate
    target_n = max(1, int(duration * SAMPLE_RATE))
    src = np.linspace(0.0, audio.size - 1, target_n)
    out = np.interp(src, np.arange(audio.size), audio)
    return np.clip(out, -32768, 32767).astype(np.int16)


def read_wav_16k(path: Path) -> np.ndarray:
    """Load a WAV as 16 kHz mono int16, whatever it started as."""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        if wf.getsampwidth() != 2:
            raise ValueError(f"{path}: need 16-bit PCM, got "
                             f"{wf.getsampwidth() * 8}-bit")
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return resample_to_16k(data, rate)


def rms(block: np.ndarray) -> float:
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))


# --------------------------------------------------------------------------
# the listener
# --------------------------------------------------------------------------
class Listener:
    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        grammar: Optional[str] = None,
        device: Optional[str] = None,
        min_confidence: float = MIN_CONFIDENCE,
        silence_rms: float = 220.0,
    ) -> None:
        self.model_path = Path(model_path) if model_path else self._find_model()
        self.grammar = grammar
        self.device = device
        self.min_confidence = min_confidence
        self.silence_rms = silence_rms
        self._model = None

    # ---------------- model and grammar ----------------

    @staticmethod
    def _find_model() -> Path:
        preferred = MODEL_DIR / DEFAULT_MODEL
        if preferred.is_dir():
            return preferred
        for d in sorted(MODEL_DIR.glob("vosk-model*")):
            if d.is_dir():
                return d
        raise FileNotFoundError(
            f"No Vosk model in {MODEL_DIR}.\n"
            f"  cd {MODEL_DIR}\n"
            f"  wget https://alphacephei.com/vosk/models/{DEFAULT_MODEL}.zip\n"
            f"  unzip {DEFAULT_MODEL}.zip"
        )

    def load(self) -> None:
        """Load the model once and keep it. Call at startup."""
        if self._model is not None:
            return
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)                      # Vosk is extremely chatty
        self._model = Model(str(self.model_path))

    def set_grammar_from_contacts(self, book) -> str:
        """Rebuild the grammar from the contact list.

        Call this whenever contacts change. A contact whose name is not in the
        grammar is literally unrecognisable, so a stale grammar looks like a
        broken recogniser.
        """
        self.grammar = book.grammar()
        return self.grammar

    def model_lexicon(self) -> Optional[set[str]]:
        """Words the model is able to produce, if we can find its word list."""
        for name in ("words.txt", "graph/words.txt"):
            path = self.model_path / name
            if path.exists():
                with path.open(encoding="utf-8", errors="replace") as fh:
                    return {line.split()[0].lower()
                            for line in fh if line.strip()}
        for path in self.model_path.rglob("words.txt"):
            with path.open(encoding="utf-8", errors="replace") as fh:
                return {line.split()[0].lower() for line in fh if line.strip()}
        return None

    def _recognizer(self, grammar: Optional[str] = None):
        from vosk import KaldiRecognizer
        self.load()
        gram = grammar if grammar is not None else self.grammar
        rec = (KaldiRecognizer(self._model, SAMPLE_RATE, gram) if gram
               else KaldiRecognizer(self._model, SAMPLE_RATE))
        rec.SetWords(True)
        return rec

    # ---------------- scoring ----------------

    def _finish(self, result: dict, elapsed: float) -> ASRResult:
        text = (result.get("text") or "").strip()
        conf = mean_confidence(result)
        words = result.get("result") or []

        if not text:
            return ASRResult("silence", elapsed=elapsed)
        if "[unk]" in text or text == "unk":
            return ASRResult("unknown", text=text, confidence=conf,
                             words=words, elapsed=elapsed)
        if conf < self.min_confidence:
            return ASRResult("low_confidence", text=text, confidence=conf,
                             words=words, elapsed=elapsed)
        return ASRResult("ok", text=text, confidence=conf,
                         words=words, elapsed=elapsed)

    # ---------------- file input ----------------

    def transcribe_file(self, path: Path | str,
                        grammar: Optional[str] = None) -> ASRResult:
        """Run a WAV through the recogniser.

        This is how asr.py gets developed and tuned before any microphone
        exists: record phrases anywhere, copy them over, iterate on the
        grammar and the confidence floor against a fixed set of samples.
        """
        t0 = time.monotonic()
        audio = read_wav_16k(Path(path))
        rec = self._recognizer(grammar)
        for i in range(0, audio.size, BLOCK):
            rec.AcceptWaveform(audio[i:i + BLOCK].tobytes())
        result = json.loads(rec.FinalResult())
        return self._finish(result, time.monotonic() - t0)

    # ---------------- live input ----------------

    def listen(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        silence_tail: float = SILENCE_TAIL,
        grammar: Optional[str] = None,
    ) -> ASRResult:
        """Listen on the microphone until the caller stops talking.

        Ends on whichever comes first: Vosk's own endpointing, silence_tail
        seconds of quiet after speech started, or the hard timeout.
        """
        import sounddevice as sd

        rec = self._recognizer(grammar)
        t0 = time.monotonic()
        heard_speech = False
        quiet_since: Optional[float] = None

        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK,
                               dtype="int16", channels=1,
                               device=self.device) as stream:
            while True:
                now = time.monotonic()
                if now - t0 > timeout:
                    partial = json.loads(rec.FinalResult())
                    res = self._finish(partial, now - t0)
                    if res.status == "silence" and heard_speech:
                        res = ASRResult("timeout", elapsed=now - t0)
                    return res

                data, _ = stream.read(BLOCK)
                block = np.frombuffer(bytes(data), dtype=np.int16)
                level = rms(block)

                if level > self.silence_rms:
                    heard_speech = True
                    quiet_since = None
                elif heard_speech and quiet_since is None:
                    quiet_since = now

                if rec.AcceptWaveform(bytes(data)):
                    return self._finish(json.loads(rec.Result()),
                                        time.monotonic() - t0)

                if (heard_speech and quiet_since is not None
                        and now - quiet_since > silence_tail):
                    return self._finish(json.loads(rec.FinalResult()),
                                        time.monotonic() - t0)

    def confirm(self, timeout: float = 6.0) -> tuple[Optional[bool], ASRResult]:
        """Listen for yes or no against a two-word grammar.

        Returns (True/False/None, result). None means ask again rather than
        assuming either way, because guessing here dials the wrong person.
        """
        res = self.listen(timeout=timeout, grammar=CONFIRM_GRAMMAR)
        if res.status in ("silence", "timeout", "unknown"):
            return None, res
        return classify_yes_no(res.text), res


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _selftest() -> int:
    """Checks needing no model, no microphone and no network."""
    fails = 0

    yes_no = [
        ("yes", True), ("yeah", True), ("that is correct", True),
        ("yes that's right", True),
        ("no", False), ("nope", False), ("no wrong number", False),
        ("margaret hale", None), ("", None),
        ("yes no", None),                       # contradictory: must not guess
    ]
    for text, want in yes_no:
        got = classify_yes_no(text)
        if got != want:
            fails += 1
            print(f"  FAIL classify_yes_no({text!r}) -> {got}, wanted {want}")
        else:
            print(f"  ok   classify_yes_no({text!r:<22}) -> {got}")

    # Confidence gating
    listener = Listener.__new__(Listener)
    listener.min_confidence = 0.70
    cases = [
        ({"text": "gavin reed",
          "result": [{"word": "gavin", "conf": 0.98},
                     {"word": "reed", "conf": 0.95}]}, "ok"),
        ({"text": "gavin reed",
          "result": [{"word": "gavin", "conf": 0.40},
                     {"word": "reed", "conf": 0.35}]}, "low_confidence"),
        ({"text": "[unk]", "result": [{"word": "[unk]", "conf": 0.9}]}, "unknown"),
        ({"text": "", "result": []}, "silence"),
    ]
    for payload, want in cases:
        got = listener._finish(payload, 0.0)
        if got.status != want:
            fails += 1
            print(f"  FAIL _finish {payload.get('text')!r} -> {got.status}, "
                  f"wanted {want}")
        else:
            print(f"  ok   _finish {payload.get('text')!r:<14} -> {got.status}")

    # Resampler: a 1 kHz tone at 48 kHz must survive the trip to 16 kHz.
    t = np.arange(48000) / 48000
    tone = (np.sin(2 * np.pi * 1000 * t) * 15000).astype(np.int16)
    out = resample_to_16k(tone, 48000)
    expected = SAMPLE_RATE
    if abs(out.size - expected) > 50:
        fails += 1
        print(f"  FAIL resample length {out.size}, wanted about {expected}")
    else:
        print(f"  ok   resample 48k -> 16k ({out.size} samples)")

    mid = out[1000:-1000].astype(np.float64)
    spectrum = np.abs(np.fft.rfft(mid * np.hanning(mid.size)))
    peak_hz = np.fft.rfftfreq(mid.size, 1 / SAMPLE_RATE)[int(np.argmax(spectrum))]
    if abs(peak_hz - 1000) > 25:
        fails += 1
        print(f"  FAIL resampled tone peaks at {peak_hz:.0f} Hz, wanted 1000")
    else:
        print(f"  ok   resampled tone still at {peak_hz:.0f} Hz")

    if resample_to_16k(np.array([], dtype=np.int16), 48000).size != 0:
        fails += 1
        print("  FAIL empty resample")
    else:
        print("  ok   empty input handled")

    # The confirmation grammar must be tight and must keep [unk].
    gram = json.loads(CONFIRM_GRAMMAR)
    if "[unk]" not in gram:
        fails += 1
        print("  FAIL confirm grammar dropped [unk]")
    elif len(gram) > 20:
        fails += 1
        print(f"  FAIL confirm grammar too broad ({len(gram)} entries)")
    else:
        print(f"  ok   confirm grammar: {len(gram)} entries, [unk] present")

    print()
    print(f"{fails} failure(s)" if fails else "all checks passed")
    return 1 if fails else 0


def _cmd_check(listener: Listener) -> int:
    """Report contact words the model has no pronunciation for."""
    if ContactBook is None:
        print("contacts.py not importable; cannot check")
        return 1
    with ContactBook() as book:
        lexicon = listener.model_lexicon()
        if lexicon is None:
            print(f"Could not find words.txt under {listener.model_path}.")
            print("Skipping the check; unknown words will simply be unheard.")
            return 1
        missing = book.unknown_words(lexicon)
        print(f"  model lexicon : {len(lexicon)} words")
        print(f"  contact words : {len(book.words())}")
        if not missing:
            print("\n  Every contact word is in the model. Nothing to do.")
            return 0
        print(f"\n  {len(missing)} word(s) the model cannot produce:")
        for w in sorted(missing):
            print(f"    {w}")
        print("\n  Vosk silently drops these from the grammar, so those names")
        print("  will never be recognised. Add a phonetic alias for each:")
        print("    python src/contacts.py alias <id> shivawn")
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Kellogg phone speech recognition")
    ap.add_argument("--model", help="path to a Vosk model directory")
    ap.add_argument("--device", help="ALSA capture device")
    ap.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    ap.add_argument("--open", action="store_true",
                    help="no grammar at all (open vocabulary, slower, worse)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="checks needing no model or microphone")
    sub.add_parser("grammar", help="print the grammar built from contacts")
    sub.add_parser("check", help="find contact names the model cannot hear")
    sub.add_parser("confirm", help="live yes/no test")

    p = sub.add_parser("file", help="transcribe a WAV")
    p.add_argument("path")

    p = sub.add_parser("listen", help="listen on the microphone")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)

    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        return _selftest()

    grammar = None
    if not args.open and ContactBook is not None:
        with ContactBook() as book:
            grammar = book.grammar()

    if args.cmd == "grammar":
        if grammar is None:
            print("contacts.py not importable")
            return 1
        items = json.loads(grammar)
        print(f"{len(items)} entries:\n")
        for item in items:
            print(f"  {item}")
        return 0

    listener = Listener(model_path=args.model, grammar=grammar,
                        device=args.device, min_confidence=args.min_confidence)

    if args.cmd == "check":
        return _cmd_check(listener)

    if args.cmd == "file":
        res = listener.transcribe_file(args.path)
        print(f"  {res}")
        print(f"  took {res.elapsed:.2f}s")
        for w in res.words:
            print(f"    {w.get('word'):<16} {w.get('conf', 0):.2f}")
        return 0 if res.ok else 1

    if args.cmd == "listen":
        print("Speak now. Ctrl-C to stop.")
        res = listener.listen(timeout=args.timeout)
        print(f"  {res}")
        print(f"  took {res.elapsed:.2f}s")
        return 0 if res.ok else 1

    if args.cmd == "confirm":
        print("Say yes or no.")
        answer, res = listener.confirm()
        print(f"  {res}")
        print(f"  answer: {answer}")
        if answer is None:
            print("  -> operator would ask again rather than guess")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())