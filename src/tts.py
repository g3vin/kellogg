#!/usr/bin/env python3
"""
tts.py - the operator's voice.

Wraps Piper with four things the raw library does not give you:

  1. The model is loaded once and stays resident. Reloading per utterance adds
     seconds of latency, and the operator answering slowly is the single most
     noticeable way this build can feel wrong.
  2. Fixed prompts are rendered to WAV ahead of time and cached on disk, so
     "Number please" plays instantly rather than being synthesised each call.
  3. An optional telephone-band filter (300-3400 Hz) that makes Piper sound
     like it is coming down a 1920s line instead of out of a smart speaker.
  4. Readback helpers that speak phone numbers digit by digit, because feeding
     a raw number to any TTS produces "seven billion five hundred seventy-five
     million..." which is useless on a phone.

Piper's Python API has changed shape across releases, so synthesis goes
through _synthesize_to_wav(), which tries each known calling convention and
falls back to the piper CLI. If a future version breaks all of them, that is
the one function to fix.

Standalone use:

    python src/tts.py --say "Number please"
    python src/tts.py --say "Number please" --telephone
    python src/tts.py readback "Margaret Hale" 7575559090
    python src/tts.py prompts          # pre-render the whole prompt library
    python src/tts.py selftest         # latency + filter checks, no audio out
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
CACHE_DIR = ROOT / "sounds" / "cache"

DEFAULT_VOICE = "en_US-lessac-medium"

# Telephone band
TELEPHONE_LOW_HZ = 300.0
TELEPHONE_HIGH_HZ = 3400.0
FIR_TAPS = 255

# Fixed phrases the operator says
PROMPTS: dict[str, str] = {
    "greeting":       "Operator. Number please.",
    "confirm_prefix": "Did you say",
    "confirm_suffix": "Is that correct?",
    "connecting":     "Thank you. Connecting you now.",
    "retry":          "I'm sorry. Could you say that again?",
    "not_found":      "I'm sorry, I don't have anyone by that name.",
    "which_one":      "I have more than one. Which did you mean?",
    "no_answer":      "I'm sorry, there's no answer.",
    "goodbye":        "Thank you. Goodbye.",
    "timeout":        "Are you still there?",
    "ready":          "Ready.",
}


# --------------------------------------------------------------------------
# audio helpers
# --------------------------------------------------------------------------
def _bandpass_kernel(rate: int, low: float, high: float, taps: int) -> np.ndarray:
    """Windowed-sinc bandpass, built with numpy alone (no scipy dependency)."""
    if taps % 2 == 0:
        taps += 1
    n = np.arange(taps) - (taps - 1) / 2.0
    window = np.hamming(taps)

    def lowpass(cutoff: float) -> np.ndarray:
        fc = cutoff / rate
        h = np.sinc(2 * fc * n) * 2 * fc
        h *= window
        return h / np.sum(h)

    high = min(high, rate / 2.0 * 0.98)
    kernel = lowpass(high) - lowpass(low)
    return kernel


def telephone_filter(samples: np.ndarray, rate: int) -> np.ndarray:
    """Band-limit int16 audio to the telephone band and renormalise.

    Filtering removes a lot of energy, so without the gain correction the
    result just sounds quiet rather than old.
    """
    if samples.size == 0:
        return samples
    kernel = _bandpass_kernel(rate, TELEPHONE_LOW_HZ, TELEPHONE_HIGH_HZ, FIR_TAPS)
    audio = samples.astype(np.float64)
    filtered = np.convolve(audio, kernel, mode="same")

    peak = np.max(np.abs(filtered))
    if peak > 0:
        target = min(0.89 * 32767.0, np.max(np.abs(audio)) or 32767.0)
        filtered *= target / peak
    return np.clip(filtered, -32768, 32767).astype(np.int16)


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16)
        if wf.getnchannels() > 1:
            data = data.reshape(-1, wf.getnchannels())[:, 0].copy()
    return data, rate


def _write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.astype(np.int16).tobytes())


# --------------------------------------------------------------------------
# number readback
# --------------------------------------------------------------------------
def speak_digits(number: str) -> str:
    """Digit-by-digit, grouped, with commas so Piper pauses naturally.

    Mirrors contacts.speak_digits but with pacing, since the operator reading
    a number back needs to be followed by ear, not parsed.
    """
    digits = "".join(c for c in str(number) if c.isdigit())
    if len(digits) == 10:
        groups = [digits[:3], digits[3:6], digits[6:]]
    elif len(digits) == 7:
        groups = [digits[:3], digits[3:]]
    else:
        groups = [digits[i:i + 3] for i in range(0, len(digits), 3)]
    return ", ".join(" ".join(g) for g in groups if g)


def readback_text(name: Optional[str], number: str) -> str:
    """The confirmation line: who, then the number, then the question."""
    spoken = speak_digits(number)
    if name:
        return f"{name}. {spoken}. Is that correct?"
    return f"{spoken}. Is that correct?"


# --------------------------------------------------------------------------
# the voice
# --------------------------------------------------------------------------
class OperatorVoice:
    def __init__(
        self,
        model: Optional[Path | str] = None,
        telephone: bool = True,
        device: Optional[str] = None,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        self.model_path = Path(model) if model else self._find_model()
        self.telephone = telephone
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._voice = None          # lazily loaded, then kept forever
        self._rate: Optional[int] = None

    # ---------------- model ----------------

    @staticmethod
    def _find_model() -> Path:
        """Prefer the default voice, else take any .onnx sitting in models/."""
        preferred = MODEL_DIR / f"{DEFAULT_VOICE}.onnx"
        if preferred.exists():
            return preferred
        found = sorted(MODEL_DIR.glob("*.onnx"))
        if found:
            return found[0]
        raise FileNotFoundError(
            f"No Piper voice in {MODEL_DIR}.\n"
            f"  mkdir -p {MODEL_DIR}\n"
            f"  python -m piper.download_voices {DEFAULT_VOICE} "
            f"--data-dir {MODEL_DIR}"
        )

    def load(self) -> None:
        """Load the model once. Call at startup, not on first utterance."""
        if self._voice is not None:
            return
        try:
            from piper import PiperVoice
        except ImportError:
            self._voice = "cli"     # fall back to the piper executable
            return
        self._voice = PiperVoice.load(str(self.model_path))

    # ---------------- synthesis ----------------

    def _synthesize_to_wav(self, text: str, out: Path) -> None:
        """Render text to a mono WAV, coping with Piper's API drift."""
        self.load()
        out.parent.mkdir(parents=True, exist_ok=True)

        if self._voice != "cli":
            v = self._voice
            # Newest API: writes into an open wave file.
            if hasattr(v, "synthesize_wav"):
                with wave.open(str(out), "wb") as wf:
                    v.synthesize_wav(text, wf)
                return
            # Older API: same idea, different name.
            try:
                with wave.open(str(out), "wb") as wf:
                    v.synthesize(text, wf)
                return
            except TypeError:
                pass
            # Streaming API: synthesize() yields audio chunks.
            chunks = list(v.synthesize(text))
            if chunks:
                rate = getattr(chunks[0], "sample_rate", 22050)
                buf = b"".join(
                    getattr(c, "audio_int16_bytes", None) or c.audio_int16_array.tobytes()
                    for c in chunks
                )
                _write_wav(out, np.frombuffer(buf, dtype=np.int16), rate)
                return

        exe = shutil.which("piper")
        if not exe:
            raise RuntimeError(
                "Could not synthesise: the Piper Python API did not match any "
                "known signature and no 'piper' executable is on PATH."
            )
        subprocess.run(
            [exe, "-m", str(self.model_path), "-f", str(out)],
            input=text.encode(), check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def render(self, text: str, out: Optional[Path] = None) -> Path:
        """Synthesise text to a WAV, applying the telephone filter if enabled."""
        target = Path(out) if out else Path(tempfile.mktemp(suffix=".wav"))
        raw = Path(tempfile.mktemp(suffix=".wav"))
        try:
            self._synthesize_to_wav(text, raw)
            samples, rate = _read_wav(raw)
            self._rate = rate
            if self.telephone:
                samples = telephone_filter(samples, rate)
            _write_wav(target, samples, rate)
        finally:
            raw.unlink(missing_ok=True)
        return target

    # ---------------- caching ----------------

    def _cache_path(self, text: str) -> Path:
        key = f"{self.model_path.name}|{self.telephone}|{text}"
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{digest}.wav"

    def cached(self, text: str) -> Path:
        """Render text, reusing the cached WAV when one exists."""
        path = self._cache_path(text)
        if not path.exists():
            self.render(text, path)
        return path

    def prerender_prompts(self, prompts: Optional[dict[str, str]] = None) -> dict[str, Path]:
        """Render the fixed prompt library. Run once at startup or at install."""
        out: dict[str, Path] = {}
        for name, text in (prompts or PROMPTS).items():
            out[name] = self.cached(text)
        return out

    # ---------------- playback ----------------

    def play_file(self, path: Path, blocking: bool = True) -> None:
        try:
            import sounddevice as sd
            samples, rate = _read_wav(Path(path))
            sd.play(samples, rate, device=self.device)
            if blocking:
                sd.wait()
            return
        except Exception:
            pass
        cmd = ["aplay", "-q"]
        if self.device:
            cmd += ["-D", self.device]
        cmd.append(str(path))
        (subprocess.run if blocking else subprocess.Popen)(cmd)

    def say(self, text: str, blocking: bool = True) -> None:
        """Speak arbitrary text. Cached, so repeats are instant."""
        self.play_file(self.cached(text), blocking=blocking)

    def prompt(self, name: str, blocking: bool = True) -> None:
        """Speak a named prompt from the library."""
        if name not in PROMPTS:
            raise KeyError(f"unknown prompt {name!r}; have {sorted(PROMPTS)}")
        self.say(PROMPTS[name], blocking=blocking)

    def readback(self, name: Optional[str], number: str, blocking: bool = True) -> None:
        """Speak the confirmation line for a resolved contact."""
        self.say(readback_text(name, number), blocking=blocking)

    def prewarm(self) -> float:
        """Load the model and synthesise once. Returns seconds taken.

        Call this at service start. The first synthesis after load is always
        slower than the rest, and you want that cost paid before anyone picks
        up the handset.
        """
        t0 = time.monotonic()
        self.load()
        self.cached(PROMPTS["ready"])
        return time.monotonic() - t0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _selftest() -> int:
    """Checks that need no model, no audio hardware and no network."""
    fails = 0

    cases = [
        ("7575551212", "7 5 7, 5 5 5, 1 2 1 2"),
        ("(757) 555-9090", "7 5 7, 5 5 5, 9 0 9 0"),
        ("5551212", "5 5 5, 1 2 1 2"),
    ]
    for raw, want in cases:
        got = speak_digits(raw)
        if got != want:
            fails += 1
            print(f"  FAIL speak_digits({raw!r}) -> {got!r}, wanted {want!r}")
        else:
            print(f"  ok   speak_digits({raw!r})")

    line = readback_text("Margaret Hale", "7575559090")
    if not line.startswith("Margaret Hale.") or not line.endswith("Is that correct?"):
        fails += 1
        print(f"  FAIL readback_text: {line!r}")
    else:
        print(f"  ok   readback_text -> {line!r}")

    if "billion" in line or "7575559090" in line:
        fails += 1
        print("  FAIL readback spoke the number as a whole quantity")

    # Filter: a 1 kHz tone must survive, 100 Hz and 8 kHz must not.
    rate = 22050
    t = np.arange(rate) / rate
    for freq, should_pass in ((1000, True), (100, False), (8000, False)):
        tone = (np.sin(2 * np.pi * freq * t) * 20000).astype(np.int16)
        out = telephone_filter(tone, rate)
        mid = out[len(out) // 4: -len(out) // 4]
        ratio = np.max(np.abs(mid)) / 20000.0
        ok = ratio > 0.5 if should_pass else ratio < 0.2
        if not ok:
            fails += 1
            print(f"  FAIL {freq} Hz passed at {ratio:.2f} "
                  f"(expected {'kept' if should_pass else 'removed'})")
        else:
            print(f"  ok   {freq:>5} Hz {'kept' if should_pass else 'removed'} "
                  f"({ratio:.2f})")

    if telephone_filter(np.array([], dtype=np.int16), rate).size != 0:
        fails += 1
        print("  FAIL empty input should return empty")
    else:
        print("  ok   empty input handled")

    for name, text in PROMPTS.items():
        if not text.strip():
            fails += 1
            print(f"  FAIL prompt {name!r} is empty")

    print()
    print(f"{fails} failure(s)" if fails else "all checks passed")
    return 1 if fails else 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Kellogg phone operator voice")
    ap.add_argument("--model", help="path to a .onnx Piper voice")
    ap.add_argument("--device", help="ALSA device for playback")
    ap.add_argument("--telephone", action="store_true",
                    help="apply the 300-3400 Hz telephone band filter")
    ap.add_argument("--flat", action="store_true",
                    help="disable the telephone filter (full bandwidth)")
    ap.add_argument("--say", help="speak this text")
    ap.add_argument("--wav", help="write to this WAV instead of playing")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("readback", help="speak a name and number for confirmation")
    p.add_argument("name")
    p.add_argument("number")

    sub.add_parser("prompts", help="pre-render the fixed prompt library")
    sub.add_parser("selftest", help="checks needing no model or audio")
    sub.add_parser("latency", help="measure load and first-utterance time")

    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "selftest":
        return _selftest()

    telephone = not args.flat
    voice = OperatorVoice(model=args.model, telephone=telephone, device=args.device)

    if args.cmd == "readback":
        text = readback_text(args.name, args.number)
        print(text)
        if args.wav:
            print(voice.render(text, Path(args.wav)))
        else:
            voice.say(text)
        return 0

    if args.cmd == "prompts":
        rendered = voice.prerender_prompts()
        for name, path in rendered.items():
            print(f"  {name:<16} {path}")
        print(f"\n{len(rendered)} prompts cached in {voice.cache_dir}")
        return 0

    if args.cmd == "latency":
        warm = voice.prewarm()
        print(f"  model load + first synthesis : {warm:6.2f} s")
        t0 = time.monotonic()
        voice.render(PROMPTS["greeting"])
        print(f"  second synthesis            : {time.monotonic() - t0:6.2f} s")
        t0 = time.monotonic()
        voice.cached(PROMPTS["greeting"])
        print(f"  cached lookup               : {time.monotonic() - t0:6.3f} s")
        print("\n  Aim for the cached lookup to be near zero. If the second")
        print("  synthesis is slow, pre-render every fixed prompt at startup.")
        return 0

    if args.say:
        if args.wav:
            print(voice.render(args.say, Path(args.wav)))
        else:
            voice.say(args.say)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())