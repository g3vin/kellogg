#!/usr/bin/env python3
"""Kellogg phone - audio device identity and diagnostics.

This module does NOT play or capture for the application. tts.py plays and
asr.py captures. What those two lack is a trustworthy answer to "which
device?", and that is the whole job here.

Why it exists at all: ALSA card indices move. On this Pi the USB sound card
was index 1 one week and index 3 the next, because the two vc4-hdmi cards
and the headphone jack enumerate first and the order is not stable. A
hard-coded hw:1,0 produces a phone that works until it is rebooted, and the
failure mode is baffling - opening plughw:1,0 on a headless Pi reaches an
HDMI device with no display attached and fails with error -524 ENOTSUPP,
which reads like broken hardware. Everything here resolves by NAME.

Standalone use:

    python src/audio.py devices     every device, with names and rates
    python src/audio.py detect      find the card, save audio.json
    python src/audio.py record out.wav --seconds 5
    python src/audio.py play out.wav
    python src/audio.py level       live meter, for setting the mic trimpot
    python src/audio.py echo        AEC leakage check
    python src/audio.py selftest    maths and config round-trip, no hardware
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "audio.json"

# The USB card runs at 44.1 or 48 kHz. 16 kHz is what Vosk wants but the
# card will usually refuse to open there, so capture native and let
# asr.resample_to_16k do the conversion.
PREFERRED_RATES = (48000, 44100, 32000, 16000)

# Hints used to recognise the USB sound card among the Pi's own outputs.
USB_HINTS = ("usb audio", "usb-audio", "generalplus", "device")
# Devices that are never what we want on a headless Pi.
EXCLUDE_HINTS = ("vc4-hdmi", "hdmi", "bcm2835")

# PipeWire's echo-cancel module creates these. Names come from the
# config in kellogg-aec.conf; keep the two in step.
AEC_SOURCE_HINT = "echo-cancel-source"
AEC_SINK_HINT = "echo-cancel-sink"

# Level targets for the mic trimpot. Speech should peak here.
TARGET_PEAK_MIN = -18.0
TARGET_PEAK_MAX = -6.0
CLIP_DBFS = -0.5

# An AEC that leaves less than this much of the far end in the near end
# is doing its job well enough for ASR.
ECHO_PASS_DB = -25.0


class AudioError(RuntimeError):
    """Something about the audio devices is wrong and worth reading."""


# --------------------------------------------------------------------------
# level maths - pure, so the self test can exercise it with no hardware
# --------------------------------------------------------------------------
def dbfs(amplitude: float) -> float:
    """Convert a 0..1 normalised amplitude to dBFS.

    Silence is -inf in theory; we floor it at -120 so reports stay
    readable and arithmetic downstream does not produce nan.
    """
    if amplitude <= 0.0:
        return -120.0
    return max(-120.0, 20.0 * math.log10(min(1.0, amplitude)))


def peak_rms(samples: Sequence[float]) -> tuple[float, float]:
    """Peak and RMS of normalised samples, both in dBFS."""
    if not len(samples):
        return -120.0, -120.0
    peak = max(abs(float(s)) for s in samples)
    mean_square = sum(float(s) * float(s) for s in samples) / len(samples)
    return dbfs(peak), dbfs(math.sqrt(mean_square))


def gain_advice(peak: float, rms: float, clipped: int) -> str:
    """A sentence telling the builder which way to turn the trimpot.

    This is the tool used to set the MAX4466 gain, so it has to say what
    to do, not just report numbers at someone holding a screwdriver.
    """
    if clipped > 0:
        return (f"Clipping on {clipped} samples. Turn the MAX4466 trimpot "
                f"DOWN until peaks sit near {TARGET_PEAK_MAX:.0f} dBFS.")
    if peak <= -60.0:
        return ("Silence. Nothing is reaching the sound card - check the "
                "mic wiring, the coupling capacitor, and that you are "
                "recording from the USB card and not the Pi's own input.")
    if peak < TARGET_PEAK_MIN:
        short = TARGET_PEAK_MAX - peak
        return (f"Too quiet by roughly {short:.0f} dB. Turn the MAX4466 "
                f"trimpot UP, then run this again.")
    if peak > TARGET_PEAK_MAX:
        return (f"Hot, though not yet clipping. Turn the trimpot DOWN a "
                f"little; aim for peaks around {TARGET_PEAK_MAX:.0f} dBFS.")
    if rms < -45.0:
        return ("Peaks are fine but the average is very low, which usually "
                "means you spoke once and then stopped. Talk continuously "
                "for the whole measurement.")
    if peak < -10.0 and rms < -30.0:
        return (f"Peak {peak:.1f} dBFS, average {rms:.1f} dBFS. That is low "
                f"for speech - if you were not talking, this is a noise "
                f"floor measurement and you want 'audio.py noise' instead, "
                f"which says what the noise is made of.")
    return (f"Good. Peak {peak:.1f} dBFS, average {rms:.1f} dBFS. "
            f"Leave the trimpot where it is.")


# A quiet room, measured through a decent mic path, should sit below this.
NOISE_FLOOR_GOOD = -55.0
NOISE_FLOOR_POOR = -45.0


def _dominant(bands: dict, margin: float = 6.0) -> bool:
    """True when one band clearly beats the rest.

    Compared band to band, never against the overall RMS: the bands are
    per-bin densities and the RMS is a total, so the two are not on the
    same scale and mixing them silently mis-sorts hum as hiss.
    """
    ordered = sorted(bands.values(), reverse=True)
    return len(ordered) > 1 and (ordered[0] - ordered[1]) >= margin


def noise_advice(peak: float, rms: float, bands: dict) -> str:
    """What the noise floor is made of, and what to do about it.

    Distinguishing hum from hiss from switching noise matters because the
    three have completely different fixes and look identical on a meter.
    """
    if rms <= NOISE_FLOOR_GOOD:
        return (f"Quiet. Noise floor {rms:.1f} dBFS leaves plenty of room "
                f"under speech.")

    worst = max(bands, key=lambda k: bands[k])
    detail = "  ".join(f"{k} {v:.1f}" for k, v in bands.items())

    # A constant has peak == rms and no spectral content anywhere. Catch
    # it before any band reasoning, because every band reads the floor and
    # whichever is nominally "highest" is meaningless.
    if peak - rms < 3.0 and max(bands.values()) < rms - 40.0:
        cause = ("Peak and average are the same and no band carries any "
                 "energy - this is a DC level, not noise. If the input was "
                 "shorted for this measurement, that is just the ADC's own "
                 "offset and it is harmless. If it was not, the coupling "
                 "capacitor is passing DC and should be checked.")
    elif worst == "sub" and _dominant(bands):
        cause = ("The energy is BELOW the mains frequency - DC offset or "
                 "subsonic drift, not audible noise. The usual cause is "
                 "the coupling capacitor passing DC: check its polarity "
                 "with a meter (the + leg must face whichever side reads "
                 "more positive), and if the sound card biases its mic "
                 "jack, fit a non-polarised capacitor instead.")
    # Impulses next. A click has energy everywhere, so it will always
    # look like whichever band is examined, and the fix is different.
    elif peak - rms > 24.0:
        cause = ("Impulsive, not steady - the peaks are far above the "
                 "average. That is clicks rather than hiss: suspect a "
                 "loose joint on the mic cord, or USB contention. Wiggle "
                 "the cord while this runs and see if it changes.")
    elif worst == "mains" and _dominant(bands):
        cause = ("Mains hum dominates. This is a grounding problem, not a "
                 "gain one: check that the mic cord's ground returns to the "
                 "same point as the MAX4466's GND pin, and that the cord is "
                 "not running alongside the 9 V printer supply.")
    elif worst == "hf" and _dominant(bands):
        cause = ("High-frequency noise dominates - switching rather than "
                 "thermal. Move the mic cord away from the Pi and the "
                 "printer supply, and add 10uF + 0.1uF decoupling right at "
                 "the MAX4466's VCC pin.")
    else:
        cause = ("Broadband hiss. Mostly the MAX4466's own noise at high "
                 "gain. Turn its trimpot DOWN and raise the card's capture "
                 "level to compensate, then re-run 'level' to confirm "
                 "speech still peaks near -6 dBFS.")

    severity = "Poor" if rms > NOISE_FLOOR_POOR else "Marginal"
    return (f"{severity}. Noise floor {rms:.1f} dBFS (want below "
            f"{NOISE_FLOOR_GOOD:.0f}).\n  Bands, dBFS: {detail}\n  {cause}")


@dataclass
class LevelReport:
    peak_dbfs: float
    rms_dbfs: float
    clipped_frames: int
    advice: str
    seconds: float = 0.0

    def __str__(self) -> str:
        return (f"peak {self.peak_dbfs:6.1f} dBFS   "
                f"rms {self.rms_dbfs:6.1f} dBFS   "
                f"clipped {self.clipped_frames}\n  {self.advice}")


@dataclass
class EchoReport:
    leak_db: float
    passed: bool
    note: str

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return f"{verdict}  playback leaks into capture at {self.leak_db:.1f} dB\n  {self.note}"


# --------------------------------------------------------------------------
# device identity
# --------------------------------------------------------------------------
@dataclass
class AudioConfig:
    """Which devices to use, resolved by name.

    aec_source and aec_sink are the PipeWire virtual devices. They are
    None until the echo-cancel module is configured. main.py prefers them
    when present and falls back to the raw card when not, which is what
    lets the phone work (badly) before AEC is set up.
    """
    capture: str
    playback: str
    capture_rate: int
    playback_rate: int
    aec_source: Optional[str] = None
    aec_sink: Optional[str] = None
    detected_at: str = ""
    # The hint that found these devices. PortAudio names embed the card
    # index - "USB Audio Device: - (hw:3,0)" - so a saved name goes stale
    # the moment the index moves, which is the exact failure this module
    # exists to prevent. Re-resolving from the hint survives that.
    hint: str = "USB"

    @property
    def capture_device(self) -> str:
        """What asr.py should open. The AEC source when there is one."""
        return self.aec_source or self.capture

    @property
    def playback_device(self) -> str:
        """What tts.py should open."""
        return self.aec_sink or self.playback

    def save(self, path: Path | str = CONFIG_PATH) -> Path:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path

    @classmethod
    def resolve(cls, path: Path | str = CONFIG_PATH) -> "AudioConfig":
        """The config to actually use: detected now, file as fallback.

        Always prefers a live scan, because card indices move between
        boots and the saved device names contain them. The file supplies
        the hint and is only used wholesale when detection fails.
        """
        hint = "USB"
        saved: Optional[AudioConfig] = None
        try:
            saved = cls.load(path)
            hint = saved.hint or hint
        except Exception:
            pass

        try:
            fresh = find_devices(prefer=hint)
        except AudioError:
            if saved is not None:
                return saved
            raise

        fresh.hint = hint
        if saved is None or (saved.capture, saved.playback) != (
                fresh.capture, fresh.playback):
            try:
                fresh.save(path)
            except Exception:
                pass
        return fresh

    @classmethod
    def load(cls, path: Path | str = CONFIG_PATH) -> "AudioConfig":
        path = Path(path)
        if not path.is_file():
            raise AudioError(
                f"{path} does not exist. Run:  python src/audio.py detect")
        data = json.loads(path.read_text())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def __str__(self) -> str:
        lines = [
            f"  capture   {self.capture}  @ {self.capture_rate} Hz",
            f"  playback  {self.playback}  @ {self.playback_rate} Hz",
        ]
        if self.aec_source or self.aec_sink:
            lines.append(f"  aec in    {self.aec_source or '-'}")
            lines.append(f"  aec out   {self.aec_sink or '-'}")
        else:
            lines.append("  aec       not configured - see kellogg-aec.conf")
        return "\n".join(lines)


def _sounddevice():
    """Import sounddevice, or explain why the module is useless without it."""
    try:
        import sounddevice as sd
    except Exception as exc:
        raise AudioError(
            f"sounddevice is unavailable ({exc}). Install it with:\n"
            f"    sudo apt install python3-sounddevice\n"
            f"or, inside the venv:  pip install sounddevice"
        ) from None
    return sd


def _score(name: str, prefer: str) -> int:
    """How much this device looks like the one we want. Higher is better."""
    low = name.lower()
    if any(h in low for h in EXCLUDE_HINTS):
        return -100
    score = 0
    if prefer and prefer.lower() in low:
        score += 50
    for hint in USB_HINTS:
        if hint in low:
            score += 10
    return score


def _usable_rate(sd, name: str, kind: str) -> int:
    """The highest preferred rate this device will actually open at.

    Asking rather than assuming, because these cards advertise more than
    they support and the failure only shows up at stream-open time.
    """
    for rate in PREFERRED_RATES:
        try:
            if kind == "input":
                sd.check_input_settings(device=name, samplerate=rate,
                                        channels=1)
            else:
                sd.check_output_settings(device=name, samplerate=rate,
                                         channels=1)
            return rate
        except Exception:
            continue
    raise AudioError(
        f"{name} would not open for {kind} at any of {PREFERRED_RATES}.")


def find_devices(prefer: str = "USB") -> AudioConfig:
    """Pick the capture and playback devices, by name.

    Prefers a device whose name contains the hint, skips the Pi's own
    HDMI and headphone outputs, and falls back to the system default
    rather than failing outright.
    """
    sd = _sounddevice()
    devices = sd.query_devices()

    best_in = best_out = None
    in_score = out_score = -1
    for dev in devices:
        name = dev["name"]
        score = _score(name, prefer)
        if dev["max_input_channels"] > 0 and score > in_score:
            best_in, in_score = name, score
        if dev["max_output_channels"] > 0 and score > out_score:
            best_out, out_score = name, score

    if best_in is None or best_out is None:
        raise AudioError(
            "No usable audio devices. Found:\n" + list_devices())
    if in_score < 0 or out_score < 0:
        raise AudioError(
            f"Only excluded devices matched {prefer!r}. The USB sound card "
            f"is probably unplugged. Found:\n" + list_devices())

    # The PipeWire virtual devices, if the AEC module is loaded.
    aec_in = aec_out = None
    for dev in devices:
        low = dev["name"].lower()
        if AEC_SOURCE_HINT in low and dev["max_input_channels"] > 0:
            aec_in = dev["name"]
        if AEC_SINK_HINT in low and dev["max_output_channels"] > 0:
            aec_out = dev["name"]

    return AudioConfig(
        capture=best_in,
        playback=best_out,
        capture_rate=_usable_rate(sd, best_in, "input"),
        playback_rate=_usable_rate(sd, best_out, "output"),
        aec_source=aec_in,
        aec_sink=aec_out,
        detected_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def list_devices() -> str:
    """Human-readable table. Indices are shown but never used."""
    try:
        sd = _sounddevice()
        devices = sd.query_devices()
    except AudioError as exc:
        return str(exc)

    rows = ["  idx  in out  rate     name",
            "  ---  -- ---  -------  " + "-" * 44]
    for i, dev in enumerate(devices):
        flag = ""
        low = dev["name"].lower()
        if any(h in low for h in EXCLUDE_HINTS):
            flag = "  (ignored)"
        elif any(h in low for h in USB_HINTS):
            flag = "  <- likely"
        rows.append(f"  {i:3d}  {dev['max_input_channels']:2d} "
                    f"{dev['max_output_channels']:3d}  "
                    f"{int(dev['default_samplerate']):7d}  "
                    f"{dev['name']}{flag}")
    rows.append("")
    rows.append("  Indices move between boots. Everything here uses names.")
    return "\n".join(rows)


# --------------------------------------------------------------------------
# recording and playback, for diagnostics only
# --------------------------------------------------------------------------
def record_wav(path: Path | str, seconds: float = 5.0,
               device: Optional[str] = None,
               rate: Optional[int] = None) -> Path:
    """Record mono to a 16-bit WAV. Returns the path."""
    sd = _sounddevice()
    import numpy as np

    if device is None or rate is None:
        cfg = find_devices()
        device = device or cfg.capture
        rate = rate or cfg.capture_rate

    frames = int(seconds * rate)
    data = sd.rec(frames, samplerate=rate, channels=1,
                  dtype="int16", device=device)
    sd.wait()

    path = Path(path)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.asarray(data, dtype="int16").tobytes())
    return path


def play_wav(path: Path | str, device: Optional[str] = None,
             blocking: bool = True) -> None:
    sd = _sounddevice()
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        raw = wav.readframes(wav.getnframes())

    samples = np.frombuffer(raw, dtype="int16")
    if channels > 1:
        samples = samples.reshape(-1, channels)

    if device is None:
        device = find_devices().playback

    sd.play(samples, samplerate=rate, device=device)
    if blocking:
        sd.wait()


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def measure_level(seconds: float = 5.0,
                  device: Optional[str] = None,
                  rate: Optional[int] = None) -> LevelReport:
    """Record and report, with advice on which way to turn the trimpot."""
    sd = _sounddevice()
    import numpy as np

    if device is None or rate is None:
        cfg = find_devices()
        device = device or cfg.capture_device
        rate = rate or cfg.capture_rate

    frames = int(seconds * rate)
    data = sd.rec(frames, samplerate=rate, channels=1,
                  dtype="float32", device=device)
    sd.wait()
    flat = np.asarray(data, dtype="float32").reshape(-1)

    peak = dbfs(float(np.max(np.abs(flat)))) if flat.size else -120.0
    rms = dbfs(float(np.sqrt(np.mean(flat * flat)))) if flat.size else -120.0
    clipped = int(np.count_nonzero(np.abs(flat) >= 0.999))

    return LevelReport(peak, rms, clipped,
                       gain_advice(peak, rms, clipped), seconds)


def band_levels(samples, rate: int) -> dict:
    """Split the noise into three bands so its character is visible.

    mains  45-65 Hz plus the 120 Hz harmonic - grounding problems
    voice  300-3400 Hz - the band that actually matters for speech
    hf     above 8 kHz - switching supplies and digital pickup
    """
    import numpy as np

    flat = np.asarray(samples, dtype="float32").reshape(-1)
    if flat.size < 64:
        return {"mains": -120.0, "voice": -120.0, "hf": -120.0}

    window = np.hanning(flat.size).astype("float32")
    spec = np.abs(np.fft.rfft(flat * window)) / (flat.size / 2)
    freq = np.fft.rfftfreq(flat.size, 1.0 / rate)

    def band(lo: float, hi: float) -> float:
        """Average energy per bin, not total.

        The high band is four times wider than the voice band, so summing
        would make flat hiss look like high-frequency switching noise -
        it would win on width alone. Density is the fair comparison.
        """
        sel = spec[(freq >= lo) & (freq < hi)]
        if not sel.size:
            return -120.0
        return dbfs(float(np.sqrt(np.mean(sel * sel))))

    return {
        # Below the mains fundamental: DC offset, subsonic drift, and the
        # signature of a coupling capacitor that is passing DC. Without
        # this band the energy simply vanishes from the report and the
        # numbers stop reconciling with the overall RMS.
        "sub": band(1.0, 45.0),
        "mains": max(band(45.0, 65.0), band(115.0, 125.0)),
        "lowmid": band(125.0, 300.0),
        "voice": band(300.0, 3400.0),
        "hf": band(8000.0, min(rate / 2 - 1, 20000.0)),
    }


def dc_offset(samples) -> float:
    """Mean sample value. Should be ~0; anything else is a DC fault."""
    import numpy as np
    flat = np.asarray(samples, dtype="float32").reshape(-1)
    if not flat.size:
        return 0.0
    return float(np.mean(flat))


def measure_noise(seconds: float = 5.0,
                  device: Optional[str] = None,
                  rate: Optional[int] = None) -> LevelReport:
    """Measure the noise floor and say what kind of noise it is.

    Run this in a silent room with nobody talking. It is a different
    question from measure_level(), which assumes you ARE talking - give
    the wrong one a silent room and it cheerfully reports "Good".
    """
    sd = _sounddevice()
    import numpy as np

    if device is None or rate is None:
        cfg = find_devices()
        device = device or cfg.capture_device
        rate = rate or cfg.capture_rate

    data = sd.rec(int(seconds * rate), samplerate=rate, channels=1,
                  dtype="float32", device=device)
    sd.wait()
    flat = np.asarray(data, dtype="float32").reshape(-1)

    peak = dbfs(float(np.max(np.abs(flat)))) if flat.size else -120.0
    rms = dbfs(float(np.sqrt(np.mean(flat * flat)))) if flat.size else -120.0
    clipped = int(np.count_nonzero(np.abs(flat) >= 0.999))
    bands = band_levels(flat, rate)

    advice = noise_advice(peak, rms, bands)
    offset = dc_offset(flat)
    if abs(offset) > 0.005:
        advice = (f"DC offset {offset:+.3f} of full scale - the capture is "
                  f"not centred on zero. Fix this before anything else; it "
                  f"eats headroom and the recogniser sees a signal riding "
                  f"on a step.\n  " + advice)
    return LevelReport(peak, rms, clipped, advice, seconds)


def check_echo_cancellation(device: Optional[str] = None,
                            playback: Optional[str] = None,
                            seconds: float = 3.0,
                            freq: float = 1000.0) -> EchoReport:
    """Play a tone, record at the same time, measure how much came back.

    The mouthpiece and earpiece are inches apart in this phone, so without
    echo cancellation the operator hears itself and the recogniser gets a
    mixture of the caller and the operator's own voice. Build-order stage
    C: nothing downstream should be trusted until this passes.
    """
    sd = _sounddevice()
    import numpy as np

    cfg = find_devices()
    device = device or cfg.capture_device
    playback = playback or cfg.playback_device
    rate = cfg.capture_rate

    frames = int(seconds * rate)
    t = np.arange(frames, dtype="float32") / rate
    tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype("float32")

    # silence first, to learn the room
    quiet = sd.rec(int(0.7 * rate), samplerate=rate, channels=1,
                   dtype="float32", device=device)
    sd.wait()
    floor = dbfs(float(np.sqrt(np.mean(quiet * quiet))) or 1e-9)

    sd.play(tone, samplerate=rate, device=playback)
    heard = sd.rec(frames, samplerate=rate, channels=1,
                   dtype="float32", device=device)
    sd.wait()
    flat = np.asarray(heard, dtype="float32").reshape(-1)

    # Energy at the tone frequency only; broadband room noise should not
    # count against the AEC.
    window = np.hanning(flat.size).astype("float32")
    spectrum = np.abs(np.fft.rfft(flat * window))
    bins = np.fft.rfftfreq(flat.size, 1.0 / rate)
    target = int(np.argmin(np.abs(bins - freq)))
    band = spectrum[max(0, target - 2): target + 3].max()
    total = spectrum.max() or 1e-9
    leak = dbfs(float(band / (flat.size / 2)))

    passed = leak <= ECHO_PASS_DB
    if passed:
        note = ("The earpiece is not bleeding into the mouthpiece. The "
                "recogniser will hear the caller, not the operator.")
    elif cfg.aec_source is None:
        note = ("No AEC device found, so this is measuring the raw card. "
                "Load the echo-cancel module (kellogg-aec.conf), then run "
                "this again.")
    else:
        note = ("Too much playback is reaching the mic. Move the earpiece "
                "away from the mouthpiece, turn the amp gain down, and "
                "confirm capture is bound to the AEC source rather than "
                "the raw card.")
    _ = (floor, total)
    return EchoReport(leak, passed, note)


# --------------------------------------------------------------------------
# self test - no hardware
# --------------------------------------------------------------------------
def _selftest() -> int:
    import tempfile

    checks: list[tuple[bool, str]] = []

    def ok(cond: bool, label: str) -> None:
        checks.append((bool(cond), label))

    # dBFS maths
    ok(abs(dbfs(1.0) - 0.0) < 1e-9, "full scale is 0 dBFS")
    ok(abs(dbfs(0.5) + 6.0206) < 1e-3, "half scale is -6.02 dBFS")
    ok(dbfs(0.0) == -120.0, "silence floors at -120, not -inf")
    ok(dbfs(2.0) == 0.0, "over-unity clamps to 0 rather than going positive")

    p, r = peak_rms([0.0, 0.5, -0.5, 0.0])
    ok(abs(p + 6.0206) < 1e-3, "peak_rms finds the peak")
    ok(r < p, "rms is below peak for a non-constant signal")
    ok(peak_rms([]) == (-120.0, -120.0), "empty input does not divide by zero")

    # advice is actionable in each regime
    ok("DOWN" in gain_advice(-2.0, -20.0, 400), "clipping says turn it down")
    ok("UP" in gain_advice(-40.0, -55.0, 0), "quiet says turn it up")
    ok("Good" in gain_advice(-12.0, -28.0, 0), "in range says leave it")
    ok("Silence" in gain_advice(-120.0, -120.0, 0), "silence is called out")
    ok("mic wiring" in gain_advice(-120.0, -120.0, 0),
       "silence advice names a cause worth checking")

    # config round-trip
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "audio.json"
        cfg = AudioConfig(capture="USB Audio Device",
                          playback="USB Audio Device",
                          capture_rate=48000, playback_rate=48000)
        cfg.save(path)
        back = AudioConfig.load(path)
        ok(back.capture == cfg.capture, "config round-trips through JSON")
        ok(back.capture_rate == 48000, "rates survive the round trip")
        ok(back.capture_device == "USB Audio Device",
           "capture_device falls back to the raw card with no AEC")

        cfg.aec_source = "echo-cancel-source"
        cfg.aec_sink = "echo-cancel-sink"
        cfg.save(path)
        back = AudioConfig.load(path)
        ok(back.capture_device == "echo-cancel-source",
           "capture_device prefers the AEC source once it exists")
        ok(back.playback_device == "echo-cancel-sink",
           "playback_device prefers the AEC sink once it exists")

        # an unknown key in the file must not crash a later version
        data = json.loads(path.read_text())
        data["something_from_the_future"] = 1
        path.write_text(json.dumps(data))
        AudioConfig.load(path)
        ok(True, "unknown keys in audio.json are ignored, not fatal")

        # A saved name with a stale card index must not be trusted.
        stale = Path(tmp) / "stale.json"
        AudioConfig(capture="USB Audio Device: - (hw:9,0)",
                    playback="USB Audio Device: - (hw:9,0)",
                    capture_rate=48000, playback_rate=48000,
                    hint="USB").save(stale)
        back = AudioConfig.load(stale)
        ok(back.hint == "USB", "the detection hint survives a save/load")
        ok("hw:9,0" in back.capture,
           "the saved name does embed a card index, which is why "
           "resolve() rescans rather than trusting it")

        missing = Path(tmp) / "nope.json"
        try:
            AudioConfig.load(missing)
            ok(False, "missing config raises")
        except AudioError as exc:
            ok("detect" in str(exc),
               "missing config tells you which command to run")

    # noise classifier - synthetic signals of each known character
    try:
        import numpy as np
        rate = 48000
        t = np.arange(rate * 2) / rate
        imp = 0.002 * np.random.randn(t.size)
        imp[::7000] = 0.25
        cases = [
            ("quiet room", 0.001 * np.random.randn(t.size),
             "plenty of room"),
            ("mains hum", 0.02 * np.sin(2 * np.pi * 60 * t)
             + 0.0005 * np.random.randn(t.size), "grounding problem"),
            ("switching noise", 0.02 * np.sin(2 * np.pi * 14000 * t)
             + 0.0005 * np.random.randn(t.size), "switching rather"),
            ("broadband hiss", 0.02 * np.random.randn(t.size),
             "Broadband hiss"),
            ("impulsive clicks", imp, "Impulsive"),
        ]
        for label, sig, expect in cases:
            sig = sig.astype("float32")
            pk = dbfs(float(np.max(np.abs(sig))))
            rm = dbfs(float(np.sqrt(np.mean(sig * sig))))
            said = noise_advice(pk, rm, band_levels(sig, rate))
            ok(expect in said, f"{label} is diagnosed correctly")
        ok(not _dominant({"a": -40.0, "b": -42.0}),
           "two close bands are not called dominant")
        ok(_dominant({"a": -30.0, "b": -45.0}),
           "a band 15 dB above the rest is dominant")
    except ImportError:
        ok(True, "numpy absent - noise classifier checks skipped")

    # device scoring
    ok(_score("vc4-hdmi-0", "USB") < 0, "HDMI is excluded outright")
    ok(_score("bcm2835 Headphones", "USB") < 0, "the Pi's own jack is excluded")
    ok(_score("USB Audio Device", "USB") > 0, "the USB card scores positively")
    ok(_score("USB Audio Device", "USB") > _score("default", "USB"),
       "the USB card beats the system default")

    width = max(len(label) for _, label in checks)
    failed = 0
    for passed, label in checks:
        if not passed:
            failed += 1
        print(f"  {'ok ' if passed else 'FAIL'}  {label:<{width}}")
    print()
    if failed:
        print(f"{failed} of {len(checks)} checks failed")
        return 1
    print(f"all {len(checks)} checks passed")
    return 0


# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Kellogg phone audio devices")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("devices", help="list every audio device")
    sub.add_parser("detect", help="find the card and save audio.json")
    sub.add_parser("selftest", help="maths and config checks, no hardware")

    rec = sub.add_parser("record", help="record a wav")
    rec.add_argument("path")
    rec.add_argument("--seconds", type=float, default=5.0)

    play = sub.add_parser("play", help="play a wav")
    play.add_argument("path")

    lvl = sub.add_parser("level", help="measure the mic, for the trimpot")
    lvl.add_argument("--seconds", type=float, default=5.0)

    nz = sub.add_parser("noise", help="measure the noise floor, in silence")
    nz.add_argument("--seconds", type=float, default=5.0)

    sub.add_parser("echo", help="check echo cancellation")

    args = ap.parse_args(argv)
    cmd = args.cmd or "devices"

    if cmd == "selftest":
        return _selftest()

    try:
        if cmd == "devices":
            # Ask for the library first so an import failure becomes a
            # non-zero exit rather than a cheerful printout of the error.
            _sounddevice()
            print(list_devices())
            return 0

        if cmd == "detect":
            cfg = find_devices()
            path = cfg.save()
            print("Detected:")
            print(cfg)
            print(f"\nSaved to {path}")
            if cfg.aec_source is None:
                print("\nNo echo-cancel device yet. The phone will work but "
                      "the operator will hear itself; set up the AEC before "
                      "trusting the recogniser.")
            return 0

        if cmd == "record":
            path = record_wav(args.path, args.seconds)
            print(f"Recorded {args.seconds:g}s to {path}")
            return 0

        if cmd == "play":
            play_wav(args.path)
            return 0

        if cmd == "level":
            print(f"Speak normally for {args.seconds:g} seconds...")
            report = measure_level(args.seconds)
            print(f"\n  {report}")
            return 0 if report.clipped_frames == 0 else 1

        if cmd == "noise":
            print(f"Stay silent for {args.seconds:g} seconds...")
            report = measure_noise(args.seconds)
            print(f"\n  peak {report.peak_dbfs:6.1f} dBFS   "
                  f"rms {report.rms_dbfs:6.1f} dBFS")
            print(f"  {report.advice}")
            return 0 if report.rms_dbfs <= NOISE_FLOOR_GOOD else 1

        if cmd == "echo":
            print("Playing a tone while recording. Keep the room quiet.")
            report = check_echo_cancellation()
            print(f"\n  {report}")
            return 0 if report.passed else 1

    except AudioError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n  stopped")
        return 130

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())