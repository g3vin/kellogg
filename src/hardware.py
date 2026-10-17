#!/usr/bin/env python3
"""
hardware.py - GPIO layer for the Kellogg magneto phone.

Pin map (BCM), matching the build plan:

    GPIO 17   hookswitch     micro switch: COM -> GND, NO -> GPIO 17
    GPIO 22   crank sensor   DRV5032 OUT (open drain, active low)
    GPIO 26   spare switch   second micro switch, unused for now
    GPIO 16   amp mute       PAM8302 SD pin; LOW = muted

All three inputs are active-low with the internal pull-up enabled, so an
unconnected pin reads as "inactive" rather than floating noise.

Run it on its own to exercise the sensors:

    python src/hardware.py           event log (what you normally want)
    python src/hardware.py --raw     every Hall edge, for magnet calibration
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from collections import deque
from typing import Callable, Optional

try:
    from gpiozero import Button, DigitalOutputDevice
except ImportError:  # pragma: no cover
    sys.exit(
        "gpiozero not found.\n"
        "  sudo apt install python3-gpiozero python3-lgpio\n"
        "and make sure your venv was created with --system-site-packages."
    )


# --------------------------------------------------------------------------
# Pin assignments
# --------------------------------------------------------------------------
_log = logging.getLogger("kellogg.hardware")

PIN_HOOK = 17

# Which way the cradle works the micro switch.
#
# The plan assumes NO wired so that the handset SITTING IN THE CRADLE
# presses the switch and grounds the pin. Real cradles differ - some
# press on lift instead, and some builds end up on NC - so this is a
# setting rather than an assumption. Flip it without touching the
# soldering by putting this in .env:
#
#     KELLOGG_HOOK_INVERTED=1
#
# Verify with:  python src/hardware.py   and lift the handset.
HOOK_INVERTED = os.environ.get("KELLOGG_HOOK_INVERTED", "") not in ("", "0")
PIN_HALL = 22
PIN_SPARE = 26
PIN_AMP = 16

# Debounce. The hookswitch is a mechanical lever and bounces for several
# milliseconds. The Hall sensor is a solid-state switch and barely bounces at
# all, so its window is kept short to avoid swallowing fast crank pulses.
HOOK_BOUNCE_S = 0.05
HALL_BOUNCE_S = 0.02

# Crank detection: how many magnet passes, inside what window, counts as
# "the caller is cranking" rather than someone bumping the furniture.
# Measured on the real crank: one pulse per revolution, gaps of 0.6 to
# 1.6 s at a natural cranking pace - roughly 36 to 100 rpm. Requiring 3
# pulses in 2 s meant sustaining about 90 rpm, which only the fastest
# bursts reached, so an ordinary crank never registered.
#
# 2 pulses in 3 s needs about 40 rpm: comfortably within a normal turn,
# while still refusing to fire on one stray pulse from a knock or from
# the magnet settling next to the sensor at rest.
CRANK_PULSES = 2
CRANK_WINDOW_S = 3.0
CRANK_COOLDOWN_S = 3.0


class CrankDetector:
    """Turns a stream of Hall pulses into a single 'crank' decision.

    One magnet on the shaft gives one pulse per revolution. Requiring several
    pulses inside a short window is what separates deliberate cranking from a
    single stray edge, and the cooldown stops one long crank from firing the
    event over and over.
    """

    def __init__(
        self,
        pulses_needed: int = CRANK_PULSES,
        window_s: float = CRANK_WINDOW_S,
        cooldown_s: float = CRANK_COOLDOWN_S,
    ) -> None:
        self.pulses_needed = pulses_needed
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self._times: deque[float] = deque(maxlen=32)
        self._last_fired = 0.0

    def reset(self) -> None:
        self._times.clear()
        self._last_fired = 0.0

    def pulse(self, now: Optional[float] = None) -> bool:
        """Record one magnet pass. Returns True if this completes a crank."""
        now = time.monotonic() if now is None else now

        if now - self._last_fired < self.cooldown_s:
            return False

        self._times.append(now)
        while self._times and now - self._times[0] > self.window_s:
            self._times.popleft()

        if len(self._times) >= self.pulses_needed:
            self._last_fired = now
            self._times.clear()
            return True
        return False


class PhoneHardware:
    """Hookswitch, crank sensor, spare switch and amplifier mute.

    Callbacks fire on a gpiozero background thread, so keep them short and
    hand anything slow off to your own queue or state machine.
    """

    def __init__(
        self,
        on_off_hook: Optional[Callable[[], None]] = None,
        on_on_hook: Optional[Callable[[], None]] = None,
        on_crank: Optional[Callable[[], None]] = None,
        on_hall_pulse: Optional[Callable[[], None]] = None,
        crank_requires_off_hook: bool = True,
        hook_inverted: Optional[bool] = None,
        auto_mute_on_hook: bool = True,
    ) -> None:
        self.on_off_hook = on_off_hook
        self.on_on_hook = on_on_hook
        self.on_crank = on_crank
        self.on_hall_pulse = on_hall_pulse
        self.crank_requires_off_hook = crank_requires_off_hook
        # Resolved here, not as a default argument: a default is bound
        # when the class is defined, so --invert-hook setting the module
        # global afterwards would silently do nothing.
        self.hook_inverted = (HOOK_INVERTED if hook_inverted is None
                              else hook_inverted)
        self.cranks_dropped_on_hook = 0
        self.auto_mute_on_hook = auto_mute_on_hook

        self.crank = CrankDetector()
        self.hall_pulse_count = 0

        # pull_up=True means "pressed" == pin pulled to ground.
        # Hookswitch wiring: handset resting in the cradle depresses the
        # lever, closing COM-NO, which grounds the pin. So:
        #     pressed  == handset DOWN  (on-hook)
        #     released == handset UP    (off-hook)
        self.hook = Button(PIN_HOOK, pull_up=True, bounce_time=HOOK_BOUNCE_S)
        self.hall = Button(PIN_HALL, pull_up=True, bounce_time=HALL_BOUNCE_S)
        self.spare = Button(PIN_SPARE, pull_up=True, bounce_time=HOOK_BOUNCE_S)

        # PAM8302 SD: LOW shuts the amplifier down. Start muted so the
        # earpiece is silent while the handset sits in the cradle.
        self.amp = DigitalOutputDevice(PIN_AMP, initial_value=False)

        if self.hook_inverted:
            self.hook.when_pressed = self._handset_lifted
            self.hook.when_released = self._handset_replaced
        else:
            self.hook.when_pressed = self._handset_replaced
            self.hook.when_released = self._handset_lifted
        self.hall.when_pressed = self._hall_edge

    # ---------------- state ----------------

    @property
    def off_hook(self) -> bool:
        """True when the handset has been lifted out of the cradle.

        is_pressed, not value: pull_up=True inverts the reading, so
        .value is 1 when the pin is pulled to GROUND. Reading .value == 0
        here would be backwards and would look like it worked.
        """
        pressed = self.hook.is_pressed
        return pressed if self.hook_inverted else not pressed

    @property
    def amp_enabled(self) -> bool:
        return bool(self.amp.value)

    def set_amp(self, enabled: bool) -> None:
        """Enable or mute the earpiece amplifier."""
        self.amp.value = bool(enabled)

    # ---------------- internal callbacks ----------------

    def _handset_lifted(self) -> None:
        self.crank.reset()
        if self.auto_mute_on_hook:
            self.set_amp(True)
        if self.on_off_hook:
            self.on_off_hook()

    def _handset_replaced(self) -> None:
        self.crank.reset()
        if self.auto_mute_on_hook:
            self.set_amp(False)
        if self.on_on_hook:
            self.on_on_hook()

    def _hall_edge(self) -> None:
        self.hall_pulse_count += 1
        if self.on_hall_pulse:
            self.on_hall_pulse()

        # On-hook cranking is ignored on purpose. Otherwise every visitor,
        # child and cat that spins the handle places a call.
        if self.crank_requires_off_hook and not self.off_hook:
            # Silently dropping these made a backwards hookswitch look
            # like a dead Hall sensor: pulses arrived, nothing happened,
            # and nothing said why.
            self.cranks_dropped_on_hook += 1
            if self.cranks_dropped_on_hook in (1, 5, 20):
                _log.warning(
                    "crank pulse ignored - the handset reads ON-HOOK "
                    "(%d so far). If it is actually lifted, the hookswitch "
                    "sense is inverted: set KELLOGG_HOOK_INVERTED=1 in "
                    ".env, or run with --invert-hook to check.",
                    self.cranks_dropped_on_hook)
            return

        if self.crank.pulse() and self.on_crank:
            self.on_crank()

    # ---------------- lifecycle ----------------

    def close(self) -> None:
        for dev in (self.hook, self.hall, self.spare, self.amp):
            try:
                dev.close()
            except Exception:
                pass

    def __enter__(self) -> "PhoneHardware":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------
def _stamp(start: float) -> str:
    return f"[{time.monotonic() - start:7.2f}s]"


def _run_events() -> None:
    start = time.monotonic()

    def off_hook() -> None:
        print(f"{_stamp(start)} HANDSET LIFTED   amp enabled")

    def on_hook() -> None:
        print(f"{_stamp(start)} HANDSET REPLACED amp muted")

    def cranked() -> None:
        print(f"{_stamp(start)} *** CRANK DETECTED - operator would answer ***")

    def pulse() -> None:
        print(f"{_stamp(start)}   hall pulse")

    with PhoneHardware(
        on_off_hook=off_hook,
        on_on_hook=on_hook,
        on_crank=cranked,
        on_hall_pulse=pulse,
    ) as hw:
        print("hardware.py self test")
        print(f"  hookswitch GPIO {PIN_HOOK}   crank GPIO {PIN_HALL}   "
              f"amp GPIO {PIN_AMP}")
        print(f"  initial state: {'OFF-HOOK' if hw.off_hook else 'ON-HOOK'}")
        print()
        print("  Lift and replace the handset. Turn the crank while off-hook.")
        print("  Every action should print exactly once. Ctrl-C to stop.")
        print()
        try:
            signal.pause()
        except KeyboardInterrupt:
            print(f"\n  total hall pulses: {hw.hall_pulse_count}")
            print("  stopped")


def _run_raw() -> None:
    """Print every Hall edge with the gap since the last one.

    Use this to position the magnet and sensor. Turning the crank at a steady
    speed should give one pulse per revolution with consistent gaps. Missing
    pulses at speed means the magnet is too far from the chip; doubled pulses
    mean the magnet is passing the sensor twice or bouncing.
    """
    last = [0.0]
    start = time.monotonic()

    def pulse() -> None:
        now = time.monotonic()
        gap = now - last[0] if last[0] else 0.0
        last[0] = now
        rpm = 60.0 / gap if gap > 0 else 0.0
        print(f"{_stamp(start)} pulse   gap {gap:6.3f}s   ~{rpm:5.1f} rpm")

    with PhoneHardware(on_hall_pulse=pulse, crank_requires_off_hook=False) as hw:
        print("raw Hall mode - turn the crank steadily. Ctrl-C to stop.")
        print()
        try:
            signal.pause()
        except KeyboardInterrupt:
            print(f"\n  total pulses: {hw.hall_pulse_count}")


def _selftest() -> int:
    """CrankDetector only - no GPIO, no crank, no magnet."""
    checks: list[tuple[bool, str]] = []

    def ok(cond, label):
        checks.append((bool(cond), label))

    # Real timestamps captured from the actual crank with --raw.
    # One pulse per revolution, gaps 0.6-1.6 s.
    real = [2.11, 3.34, 4.98, 8.05, 9.48, 10.30, 10.92, 11.94, 12.97,
            13.58, 14.20, 15.01, 15.63, 16.24, 17.88, 18.91]

    d = CrankDetector()
    fired = [t for t in real if d.pulse(now=t)]
    ok(len(fired) >= 3, "an ordinary crank fires repeatedly")
    ok(fired and fired[0] - real[0] < 4.0,
       "it fires within a few seconds of starting to crank")

    old = CrankDetector(pulses_needed=3, window_s=2.0)
    old_fired = [t for t in real if old.pulse(now=t)]
    ok(len(fired) > len(old_fired),
       "the current threshold is more forgiving than 3-in-2s was")

    ok(not CrankDetector().pulse(now=100.0),
       "one stray pulse never fires")

    d = CrankDetector()
    d.pulse(now=200.0)
    ok(not d.pulse(now=205.0),
       "two pulses far apart are not a crank")

    d = CrankDetector()
    ok(not d.pulse(now=300.0), "first of a pair does not fire")
    ok(d.pulse(now=301.0), "second within the window does fire")
    ok(not d.pulse(now=301.5), "cooldown suppresses an immediate refire")
    ok(not d.pulse(now=306.0),
       "after the cooldown, one pulse alone still does not fire")
    ok(d.pulse(now=307.0),
       "a second pulse after the cooldown fires again")

    d = CrankDetector()
    for t in (400.0, 400.5):
        d.pulse(now=t)
    d.reset()
    ok(not d.pulse(now=400.7), "reset clears the accumulated pulses")

    width = max(len(l) for _, l in checks)
    failed = sum(1 for p, _ in checks if not p)
    for passed, label in checks:
        print(f"  {'ok ' if passed else 'FAIL'}  {label:<{width}}")
    print()
    print(f"all {len(checks)} checks passed" if not failed
          else f"{failed} of {len(checks)} checks failed")
    return 1 if failed else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Kellogg phone GPIO self test")
    ap.add_argument("--raw", action="store_true",
                    help="log every Hall edge, for magnet calibration")
    ap.add_argument("--selftest", action="store_true",
                    help="CrankDetector checks, no hardware")
    ap.add_argument("--invert-hook", action="store_true",
                    help="flip the hookswitch sense for this run")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())
    if args.invert_hook:
        globals()["HOOK_INVERTED"] = True
    _run_raw() if args.raw else _run_events()


if __name__ == "__main__":
    main()