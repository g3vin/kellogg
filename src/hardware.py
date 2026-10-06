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
PIN_HOOK = 17
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
CRANK_PULSES = 3
CRANK_WINDOW_S = 2.0
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
        auto_mute_on_hook: bool = True,
    ) -> None:
        self.on_off_hook = on_off_hook
        self.on_on_hook = on_on_hook
        self.on_crank = on_crank
        self.on_hall_pulse = on_hall_pulse
        self.crank_requires_off_hook = crank_requires_off_hook
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

        self.hook.when_pressed = self._handset_replaced
        self.hook.when_released = self._handset_lifted
        self.hall.when_pressed = self._hall_edge

    # ---------------- state ----------------

    @property
    def off_hook(self) -> bool:
        """True when the handset has been lifted out of the cradle."""
        return not self.hook.is_pressed

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Kellogg phone GPIO self test")
    ap.add_argument("--raw", action="store_true",
                    help="log every Hall edge, for magnet calibration")
    args = ap.parse_args()
    _run_raw() if args.raw else _run_events()


if __name__ == "__main__":
    main()