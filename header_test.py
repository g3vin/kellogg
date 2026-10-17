#!/usr/bin/env python3
"""
header_test.py - verify every soldered joint on the Perma-Proto HAT.

Bench utility, not part of the phone. Run it with the HAT seated on the Pi,
then touch a grounded jumper wire into each hole of the labelled breakout
column, working top to bottom. The pin that goes low is the one that hole is
really connected to.

That tests the whole path at once: Pi pad, header socket, solder joint, and
the board trace. A meter only tests the last two.

What the results mean:

    a hole pulls its own pin low      the joint is good
    a hole pulls nothing low          open joint, or the wire is not grounded
    a hole pulls the wrong pin low    the column is offset, recount
    a hole pulls two pins low         solder bridge between those pins

Usage:

    python header_test.py                live test, Ctrl-C for a summary
    python header_test.py --order pin    list in header pin order instead
    python header_test.py selftest       table checks, no GPIO needed
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time

# (breakout label, header pin, BCM) in the order the labels appear down the
# column on the board, so you can work straight down it.
BREAKOUT = [
    ("SDA",   3,  2),
    ("SCL",   5,  3),
    ("TXD",   8, 14),
    ("RXD",  10, 15),
    ("#4",    7,  4),
    ("#17",  11, 17),
    ("#18",  12, 18),
    ("#27",  13, 27),
    ("#22",  15, 22),
    ("#23",  16, 23),
    ("#24",  18, 24),
    ("#25",  22, 25),
    ("MOSI", 19, 10),
    ("MISO", 21,  9),
    ("CLK",  23, 11),
    ("CE0",  24,  8),
    ("CE1",  26,  7),
    ("#5",   29,  5),
    ("#6",   31,  6),
    ("#12",  32, 12),
    ("#13",  33, 13),
    ("#16",  36, 16),
    ("#19",  35, 19),
    ("#20",  38, 20),
    ("#21",  40, 21),
]

# Pins with no trace on this board, so nothing to test electrically.
NO_BREAKOUT = [
    (27, "ID_SD", "HAT EEPROM line, no breakout"),
    (28, "ID_SC", "HAT EEPROM line, no breakout"),
    (37, "GPIO 26", "did not make the cut on this board"),
]

POLL_S = 0.04
RAIL_NOTE = """Rails are not covered here. Check those with the meter:
  pins 1, 17  -> +3V rail      pins 2, 4 -> +5V rail
  pins 6, 9, 14, 20, 25, 30, 34, 39 -> either GND rail"""


class Row:
    __slots__ = ("label", "pin", "bcm", "dev", "error", "verified",
                 "grounded", "stuck")

    def __init__(self, label: str, pin: int, bcm: int) -> None:
        self.label = label
        self.pin = pin
        self.bcm = bcm
        self.dev = None
        self.error: str | None = None
        self.verified = False
        self.grounded = False
        self.stuck = False

    @property
    def state(self) -> str:
        if self.error:
            return "n/a"
        if self.stuck:
            return "STUCK"
        if self.grounded:
            return "<<<"
        return "ok" if self.verified else "--"


def claim(rows: list[Row]) -> None:
    """Open every pin as an input with the pull-up enabled."""
    from gpiozero import DigitalInputDevice
    from gpiozero.exc import GPIOZeroError

    for r in rows:
        try:
            r.dev = DigitalInputDevice(r.bcm, pull_up=True)
        except GPIOZeroError as exc:
            r.error = type(exc).__name__
        except Exception as exc:  # a busy pin can raise from the backend
            r.error = type(exc).__name__


def release(rows: list[Row]) -> None:
    for r in rows:
        if r.dev is not None:
            try:
                r.dev.close()
            except Exception:
                pass


def baseline(rows: list[Row]) -> list[Row]:
    """Read every pin before anything is touched.

    With the pull-up on and nothing connected, every pin should read high.
    Any pin already reading grounded is either shorted to a ground rail or
    the backend is not applying the pull-up.
    """
    time.sleep(0.15)
    stuck = []
    for r in rows:
        if r.dev is not None and r.dev.is_active:
            r.stuck = True
            stuck.append(r)
    return stuck


def render(rows: list[Row], warning: str = "") -> list[str]:
    """Two columns of rows, plus a count and any warning."""
    done = sum(1 for r in rows if r.verified)
    testable = sum(1 for r in rows if not r.error)
    out = [
        f"  HEADER JOINT TEST        verified {done} of {testable}",
        "",
        "  hole   pin  bcm  state      hole   pin  bcm  state",
        "  " + "-" * 25 + "  " + "-" * 25,
    ]
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]
    for i in range(half):
        a = left[i]
        cell_a = f"  {a.label:<5} {a.pin:>4} {a.bcm:>4}  {a.state:<6}"
        if i < len(right):
            b = right[i]
            cell_b = f"  {b.label:<5} {b.pin:>4} {b.bcm:>4}  {b.state:<6}"
        else:
            cell_b = ""
        out.append(cell_a + " " + cell_b)
    out.append("")
    out.append(f"  {warning}" if warning else "  touch the grounded wire into"
                                              " each hole, top to bottom")
    out.append("  Ctrl-C for a summary")
    return out


def live(rows: list[Row]) -> int:
    claim(rows)

    blocked = [r for r in rows if r.error]
    if blocked:
        print("\n  Could not open some pins. They are probably in use by an")
        print("  interface enabled in raspi-config:\n")
        for r in blocked:
            print(f"    {r.label:<5} BCM {r.bcm:<3} {r.error}")
        print("\n  Serial console claims TXD and RXD; I2C claims SDA and SCL;")
        print("  SPI claims MOSI, MISO, CLK, CE0 and CE1. Turn the interface")
        print("  off in raspi-config to test those joints, then turn it back on.\n")

    stuck = baseline(rows)
    if len(stuck) > 3:
        print("\n  Every pin reads grounded before anything was touched.")
        print("  That is not 25 bad joints. Either the pull-ups are not being")
        print("  applied by the GPIO backend, or something is holding the whole")
        print("  header low. Try:\n")
        print("    sudo apt install python3-lgpio")
        print("    GPIOZERO_PIN_FACTORY=lgpio python header_test.py\n")
        release(rows)
        return 1
    if stuck:
        print("\n  These pins already read grounded with nothing connected.")
        print("  Check them for a solder bridge to a ground rail:\n")
        for r in stuck:
            print(f"    {r.label:<5} header pin {r.pin:<3} BCM {r.bcm}")
        print()

    lines = render(rows)
    print("\n".join(lines))
    height = len(lines)

    try:
        while True:
            time.sleep(POLL_S)
            touched = []
            for r in rows:
                if r.dev is None or r.stuck:
                    continue
                # pull_up=True inverts the logic: is_active is True when the
                # pin is pulled down to ground, which is what the jumper does.
                r.grounded = r.dev.is_active
                if r.grounded:
                    touched.append(r)
                    r.verified = True

            warning = ""
            if len(touched) > 1:
                names = ", ".join(x.label for x in touched[:6])
                if len(touched) > 6:
                    names += f" +{len(touched) - 6} more"
                warning = f"{len(touched)} PINS AT ONCE: {names}  bridge?"

            width = max(40, shutil.get_terminal_size((80, 24)).columns - 1)
            lines = [ln[:width] for ln in render(rows, warning)]
            sys.stdout.write(f"\033[{height}A")
            sys.stdout.write("\n".join(f"\033[2K{ln}" for ln in lines) + "\n")
            sys.stdout.flush()
            height = len(lines)
    except KeyboardInterrupt:
        pass
    finally:
        release(rows)

    missing = [r for r in rows if not r.verified and not r.error
               and not r.stuck]
    skipped = [r for r in rows if r.error]

    print()
    if not missing and not skipped:
        print("  All joints verified. Every hole pulled its own pin low.")
    if missing:
        print(f"  {len(missing)} hole(s) never responded:\n")
        for r in missing:
            print(f"    {r.label:<5} header pin {r.pin:<3} BCM {r.bcm}")
        print("\n  Either the joint is open, or the jumper was not actually")
        print("  grounded. Confirm the wire first by touching it to a hole")
        print("  that already passed, then reflow the ones that still fail.")
    if skipped:
        print(f"\n  {len(skipped)} pin(s) not testable while their interface"
              " is enabled.")

    print("\n  Not covered, no trace on this board:")
    for pin, name, why in NO_BREAKOUT:
        print(f"    header pin {pin:<3} {name:<8} {why}")
    print()
    print("  " + RAIL_NOTE.replace("\n", "\n  "))
    print()
    return 1 if missing else 0


def selftest() -> int:
    fails = 0
    seen_pins: dict[int, str] = {}
    seen_bcm: dict[int, str] = {}

    if len(BREAKOUT) != 25:
        fails += 1
        print(f"  FAIL expected 25 breakout entries, have {len(BREAKOUT)}")
    else:
        print("  ok   25 breakout entries")

    for label, pin, bcm in BREAKOUT:
        if not 1 <= pin <= 40:
            fails += 1
            print(f"  FAIL {label}: header pin {pin} out of range")
        if pin in seen_pins:
            fails += 1
            print(f"  FAIL header pin {pin} used by {seen_pins[pin]} and {label}")
        if bcm in seen_bcm:
            fails += 1
            print(f"  FAIL BCM {bcm} used by {seen_bcm[bcm]} and {label}")
        seen_pins[pin] = label
        seen_bcm[bcm] = label
    if not fails:
        print("  ok   no duplicate header pins or BCM numbers")

    # Breakout pins are all on the GPIO side, never power or ground.
    power_ground = {1, 2, 4, 6, 9, 14, 17, 20, 25, 30, 34, 39}
    clash = power_ground & set(seen_pins)
    if clash:
        fails += 1
        print(f"  FAIL power/ground pins listed as breakouts: {sorted(clash)}")
    else:
        print("  ok   no power or ground pins in the breakout list")

    # Every header pin is accounted for somewhere.
    orphan = {p for p, _, _ in NO_BREAKOUT}
    covered = set(seen_pins) | power_ground | orphan
    if covered != set(range(1, 41)):
        fails += 1
        print(f"  FAIL unaccounted header pins: "
              f"{sorted(set(range(1, 41)) - covered)}")
    else:
        print("  ok   all 40 header pins accounted for")

    # Spot-check a few mappings against the Pi pinout.
    expect = {"#17": (11, 17), "#22": (15, 22), "#16": (36, 16),
              "MOSI": (19, 10), "CE1": (26, 7), "#19": (35, 19)}
    for label, (pin, bcm) in expect.items():
        got = next(((p, b) for lb, p, b in BREAKOUT if lb == label), None)
        if got != (pin, bcm):
            fails += 1
            print(f"  FAIL {label} maps to {got}, expected {(pin, bcm)}")
    if not fails:
        print("  ok   spot-checked mappings match the Pi pinout")

    rows = [Row(lb, p, b) for lb, p, b in BREAKOUT]
    rows[0].verified = True
    rows[1].error = "Busy"
    body = render(rows, "test")
    if len(body) < 10 or "verified 1 of 24" not in body[0]:
        fails += 1
        print(f"  FAIL render header: {body[0]!r}")
    else:
        print("  ok   render counts testable pins and skips blocked ones")

    print()
    print(f"{fails} failure(s)" if fails else "all checks passed")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Perma-Proto HAT joint test")
    ap.add_argument("--order", choices=("column", "pin"), default="column",
                    help="column: the order the labels run down the board "
                         "(default). pin: header pin order.")
    ap.add_argument("cmd", nargs="?", choices=("selftest",))
    args = ap.parse_args()

    if args.cmd == "selftest":
        return selftest()

    entries = sorted(BREAKOUT, key=lambda e: e[1]) if args.order == "pin" \
        else BREAKOUT
    rows = [Row(lb, p, b) for lb, p, b in entries]

    try:
        import gpiozero  # noqa: F401
    except ImportError:
        print("gpiozero not found.\n"
              "  sudo apt install python3-gpiozero python3-lgpio\n"
              "and create the venv with --system-site-packages.")
        return 1

    print()
    print("  Seat the HAT on the Pi. Put one end of a jumper in any GND pin,")
    print("  then touch the other end into each hole of the labelled column.")
    print()
    return live(rows)


if __name__ == "__main__":
    sys.exit(main())