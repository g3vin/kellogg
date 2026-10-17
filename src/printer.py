#!/usr/bin/env python3
"""
printer.py - the thermal printer in the cabinet.

Prints incoming messages on the 58 mm embedded module, reliably, including
when the printer is unplugged, out of paper, or a message arrives mid-call.

Three things shape this module:

  * The print head is 384 dots wide at 8 dots/mm, so the printable area is
    48 mm, NOT the 58 mm paper width. That is 32 characters in Font A and 42
    in Font B. Everything wraps to 32 by default.

  * The printer peaks at 2.5 A on the 9 V rail. Printing during a live call
    puts that spike next to the microphone preamp, so main.py queues instead.
    The queue lives here; the decision to use it lives there.

  * These modules sometimes enumerate on a non-default USB endpoint, and
    python-escpos then fails silently - it accepts the bytes and nothing
    comes out. detect() finds the real endpoints rather than assuming.

Two backends. When the kernel's usblp driver has bound the printer it
appears as /dev/usb/lp0 and holds the USB interface exclusively, so raw
pyusb access is impossible; we write ESC/POS straight to the node. That is
the normal case on this build and needs no udev rule, only membership of
group lp. Raw USB is the fallback for when no such node exists.

Nothing is ever dropped. A queued message is removed only after the print
succeeds, so a power cut mid-print costs you a duplicate, never a loss.

Standalone use:

    python src/printer.py selftest          formatting + queue, no hardware
    python src/printer.py detect            scan USB, report ids + endpoints
    python src/printer.py test              print a real test page
    python src/printer.py message "Gavin" "hello there"
    python src/printer.py image photo.jpg
    python src/printer.py queue             show pending
    python src/printer.py flush             print everything pending
    python src/printer.py preview "Gavin" "hi"    render to stdout, no printer
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import textwrap
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
QUEUE_PATH = ROOT / "print_queue.json"

# Print head geometry. These are not adjustable - they are the hardware.
DOTS_PER_LINE = 384
DOTS_PER_MM = 8
PRINT_WIDTH_MM = DOTS_PER_LINE // DOTS_PER_MM      # 48 mm
CHARS_FONT_A = 32
CHARS_FONT_B = 42

# python-escpos defaults. The whole point of detect() is that these are
# frequently wrong on these cheap modules.
DEFAULT_IN_EP = 0x82
DEFAULT_OUT_EP = 0x01

FEED_AFTER_PRINT = 4        # blank lines so the text clears the tear bar
MAX_ATTEMPTS = 5            # before a queued message is parked, not dropped

RULE = "-" * CHARS_FONT_A
HEAVY = "=" * CHARS_FONT_A

# Where the kernel puts a printer it has bound with the usblp driver.
USBLP_GLOB = "/dev/usb/lp*"


class PyusbMissing(RuntimeError):
    """pyusb is absent, so the raw USB backend cannot even be attempted.

    This exists so the absence is reported rather than silently becoming
    "no printer found", which is a different and much more alarming
    diagnosis.
    """


# --------------------------------------------------------------------------
# text layout
# --------------------------------------------------------------------------
def wrap(text: str, width: int = CHARS_FONT_A) -> list[str]:
    """Wrap to the printable width, preserving deliberate blank lines.

    textwrap alone collapses paragraph breaks, which matters when someone
    sends a two-paragraph text.
    """
    out: list[str] = []
    for para in str(text).split("\n"):
        if not para.strip():
            out.append("")
            continue
        out.extend(textwrap.wrap(
            para, width=width,
            break_long_words=True,     # a URL must not run off the paper
            break_on_hyphens=False,
        ) or [""])
    return out


def centre(text: str, width: int = CHARS_FONT_A) -> str:
    text = text[:width]
    pad = (width - len(text)) // 2
    return " " * pad + text


def format_message(sender: str, body: str,
                   when: Optional[datetime] = None,
                   width: int = CHARS_FONT_A) -> str:
    """The telegram-style layout, as plain text at the printable width."""
    when = when or datetime.now()
    stamp = when.strftime("%a %d %b   %-I:%M %p") if os.name != "nt" \
        else when.strftime("%a %d %b   %I:%M %p")

    lines = [
        HEAVY[:width],
        centre("KELLOGG EXCHANGE", width),
        HEAVY[:width],
        "",
        f"FROM  {sender}"[:width],
        f"RECD  {stamp}"[:width],
        RULE[:width],
        "",
    ]
    lines.extend(wrap(body, width))
    lines.extend(["", RULE[:width]])
    return "\n".join(lines)


# --------------------------------------------------------------------------
# device discovery
# --------------------------------------------------------------------------
@dataclass
class PrinterConfig:
    """Where and how to reach the printer.

    Two backends, and the choice is not ours to make. When the kernel's
    usblp driver binds the printer it claims the USB interface
    EXCLUSIVELY, so pyusb cannot have it. In that case the device turns up
    as /dev/usb/lp0 and we write ESC/POS bytes straight to the file. Only
    when no such node exists do we fall back to raw USB.
    """
    vendor_id: int = 0
    product_id: int = 0
    in_ep: int = DEFAULT_IN_EP
    out_ep: int = DEFAULT_OUT_EP
    profile: Optional[str] = None
    note: str = ""
    backend: str = "usb"          # "usblp" | "usb"
    device_path: str = ""         # /dev/usb/lp0, when backend is usblp

    def __str__(self) -> str:
        if self.backend == "usblp":
            ident = ""
            if self.vendor_id or self.product_id:
                ident = f" {self.vendor_id:04x}:{self.product_id:04x}"
            return (f"{self.device_path}{ident}"
                    + (f"  ({self.note})" if self.note else ""))
        return (f"{self.vendor_id:04x}:{self.product_id:04x} "
                f"in_ep=0x{self.in_ep:02x} out_ep=0x{self.out_ep:02x}"
                + (f"  ({self.note})" if self.note else ""))

    @property
    def writable(self) -> bool:
        """usblp only: can this process actually write to the node?"""
        if self.backend != "usblp":
            return True
        return os.access(self.device_path, os.W_OK)

    @classmethod
    def scan_usblp(cls) -> list["PrinterConfig"]:
        """Printers the kernel has already bound as line printers.

        Reads the ids out of sysfs when it can, purely so the listing is
        recognisable. Their absence is not an error.
        """
        found: list[PrinterConfig] = []
        for node in sorted(glob.glob(USBLP_GLOB)):
            vid = pid = 0
            name = os.path.basename(node)
            sysfs = Path(f"/sys/class/usbmisc/{name}/device/../")
            try:
                vid = int((sysfs / "idVendor").read_text().strip(), 16)
                pid = int((sysfs / "idProduct").read_text().strip(), 16)
            except Exception:
                pass
            found.append(cls(vendor_id=vid, product_id=pid,
                             backend="usblp", device_path=node,
                             note="kernel usblp"))
        return found

    @classmethod
    def scan(cls) -> list["PrinterConfig"]:
        """Every USB device that could plausibly be the printer.

        Looks for the USB printer class (7) first, then vendor-specific
        interfaces, because these modules use both. Reads the real endpoint
        addresses off the descriptor instead of assuming the defaults.
        """
        try:
            import usb.core
            import usb.util
        except ImportError:
            # Returning [] here would surface as "no printer found", which
            # is a lie and sent a real debugging session down the wrong
            # path. Say what is actually wrong.
            raise PyusbMissing(
                "pyusb is not installed, so raw USB printers cannot be "
                "found. Install it with:  sudo apt install python3-usb"
            ) from None

        found: list[PrinterConfig] = []
        try:
            devices = list(usb.core.find(find_all=True))
        except Exception:
            return []

        for dev in devices:
            try:
                for cfg in dev:
                    for intf in cfg:
                        klass = intf.bInterfaceClass
                        if klass not in (7, 0xFF):      # printer, vendor
                            continue
                        out = usb.util.find_descriptor(
                            intf, custom_match=lambda e:
                            usb.util.endpoint_direction(e.bEndpointAddress)
                            == usb.util.ENDPOINT_OUT)
                        inp = usb.util.find_descriptor(
                            intf, custom_match=lambda e:
                            usb.util.endpoint_direction(e.bEndpointAddress)
                            == usb.util.ENDPOINT_IN)
                        if out is None:
                            continue
                        found.append(cls(
                            vendor_id=dev.idVendor,
                            product_id=dev.idProduct,
                            out_ep=out.bEndpointAddress,
                            in_ep=(inp.bEndpointAddress if inp
                                   else DEFAULT_IN_EP),
                            note=("printer class" if klass == 7
                                  else "vendor specific"),
                        ))
            except Exception:
                continue
        return found

    @classmethod
    def detect(cls) -> Optional["PrinterConfig"]:
        """Best guess at the printer.

        /dev/usb/lp* wins when it exists. The kernel holds the interface
        in that case, so a raw USB attempt would fail anyway, and writing
        to the node needs no endpoint detection and no udev rule.
        """
        lp = cls.scan_usblp()
        if lp:
            return lp[0]

        try:
            candidates = cls.scan()
        except PyusbMissing:
            # No kernel node and no pyusb: nothing can be opened, but the
            # caller deserves the real reason rather than a bare None.
            raise

        if not candidates:
            return None
        for c in candidates:
            if c.note == "printer class":
                return c
        return candidates[0]


# --------------------------------------------------------------------------
# the queue
# --------------------------------------------------------------------------
@dataclass
class QueuedMessage:
    id: str
    sender: str
    body: str
    when: str
    media: list[str] = field(default_factory=list)
    attempts: int = 0
    error: str = ""

    @property
    def timestamp(self) -> datetime:
        try:
            return datetime.fromisoformat(self.when)
        except ValueError:
            return datetime.now()

    @property
    def parked(self) -> bool:
        return self.attempts >= MAX_ATTEMPTS


class MessageQueue:
    """Messages waiting to print, persisted so a power cut cannot lose one.

    Written with a temp file and os.replace so a cut mid-write leaves the
    previous queue intact rather than a half-written file.
    """

    def __init__(self, path: Path | str = QUEUE_PATH) -> None:
        self.path = Path(path)
        self._items: list[QueuedMessage] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._items = []
            return
        try:
            raw = json.loads(self.path.read_text())
            self._items = [QueuedMessage(**r) for r in raw]
        except Exception:
            # A corrupt queue is kept aside rather than silently discarded.
            try:
                self.path.rename(self.path.with_suffix(".corrupt"))
            except Exception:
                pass
            self._items = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump([asdict(i) for i in self._items], fh, indent=1)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    def add(self, sender: str, body: str,
            media: Optional[Sequence[str]] = None,
            when: Optional[datetime] = None) -> QueuedMessage:
        msg = QueuedMessage(
            id=uuid.uuid4().hex[:12],
            sender=sender,
            body=body,
            when=(when or datetime.now()).isoformat(timespec="seconds"),
            media=list(media or []),
        )
        self._items.append(msg)
        self.save()
        return msg

    def remove(self, msg_id: str) -> bool:
        before = len(self._items)
        self._items = [i for i in self._items if i.id != msg_id]
        if len(self._items) != before:
            self.save()
            return True
        return False

    def pending(self, include_parked: bool = False) -> list[QueuedMessage]:
        if include_parked:
            return list(self._items)
        return [i for i in self._items if not i.parked]

    def all(self) -> list[QueuedMessage]:
        return list(self._items)

    def mark_failed(self, msg_id: str, error: str) -> None:
        for i in self._items:
            if i.id == msg_id:
                i.attempts += 1
                i.error = error[:200]
        self.save()

    def clear(self) -> int:
        n = len(self._items)
        self._items = []
        self.save()
        return n

    def __len__(self) -> int:
        return len(self._items)


# --------------------------------------------------------------------------
# the printer
# --------------------------------------------------------------------------
class Printer:
    def __init__(self, config: Optional[PrinterConfig] = None,
                 dummy: bool = False,
                 queue_path: Path | str = QUEUE_PATH,
                 width: int = CHARS_FONT_A) -> None:
        self.config = config
        self.dummy = dummy
        self.width = width
        self.queue = MessageQueue(queue_path)
        self._dev = None
        self._last_error = ""

    # ---------------- connection ----------------

    def _apply_profile(self) -> None:
        """Tell escpos the real head geometry.

        Without this it does not know the paper width, so centring silently
        does nothing and oversized images are sent rather than rejected.
        """
        try:
            media = self._dev.profile.profile_data["media"]["width"]
            media["pixels"] = DOTS_PER_LINE
            media["mm"] = PRINT_WIDTH_MM
        except Exception:
            pass

    def open(self) -> bool:
        """Open the device, detecting it if no config was supplied."""
        if self._dev is not None:
            return True

        if self.dummy:
            from escpos.printer import Dummy
            self._dev = Dummy()
            self._apply_profile()
            return True

        if self.config is None:
            try:
                self.config = PrinterConfig.detect()
            except PyusbMissing as exc:
                self._last_error = str(exc)
                return False
        if self.config is None:
            self._last_error = (
                "no printer found: nothing at /dev/usb/lp* and no USB "
                "printer-class device. Is the 9 V supply connected?")
            return False

        if self.config.backend == "usblp":
            return self._open_usblp()
        return self._open_rawusb()

    def _open_usblp(self) -> bool:
        """Write ESC/POS straight to the kernel's line-printer node."""
        path = self.config.device_path
        if not os.path.exists(path):
            self._last_error = f"{path} has gone away; is it unplugged?"
            return False
        if not os.access(path, os.W_OK):
            self._last_error = (
                f"{path} is not writable by this user. Fix it with:\n"
                f"    sudo usermod -aG lp $USER\n"
                f"then log out and back in - group changes only apply to "
                f"new sessions.")
            return False

        try:
            from escpos.printer import File
            # auto_flush so a message is on paper before we mark it sent;
            # the queue's whole contract depends on that ordering.
            self._dev = File(path, auto_flush=True)
            self._apply_profile()
            return True
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._dev = None
            return False

    def _open_rawusb(self) -> bool:
        try:
            import usb.core  # noqa: F401
        except ImportError:
            self._last_error = (
                "pyusb is not installed, so the raw USB backend is "
                "unavailable. Install it with:  "
                "sudo apt install python3-usb")
            return False

        try:
            from escpos.printer import Usb
            self._dev = Usb(
                self.config.vendor_id,
                self.config.product_id,
                in_ep=self.config.in_ep,
                out_ep=self.config.out_ep,
            )
            self._apply_profile()
            return True
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._dev = None
            return False

    def close(self) -> None:
        if self._dev is not None:
            try:
                self._dev.close()
            except Exception:
                pass
            self._dev = None

    @property
    def available(self) -> bool:
        return self.open()

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def output(self) -> bytes:
        """Captured ESC/POS bytes. Dummy mode only."""
        if self.dummy and self._dev is not None:
            return self._dev.output
        return b""

    def _reset(self) -> None:
        """Drop the handle so the next call re-opens it.

        A printer that was unplugged leaves a dead handle behind; without
        this, every later print fails even after it is plugged back in.
        """
        self.close()

    # ---------------- primitives ----------------

    def feed(self, lines: int = FEED_AFTER_PRINT) -> bool:
        if not self.open():
            return False
        try:
            self._dev.ln(lines)
            return True
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._reset()
            return False

    def print_text(self, text: str, header: Optional[str] = None,
                   feed: int = FEED_AFTER_PRINT) -> bool:
        """Print plain text, wrapped to the printable width."""
        if not self.open():
            return False
        try:
            if header:
                self._dev.set(align="center", bold=True)
                self._dev.text(centre(header, self.width) + "\n")
                self._dev.set(align="left", bold=False)
            for line in wrap(text, self.width):
                self._dev.text(line + "\n")
            if feed:
                self._dev.ln(feed)
            return True
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._reset()
            return False

    def print_message(self, sender: str, body: str,
                      when: Optional[datetime] = None,
                      media: Optional[Sequence[str]] = None) -> bool:
        """Print one message in the telegram layout.

        On failure the caller should enqueue. deliver() does that for you.
        """
        if not self.open():
            return False
        try:
            self._dev.set(align="left", bold=False)
            for line in format_message(sender, body, when,
                                       self.width).split("\n"):
                self._dev.text(line + "\n")
            for path in (media or []):
                if Path(path).exists():
                    self._dev.ln(1)
                    self.print_image(path, feed=0)
            self._dev.ln(FEED_AFTER_PRINT)
            return True
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._reset()
            return False

    def print_image(self, source, max_width: int = DOTS_PER_LINE,
                    feed: int = FEED_AFTER_PRINT) -> bool:
        """Print an image, dithered to 1 bit at the head's width.

        Pillow's convert("1") is Floyd-Steinberg, which is what makes a photo
        read as a 1980s wirephoto rather than a black smear.
        """
        if not self.open():
            return False
        try:
            from PIL import Image, ImageOps

            img = source if hasattr(source, "convert") else Image.open(source)
            img = ImageOps.exif_transpose(img)
            if img.mode not in ("L", "1"):
                img = img.convert("L")
            if img.width > max_width:
                h = max(1, round(img.height * max_width / img.width))
                img = img.resize((max_width, h), Image.LANCZOS)
            img = ImageOps.autocontrast(img.convert("L")).convert("1")
            if img.width > DOTS_PER_LINE:
                img = img.resize(
                    (DOTS_PER_LINE,
                     max(1, round(img.height * DOTS_PER_LINE / img.width))))

            self._dev.image(img, impl="bitImageRaster", center=True)
            if feed:
                self._dev.ln(feed)
            return True
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._reset()
            return False

    def selftest_page(self) -> bool:
        """A page that proves the whole path, with a width ruler."""
        if not self.open():
            return False
        try:
            self._dev.set(align="center", bold=True)
            self._dev.text(centre("KELLOGG EXCHANGE", self.width) + "\n")
            self._dev.set(align="left", bold=False)
            self._dev.text(centre("printer self test", self.width) + "\n")
            self._dev.text(HEAVY[:self.width] + "\n")
            self._dev.text(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            cfg = str(self.config) if self.config else "dummy"
            for line in wrap(f"device {cfg}", self.width):
                self._dev.text(line + "\n")
            self._dev.text(RULE[:self.width] + "\n")
            self._dev.text("width ruler, should end flush:\n")
            ruler = "".join(str(i % 10) for i in range(1, self.width + 1))
            self._dev.text(ruler + "\n")
            self._dev.text("1234567890" * (self.width // 10) + "\n")
            self._dev.text(RULE[:self.width] + "\n")
            self._dev.text("If the ruler wraps onto a second line the\n")
            self._dev.text("width is wrong for this printer.\n")
            self._dev.ln(FEED_AFTER_PRINT)
            return True
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._reset()
            return False

    # ---------------- queue-aware delivery ----------------

    def deliver(self, sender: str, body: str,
                media: Optional[Sequence[str]] = None,
                when: Optional[datetime] = None,
                force_queue: bool = False) -> tuple[bool, str]:
        """Print now if possible, otherwise queue. Never drops the message.

        force_queue is what main.py passes during a live call, so a 2.5 A
        print spike never lands next to the microphone preamp.

        Returns (printed_now, message_id_or_empty).
        """
        if force_queue or not self.available:
            msg = self.queue.add(sender, body, media, when)
            return False, msg.id
        if self.print_message(sender, body, when, media):
            return True, ""
        msg = self.queue.add(sender, body, media, when)
        self.queue.mark_failed(msg.id, self._last_error)
        return False, msg.id

    def flush(self, limit: Optional[int] = None) -> tuple[int, int]:
        """Print everything pending. Returns (printed, still_waiting).

        A message is removed only after the print succeeds. A power cut
        mid-flush costs a duplicate, never a lost message.
        """
        pending = self.queue.pending()
        if limit is not None:
            pending = pending[:limit]
        printed = 0
        for msg in pending:
            if not self.available:
                break
            ok = self.print_message(msg.sender, msg.body, msg.timestamp,
                                    msg.media)
            if ok:
                self.queue.remove(msg.id)
                printed += 1
                time.sleep(0.3)      # let the head and the rail recover
            else:
                self.queue.mark_failed(msg.id, self._last_error)
        return printed, len(self.queue.pending())

    def __enter__(self) -> "Printer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _cmd_detect() -> int:
    lp = PrinterConfig.scan_usblp()
    if lp:
        print(f"  Kernel line printer{'s' if len(lp) > 1 else ''}:\n")
        for c in lp:
            mark = "" if c.writable else "   NOT WRITABLE BY YOU"
            print(f"    {c}{mark}")
        best = lp[0]
        print(f"\n  Would use: {best}")
        if not best.writable:
            print("\n  You cannot write to it yet:")
            print("      sudo usermod -aG lp $USER")
            print("  then log out and back in. Group changes only apply to")
            print("  new sessions, so reconnecting your SSH session is not")
            print("  enough on its own if it was already open.")
            return 1
        print("\n  The kernel holds this interface, so pyusb cannot have it.")
        print("  That is normal and the usblp backend is the better path.")
        return 0

    print("  Nothing at /dev/usb/lp*. Looking for a raw USB device.\n")
    try:
        found = PrinterConfig.scan()
    except PyusbMissing as exc:
        print(f"  {exc}")
        return 1

    if not found:
        print("  No printer found.\n")
        print("  Check in order:")
        print("    1. the 9 V supply is connected (USB alone will not do it)")
        print("       and the barrel polarity is right - reversed reads -9 V")
        print("    2. lsusb lists the device")
        print("    3. the USB cable carries data, not just power")
        return 1

    print(f"  {len(found)} candidate device(s):\n")
    for c in found:
        print(f"    {c}")
    best = PrinterConfig.detect()
    print(f"\n  Would use: {best}")
    print("\n  If printing enumerates but nothing comes out, the out_ep above")
    print("  is the value python-escpos needs and does not guess correctly.")
    return 0


def _cmd_queue(p: Printer) -> int:
    items = p.queue.all()
    if not items:
        print("  Queue is empty.")
        return 0
    print(f"  {len(items)} message(s) waiting:\n")
    for m in items:
        flag = "  PARKED" if m.parked else ""
        body = m.body.replace("\n", " ")
        print(f"    [{m.id}] {m.timestamp:%d %b %H:%M}  {m.sender}{flag}")
        print(f"             {body[:52]}")
        if m.error:
            print(f"             last error: {m.error[:52]}")
    return 0


def _selftest() -> int:
    """Everything that needs no printer: layout, queue, dithering."""
    fails = 0

    # --- width ---
    if PRINT_WIDTH_MM != 48:
        fails += 1
        print(f"  FAIL printable width {PRINT_WIDTH_MM} mm, expected 48")
    else:
        print("  ok   384 dots at 8 dots/mm = 48 mm printable")

    # --- wrapping ---
    long = "The quick brown fox jumps over the lazy dog and keeps on going"
    lines = wrap(long)
    if max(len(l) for l in lines) > CHARS_FONT_A:
        fails += 1
        print(f"  FAIL wrap produced a {max(len(l) for l in lines)} char line")
    else:
        print(f"  ok   wrapped to {CHARS_FONT_A} cols in {len(lines)} lines")

    url = "https://example.com/" + "x" * 80
    if max(len(l) for l in wrap(url)) > CHARS_FONT_A:
        fails += 1
        print("  FAIL a long unbroken token ran off the paper")
    else:
        print("  ok   long unbroken token is broken, not overflowed")

    if wrap("a\n\nb") != ["a", "", "b"]:
        fails += 1
        print(f"  FAIL blank line not preserved: {wrap('a\\n\\nb')}")
    else:
        print("  ok   paragraph breaks preserved")

    # --- message layout ---
    when = datetime(2026, 10, 11, 13, 23)
    body = format_message("Gavin", "hello there, this is a test message "
                                   "that is long enough to wrap", when)
    over = [l for l in body.split("\n") if len(l) > CHARS_FONT_A]
    if over:
        fails += 1
        print(f"  FAIL message line too wide: {over[0]!r}")
    else:
        print("  ok   message layout fits the printable width")
    if "Gavin" not in body or "FROM" not in body:
        fails += 1
        print("  FAIL sender missing from layout")

    # --- queue persistence ---
    tmpdir = Path(tempfile.mkdtemp())
    qp = tmpdir / "q.json"
    q = MessageQueue(qp)
    a = q.add("Gavin", "first")
    q.add("Maggie", "second")
    if len(q) != 2:
        fails += 1
        print("  FAIL queue did not hold two messages")
    q2 = MessageQueue(qp)          # reopen, as if after a reboot
    if len(q2) != 2 or q2.all()[0].sender != "Gavin":
        fails += 1
        print("  FAIL queue did not survive a reload")
    else:
        print("  ok   queue persists across a reload")
    q2.remove(a.id)
    if len(MessageQueue(qp)) != 1:
        fails += 1
        print("  FAIL remove did not persist")
    else:
        print("  ok   remove persists")

    # a corrupt queue is set aside, not silently dropped
    qp.write_text("{ not json")
    q3 = MessageQueue(qp)
    if len(q3) != 0 or not qp.with_suffix(".corrupt").exists():
        fails += 1
        print("  FAIL corrupt queue not set aside")
    else:
        print("  ok   corrupt queue moved aside, not discarded")

    # --- parking ---
    qp2 = tmpdir / "q2.json"
    q4 = MessageQueue(qp2)
    m = q4.add("X", "y")
    for _ in range(MAX_ATTEMPTS):
        q4.mark_failed(m.id, "boom")
    if q4.pending() or not q4.pending(include_parked=True):
        fails += 1
        print("  FAIL parking after repeated failures is wrong")
    else:
        print(f"  ok   parked after {MAX_ATTEMPTS} attempts, still retained")

    # --- dummy printer end to end ---
    try:
        p = Printer(dummy=True, queue_path=tmpdir / "q3.json")
        if not p.available:
            fails += 1
            print("  FAIL dummy printer not available")
        p.print_message("Gavin", "hello there", when)
        out = p.output
        if b"Gavin" not in out:
            fails += 1
            print("  FAIL dummy output missing the sender")
        elif len(out) < 100:
            fails += 1
            print(f"  FAIL dummy output suspiciously short: {len(out)}")
        else:
            print(f"  ok   dummy printer captured {len(out)} ESC/POS bytes")

        p.selftest_page()
        if b"ruler" not in p.output:
            fails += 1
            print("  FAIL self test page missing the ruler")
        else:
            print("  ok   self test page renders")
    except Exception as exc:
        fails += 1
        print(f"  FAIL dummy printer raised {type(exc).__name__}: {exc}")

    # --- image dithering ---
    try:
        from PIL import Image
        img = Image.new("RGB", (900, 400))
        for x in range(900):
            for y in range(0, 400, 40):
                img.putpixel((x, y), (255, 255, 255))
        p2 = Printer(dummy=True, queue_path=tmpdir / "q4.json")
        before = len(p2.output)
        if not p2.print_image(img):
            fails += 1
            print(f"  FAIL image print: {p2.last_error}")
        elif len(p2.output) - before < 500:
            fails += 1
            print("  FAIL image produced almost no data")
        else:
            print(f"  ok   image dithered and emitted "
                  f"{len(p2.output) - before} bytes")
    except ImportError:
        print("  --   Pillow not installed, image test skipped")

    # --- deliver falls back to the queue ---
    qp5 = tmpdir / "q5.json"
    broken = Printer(config=PrinterConfig(0xDEAD, 0xBEEF), queue_path=qp5)
    printed, mid = broken.deliver("Gavin", "should be queued")
    if printed or not mid:
        fails += 1
        print("  FAIL deliver did not queue when the printer is absent")
    elif len(MessageQueue(qp5)) != 1:
        fails += 1
        print("  FAIL queued message not persisted")
    else:
        print("  ok   unreachable printer queues instead of dropping")

    # force_queue, which is what main.py uses mid-call
    qp6 = tmpdir / "q6.json"
    good = Printer(dummy=True, queue_path=qp6)
    printed, mid = good.deliver("Gavin", "mid call", force_queue=True)
    if printed:
        fails += 1
        print("  FAIL force_queue printed anyway")
    else:
        print("  ok   force_queue holds the message during a call")
    n, left = good.flush()
    if n != 1 or left != 0:
        fails += 1
        print(f"  FAIL flush printed {n}, {left} left")
    else:
        print("  ok   flush prints the held message and empties the queue")

    print()
    print(f"{fails} failure(s)" if fails else "all checks passed")
    return 1 if fails else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Kellogg phone thermal printer")
    ap.add_argument("--dummy", action="store_true",
                    help="capture ESC/POS bytes instead of printing")
    ap.add_argument("--vid", type=lambda s: int(s, 16),
                    help="USB vendor id in hex, overrides detection")
    ap.add_argument("--pid", type=lambda s: int(s, 16), help="USB product id")
    ap.add_argument("--out-ep", type=lambda s: int(s, 16),
                    help="USB OUT endpoint in hex, e.g. 03")
    ap.add_argument("--in-ep", type=lambda s: int(s, 16))
    ap.add_argument("--width", type=int, default=CHARS_FONT_A)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="layout and queue checks, no hardware")
    sub.add_parser("detect", help="scan USB for the printer")
    sub.add_parser("test", help="print a self test page")
    sub.add_parser("queue", help="list pending messages")
    sub.add_parser("flush", help="print everything pending")

    p = sub.add_parser("message", help="print one message")
    p.add_argument("sender")
    p.add_argument("body")
    p.add_argument("--media", action="append", default=[])
    p.add_argument("--queue-only", action="store_true",
                   help="do not print, just enqueue, as during a call")

    p = sub.add_parser("preview", help="render to stdout, no printer needed")
    p.add_argument("sender")
    p.add_argument("body")

    p = sub.add_parser("image", help="print an image")
    p.add_argument("path")

    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        return _selftest()
    if args.cmd == "detect":
        return _cmd_detect()
    if args.cmd == "preview":
        bar = "+" + "-" * args.width + "+"
        print(bar)
        for line in format_message(args.sender, args.body,
                                   width=args.width).split("\n"):
            print("|" + line.ljust(args.width) + "|")
        print(bar)
        print(f"  {args.width} columns, the real printable width")
        return 0

    cfg = None
    if args.vid and args.pid:
        cfg = PrinterConfig(args.vid, args.pid,
                            in_ep=args.in_ep or DEFAULT_IN_EP,
                            out_ep=args.out_ep or DEFAULT_OUT_EP,
                            note="from command line")

    with Printer(config=cfg, dummy=args.dummy, width=args.width) as pr:
        if args.cmd == "queue":
            return _cmd_queue(pr)

        if args.cmd == "flush":
            n, left = pr.flush()
            print(f"  printed {n}, {left} still waiting")
            if left and not pr.available:
                print(f"  printer unreachable: {pr.last_error}")
            return 0 if not left else 1

        if args.cmd == "message":
            if args.queue_only:
                m = pr.queue.add(args.sender, args.body, args.media)
                print(f"  queued [{m.id}]")
                return 0
            ok, mid = pr.deliver(args.sender, args.body, args.media)
            if ok:
                print("  printed")
                return 0
            print(f"  queued [{mid}] - printer unreachable")
            print(f"  {pr.last_error}")
            return 1

        if args.cmd == "image":
            if pr.print_image(args.path):
                print("  printed")
                return 0
            print(f"  failed: {pr.last_error}")
            return 1

        if args.cmd == "test":
            if pr.selftest_page():
                print("  test page sent")
                if args.dummy:
                    print(f"  {len(pr.output)} bytes captured")
                return 0
            print(f"  failed: {pr.last_error}")
            print("\n  Run 'detect' to find the real vendor id, product id")
            print("  and OUT endpoint, then pass them explicitly:")
            print("    python src/printer.py --vid 0416 --pid 5011 "
                  "--out-ep 03 test")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())