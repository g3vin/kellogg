#!/usr/bin/env python3
"""Kellogg phone - telephony, behind one interface.

The decision, settled and not to be reopened: TWILIO, REACHED BY SIP
REGISTRATION. Not Twilio's REST voice API. A SIP client is needed either
way to get real-time bidirectional audio onto the Pi; routing media through
a websocket bridge instead would be more work for the voice path and buy
nothing. Registering as a SIP endpoint gets the same number management and
SMS handling with less plumbing.

main.py imports Telephony and nothing else from here. It never sees baresip
or Twilio. If the provider is ever swapped, only this file changes.

Three backends:

    NullTelephony       logs instead of dialling. This is how main.py gets
                        written and the state machine exercised before an
                        account exists or a campaign is approved. It is not
                        a stub to be deleted later - it stays, because it
                        is also how the phone behaves when the network is
                        down, and that path must be tested.

    SipTelephony        voice, through a baresip process on localhost,
                        driven over its ctrl_tcp JSON interface.

    TwilioSms           inbound messages, by polling the REST API. Polling
                        rather than webhooks deliberately: a webhook needs
                        a public endpoint and a tunnel into the house, and
                        a telephone that prints texts a minute late is not
                        meaningfully worse than one that prints them
                        instantly.

Credentials live in .env at the project root, which is gitignored. The
repo is public - it serves the A2P privacy policy from GitHub Pages - so
nothing secret may ever be committed.

Standalone use:

    python src/telephony.py selftest    state machine, no network
    python src/telephony.py null        drive NullTelephony interactively
    python src/telephony.py status      registration + credential check
    python src/telephony.py messages    poll Twilio once and print
"""

from __future__ import annotations

import abc
import argparse
import json
import logging
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
SEEN_PATH = ROOT / "seen_messages.json"

TWILIO_API = "https://api.twilio.com/2010-04-01/Accounts"

# baresip's ctrl_tcp module listens here by default.
BARESIP_HOST = "127.0.0.1"
BARESIP_PORT = 4444

CALL_STATES = ("dialing", "ringing", "connected", "ended")
DIRECTIONS = ("outbound", "inbound")


_log = logging.getLogger("kellogg.telephony")


class TelephonyError(RuntimeError):
    """Something is wrong with the telephony setup and worth reading."""


class MissingCredential(TelephonyError):
    """A required .env variable is absent. Names which one."""


# --------------------------------------------------------------------------
# value types
# --------------------------------------------------------------------------
@dataclass
class Call:
    """One call, in or out.

    `state` moves forward only: dialing or ringing, then connected, then
    ended. main.py keys its own state machine off this, so a backend must
    never skip backwards.
    """
    id: str
    number: str
    direction: str
    started_at: float
    state: str = "dialing"
    name: Optional[str] = None
    ended_reason: str = ""

    @property
    def active(self) -> bool:
        return self.state != "ended"

    @property
    def duration(self) -> float:
        return time.monotonic() - self.started_at

    def __str__(self) -> str:
        who = self.name or self.number
        return f"{self.direction} call to {who} [{self.state}]"


@dataclass
class Message:
    """An inbound SMS, on its way to the receipt printer."""
    id: str
    sender: str
    body: str
    received_at: str
    media_urls: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        head = self.body.replace("\n", " ")
        if len(head) > 40:
            head = head[:37] + "..."
        return f"{self.sender}: {head}"


@dataclass
class Status:
    """Why the phone can or cannot make calls right now.

    A status dataclass rather than a bare bool, per the project's rule
    that a wrong answer is worse than no answer: main.py needs to tell
    "not registered yet" from "no credentials at all" from "working".
    """
    backend: str
    registered: bool
    detail: str
    number: str = ""

    def __str__(self) -> str:
        mark = "registered" if self.registered else "NOT registered"
        line = f"{self.backend}: {mark}"
        if self.number:
            line += f"  ({self.number})"
        return f"{line}\n  {self.detail}"


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------
def load_env(path: Path | str = ENV_PATH) -> dict[str, str]:
    """Read .env into os.environ without overwriting the real environment.

    setdefault, not assignment: an operator who exports a variable to
    override the file should win.
    """
    path = Path(path)
    found: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            found[key] = value
            os.environ.setdefault(key, value)
    return found


def require(name: str, hint: str = "") -> str:
    """Fetch a credential or fail naming exactly what is missing."""
    value = os.environ.get(name, "").strip()
    if not value:
        extra = f"\n  {hint}" if hint else ""
        raise MissingCredential(
            f"{name} is not set. Add it to {ENV_PATH}:{extra}")
    return value


# --------------------------------------------------------------------------
# the interface main.py sees
# --------------------------------------------------------------------------
class Telephony(abc.ABC):
    """What main.py is allowed to know about making calls.

    Callbacks fire on background threads. They must push onto a queue and
    return - no blocking work inside one, per the project convention.
    """

    def __init__(self) -> None:
        self.on_incoming_call: Optional[Callable[[str, Optional[str]], None]] = None
        self.on_call_ended: Optional[Callable[[str], None]] = None
        self.on_message: Optional[Callable[[str, str, list], None]] = None
        self.on_registration: Optional[Callable[[bool], None]] = None
        self.current: Optional[Call] = None
        self._lock = threading.RLock()

    # ---- lifecycle ----
    @abc.abstractmethod
    def start(self) -> None:
        """Connect, register, begin polling. Idempotent."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Tear everything down. Safe to call twice."""

    @property
    @abc.abstractmethod
    def registered(self) -> bool:
        """Can we place a call this instant?"""

    @abc.abstractmethod
    def status(self) -> Status:
        """Why, in words."""

    # ---- calls ----
    @abc.abstractmethod
    def place_call(self, number: str, name: Optional[str] = None) -> Call:
        """Dial. Returns immediately with the call in `dialing`."""

    @abc.abstractmethod
    def answer(self) -> Optional[Call]:
        """Answer the ringing call, if there is one."""

    @abc.abstractmethod
    def hangup(self, reason: str = "local") -> None:
        """End whatever is happening. Safe when nothing is."""

    # ---- helpers shared by every backend ----
    def _emit_incoming(self, number: str, name: Optional[str] = None) -> None:
        with self._lock:
            self.current = Call(id=uuid.uuid4().hex[:12], number=number,
                                direction="inbound",
                                started_at=time.monotonic(),
                                state="ringing", name=name)
        if self.on_incoming_call:
            self.on_incoming_call(number, name)

    def _emit_ended(self, reason: str) -> None:
        with self._lock:
            if self.current is not None:
                self.current.state = "ended"
                self.current.ended_reason = reason
        if self.on_call_ended:
            self.on_call_ended(reason)

    def _emit_message(self, msg: Message) -> None:
        if self.on_message:
            self.on_message(msg.sender, msg.body, msg.media_urls)

    def __enter__(self) -> "Telephony":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


# --------------------------------------------------------------------------
# NullTelephony
# --------------------------------------------------------------------------
class NullTelephony(Telephony):
    """Everything the interface promises, except a network.

    Calls connect after a short delay and stay connected until hung up.
    Incoming calls and messages can be injected, which is how main.py's
    RINGING path gets tested without anyone dialling the house.
    """

    def __init__(self, connect_after: float = 1.5,
                 log: Optional[Callable[[str], None]] = None) -> None:
        super().__init__()
        self.connect_after = connect_after
        self.log = log or (lambda line: print(f"  [null] {line}"))
        self._running = False
        self._timer: Optional[threading.Timer] = None

    def start(self) -> None:
        self._running = True
        self.log("started - no calls will actually be placed")
        if self.on_registration:
            self.on_registration(True)

    def stop(self) -> None:
        self._cancel_timer()
        self._running = False
        self.log("stopped")

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    @property
    def registered(self) -> bool:
        return self._running

    def status(self) -> Status:
        return Status("null", self._running,
                      "Simulated. Nothing leaves the Pi.", "+15555550100")

    def place_call(self, number: str, name: Optional[str] = None) -> Call:
        if not self._running:
            raise TelephonyError("NullTelephony.start() was never called")
        with self._lock:
            self.current = Call(id=uuid.uuid4().hex[:12], number=number,
                                direction="outbound",
                                started_at=time.monotonic(), name=name)
            call = self.current
        self.log(f"dialing {name or number}")

        def connect() -> None:
            with self._lock:
                if self.current is call and call.state == "dialing":
                    call.state = "connected"
                    self.log("connected")

        self._cancel_timer()
        self._timer = threading.Timer(self.connect_after, connect)
        self._timer.daemon = True
        self._timer.start()
        return call

    def answer(self) -> Optional[Call]:
        with self._lock:
            if self.current is None or self.current.state != "ringing":
                return None
            self.current.state = "connected"
            self.log("answered")
            return self.current

    def hangup(self, reason: str = "local") -> None:
        self._cancel_timer()
        with self._lock:
            had = self.current is not None and self.current.active
        if had:
            self.log(f"hung up ({reason})")
            self._emit_ended(reason)

    # ---- injection, for testing main.py ----
    def simulate_incoming(self, number: str = "+15555550123",
                          name: Optional[str] = None) -> None:
        self.log(f"incoming call from {name or number}")
        self._emit_incoming(number, name)

    def simulate_message(self, sender: str = "+15555550123",
                         body: str = "test message") -> None:
        self.log(f"message from {sender}")
        self._emit_message(Message(
            id=uuid.uuid4().hex[:12], sender=sender, body=body,
            received_at=datetime.now(timezone.utc).isoformat()))

    def simulate_remote_hangup(self) -> None:
        self.log("far end hung up")
        self._emit_ended("remote")


# --------------------------------------------------------------------------
# baresip control
# --------------------------------------------------------------------------
class BaresipControl:
    """Talk to a running baresip over its ctrl_tcp interface.

    The protocol is netstrings carrying JSON: `<length>:<payload>,`. Events
    arrive unsolicited on the same socket, so a reader thread drains it and
    hands events to a callback.

    baresip is expected to be already running and registered against the
    Twilio SIP domain, started by its own systemd unit. This class does not
    launch it - a telephone that silently restarts its SIP stack mid-call
    is worse than one that reports being down.
    """

    def __init__(self, host: str = BARESIP_HOST, port: int = BARESIP_PORT,
                 on_event: Optional[Callable[[dict], None]] = None) -> None:
        self.host = host
        self.port = port
        self.on_event = on_event
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._buf = b""

    def connect(self, timeout: float = 3.0) -> None:
        try:
            self._sock = socket.create_connection(
                (self.host, self.port), timeout=timeout)
        except OSError as exc:
            raise TelephonyError(
                f"Cannot reach baresip at {self.host}:{self.port} ({exc}).\n"
                f"  Is it running?  systemctl --user status baresip\n"
                f"  Does its config load ctrl_tcp?  "
                f"grep ctrl_tcp ~/.baresip/config"
            ) from None
        self._sock.settimeout(0.5)
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def command(self, cmd: str, params: str = "") -> None:
        """Send one command. Fire and forget; results arrive as events."""
        if self._sock is None:
            raise TelephonyError("not connected to baresip")
        payload = json.dumps({"command": cmd, "params": params}).encode()
        frame = str(len(payload)).encode() + b":" + payload + b","
        self._sock.sendall(frame)

    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._sock.recv(4096) if self._sock else b""
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            self._buf += chunk
            for obj in self._drain():
                if self.on_event:
                    try:
                        self.on_event(obj)
                    except Exception:
                        # A callback must never kill the reader; a dead
                        # reader means the phone stops noticing calls.
                        pass

    def _drain(self) -> list[dict]:
        """Pull every complete netstring out of the buffer."""
        out: list[dict] = []
        while True:
            sep = self._buf.find(b":")
            if sep < 0:
                return out
            try:
                length = int(self._buf[:sep])
            except ValueError:
                self._buf = self._buf[sep + 1:]
                continue
            end = sep + 1 + length
            if len(self._buf) < end + 1:
                return out
            payload = self._buf[sep + 1:end]
            self._buf = self._buf[end + 1:]
            try:
                out.append(json.loads(payload.decode()))
            except (ValueError, UnicodeDecodeError):
                continue


# --------------------------------------------------------------------------
# Twilio inbound SMS, by polling
# --------------------------------------------------------------------------
class TwilioSms:
    """Fetch inbound messages. No webhook, no tunnel into the house.

    Keeps the ids it has already seen in a file, so a restart does not
    reprint the last twenty texts on the receipt printer.
    """

    def __init__(self, account_sid: str, auth_token: str, number: str,
                 seen_path: Path | str = SEEN_PATH,
                 interval: float = 20.0) -> None:
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.number = number
        self.seen_path = Path(seen_path)
        self.interval = interval
        self.on_message: Optional[Callable[[Message], None]] = None
        self._seen: set[str] = set()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._load_seen()

    def _load_seen(self) -> None:
        try:
            self._seen = set(json.loads(self.seen_path.read_text()))
        except Exception:
            self._seen = set()

    def _save_seen(self) -> None:
        # Keep the file bounded; the last few hundred ids is plenty to
        # stop duplicates across a restart.
        recent = sorted(self._seen)[-500:]
        tmp = self.seen_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(recent))
        os.replace(tmp, self.seen_path)

    def _get(self, url: str) -> dict:
        req = urllib.request.Request(url)
        import base64
        auth = base64.b64encode(
            f"{self.account_sid}:{self.auth_token}".encode()).decode()
        req.add_header("Authorization", f"Basic {auth}")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            raise TelephonyError(
                f"Twilio returned {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            raise TelephonyError(f"Cannot reach Twilio: {exc.reason}") from None

    def poll_once(self, mark_seen: bool = True) -> list[Message]:
        """Return messages not seen before, oldest first."""
        query = urllib.parse.urlencode({"To": self.number, "PageSize": 20})
        url = f"{TWILIO_API}/{self.account_sid}/Messages.json?{query}"
        data = self._get(url)

        fresh: list[Message] = []
        for row in reversed(data.get("messages", [])):
            if row.get("direction") != "inbound":
                continue
            sid = row.get("sid", "")
            if not sid or sid in self._seen:
                continue
            media: list[str] = []
            if int(row.get("num_media") or 0) > 0:
                media = [f"{TWILIO_API}/{self.account_sid}"
                         f"/Messages/{sid}/Media"]
            fresh.append(Message(
                id=sid,
                sender=row.get("from") or "unknown",
                body=row.get("body") or "",
                received_at=row.get("date_sent") or row.get("date_created")
                or "",
                media_urls=media,
            ))
            if mark_seen:
                self._seen.add(sid)
        if fresh and mark_seen:
            self._save_seen()
        return fresh

    def prime(self) -> int:
        """Mark everything currently on the account as already seen.

        Run once on first install so the printer does not spool out the
        entire message history the first time the phone boots.
        """
        before = len(self._seen)
        self.poll_once(mark_seen=True)
        self._save_seen()
        return len(self._seen) - before

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                for msg in self.poll_once():
                    if self.on_message:
                        self.on_message(msg)
            except TelephonyError:
                # Network blips are expected and not worth dying over;
                # the next poll will pick up anything missed.
                pass
            except Exception:
                pass
            self._stop.wait(self.interval)


# --------------------------------------------------------------------------
# the real backend
# --------------------------------------------------------------------------
class SipTelephony(Telephony):
    """Voice over SIP through baresip, messages over the Twilio REST API.

    Not exercised against a live trunk yet - the A2P campaign is still in
    review, so outbound is blocked at the carrier with error 30034. The
    event names below are baresip's and the shapes are from its ctrl_tcp
    documentation; expect to adjust `_on_baresip_event` the first time a
    real call comes in, and use `python src/telephony.py status` to see
    what is actually arriving.
    """

    def __init__(self, sip_uri: Optional[str] = None,
                 poll_interval: float = 20.0) -> None:
        super().__init__()
        load_env()
        self.sip_uri = sip_uri or os.environ.get("SIP_URI", "")
        self._registered = False
        self._detail = "not started"
        self._ctrl = BaresipControl(on_event=self._on_baresip_event)
        self._sms: Optional[TwilioSms] = None

    def _build_sms(self) -> TwilioSms:
        sms = TwilioSms(
            require("TWILIO_ACCOUNT_SID", "from the Twilio console"),
            require("TWILIO_AUTH_TOKEN", "from the Twilio console"),
            require("TWILIO_PHONE_NUMBER", "in E.164, e.g. +17573839589"),
        )
        sms.on_message = self._emit_message
        return sms

    def start(self) -> None:
        self._sms = self._build_sms()
        self._ctrl.connect()
        self._detail = "connected to baresip, asking for registration state"
        self._sms.start()

        # ASK, do not wait. baresip announces REGISTER_OK once, when it
        # registers at startup - which is before this process connects,
        # so the announcement is long gone and waiting for it means
        # believing we are unregistered forever.
        try:
            self._ctrl.command("reginfo")
        except TelephonyError:
            return

        # Wait briefly for the answer. The reply lands on the reader
        # thread, so without this start() returns before it arrives and
        # the status logged at boot says NOT registered regardless of
        # the truth - which is alarming and wrong.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not self._registered:
            time.sleep(0.05)
        if not self._registered:
            self._detail = ("baresip did not report a registered account "
                            "within 2s - check: systemctl --user status "
                            "baresip")

    def stop(self) -> None:
        if self._sms is not None:
            self._sms.stop()
        self._ctrl.close()
        self._registered = False
        self._detail = "stopped"

    @property
    def registered(self) -> bool:
        return self._registered

    def status(self) -> Status:
        return Status("sip+twilio", self._registered, self._detail,
                      os.environ.get("TWILIO_PHONE_NUMBER", ""))

    def place_call(self, number: str, name: Optional[str] = None) -> Call:
        if not self._registered:
            raise TelephonyError(
                "Not registered with the SIP trunk, so this call would go "
                "nowhere. Check:  python src/telephony.py status")
        with self._lock:
            self.current = Call(id=uuid.uuid4().hex[:12], number=number,
                                direction="outbound",
                                started_at=time.monotonic(), name=name)
            call = self.current
        self._ctrl.command("dial", number)
        return call

    def answer(self) -> Optional[Call]:
        with self._lock:
            if self.current is None or self.current.state != "ringing":
                return None
            call = self.current
        self._ctrl.command("accept")
        with self._lock:
            call.state = "connected"
        return call

    def hangup(self, reason: str = "local") -> None:
        try:
            self._ctrl.command("hangup")
        except TelephonyError:
            pass
        self._emit_ended(reason)

    def _note_registered(self, value: bool, detail: str) -> None:
        """Set the flag and fire the callback only on a real change."""
        changed = value != self._registered
        self._registered = value
        self._detail = detail
        if changed and self.on_registration:
            self.on_registration(value)

    def _on_baresip_event(self, event: dict) -> None:
        """Translate baresip's vocabulary into ours. Short, as required."""
        kind = event.get("type") or event.get("event") or ""
        peer = event.get("peeruri") or event.get("param") or ""

        # Every event, verbatim, at debug. This is the one part of the
        # build that has never been checked against real traffic, so the
        # raw stream is the only way to find out what baresip actually
        # says rather than what its documentation implies.
        _log.debug("baresip: %s", json.dumps(event, sort_keys=True))

        # A reginfo reply is a RESPONSE, not an event, and it is a
        # rendered text table rather than structured data:
        #
        #   --- User Agents (2) ---
        #   0 - sip:kellogg@domain   \x1b[32mOK \x1b[;m Twilio Expires 600s
        #
        # It never contains the word "REGISTER", and the status is
        # wrapped in ANSI colour codes, so anything looking for a tidy
        # REGISTER_OK finds nothing and concludes we are offline.
        if event.get("response") and isinstance(event.get("data"), str):
            if reginfo_registered(event["data"]):
                self._note_registered(True, "registered with the SIP trunk")
            elif "User Agents" in event["data"]:
                self._note_registered(
                    False, "baresip has accounts but none are registered")
            return

        if kind in ("REGISTER_OK",):
            self._note_registered(True, "registered with the SIP trunk")
        elif kind in ("REGISTER_FAIL", "UNREGISTERING"):
            self._note_registered(
                False, f"registration failed: {event.get('param', '')}")
        elif kind == "CALL_INCOMING":
            self._emit_incoming(_sip_number(peer))
        elif kind == "CALL_ESTABLISHED":
            with self._lock:
                if self.current is not None:
                    self.current.state = "connected"
        elif kind in ("CALL_CLOSED", "CALL_TERMINATED"):
            self._emit_ended(event.get("param") or "remote")


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def reginfo_registered(data: str) -> bool:
    """True when baresip's reginfo table shows a registered account.

    Parsed rather than pattern-matched on the whole blob, because the
    table lists every account and one failing account must not be hidden
    by another that worked.
    """
    clean = _ANSI.sub("", data)
    for line in clean.splitlines():
        if "sip:" not in line:
            continue
        if re.search(r"(?<![A-Za-z])OK(?![A-Za-z])", line):
            return True
    return False


def _sip_number(uri: str) -> str:
    """Pull a dialable number out of sip:+15555550123@domain."""
    if not uri:
        return ""
    body = uri.split(":", 1)[-1].split("@", 1)[0]
    keep = "".join(ch for ch in body if ch.isdigit() or ch == "+")
    return keep or body


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------
def build(backend: str = "auto", **kwargs) -> Telephony:
    """Pick a backend. main.py calls this, not the classes.

    "auto" uses the real one when credentials exist and falls back to
    NullTelephony when they do not, so a fresh clone runs out of the box.
    """
    if backend == "null":
        return NullTelephony(**kwargs)
    if backend == "sip":
        return SipTelephony(**kwargs)
    if backend != "auto":
        raise TelephonyError(f"unknown backend {backend!r}")

    load_env()
    needed = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
              "TWILIO_PHONE_NUMBER")
    if all(os.environ.get(k) for k in needed):
        return SipTelephony()
    return NullTelephony()


# --------------------------------------------------------------------------
# self test - no network
# --------------------------------------------------------------------------
def _selftest() -> int:
    import tempfile

    checks: list[tuple[bool, str]] = []

    def ok(cond: bool, label: str) -> None:
        checks.append((bool(cond), label))

    # --- Call value type ---
    c = Call("abc", "+15555550123", "outbound", time.monotonic())
    ok(c.active, "a new call is active")
    ok(c.state == "dialing", "a new outbound call starts dialing")
    c.state = "ended"
    ok(not c.active, "an ended call is not active")

    # --- NullTelephony drives the whole interface ---
    lines: list[str] = []
    null = NullTelephony(connect_after=0.05, log=lines.append)
    events: list[tuple] = []
    null.on_incoming_call = lambda n, nm: events.append(("in", n, nm))
    null.on_call_ended = lambda r: events.append(("end", r))
    null.on_message = lambda s, b, m: events.append(("msg", s, b))

    ok(not null.registered, "not registered before start")
    null.start()
    ok(null.registered, "registered after start")

    call = null.place_call("+15555550123", name="Jamie")
    ok(call.state == "dialing", "place_call returns immediately, dialing")
    ok(call.name == "Jamie", "the contact name rides along with the call")
    time.sleep(0.2)
    ok(call.state == "connected", "the call connects on its own")

    null.hangup("local")
    ok(call.state == "ended", "hangup ends the call")
    ok(("end", "local") in events, "hangup fires on_call_ended")

    before = len(events)
    null.hangup("local")
    ok(len(events) == before, "hanging up twice fires one event, not two")

    null.simulate_incoming("+15555559999", "Mum")
    ok(("in", "+15555559999", "Mum") in events, "incoming call fires")
    ok(null.current is not None and null.current.state == "ringing",
       "an incoming call sits in ringing until answered")
    answered = null.answer()
    ok(answered is not None and answered.state == "connected",
       "answer connects the ringing call")
    ok(null.answer() is None, "answering when nothing rings returns None")

    null.simulate_message("+15555550123", "hello from the bench")
    ok(("msg", "+15555550123", "hello from the bench") in events,
       "a simulated message reaches on_message")

    null.stop()
    ok(not null.registered, "stop deregisters")

    # place_call before start must refuse rather than pretend
    fresh = NullTelephony(log=lambda _: None)
    try:
        fresh.place_call("+15555550123")
        ok(False, "place_call before start raises")
    except TelephonyError:
        ok(True, "place_call before start raises")

    # --- credentials ---
    with tempfile.TemporaryDirectory() as tmp:
        env = Path(tmp) / ".env"
        env.write_text("# a comment\n"
                       "TWILIO_ACCOUNT_SID=ACtest\n"
                       "EMPTY=\n"
                       "QUOTED='value'\n"
                       "WITH_EQUALS=a=b\n")
        got = load_env(env)
        ok(got.get("TWILIO_ACCOUNT_SID") == "ACtest", ".env parses")
        ok(got.get("QUOTED") == "value", "quotes are stripped")
        ok(got.get("WITH_EQUALS") == "a=b", "only the first = splits")
        ok("# a comment" not in got, "comments are skipped")

    os.environ.pop("DEFINITELY_NOT_SET", None)
    try:
        require("DEFINITELY_NOT_SET")
        ok(False, "require names the missing variable")
    except MissingCredential as exc:
        ok("DEFINITELY_NOT_SET" in str(exc),
           "require names the missing variable")

    # --- netstring framing ---
    ctrl = BaresipControl()
    payload = json.dumps({"event": True, "type": "REGISTER_OK"}).encode()
    ctrl._buf = str(len(payload)).encode() + b":" + payload + b","
    drained = ctrl._drain()
    ok(len(drained) == 1 and drained[0]["type"] == "REGISTER_OK",
       "a netstring frame decodes")
    ok(ctrl._buf == b"", "the buffer empties after a complete frame")

    ctrl._buf = str(len(payload)).encode() + b":" + payload[:5]
    ok(ctrl._drain() == [], "a partial frame yields nothing")
    ok(ctrl._buf != b"", "a partial frame is kept for the next read")

    two = (str(len(payload)).encode() + b":" + payload + b",") * 2
    ctrl._buf = two
    ok(len(ctrl._drain()) == 2, "two frames in one read both decode")

    ctrl._buf = b"notanumber:garbage,"
    ctrl._drain()
    ok(True, "malformed framing does not raise")

    # --- reginfo parsing, against the real reply from the Pi ---
    real = ("\n--- User Agents (2) ---\n"
            "0 - sip:kellogg@kellogg-g3vin.sip.twilio.com   "
            "\x1b[32mOK \x1b[;m Twilio Expires 600s\n\n")
    ok(reginfo_registered(real),
       "baresip's reginfo table is recognised as registered")
    ok(not reginfo_registered(
        "--- User Agents (1) ---\n0 - sip:a@b  \x1b[31mERR \x1b[;m 401\n"),
       "a failed registration is not read as success")
    ok(not reginfo_registered("--- User Agents (0) ---\n"),
       "no accounts means not registered")
    ok(not reginfo_registered("something OKAY without a uri"),
       "OKAY inside a word is not a status of OK")
    ok(reginfo_registered("0 - sip:a@b   OK  Expires 600s"),
       "the table parses without ansi colour codes too")

    # the response path sets the flag, which is what the event path missed
    t = SipTelephony.__new__(SipTelephony)
    Telephony.__init__(t)
    t._registered = False
    t._detail = ""
    t._ctrl = None
    fired: list = []
    t.on_registration = fired.append
    t._on_baresip_event({"response": True, "ok": True, "data": real})
    ok(t.registered, "a reginfo response marks the trunk registered")
    ok(fired == [True], "and fires on_registration exactly once")
    t._on_baresip_event({"response": True, "ok": True, "data": real})
    ok(fired == [True], "a repeat response does not fire it again")

    # --- SIP uri parsing ---
    ok(_sip_number("sip:+15555550123@kellogg.sip.twilio.com")
       == "+15555550123", "a sip uri yields the number")
    ok(_sip_number("") == "", "an empty uri is handled")
    ok(_sip_number("sip:anonymous@example.com") == "anonymous",
       "a non-numeric user falls back to the raw string")

    # --- seen-message bookkeeping ---
    with tempfile.TemporaryDirectory() as tmp:
        seen = Path(tmp) / "seen.json"
        sms = TwilioSms("ACtest", "tok", "+15555550100", seen_path=seen)
        sms._seen = {f"SM{i:032d}" for i in range(600)}
        sms._save_seen()
        again = TwilioSms("ACtest", "tok", "+15555550100", seen_path=seen)
        ok(len(again._seen) == 500, "the seen list is bounded at 500")

        seen.write_text("this is not json")
        broken = TwilioSms("ACtest", "tok", "+1", seen_path=seen)
        ok(broken._seen == set(),
           "a corrupt seen file starts empty rather than crashing")

    # --- factory ---
    for key in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
                "TWILIO_PHONE_NUMBER"):
        os.environ.pop(key, None)
    saved, globals()["ENV_PATH"] = ENV_PATH, Path("/nonexistent/.env")
    ok(isinstance(build("auto"), NullTelephony),
       "auto falls back to NullTelephony with no credentials")
    globals()["ENV_PATH"] = saved
    ok(isinstance(build("null"), NullTelephony), "explicit null works")
    try:
        build("carrier pigeon")
        ok(False, "an unknown backend raises")
    except TelephonyError:
        ok(True, "an unknown backend raises")

    width = max(len(label) for _, label in checks)
    failed = sum(1 for passed, _ in checks if not passed)
    for passed, label in checks:
        print(f"  {'ok ' if passed else 'FAIL'}  {label:<{width}}")
    print()
    if failed:
        print(f"{failed} of {len(checks)} checks failed")
        return 1
    print(f"all {len(checks)} checks passed")
    return 0


# --------------------------------------------------------------------------
def _cmd_null() -> int:
    """Drive NullTelephony by hand, to watch the event model work."""
    phone = NullTelephony(connect_after=1.0)
    phone.on_incoming_call = lambda n, nm: print(f"  -> incoming {nm or n}")
    phone.on_call_ended = lambda r: print(f"  -> ended ({r})")
    phone.on_message = lambda s, b, m: print(f"  -> message {s}: {b}")
    phone.start()
    print("\n  commands: dial <number> | answer | hangup | ring | text | quit")
    try:
        while True:
            try:
                raw = input("  > ").strip()
            except EOFError:
                break
            if not raw:
                continue
            word, _, rest = raw.partition(" ")
            if word in ("quit", "exit"):
                break
            if word == "dial":
                phone.place_call(rest or "+15555550123")
            elif word == "answer":
                phone.answer()
            elif word == "hangup":
                phone.hangup()
            elif word == "ring":
                phone.simulate_incoming(rest or "+15555559999")
            elif word == "text":
                phone.simulate_message(body=rest or "hello")
            else:
                print("  ?")
    except KeyboardInterrupt:
        pass
    phone.stop()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Kellogg phone telephony")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("selftest", help="state machine checks, no network")
    sub.add_parser("null", help="drive NullTelephony interactively")
    sub.add_parser("status", help="credentials and registration")
    sub.add_parser("messages", help="poll Twilio once and print")
    sub.add_parser("prime", help="mark existing messages as already seen")

    args = ap.parse_args(argv)
    cmd = args.cmd or "selftest"

    if cmd == "selftest":
        return _selftest()
    if cmd == "null":
        return _cmd_null()

    try:
        load_env()
        if cmd == "status":
            phone = build("auto")
            print(f"  {phone.status()}")
            if isinstance(phone, NullTelephony):
                print("\n  No Twilio credentials found, so the phone would "
                      f"run in null mode.\n  Add them to {ENV_PATH}.")
            return 0

        sms = TwilioSms(
            require("TWILIO_ACCOUNT_SID"),
            require("TWILIO_AUTH_TOKEN"),
            require("TWILIO_PHONE_NUMBER"),
        )
        if cmd == "prime":
            n = sms.prime()
            print(f"  marked {n} existing message(s) as seen")
            return 0
        if cmd == "messages":
            fresh = sms.poll_once(mark_seen=False)
            if not fresh:
                print("  no new inbound messages")
                return 0
            for msg in fresh:
                print(f"  {msg}")
            return 0

    except TelephonyError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())