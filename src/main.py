#!/usr/bin/env python3
"""Kellogg phone - the state machine.

Every transition in the phone happens here. The feature modules stay
unaware of each other; main.py is the only place that knows the call flow.

Architecture: one thread owns all state. Hardware callbacks, telephony
callbacks and timers do nothing but push an Event onto a queue and return.
The loop pops and dispatches. No state is mutated from a callback thread,
which is why there is no locking anywhere below.

    IDLE        on-hook, mic muted, amp off, crank ignored
    OFF_HOOK    handset lifted, amp on, waiting for the crank
    OPERATOR    greeting played, listening for a name or number
    CONFIRM     reading back, listening for yes or no
    CONNECTED   full duplex
    RINGING     inbound call, waiting for the handset to be lifted

A magneto phone never had a dial tone. Cranking summoned a human operator,
which is exactly what this recreates.

Standalone use:

    python src/main.py                   run it
    python src/main.py --null-telephony  run with no real calls
    python src/main.py --simulate        drive from stdin, no GPIO
    python src/main.py selftest          transition table, no hardware
"""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("kellogg")

# How long the operator waits for speech before giving up on a turn.
LISTEN_TIMEOUT = 8.0
CONFIRM_TIMEOUT = 6.0

# Three failures and the operator goes quiet. A phone that nags is worse
# than one that waits.
MAX_ATTEMPTS = 3

# After this long with no speech at all, re-prompt once then sit silent.
IDLE_PROMPT_AFTER = 30.0


class State(str, Enum):
    IDLE = "IDLE"
    OFF_HOOK = "OFF_HOOK"
    OPERATOR = "OPERATOR"
    CONFIRM = "CONFIRM"
    CONNECTED = "CONNECTED"
    RINGING = "RINGING"

    def __str__(self) -> str:
        return self.value


class EventKind(str, Enum):
    OFF_HOOK = "off_hook"
    ON_HOOK = "on_hook"
    CRANK = "crank"
    INCOMING_CALL = "incoming_call"
    CALL_ENDED = "call_ended"
    MESSAGE = "message"
    HEARD = "heard"              # an ASR turn completed
    TICK = "tick"
    SHUTDOWN = "shutdown"


@dataclass
class Event:
    kind: EventKind
    payload: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.kind.value}{self.payload or ''}"


# --------------------------------------------------------------------------
# the decision table, kept pure so it can be tested without a microphone
# --------------------------------------------------------------------------
@dataclass
class Decision:
    """What the operator should do about one heard utterance.

    prompt      a key from tts.PROMPTS, or None to say nothing
    next_state  where to go afterwards
    costs_try   whether this burns one of the three attempts
    contact     the contact to read back, when there is one
    candidates  names to offer when the match was ambiguous
    """
    prompt: Optional[str]
    next_state: State
    costs_try: bool = False
    contact: Any = None
    candidates: list = field(default_factory=list)


def plan_response(asr_status: str, match_status: Optional[str] = None,
                  contact: Any = None, candidates: Optional[list] = None,
                  attempts: int = 0) -> Decision:
    """Map an ASR result and a contact lookup onto what to say next.

    This table is the heart of the phone. Get it right and it feels
    intelligent; get it wrong and it feels broken.

    The rule that matters most: a low-confidence or unknown transcription
    is NEVER looked up. Feeding noisy text into fuzzy matching is exactly
    how a cough becomes a phone call.
    """
    candidates = candidates or []
    out_of_tries = attempts + 1 >= MAX_ATTEMPTS

    if asr_status in ("low_confidence", "unknown"):
        if out_of_tries:
            return Decision("goodbye", State.OFF_HOOK, costs_try=True)
        return Decision("retry", State.OPERATOR, costs_try=True)

    if asr_status in ("silence", "timeout"):
        if out_of_tries:
            return Decision("goodbye", State.OFF_HOOK, costs_try=True)
        return Decision("timeout", State.OPERATOR, costs_try=True)

    if asr_status != "ok":
        # An unfamiliar status is a bug upstream, not a reason to dial.
        return Decision("retry", State.OPERATOR, costs_try=True)

    if match_status in ("exact", "confident"):
        return Decision(None, State.CONFIRM, contact=contact)

    if match_status == "ambiguous":
        if out_of_tries:
            return Decision("goodbye", State.OFF_HOOK, costs_try=True)
        return Decision("which_one", State.OPERATOR, costs_try=True,
                        candidates=candidates)

    if out_of_tries:
        return Decision("goodbye", State.OFF_HOOK, costs_try=True)
    return Decision("not_found", State.OPERATOR, costs_try=True)


def plan_confirm(answer: Optional[bool], attempts: int = 0) -> Decision:
    """yes, no, or could not tell."""
    if answer is True:
        return Decision("connecting", State.CONNECTED)
    if answer is False:
        return Decision("greeting", State.OPERATOR)
    if attempts + 1 >= MAX_ATTEMPTS:
        return Decision("goodbye", State.OFF_HOOK, costs_try=True)
    return Decision("retry", State.CONFIRM, costs_try=True)


# --------------------------------------------------------------------------
# the phone
# --------------------------------------------------------------------------
class Phone:
    """Owns the state. Nothing else mutates it.

    Services are injected rather than constructed here, so the self test
    can drive the whole machine with fakes and no hardware at all.
    """

    def __init__(self, hardware=None, voice=None, listener=None,
                 book=None, printer=None, telephony=None,
                 on_state: Optional[Callable[[State, State], None]] = None):
        self.hardware = hardware
        self.voice = voice
        self.listener = listener
        self.book = book
        self.printer = printer
        self.telephony = telephony
        self.on_state = on_state

        self.state = State.IDLE
        self.events: "queue.Queue[Event]" = queue.Queue()
        self.attempts = 0
        self.pending_contact = None
        self.running = False
        self._listen_thread: Optional[threading.Thread] = None
        self._turn = 0           # invalidates in-flight ASR after a change

    # ---- event plumbing: callbacks only ever do this ----
    def push(self, kind: EventKind, **payload) -> None:
        self.events.put(Event(kind, payload))

    # ---- state ----
    def _go(self, new: State) -> None:
        if new == self.state:
            return
        old, self.state = self.state, new
        self._turn += 1          # any in-flight listen result is now stale
        log.info("%s -> %s", old, new)
        if self.on_state:
            self.on_state(old, new)
        self._on_enter(new)

    def _on_enter(self, state: State) -> None:
        if state is State.IDLE:
            self.attempts = 0
            self.pending_contact = None
            self._set_amp(False)
            self._flush_printer()
        elif state is State.OFF_HOOK:
            self._set_amp(True)
        elif state is State.OPERATOR:
            self._say("greeting")
            self._listen_async(LISTEN_TIMEOUT)
        elif state is State.CONFIRM:
            self._readback()
            self._listen_async(CONFIRM_TIMEOUT, confirm=True)
        elif state is State.CONNECTED:
            self.attempts = 0
        elif state is State.RINGING:
            self._set_amp(True)

    # ---- thin wrappers so fakes stay simple and None is survivable ----
    def _set_amp(self, on: bool) -> None:
        if self.hardware is not None:
            try:
                self.hardware.set_amp(on)
            except Exception:
                log.exception("amp")

    def _say(self, prompt: str) -> None:
        if self.voice is None:
            log.info("[say] %s", prompt)
            return
        try:
            self.voice.prompt(prompt)
        except Exception:
            log.exception("tts")

    def _readback(self) -> None:
        c = self.pending_contact
        if c is None:
            return
        if self.voice is None:
            log.info("[readback] %s %s", getattr(c, "name", c),
                     getattr(c, "number", ""))
            return
        try:
            self.voice.readback(c.name, c.number)
        except Exception:
            log.exception("readback")

    def _flush_printer(self) -> None:
        """Print anything held during a call. Never during one."""
        if self.printer is None:
            return
        try:
            printed, failed = self.printer.flush()
            if printed:
                log.info("printed %d queued message(s)", printed)
            if failed:
                log.warning("%d message(s) still queued", failed)
        except Exception:
            log.exception("printer flush")

    def _listen_async(self, timeout: float, confirm: bool = False) -> None:
        """Listen on a worker, push the result back as an event.

        The turn number is captured now and checked on arrival, so a
        result from before the handset went down is discarded instead of
        driving a transition in a state that no longer exists.
        """
        if self.listener is None:
            return
        turn = self._turn

        def work() -> None:
            try:
                if confirm:
                    answer, result = self.listener.confirm(timeout=timeout)
                    self.push(EventKind.HEARD, turn=turn, confirm=True,
                              answer=answer, status=result.status,
                              text=result.text)
                else:
                    result = self.listener.listen(timeout=timeout)
                    self.push(EventKind.HEARD, turn=turn, confirm=False,
                              status=result.status, text=result.text)
            except Exception:
                log.exception("listen")
                self.push(EventKind.HEARD, turn=turn, confirm=confirm,
                          status="timeout", text="")

        self._listen_thread = threading.Thread(target=work, daemon=True)
        self._listen_thread.start()

    # ---- event handling ----
    def handle(self, event: Event) -> None:
        kind = event.kind

        # The handset going down wins from any state, always, immediately.
        if kind is EventKind.ON_HOOK:
            if self.state is not State.IDLE:
                self._hangup("handset replaced")
            self._go(State.IDLE)
            return

        if kind is EventKind.SHUTDOWN:
            self.running = False
            return

        if kind is EventKind.MESSAGE:
            self._on_message(event.payload)
            return

        if kind is EventKind.OFF_HOOK:
            if self.state is State.RINGING:
                self._answer()
                self._go(State.CONNECTED)
            elif self.state is State.IDLE:
                self._go(State.OFF_HOOK)
            return

        if kind is EventKind.CRANK:
            # Only from OFF_HOOK. A second crank mid-prompt is ignored,
            # not queued - hardware.py already drops cranks while on-hook.
            if self.state is State.OFF_HOOK:
                self.attempts = 0
                self._go(State.OPERATOR)
            else:
                log.debug("crank ignored in %s", self.state)
            return

        if kind is EventKind.INCOMING_CALL:
            if self.state is State.IDLE:
                self._go(State.RINGING)
            else:
                log.info("incoming call while %s - not answering", self.state)
            return

        if kind is EventKind.CALL_ENDED:
            if self.state in (State.CONNECTED, State.RINGING):
                self._say("goodbye")
                self._go(State.OFF_HOOK if self._off_hook() else State.IDLE)
            return

        if kind is EventKind.HEARD:
            self._on_heard(event.payload)
            return

    def _off_hook(self) -> bool:
        if self.hardware is None:
            return self.state not in (State.IDLE,)
        try:
            return bool(self.hardware.off_hook)
        except Exception:
            return False

    def _on_heard(self, payload: dict) -> None:
        if payload.get("turn") != self._turn:
            log.debug("discarding stale ASR result")
            return

        if payload.get("confirm"):
            if self.state is not State.CONFIRM:
                return
            decision = plan_confirm(payload.get("answer"), self.attempts)
            self._apply(decision)
            return

        if self.state is not State.OPERATOR:
            return

        status = payload.get("status", "timeout")
        match_status = None
        contact = None
        candidates: list = []
        if status == "ok" and self.book is not None:
            try:
                match = self.book.lookup(payload.get("text", ""))
                match_status = match.status
                contact = match.contact
                candidates = list(match.candidates or [])
            except Exception:
                log.exception("lookup")
                match_status = "none"
        elif status == "ok":
            match_status = "none"

        decision = plan_response(status, match_status, contact,
                                 candidates, self.attempts)
        self._apply(decision)

    def _apply(self, decision: Decision) -> None:
        if decision.costs_try:
            self.attempts += 1
        if decision.contact is not None:
            self.pending_contact = decision.contact

        if decision.prompt == "which_one" and decision.candidates:
            self._say("which_one")
            for cand in decision.candidates[:3]:
                if self.voice is not None:
                    try:
                        self.voice.say(cand.name)
                    except Exception:
                        log.exception("tts")
                else:
                    log.info("[say] %s", cand.name)
        elif decision.prompt:
            self._say(decision.prompt)

        if decision.next_state is State.CONNECTED:
            if not self._dial():
                self._say("no_answer")
                self._go(State.OFF_HOOK)
                return

        # Re-entering OPERATOR must re-trigger the prompt and listen, and
        # _go() short-circuits when the state is unchanged, so nudge it.
        if decision.next_state is self.state is State.OPERATOR:
            self._listen_async(LISTEN_TIMEOUT)
            return
        if decision.next_state is self.state is State.CONFIRM:
            self._listen_async(CONFIRM_TIMEOUT, confirm=True)
            return

        self._go(decision.next_state)

    def _dial(self) -> bool:
        contact = self.pending_contact
        if contact is None:
            return False
        if self.telephony is None:
            log.info("[dial] %s", getattr(contact, "number", contact))
            return True
        try:
            if not self.telephony.registered:
                log.warning("not registered - cannot connect")
                return False
            self.telephony.place_call(contact.number,
                                      name=getattr(contact, "name", None))
            return True
        except Exception:
            log.exception("dial")
            return False

    def _answer(self) -> None:
        if self.telephony is None:
            return
        try:
            self.telephony.answer()
        except Exception:
            log.exception("answer")

    def _hangup(self, reason: str) -> None:
        if self.telephony is None:
            return
        try:
            self.telephony.hangup(reason)
        except Exception:
            log.exception("hangup")

    def _on_message(self, payload: dict) -> None:
        """A text arrived. Print it, unless a call is live.

        A 2.5 A print spike next to live audio is not worth the
        immediacy, so during a call it queues and prints on hangup.
        """
        sender = payload.get("sender", "unknown")
        body = payload.get("body", "")
        busy = self.state in (State.CONNECTED, State.RINGING)
        if self.printer is None:
            log.info("[print%s] %s: %s", " later" if busy else "",
                     sender, body)
            return
        try:
            self.printer.deliver(sender, body, force_queue=busy)
        except Exception:
            log.exception("printer")

    # ---- loop ----
    def run(self) -> None:
        self.running = True
        while self.running:
            try:
                event = self.events.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.handle(event)
            except Exception:
                log.exception("while handling %s", event)

    def stop(self) -> None:
        self.push(EventKind.SHUTDOWN)


# --------------------------------------------------------------------------
# wiring the real thing
# --------------------------------------------------------------------------
def build_phone(null_telephony: bool = False, no_gpio: bool = False) -> Phone:
    """Construct every service in the order that works.

    Order matters: the voice takes seconds to warm up, and the hardware
    goes last so no GPIO event arrives before there is something able to
    serve it.
    """
    import audio
    import contacts
    import printer as printer_mod
    import telephony as tel_mod

    # Load .env FIRST, before importing hardware.
    #
    # hardware.py reads KELLOGG_HOOK_INVERTED at import time, and
    # telephony.build("null") returns without touching .env - so running
    # with --null-telephony used to leave the file unread and the
    # hookswitch sense wrong, which then silently swallowed every crank.
    # Configuration belongs to the entry point, not to whichever module
    # happens to load first.
    found = tel_mod.load_env()
    if found:
        log.debug("loaded %d setting(s) from .env", len(found))

    phone = Phone()

    # 1. audio devices, by name
    # Resolve, not load. Card indices move across reboots and PortAudio
    # device names contain them, so a name saved last week can match
    # nothing today. resolve() rescans and only falls back to the file.
    try:
        cfg = audio.AudioConfig.resolve()
        log.info("audio: %s", cfg.capture_device)
    except Exception as exc:
        log.warning("no usable audio device (%s) - run: "
                    "python src/audio.py devices", exc)
        cfg = None

    # 2. contacts
    phone.book = contacts.ContactBook()
    log.info("contacts: %d", len(phone.book.all()))

    # 3. the operator's voice, slowest to start
    try:
        import tts
        phone.voice = tts.OperatorVoice(
            device=cfg.playback_device if cfg else None)
        took = phone.voice.prewarm()
        log.info("voice ready in %.1fs", took)
    except Exception:
        log.exception("tts unavailable - the operator will be silent")

    # 4. recogniser, with the grammar built from contacts
    try:
        import asr
        phone.listener = asr.Listener(
            device=cfg.capture_device if cfg else None)
        phone.listener.load()
        phone.listener.set_grammar_from_contacts(phone.book)
        lexicon = phone.listener.model_lexicon()
        if lexicon:
            unknown = phone.book.unknown_words(lexicon)
            if unknown:
                log.warning("names the model cannot hear: %s",
                            ", ".join(sorted(unknown)))
    except Exception:
        log.exception("asr unavailable - the phone cannot take requests")

    # 5. printer, and anything stranded from last run
    phone.printer = printer_mod.Printer()
    if phone.printer.available:
        printed, failed = phone.printer.flush()
        if printed or failed:
            log.info("startup flush: %d printed, %d still queued",
                     printed, failed)
    else:
        log.warning("printer: %s", phone.printer.last_error)

    # 6. telephony
    #
    # A backend that cannot start must NOT take the phone down with it.
    # The rule is in CLAUDE.md: unregistered telephony still answers
    # off-hook and the operator still speaks, it just cannot connect. A
    # missing baresip, a dead trunk or no network are all normal states
    # for a telephone in a house, not reasons to refuse to boot.
    def _wire(backend) -> None:
        backend.on_incoming_call = lambda num, name: phone.push(
            EventKind.INCOMING_CALL, number=num, name=name)
        backend.on_call_ended = lambda reason: phone.push(
            EventKind.CALL_ENDED, reason=reason)
        backend.on_message = lambda s, b, m: phone.push(
            EventKind.MESSAGE, sender=s, body=b, media=m)

    phone.telephony = tel_mod.build("null" if null_telephony else "auto")
    _wire(phone.telephony)
    try:
        phone.telephony.start()
    except Exception as exc:
        log.warning("telephony could not start (%s)", exc)
        log.warning("falling back to null - the operator will answer and "
                    "speak, but cannot connect a call")
        try:
            phone.telephony.stop()
        except Exception:
            pass
        phone.telephony = tel_mod.NullTelephony(log=lambda _: None)
        _wire(phone.telephony)
        phone.telephony.start()
    log.info("telephony: %s", phone.telephony.status())

    # 7. hardware last
    if not no_gpio:
        import hardware
        phone.hardware = hardware.PhoneHardware(
            on_off_hook=lambda: phone.push(EventKind.OFF_HOOK),
            on_on_hook=lambda: phone.push(EventKind.ON_HOOK),
            on_crank=lambda: phone.push(EventKind.CRANK),
        )
        log.info("gpio ready, handset is %s",
                 "up" if phone.hardware.off_hook else "down")

    return phone


# --------------------------------------------------------------------------
# simulate: drive the machine from stdin
# --------------------------------------------------------------------------
def _simulate() -> int:
    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    import telephony as tel_mod

    # Use the real contact book when there is one, so "? name" exercises
    # the actual lookup including the ambiguous-surname path. Falling back
    # to a couple of invented contacts keeps simulate usable on a fresh
    # clone with an empty database.
    book = None
    try:
        import contacts
        book = contacts.ContactBook()
        if not book.all():
            book.close()
            raise RuntimeError("empty")
        print(f"  using {len(book.all())} real contact(s)")
    except Exception:
        book = _FakeBook(_FakeMatch("exact", _FakeContact("Jamie")))
        print("  no contacts database - 'jamie' is the only name that works")

    phone = Phone(telephony=tel_mod.NullTelephony(log=lambda _: None),
                  book=book, voice=_FakeVoice(), printer=_FakePrinter(),
                  hardware=_FakeHardware())
    phone.telephony.start()
    phone.on_state = lambda old, new: print(f"    [{old} -> {new}]")

    print("\n  Keys:  u lift   d replace   c crank   r ring   t text")
    print("         y yes    n no        ?  say a name    q quit")
    print(f"\n  state: {phone.state}")

    worker = threading.Thread(target=phone.run, daemon=True)
    worker.start()

    try:
        while phone.running or worker.is_alive():
            try:
                raw = input("  > ").strip()
            except EOFError:
                break
            if not raw:
                continue
            key, _, rest = raw.partition(" ")
            turn = phone._turn
            if key == "q":
                break
            elif key == "u":
                phone.push(EventKind.OFF_HOOK)
            elif key == "d":
                phone.push(EventKind.ON_HOOK)
            elif key == "c":
                phone.push(EventKind.CRANK)
            elif key == "r":
                phone.push(EventKind.INCOMING_CALL, number="+15555559999")
            elif key == "t":
                phone.push(EventKind.MESSAGE, sender="+15555550123",
                           body=rest or "hello")
            elif key == "y":
                phone.push(EventKind.HEARD, turn=turn, confirm=True,
                           answer=True, status="ok", text="yes")
            elif key == "n":
                phone.push(EventKind.HEARD, turn=turn, confirm=True,
                           answer=False, status="ok", text="no")
            elif key == "?":
                phone.push(EventKind.HEARD, turn=turn, confirm=False,
                           status="ok", text=rest or "jamie")
            else:
                print("  ?")
            time.sleep(0.15)
            print(f"  state: {phone.state}  attempts: {phone.attempts}")
    except KeyboardInterrupt:
        pass
    phone.stop()
    return 0


# --------------------------------------------------------------------------
# self test
# --------------------------------------------------------------------------
class _FakeContact:
    def __init__(self, name: str, number: str = "+15555550123"):
        self.name = name
        self.number = number


class _FakeMatch:
    def __init__(self, status, contact=None, candidates=None):
        self.status = status
        self.contact = contact
        self.candidates = candidates or []


class _FakeBook:
    def __init__(self, result): self.result = result
    def lookup(self, text): return self.result
    def all(self): return []


class _FakeVoice:
    def __init__(self): self.said: list[str] = []
    def prompt(self, name): self.said.append(name)
    def say(self, text): self.said.append(f"say:{text}")
    def readback(self, name, number): self.said.append(f"readback:{name}")


class _FakePrinter:
    def __init__(self):
        self.printed: list[tuple] = []
        self.queued: list[tuple] = []
    def deliver(self, sender, body, force_queue=False):
        (self.queued if force_queue else self.printed).append((sender, body))
        return (not force_queue, "id")
    def flush(self):
        n = len(self.queued)
        self.printed.extend(self.queued)
        self.queued.clear()
        return n, 0


class _FakeTelephony:
    def __init__(self, registered=True):
        self.registered = registered
        self.calls: list[str] = []
        self.hangups: list[str] = []
        self.answered = 0
    def place_call(self, number, name=None): self.calls.append(number)
    def answer(self): self.answered += 1
    def hangup(self, reason="local"): self.hangups.append(reason)


class _FakeHardware:
    def __init__(self, off_hook=False):
        self.off_hook = off_hook
        self.amp: list[bool] = []
    def set_amp(self, on): self.amp.append(on)


def _selftest() -> int:
    checks: list[tuple[bool, str]] = []

    def ok(cond, label): checks.append((bool(cond), label))

    def fresh(**kw):
        p = Phone(hardware=_FakeHardware(), voice=_FakeVoice(),
                  printer=_FakePrinter(), telephony=_FakeTelephony(), **kw)
        return p

    # ---- decision table ----
    d = plan_response("ok", "exact", _FakeContact("Jamie"))
    ok(d.next_state is State.CONFIRM, "an exact match goes to CONFIRM")
    ok(d.prompt is None, "an exact match reads back rather than prompting")
    ok(not d.costs_try, "a successful turn costs no attempt")

    d = plan_response("ok", "confident", _FakeContact("Jamie"))
    ok(d.next_state is State.CONFIRM, "a confident match goes to CONFIRM")

    d = plan_response("ok", "ambiguous", None,
                      [_FakeContact("Ann Reed"), _FakeContact("Bob Reed")])
    ok(d.prompt == "which_one", "ambiguity asks which one")
    ok(len(d.candidates) == 2, "ambiguity carries the candidates")
    ok(d.next_state is State.OPERATOR, "ambiguity stays in OPERATOR")

    d = plan_response("ok", "none")
    ok(d.prompt == "not_found", "an unknown name says so")

    for status in ("low_confidence", "unknown"):
        d = plan_response(status, "exact", _FakeContact("Jamie"))
        ok(d.prompt == "retry", f"{status} retries")
        ok(d.next_state is State.OPERATOR, f"{status} does not go to CONFIRM")
        ok(d.contact is None,
           f"{status} never carries a contact, however good the match")

    for status in ("silence", "timeout"):
        d = plan_response(status)
        ok(d.prompt == "timeout", f"{status} says timeout")

    d = plan_response("ok", "none", attempts=MAX_ATTEMPTS - 1)
    ok(d.next_state is State.OFF_HOOK, "the last attempt gives up")
    ok(d.prompt == "goodbye", "giving up says goodbye")

    d = plan_response("something_new")
    ok(d.next_state is State.OPERATOR,
       "an unrecognised ASR status retries rather than dialling")

    ok(plan_confirm(True).next_state is State.CONNECTED, "yes connects")
    ok(plan_confirm(False).next_state is State.OPERATOR, "no starts over")
    ok(plan_confirm(False).prompt == "greeting", "no re-greets")
    ok(plan_confirm(None).next_state is State.CONFIRM, "unclear re-asks")
    ok(plan_confirm(None, MAX_ATTEMPTS - 1).next_state is State.OFF_HOOK,
       "unclear too often gives up")

    # ---- transitions ----
    p = fresh()
    p.handle(Event(EventKind.CRANK))
    ok(p.state is State.IDLE, "a crank while on-hook is ignored")

    p.handle(Event(EventKind.OFF_HOOK))
    ok(p.state is State.OFF_HOOK, "lifting the handset leaves IDLE")
    ok(p.hardware.amp[-1] is True, "the amp comes on off-hook")

    p.handle(Event(EventKind.CRANK))
    ok(p.state is State.OPERATOR, "cranking off-hook summons the operator")
    ok("greeting" in p.voice.said, "the operator greets")

    p.handle(Event(EventKind.CRANK))
    ok(p.state is State.OPERATOR, "a second crank mid-prompt is ignored")

    p.handle(Event(EventKind.ON_HOOK))
    ok(p.state is State.IDLE, "replacing the handset returns to IDLE")
    ok(p.hardware.amp[-1] is False, "the amp goes off on-hook")
    ok(p.telephony.hangups, "replacing the handset hangs up")

    # hang up from every state
    for state in (State.OFF_HOOK, State.OPERATOR, State.CONFIRM,
                  State.CONNECTED, State.RINGING):
        p = fresh()
        p.state = state
        p.handle(Event(EventKind.ON_HOOK))
        ok(p.state is State.IDLE, f"ON_HOOK from {state} reaches IDLE")

    # inbound
    p = fresh()
    p.handle(Event(EventKind.INCOMING_CALL, {"number": "+1555"}))
    ok(p.state is State.RINGING, "an inbound call rings")
    p.handle(Event(EventKind.OFF_HOOK))
    ok(p.state is State.CONNECTED, "lifting the handset answers")
    ok(p.telephony.answered == 1, "answering calls through to telephony")

    p = fresh()
    p.state = State.CONNECTED
    p.handle(Event(EventKind.INCOMING_CALL, {"number": "+1555"}))
    ok(p.state is State.CONNECTED, "an inbound call mid-call is ignored")

    p = fresh()
    p.hardware.off_hook = True
    p.state = State.CONNECTED
    p.handle(Event(EventKind.CALL_ENDED, {"reason": "remote"}))
    ok(p.state is State.OFF_HOOK,
       "the far end hanging up leaves you off-hook, not idle")

    # ---- stale ASR results ----
    p = fresh(book=_FakeBook(_FakeMatch("exact", _FakeContact("Jamie"))))
    p.state = State.OPERATOR
    stale = p._turn - 1
    p.handle(Event(EventKind.HEARD,
                   {"turn": stale, "status": "ok", "text": "jamie"}))
    ok(p.state is State.OPERATOR,
       "an ASR result from a previous turn is discarded")

    p.handle(Event(EventKind.HEARD,
                   {"turn": p._turn, "status": "ok", "text": "jamie"}))
    ok(p.state is State.CONFIRM, "a current ASR result is acted on")
    ok(p.pending_contact is not None, "the contact is held for readback")
    ok(any(s.startswith("readback:") for s in p.voice.said),
       "the number is read back before connecting")

    p.handle(Event(EventKind.HEARD,
                   {"turn": p._turn, "confirm": True, "answer": True,
                    "status": "ok", "text": "yes"}))
    ok(p.state is State.CONNECTED, "yes connects the call")
    ok(p.telephony.calls == ["+15555550123"], "the right number was dialled")

    # ---- a cough must not place a call ----
    p = fresh(book=_FakeBook(_FakeMatch("exact", _FakeContact("Jamie"))))
    p.state = State.OPERATOR
    p.handle(Event(EventKind.HEARD,
                   {"turn": p._turn, "status": "low_confidence",
                    "text": "jamie"}))
    ok(p.state is State.OPERATOR, "a low-confidence turn does not dial")
    ok(p.telephony.calls == [], "nothing was dialled on a bad transcription")

    # ---- retry budget ----
    p = fresh(book=_FakeBook(_FakeMatch("none")))
    p.state = State.OPERATOR
    for _ in range(MAX_ATTEMPTS):
        p.handle(Event(EventKind.HEARD,
                       {"turn": p._turn, "status": "ok", "text": "nobody"}))
    ok(p.state is State.OFF_HOOK,
       f"{MAX_ATTEMPTS} failures return to OFF_HOOK rather than looping")
    ok("goodbye" in p.voice.said, "giving up is announced")

    # ---- unregistered telephony ----
    p = fresh(book=_FakeBook(_FakeMatch("exact", _FakeContact("Jamie"))))
    p.telephony.registered = False
    p.state = State.CONFIRM
    p.pending_contact = _FakeContact("Jamie")
    p.handle(Event(EventKind.HEARD,
                   {"turn": p._turn, "confirm": True, "answer": True,
                    "status": "ok", "text": "yes"}))
    ok(p.state is State.OFF_HOOK, "an unconnectable call does not pretend")
    ok("no_answer" in p.voice.said, "the operator says it could not connect")

    # ---- messages ----
    p = fresh()
    p.handle(Event(EventKind.MESSAGE,
                   {"sender": "+1555", "body": "hello"}))
    ok(p.printer.printed, "a text while idle prints immediately")

    p = fresh()
    p.state = State.CONNECTED
    p.handle(Event(EventKind.MESSAGE,
                   {"sender": "+1555", "body": "during a call"}))
    ok(not p.printer.printed, "a text during a call does not print")
    ok(p.printer.queued, "a text during a call is queued instead")
    p.handle(Event(EventKind.ON_HOOK))
    ok(p.printer.printed, "the queue flushes when the handset goes down")
    ok(not p.printer.queued, "nothing is left queued after the flush")

    # ---- a phone with no services at all must not crash ----
    bare = Phone()
    for kind in EventKind:
        if kind is EventKind.SHUTDOWN:
            continue
        bare.handle(Event(kind, {"sender": "x", "body": "y", "turn": 0}))
    ok(True, "every event is survivable with no services wired up")

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
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Kellogg magneto phone")
    ap.add_argument("command", nargs="?", default="run",
                    choices=["run", "selftest"])
    ap.add_argument("--null-telephony", action="store_true",
                    help="never place a real call")
    ap.add_argument("--simulate", action="store_true",
                    help="drive the state machine from stdin, no GPIO")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    if args.command == "selftest":
        return _selftest()
    if args.simulate:
        return _simulate()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S")

    phone = build_phone(null_telephony=args.null_telephony)

    def shutdown(_sig, _frm):
        log.info("shutting down")
        phone.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # If the handset is already up at boot, do not pretend it is down.
    if phone.hardware is not None and phone.hardware.off_hook:
        phone.push(EventKind.OFF_HOOK)

    log.info("ready - lift the handset and turn the crank")
    try:
        phone.run()
    finally:
        if phone.telephony is not None:
            phone.telephony.stop()
        if phone.hardware is not None:
            phone.hardware.close()
        if phone.book is not None:
            phone.book.close()
        if phone.printer is not None:
            phone.printer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())