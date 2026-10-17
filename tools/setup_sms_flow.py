#!/usr/bin/env python3
"""Build the Kellogg SMS keyword flow in Twilio Studio, from the API.

This replaces clicking widgets around the Studio canvas. The flow it builds
is the one described to the carriers during A2P 10DLC registration, so the
message text here and the message text in the campaign filing must stay in
step. Edit MESSAGES below and re-run with --update to change both.

Usage
    python tools/setup_sms_flow.py --dry-run
        print the flow definition and the compliance check, touch nothing

    python tools/setup_sms_flow.py
        create the flow as a DRAFT. Open the printed URL, look at it, then

    python tools/setup_sms_flow.py --update FWxxxx --publish
        publish it

    python tools/setup_sms_flow.py --attach FWxxxx
        point the phone number at the flow

Credentials come from the environment or a .env file beside the project root:
    TWILIO_ACCOUNT_SID=ACxxxxxxxx
    TWILIO_AUTH_TOKEN=xxxxxxxx
    TWILIO_PHONE_NUMBER=+17573839589

.env must be gitignored. This script never prints the auth token.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------
# the campaign. these strings are what was filed with the carriers.
# --------------------------------------------------------------------------
BRAND = "Kellogg"
KEYWORD = "KELLOGG"
CONFIRM = "Y"
PRIVACY_URL = "https://g3vin.github.io/kellogg/"
TERMS_URL = "https://g3vin.github.io/kellogg/terms.html"
CONTACT_EMAIL = "hello@gavinbowden.me"

MESSAGES = {
    "welcome": (
        f"{BRAND}: Reply {CONFIRM} to confirm you want texts from our "
        f"restored 1900s telephone. Msg frequency varies. Msg&data rates "
        f"may apply. Reply STOP to cancel, HELP for help. "
        f"Terms: {TERMS_URL} Privacy: {PRIVACY_URL}"
    ),
    "confirmed": (
        f"{BRAND}: You're confirmed. Your messages will print on the "
        f"telephone's receipt printer. Msg&data rates may apply. "
        f"Reply STOP to cancel, HELP for help."
    ),
    "help": (
        f"{BRAND}: A personal restored telephone project. Questions: "
        f"{CONTACT_EMAIL}. Msg&data rates may apply. Reply STOP to cancel."
    ),
    "stop": (
        f"{BRAND}: You've been unsubscribed and will receive no further "
        f"messages. Reply {KEYWORD} to rejoin."
    ),
    # anything that is not a keyword. this is the path a real text message
    # takes, so it must not look like a bot wall.
    "passthrough": (
        f"{BRAND}: Message received - it is printing on the telephone now. "
        f"Reply STOP to cancel, HELP for help."
    ),
}

# What the carriers require to appear in the first message a subscriber sees.
REQUIRED_IN_WELCOME = [BRAND, "Msg&data rates may apply", "STOP", "HELP"]

STUDIO_API = "https://studio.twilio.com/v2/Flows"
API_BASE = "https://api.twilio.com/2010-04-01/Accounts"
FLOW_WEBHOOK = "https://webhooks.twilio.com/v1/Accounts/{sid}/Flows/{flow}"


# --------------------------------------------------------------------------
# flow definition
# --------------------------------------------------------------------------
def _transition(event: str, target: str | None = None,
                conditions: list | None = None) -> dict:
    """One transition.

    The `next` key must be OMITTED for a terminal transition, not set to
    null. Twilio's own documented example shows "next": null, and the API
    rejects it - the validation error surfaces as a pile of misleading
    "must be a constant value" messages on the event field.
    """
    t: dict = {"event": event}
    if target is not None:
        t["next"] = target
    if conditions is not None:
        t["conditions"] = conditions
    return t


def _send(name: str, body: str, x: int, y: int) -> dict:
    return {
        "type": "send-message",
        "name": name,
        "properties": {
            "body": body,
            "from": "{{flow.channel.address}}",
            "to": "{{contact.channel.address}}",
            "offset": {"x": x, "y": y},
        },
        "transitions": [
            _transition("sent"),
            _transition("failed"),
        ],
    }


def _match(label: str, value: str, target: str, test: str) -> dict:
    return _transition("match", target, [{
        "friendly_name": label,
        "arguments": [test],
        "type": "equal_to",
        "value": value,
    }])


def build_definition() -> dict:
    """The whole flow. One split, five replies."""
    # strip and upcase so " kellogg " matches KELLOGG. Studio runs Liquid.
    test = "{{trigger.message.Body | strip | upcase}}"

    states = [
        {
            "type": "trigger",
            "name": "Trigger",
            "properties": {"offset": {"x": 0, "y": 0}},
            "transitions": [
                _transition("incomingMessage", "route_keyword"),
                _transition("incomingCall"),
                _transition("incomingRequest"),
            ],
        },
        {
            "type": "split-based-on",
            "name": "route_keyword",
            "properties": {
                "input": test,
                "offset": {"x": 0, "y": 200},
            },
            "transitions": [
                # noMatch first: a plain text message is the common case.
                _transition("noMatch", "reply_passthrough"),
                _match("keyword", KEYWORD, "reply_welcome", test),
                _match("confirm", CONFIRM, "reply_confirmed", test),
                _match("help", "HELP", "reply_help", test),
                _match("stop", "STOP", "reply_stop", test),
            ],
        },
        _send("reply_welcome", MESSAGES["welcome"], -520, 430),
        _send("reply_confirmed", MESSAGES["confirmed"], -260, 430),
        _send("reply_help", MESSAGES["help"], 0, 430),
        _send("reply_stop", MESSAGES["stop"], 260, 430),
        _send("reply_passthrough", MESSAGES["passthrough"], 520, 430),
    ]

    return {
        "description": f"{BRAND} SMS keyword flow - A2P 10DLC registered",
        "states": states,
        "initial_state": "Trigger",
        "flags": {"allow_concurrent_calls": True},
    }


# --------------------------------------------------------------------------
# compliance check - run before anything is sent to Twilio
# --------------------------------------------------------------------------
def check_messages() -> list[str]:
    """Return a list of problems. Empty list means the copy is compliant."""
    problems: list[str] = []

    welcome = MESSAGES["welcome"]
    for needle in REQUIRED_IN_WELCOME:
        if needle.lower() not in welcome.lower():
            problems.append(f"welcome message is missing {needle!r}")

    for key, body in MESSAGES.items():
        if BRAND.lower() not in body.lower():
            problems.append(f"{key!r} does not identify the brand {BRAND!r}")
        if "[" in body or "]" in body:
            problems.append(f"{key!r} still contains a [PLACEHOLDER]")

    if TERMS_URL not in welcome or PRIVACY_URL not in welcome:
        problems.append("welcome message should link terms and privacy")

    return problems


def segments(body: str) -> int:
    """How many SMS segments this body will bill as."""
    gsm = all(ord(c) < 128 for c in body)
    limit_single, limit_multi = (160, 153) if gsm else (70, 67)
    if len(body) <= limit_single:
        return 1
    return -(-len(body) // limit_multi)


# --------------------------------------------------------------------------
# credentials and HTTP
# --------------------------------------------------------------------------
def load_env() -> tuple[str, str, str]:
    """Read credentials from the environment, falling back to .env."""
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        env = parent / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
            break

    missing = [k for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN")
               if not os.environ.get(k)]
    if missing:
        sys.exit(
            "Missing credentials: " + ", ".join(missing) + "\n"
            "Put them in the environment or in a gitignored .env file:\n"
            "    TWILIO_ACCOUNT_SID=ACxxxxxxxx\n"
            "    TWILIO_AUTH_TOKEN=xxxxxxxx\n"
            "    TWILIO_PHONE_NUMBER=+17573839589"
        )
    return (os.environ["TWILIO_ACCOUNT_SID"],
            os.environ["TWILIO_AUTH_TOKEN"],
            os.environ.get("TWILIO_PHONE_NUMBER", ""))


def request(method: str, url: str, sid: str, token: str,
            data: dict | None = None) -> dict:
    """One Twilio REST call. Raises SystemExit with the real error text."""
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    auth = base64.b64encode(f"{sid}:{token}".encode()).decode()
    req.add_header("Authorization", f"Basic {auth}")
    if body:
        req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        try:
            parsed = json.loads(detail)
            detail = json.dumps(parsed, indent=2)
        except Exception:
            pass
        sys.exit(f"Twilio returned {exc.code} for {method} {url}\n{detail}")
    except urllib.error.URLError as exc:
        sys.exit(f"Could not reach Twilio: {exc.reason}")


# --------------------------------------------------------------------------
# actions
# --------------------------------------------------------------------------
def create_flow(sid: str, token: str, publish: bool) -> dict:
    return request("POST", STUDIO_API, sid, token, {
        "FriendlyName": f"{BRAND.lower()}-sms",
        "Status": "published" if publish else "draft",
        "Definition": json.dumps(build_definition()),
        "CommitMessage": "created by setup_sms_flow.py",
    })


def update_flow(sid: str, token: str, flow_sid: str, publish: bool) -> dict:
    return request("POST", f"{STUDIO_API}/{flow_sid}", sid, token, {
        "Status": "published" if publish else "draft",
        "Definition": json.dumps(build_definition()),
        "CommitMessage": "updated by setup_sms_flow.py",
    })


def attach_number(sid: str, token: str, number: str, flow_sid: str) -> dict:
    if not number:
        sys.exit("TWILIO_PHONE_NUMBER is not set, so there is nothing to "
                 "attach the flow to.")

    # Pointing a live number at a flow that does not exist silently breaks
    # every incoming message, so prove the flow is real and published first.
    if not (flow_sid.startswith("FW") and len(flow_sid) == 34):
        sys.exit(f"{flow_sid!r} is not a flow SID. It should start with FW "
                 "and be 34 characters. Did you paste the placeholder?")
    flow = request("GET", f"{STUDIO_API}/{flow_sid}", sid, token)
    if flow.get("status") != "published":
        sys.exit(f"Flow {flow_sid} is {flow.get('status')!r}, not published. "
                 "Publish it before attaching, or incoming messages will "
                 "reach a flow that cannot run:\n"
                 f"  python tools/setup_sms_flow.py --update {flow_sid} "
                 "--publish")

    query = urllib.parse.urlencode({"PhoneNumber": number})
    listing = request(
        "GET", f"{API_BASE}/{sid}/IncomingPhoneNumbers.json?{query}",
        sid, token)
    numbers = listing.get("incoming_phone_numbers", [])
    if not numbers:
        sys.exit(f"{number} is not an active number on this account. "
                 "Check the format, which must be E.164 (+17573839589).")

    pn_sid = numbers[0]["sid"]
    hook = FLOW_WEBHOOK.format(sid=sid, flow=flow_sid)
    return request(
        "POST", f"{API_BASE}/{sid}/IncomingPhoneNumbers/{pn_sid}.json",
        sid, token, {"SmsUrl": hook, "SmsMethod": "POST"})


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create the Kellogg SMS keyword flow in Twilio Studio")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the definition and checks, call nothing")
    ap.add_argument("--update", metavar="FLOW_SID",
                    help="update an existing flow instead of creating one")
    ap.add_argument("--publish", action="store_true",
                    help="publish rather than leaving it a draft")
    ap.add_argument("--attach", metavar="FLOW_SID",
                    help="point TWILIO_PHONE_NUMBER at this flow")
    args = ap.parse_args()

    problems = check_messages()
    print("Message copy")
    for key, body in MESSAGES.items():
        print(f"  {key:12s} {len(body):4d} chars, {segments(body)} segment(s)")
    if problems:
        print("\nCompliance problems:")
        for p in problems:
            print(f"  - {p}")
        sys.exit("\nFix the copy before registering or sending anything.")
    print("  all messages carry the brand, rates notice, STOP and HELP\n")

    if args.dry_run:
        print(json.dumps(build_definition(), indent=2))
        print("\nDry run. Nothing was sent to Twilio.")
        return

    account_sid, token, number = load_env()

    if args.attach:
        attach_number(account_sid, token, number, args.attach)
        print(f"{number} now routes incoming SMS to {args.attach}")
        print("Text the keyword to it and watch the Studio execution log.")
        return

    if args.update:
        if not (args.update.startswith("FW") and len(args.update) == 34):
            sys.exit(f"{args.update!r} is not a flow SID. Use the one the "
                     "create step printed.")
        flow = update_flow(account_sid, token, args.update, args.publish)
    else:
        flow = create_flow(account_sid, token, args.publish)

    status = flow.get("status", "?")
    flow_sid = flow["sid"]
    print(f"Flow {flow_sid} is {status}")
    print(f"  https://console.twilio.com/us1/develop/studio/flows/{flow_sid}")
    print()
    if status != "published":
        print("Open it, check the canvas looks right, then publish:")
        print(f"  python tools/setup_sms_flow.py --update {flow_sid} --publish")
    else:
        print("Attach it to the phone number:")
        print(f"  python tools/setup_sms_flow.py --attach {flow_sid}")


if __name__ == "__main__":
    main()