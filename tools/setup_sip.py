#!/usr/bin/env python3
"""Set up the SIP trunk: Twilio side and baresip side.

The phone reaches the telephone network by registering a SIP client
against a Twilio SIP Domain. telephony.py drives that client over its
ctrl_tcp interface on 127.0.0.1:4444; this script is what puts the client
and the Twilio configuration in place.

Run it in three steps, checking each before the next:

    python tools/setup_sip.py plan
        print what will be created and what it needs. Touches nothing.

    python tools/setup_sip.py twilio
        create the SIP domain, credential list and mappings on Twilio.
        Writes the generated SIP password into .env.

    python tools/setup_sip.py baresip
        write ~/.baresip/accounts and config, and the user unit, with
        the audio device filled in from audio.py.

    python tools/setup_sip.py status
        is baresip running, is it registered, does Twilio agree.

One thing Twilio does NOT expose over the API is a TwiML Bin, so the
outbound dial plan has to be pasted into the console once. `plan` prints
the exact TwiML and the link; after that, pass --voice-url.

Credentials come from .env:
    TWILIO_ACCOUNT_SID   TWILIO_AUTH_TOKEN   TWILIO_PHONE_NUMBER
and this script adds:
    SIP_DOMAIN   SIP_USERNAME   SIP_PASSWORD
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import re
import secrets
import socket
import string
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
BARESIP_DIR = Path.home() / ".baresip"
UNIT_DIR = Path.home() / ".config" / "systemd" / "user"

API = "https://api.twilio.com/2010-04-01/Accounts"
CTRL_HOST, CTRL_PORT = "127.0.0.1", 4444

DEFAULT_USER = "kellogg"


class SetupError(RuntimeError):
    """Something is wrong and the message says what."""


# --------------------------------------------------------------------------
# env
# --------------------------------------------------------------------------
def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    found: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            found[k.strip()] = v.strip().strip("'\"")
            os.environ.setdefault(k.strip(), found[k.strip()])
    return found


def save_env(updates: dict[str, str], path: Path = ENV_PATH) -> None:
    """Add or replace keys in .env, preserving everything else.

    Rewrites in place rather than appending, so running this twice does
    not leave two SIP_PASSWORD lines with the later one silently winning.
    """
    lines = path.read_text().splitlines() if path.is_file() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")
    path.write_text("\n".join(out).rstrip("\n") + "\n")
    os.chmod(path, 0o600)


def require(name: str, hint: str = "") -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SetupError(f"{name} is not set in {ENV_PATH}."
                         + (f"\n  {hint}" if hint else ""))
    return value


def make_password(n: int = 24) -> str:
    """A SIP password Twilio will accept.

    Letters and digits only: some SIP stacks mangle punctuation in the
    digest calculation and the failure looks like a wrong password.
    """
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))


# --------------------------------------------------------------------------
# twilio rest
# --------------------------------------------------------------------------
def api(method: str, url: str, data: Optional[dict] = None) -> dict:
    sid = require("TWILIO_ACCOUNT_SID")
    token = require("TWILIO_AUTH_TOKEN")
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    auth = base64.b64encode(f"{sid}:{token}".encode()).decode()
    req.add_header("Authorization", f"Basic {auth}")
    if body:
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        try:
            detail = json.dumps(json.loads(detail), indent=2)
        except Exception:
            pass
        raise SetupError(
            f"Twilio returned {exc.code} for {method} {url}\n{detail}")
    except urllib.error.URLError as exc:
        raise SetupError(f"Could not reach Twilio: {exc.reason}")


def acct() -> str:
    return require("TWILIO_ACCOUNT_SID")


def find_domain(name: str) -> Optional[dict]:
    data = api("GET", f"{API}/{acct()}/SIP/Domains.json?PageSize=50")
    for dom in data.get("domains", []):
        if dom.get("domain_name", "").lower() == name.lower():
            return dom
    return None


def find_credential_list(friendly: str) -> Optional[dict]:
    data = api("GET", f"{API}/{acct()}/SIP/CredentialLists.json?PageSize=50")
    for cl in data.get("credential_lists", []):
        if cl.get("friendly_name") == friendly:
            return cl
    return None


# --------------------------------------------------------------------------
# the TwiML Twilio cannot be given over the API
# --------------------------------------------------------------------------
def outbound_twiml(caller_id: str) -> str:
    """Dial plan for calls PLACED by the phone.

    When the handset dials, Twilio POSTs to the domain's Voice URL with
    the dialled SIP URI. This dials it onward to the PSTN, showing your
    Twilio number as the caller.

    The dialled SIP URI arrives as sip:+15555550123@domain, and a Bin
    cannot run code to strip it. Twilio exposes the user part separately
    as {{To}} for SIP-originated calls in most setups; if your Bin
    produces a malformed number, switch the Voice URL to a Function and
    parse it there.
    """
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Dial callerId="{caller_id}" answerOnBridge="true">
    <Number>{{{{To}}}}</Number>
  </Dial>
</Response>'''


def inbound_twiml(sip_user: str, domain: str) -> str:
    """Dial plan for calls arriving AT your Twilio number."""
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Dial timeout="30" answerOnBridge="true">
    <Sip>sip:{sip_user}@{domain}</Sip>
  </Dial>
</Response>'''


# --------------------------------------------------------------------------
# baresip
# --------------------------------------------------------------------------
def baresip_binary() -> Optional[str]:
    for path in ("/usr/bin/baresip", "/usr/local/bin/baresip"):
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def module_path() -> Optional[str]:
    """Where this build of baresip keeps its modules.

    The path is architecture-specific - aarch64-linux-gnu on a 64-bit
    Pi - so it is found rather than assumed. A wrong module_path makes
    baresip start and silently load nothing, which looks like a
    configuration error anywhere but here.
    """
    for pattern in ("/usr/lib/*/baresip/modules",
                    "/usr/local/lib/baresip/modules",
                    "/usr/lib/baresip/modules"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


def alsa_device() -> str:
    """The ALSA device baresip should use.

    Prefers the shared "kellogg" device from ~/.asoundrc when it exists.
    That matters: a raw hw/plughw device is EXCLUSIVE, so baresip holding
    it during a call locks the Python app out of the operator's voice
    entirely. dmix/dsnoop let both have it.

    Falls back to the card by NAME, never by index - indices move
    between boots and this card has been both 1 and 3 on the same Pi.
    """
    asoundrc = Path.home() / ".asoundrc"
    if asoundrc.is_file() and "pcm.kellogg " in asoundrc.read_text():
        return "kellogg"
    return "plughw:CARD=Device,DEV=0"


# baresip writes a complete default config on first run. Patching that
# is far safer than writing one from scratch: the module set is large,
# split between `module` and `module_app`, and version-dependent. The
# first attempt here omitted account.so, so baresip started cleanly,
# read no accounts, and never registered - a silent failure that looks
# exactly like bad credentials.
REQUIRED_MODULES = (
    "account.so",      # reads ~/.baresip/accounts. Without it, nothing
                       # registers and baresip still says it is ready.
    "contact.so",
    "menu.so",
    "ctrl_tcp.so",     # how telephony.py drives the phone
)


def generate_default_config(binary: str) -> None:
    """Start baresip once so it writes its own config, then stop it."""
    proc = subprocess.Popen(
        [binary, "-f", str(BARESIP_DIR)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def patch_config(text: str, device: str) -> str:
    """Turn baresip's default config into ours.

    Every edit is a substitution on a line that already exists, because
    appending a duplicate key makes baresip use whichever it read first
    and the other silently does nothing.
    """
    lines = text.splitlines()

    def set_key(key: str, value: str) -> None:
        pattern = re.compile(rf"^#?\s*{re.escape(key)}\s+\S.*$")
        for i, line in enumerate(lines):
            if pattern.match(line):
                lines[i] = f"{key}\t\t\t{value}"
                return
        lines.append(f"{key}\t\t\t{value}")

    set_key("audio_player", f"alsa,{device}")
    set_key("audio_source", f"alsa,{device}")
    set_key("audio_alert", f"alsa,{device}")
    set_key("ctrl_tcp_listen", f"{CTRL_HOST}:{CTRL_PORT}")

    # Uncomment the modules we need, keeping whichever keyword the
    # generated file used for each - module vs module_app is not ours
    # to decide and differs between builds.
    for mod in REQUIRED_MODULES:
        found = False
        for i, line in enumerate(lines):
            m = re.match(rf"^#\s*(module(?:_app)?)\s+{re.escape(mod)}\s*$",
                         line.strip())
            if m:
                lines[i] = f"{m.group(1)}\t\t\t{mod}"
                found = True
                break
            if re.match(rf"^(module(?:_app)?)\s+{re.escape(mod)}\s*$",
                        line.strip()):
                found = True
                break
        if not found:
            keyword = "module_app" if mod in (
                "account.so", "contact.so", "menu.so",
                "ctrl_tcp.so") else "module"
            lines.append(f"{keyword}\t\t\t{mod}")

    return "\n".join(lines) + "\n"


BARESIP_ACCOUNT = (
    "<sip:{user}@{domain}>"
    ";auth_user={user}"
    ";auth_pass={password}"
    ";answermode=manual"
    ";regint=600"
    ";medianat=stun"
    ";stunserver=stun:global.stun.twilio.com:3478"
    ";audio_codecs=PCMU/8000/1,PCMA/8000/1,opus/48000/2"
    "\n"
)

BARESIP_UNIT = """\
# baresip for the Kellogg phone. Generated by tools/setup_sip.py.
#
# A user unit, matching kellogg.service, because both need the same
# session to reach the audio device.

[Unit]
Description=baresip SIP client for the Kellogg phone
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={binary} -f %h/.baresip
Restart=always
RestartSec=5
StartLimitIntervalSec=300
StartLimitBurst=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=baresip

[Install]
WantedBy=default.target
"""


def write_baresip(user: str, domain: str, password: str) -> list[Path]:
    binary = baresip_binary()
    if not binary:
        raise SetupError(
            "baresip is not installed.\n"
            "    sudo apt install -y baresip\n"
            "then run this again.")
    modules = module_path()
    if not modules:
        raise SetupError(
            "Found the baresip binary but not its modules directory. "
            "Look for it with:  dpkg -L baresip-core | grep modules")

    BARESIP_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    cfg = BARESIP_DIR / "config"
    if not cfg.is_file() or cfg.stat().st_size < 500:
        print("  asking baresip to write its default config...")
        generate_default_config(binary)
    if not cfg.is_file():
        raise SetupError(
            f"baresip did not write {cfg}. Try running it once by hand:\n"
            f"    {binary} -f {BARESIP_DIR}\n"
            f"then press q to quit, and run this again.")

    device = alsa_device()
    cfg.write_text(patch_config(cfg.read_text(), device))
    written.append(cfg)
    _ = modules

    acc = BARESIP_DIR / "accounts"
    acc.write_text(BARESIP_ACCOUNT.format(
        user=user, domain=domain, password=password))
    # The SIP password is in here in clear text, as baresip requires.
    os.chmod(acc, 0o600)
    written.append(acc)

    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    unit = UNIT_DIR / "baresip.service"
    unit.write_text(BARESIP_UNIT.format(binary=binary))
    written.append(unit)

    return written


def ctrl_reachable(timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((CTRL_HOST, CTRL_PORT), timeout):
            return True
    except OSError:
        return False


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_plan(args) -> int:
    load_env()
    number = os.environ.get("TWILIO_PHONE_NUMBER", "+1XXXXXXXXXX")
    domain = args.domain or f"{DEFAULT_USER}-{os.urandom(3).hex()}"
    full = domain if "." in domain else f"{domain}.sip.twilio.com"
    user = args.user

    print(f"""
Plan
----
  SIP domain        {full}
  SIP username      {user}
  SIP password      generated, written to {ENV_PATH}
  Caller ID         {number}

On Twilio, this script will create:
  * the SIP Domain, with registration enabled
  * a credential list holding {user}
  * mappings so that credential list authorises BOTH registration and
    calls - both are needed, and missing the second gives a client that
    registers happily and cannot dial

On this Pi, it will write:
  * ~/.baresip/config     with ctrl_tcp on {CTRL_HOST}:{CTRL_PORT}
  * ~/.baresip/accounts   mode 600, SIP password in clear (baresip
                          has no other option)
  * ~/.config/systemd/user/baresip.service

One thing it CANNOT do
----------------------
Twilio has no REST API for TwiML Bins, so the dial plans are pasted in
by hand, once. Create these two at:

  https://console.twilio.com/us1/develop/twiml-bins

OUTBOUND - name it "kellogg-outbound". This is the domain's Voice URL,
used when the handset places a call:

{outbound_twiml(number)}

INBOUND - name it "kellogg-inbound". Set it as the Voice webhook on
{number}, so calls to your number ring the handset:

{inbound_twiml(user, full)}

Then:
  python tools/setup_sip.py twilio --domain {domain} \\
      --voice-url https://handler.twilio.com/twiml/EHxxxxxxxx
  sudo apt install -y baresip
  python tools/setup_sip.py baresip
  systemctl --user enable --now baresip
  python tools/setup_sip.py status
""")
    return 0


def cmd_twilio(args) -> int:
    load_env()
    number = require("TWILIO_PHONE_NUMBER")
    if not args.voice_url:
        raise SetupError(
            "--voice-url is required: the TwiML Bin URL for outbound "
            "calls.\n  Run 'plan' to see the TwiML to paste, then come "
            "back with the Bin's URL.")

    domain = args.domain or DEFAULT_USER
    full = domain if "." in domain else f"{domain}.sip.twilio.com"
    user = args.user

    existing = find_domain(full)
    if existing:
        print(f"  SIP domain {full} already exists ({existing['sid']})")
        dom = api("POST", f"{API}/{acct()}/SIP/Domains/{existing['sid']}.json",
                  {"VoiceUrl": args.voice_url, "VoiceMethod": "POST",
                   "SipRegistration": "true"})
        print("  updated its voice url and registration setting")
    else:
        try:
            dom = api("POST", f"{API}/{acct()}/SIP/Domains.json", {
                "DomainName": full,
                "FriendlyName": "Kellogg magneto phone",
                "VoiceUrl": args.voice_url,
                "VoiceMethod": "POST",
                "SipRegistration": "true",
            })
        except SetupError as exc:
            if "21232" in str(exc) or "already exists" in str(exc):
                raise SetupError(
                    f"{full} is taken.\n\n"
                    f"  SIP domain names are unique across ALL of Twilio, "
                    f"not just your account,\n"
                    f"  so a common word like 'kellogg' is likely gone. "
                    f"It is not on your\n"
                    f"  account either - this script checked.\n\n"
                    f"  Pick something unlikely to collide:\n"
                    f"      --domain kellogg-{os.urandom(3).hex()}\n"
                ) from None
            raise
        print(f"  created SIP domain {full} ({dom['sid']})")
    dom_sid = dom["sid"]

    friendly = "kellogg-phone"
    cl = find_credential_list(friendly)
    password = make_password()
    if cl:
        cl_sid = cl["sid"]
        print(f"  credential list {friendly} exists ({cl_sid})")
        creds = api("GET", f"{API}/{acct()}/SIP/CredentialLists/"
                           f"{cl_sid}/Credentials.json")
        mine = [c for c in creds.get("credentials", [])
                if c.get("username") == user]
        if mine:
            api("POST", f"{API}/{acct()}/SIP/CredentialLists/{cl_sid}"
                        f"/Credentials/{mine[0]['sid']}.json",
                {"Password": password})
            print(f"  rotated the password for {user}")
        else:
            api("POST", f"{API}/{acct()}/SIP/CredentialLists/{cl_sid}"
                        f"/Credentials.json",
                {"Username": user, "Password": password})
            print(f"  added credential {user}")
    else:
        cl = api("POST", f"{API}/{acct()}/SIP/CredentialLists.json",
                 {"FriendlyName": friendly})
        cl_sid = cl["sid"]
        api("POST", f"{API}/{acct()}/SIP/CredentialLists/{cl_sid}"
                    f"/Credentials.json",
            {"Username": user, "Password": password})
        print(f"  created credential list {friendly} with user {user}")

    # Both mappings. Registration alone produces a client that registers
    # and then gets 403 on every INVITE.
    for kind, label in (("Registrations", "registration"),
                        ("Calls", "calls")):
        url = (f"{API}/{acct()}/SIP/Domains/{dom_sid}/Auth/{kind}"
               f"/CredentialListMappings.json")
        try:
            api("POST", url, {"CredentialListSid": cl_sid})
            print(f"  authorised the credential list for {label}")
        except SetupError as exc:
            if "already" in str(exc).lower() or "21210" in str(exc):
                print(f"  {label} mapping already in place")
            else:
                raise

    save_env({"SIP_DOMAIN": full, "SIP_USERNAME": user,
              "SIP_PASSWORD": password})
    print(f"\n  wrote SIP_DOMAIN, SIP_USERNAME and SIP_PASSWORD to {ENV_PATH}")
    print(f"  (mode 600 - it holds a password and this repo is public)")
    print("\n  Still to do by hand: point the voice webhook on "
          f"{number}\n  at your kellogg-inbound TwiML Bin, so calls to "
          "the number ring the handset.")
    print("\n  Next:  sudo apt install -y baresip"
          "\n         python tools/setup_sip.py baresip")
    return 0


def cmd_baresip(args) -> int:
    load_env()
    user = os.environ.get("SIP_USERNAME") or args.user
    domain = require("SIP_DOMAIN",
                     "Run 'twilio' first - it writes this.")
    password = require("SIP_PASSWORD",
                       "Run 'twilio' first - it writes this.")

    written = write_baresip(user, domain, password)
    for path in written:
        print(f"  wrote {path}")
    print(f"\n  audio device: {alsa_device()}")
    print(f"  control port: {CTRL_HOST}:{CTRL_PORT}")
    print("\n  Start it:")
    print("      systemctl --user daemon-reload")
    print("      systemctl --user enable --now baresip")
    print("      python tools/setup_sip.py status")
    print("\n  Then make kellogg.service wait for it: uncomment the two")
    print("  pipewire-style lines in kellogg.service, but naming")
    print("  baresip.service instead.")
    return 0


def cmd_status(args) -> int:
    load_env()
    ok = True

    binary = baresip_binary()
    print(f"  baresip binary     {binary or 'NOT INSTALLED'}")
    ok &= bool(binary)

    mods = module_path()
    print(f"  modules            {mods or 'NOT FOUND'}")
    ok &= bool(mods)

    for name in ("config", "accounts"):
        path = BARESIP_DIR / name
        print(f"  ~/.baresip/{name:9s} {'present' if path.is_file() else 'MISSING'}")
        ok &= path.is_file()

    try:
        out = subprocess.run(
            ["systemctl", "--user", "is-active", "baresip"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        out = "unknown"
    print(f"  baresip service    {out}")

    reach = ctrl_reachable()
    print(f"  ctrl_tcp {CTRL_HOST}:{CTRL_PORT}  "
          f"{'reachable' if reach else 'NOT REACHABLE'}")
    ok &= reach

    domain = os.environ.get("SIP_DOMAIN", "")
    print(f"  SIP domain         {domain or 'not set in .env'}")

    if domain:
        try:
            data = api("GET", f"{API}/{acct()}/SIP/Domains.json?PageSize=50")
            match = [d for d in data.get("domains", [])
                     if d["domain_name"].lower() == domain.lower()]
            if match:
                d = match[0]
                print(f"  twilio domain      {d['sid']}")
                print(f"  registration       "
                      f"{'enabled' if d.get('sip_registration') else 'DISABLED'}")
                print(f"  voice url          {d.get('voice_url') or 'NOT SET'}")
                ok &= bool(d.get("sip_registration") and d.get("voice_url"))
            else:
                print("  twilio domain      NOT FOUND")
                ok = False
        except SetupError as exc:
            print(f"  twilio             {exc}")

    print()
    if not reach:
        print("  baresip is not answering on its control port, so the")
        print("  phone cannot place calls. Check:")
        print("      systemctl --user status baresip --no-pager -l")
        print("      grep ctrl_tcp ~/.baresip/config")
    elif ok:
        print("  Everything the phone needs is in place. Lift, crank,")
        print("  say a name - and this time it should actually ring.")
    return 0 if ok else 1


# --------------------------------------------------------------------------
def _selftest() -> int:
    import tempfile

    checks: list[tuple[bool, str]] = []

    def ok(c, label):
        checks.append((bool(c), label))

    pw = make_password()
    ok(len(pw) == 24, "generated password is 24 characters")
    ok(pw.isalnum(), "password avoids punctuation that breaks SIP digest")
    ok(make_password() != make_password(), "passwords are not repeated")

    with tempfile.TemporaryDirectory() as tmp:
        env = Path(tmp) / ".env"
        env.write_text("TWILIO_ACCOUNT_SID=ACold\nKEEP=yes\n")
        save_env({"SIP_PASSWORD": "abc", "TWILIO_ACCOUNT_SID": "ACnew"}, env)
        text = env.read_text()
        ok("KEEP=yes" in text, "save_env preserves unrelated keys")
        ok(text.count("TWILIO_ACCOUNT_SID=") == 1,
           "save_env replaces rather than duplicating")
        ok("ACnew" in text and "ACold" not in text,
           "save_env writes the new value")
        save_env({"SIP_PASSWORD": "def"}, env)
        ok(env.read_text().count("SIP_PASSWORD=") == 1,
           "running twice does not leave two password lines")
        ok(oct(env.stat().st_mode)[-3:] == "600",
           ".env is tightened to 600 because it holds a password")

    tw = outbound_twiml("+15555550100")
    ok("<Dial" in tw and "callerId=\"+15555550100\"" in tw,
       "outbound twiml dials with the caller id")
    ok("{{To}}" in tw, "outbound twiml templates the dialled number")
    inb = inbound_twiml("kellogg", "k.sip.twilio.com")
    ok("<Sip>sip:kellogg@k.sip.twilio.com</Sip>" in inb,
       "inbound twiml dials the sip endpoint")

    # Patch a realistic slice of baresip's generated config.
    default_cfg = """\
#------------------------------------------------------------------------------
audio_player            alsa,default
audio_source            alsa,default
audio_alert             alsa,default
#module_app             account.so
#module_app             contact.so
#module_app             menu.so
#module_app             ctrl_tcp.so
module                  alsa.so
"""
    cfg = patch_config(default_cfg, "plughw:3,0")
    ok("alsa,plughw:3,0" in cfg, "the audio device is substituted in")
    ok("alsa,default" not in cfg, "the default device is replaced, not added")
    ok(cfg.count("audio_player") == 1,
       "audio_player appears once - a duplicate key is silently ignored")
    for mod in REQUIRED_MODULES:
        ok(re.search(rf"^module(?:_app)?\s+{re.escape(mod)}$", cfg,
                     re.M) is not None,
           f"{mod} is uncommented")
    ok("account.so" in cfg,
       "account.so is required: without it baresip reads no accounts "
       "and never registers")
    ok(f"ctrl_tcp_listen\t\t\t{CTRL_HOST}:{CTRL_PORT}" in cfg,
       "the control port telephony.py expects is bound")

    # already-enabled modules must not be duplicated
    twice = patch_config(cfg, "plughw:3,0")
    for mod in REQUIRED_MODULES:
        ok(len(re.findall(rf"^module(?:_app)?\s+{re.escape(mod)}$", twice,
                          re.M)) == 1,
           f"{mod} is not duplicated on a second run")

    acc = BARESIP_ACCOUNT.format(user="kellogg", domain="d.example",
                                 password="secret")
    ok("answermode=manual" in acc,
       "manual answer - main.py decides when to pick up")
    ok("auth_pass=secret" in acc, "the account carries the password")
    ok("PCMU/8000/1" in acc, "a codec Twilio always offers is listed")

    unit = BARESIP_UNIT.format(binary="/usr/bin/baresip")
    ok("Restart=always" in unit, "baresip restarts on failure")
    ok("-f %h/.baresip" in unit, "the unit points at the config directory")

    width = max(len(l) for _, l in checks)
    failed = sum(1 for p, _ in checks if not p)
    for passed, label in checks:
        print(f"  {'ok ' if passed else 'FAIL'}  {label:<{width}}")
    print()
    print(f"all {len(checks)} checks passed" if not failed
          else f"{failed} of {len(checks)} checks failed")
    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    # The shared flags go on BOTH the top-level parser and every
    # subparser, so `twilio --domain x` and `--domain x twilio` both
    # work. Defined only on the parent, argparse silently rejects them
    # after the verb - which is the order everyone types first.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--domain", help="SIP domain name, e.g. kellogg "
                                         "(.sip.twilio.com is added)")
    common.add_argument("--user", default=None, help="SIP username")
    common.add_argument("--voice-url",
                        help="TwiML Bin URL for outbound calls")

    ap = argparse.ArgumentParser(
        description="Set up the SIP trunk for the Kellogg phone",
        parents=[common])
    sub = ap.add_subparsers(dest="cmd")
    for name, help_text in (
            ("plan", "show what will happen, touch nothing"),
            ("twilio", "create the domain, credentials and mappings"),
            ("baresip", "write the baresip config and unit"),
            ("status", "check both ends"),
            ("selftest", "offline checks")):
        sub.add_parser(name, help=help_text, parents=[common])

    # Pre-parse with `common`, which has NO subparsers: it picks a flag
    # out of the argument list wherever it sits and ignores the verb.
    # Using `ap` here would run the subparser again and come back with
    # the same None it is meant to be filling in.
    pre, _ = common.parse_known_args(argv)
    args = ap.parse_args(argv)
    for field in ("domain", "voice_url", "user"):
        if getattr(args, field, None) is None:
            setattr(args, field, getattr(pre, field, None))
    if args.user is None:
        args.user = DEFAULT_USER
    cmd = args.cmd or "plan"

    if cmd == "selftest":
        return _selftest()
    try:
        return {"plan": cmd_plan, "twilio": cmd_twilio,
                "baresip": cmd_baresip, "status": cmd_status}[cmd](args)
    except SetupError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())