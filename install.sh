#!/usr/bin/env bash
# Kellogg magneto phone - install the service and the echo canceller.
#
# Run from the project root on the Pi:
#     bash install.sh
#
# Idempotent: safe to run again after editing either config file.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_SYSTEMD="$HOME/.config/systemd/user"
PW_CONF_DIR="$HOME/.config/pipewire/pipewire.conf.d"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m %s\n' "$*"; }

# --------------------------------------------------------------------------
say "Checking the project layout"
for f in src/main.py src/hardware.py kellogg.service kellogg-aec.conf; do
    [[ -f "$PROJECT/$f" ]] || { warn "missing $f - run this from the project root"; exit 1; }
done
ok "all expected files present"

if [[ ! -x "$PROJECT/venv/bin/python" ]]; then
    warn "no venv at $PROJECT/venv"
    warn "create it with:  python3 -m venv --system-site-packages venv"
    exit 1
fi
ok "venv found"

# --------------------------------------------------------------------------
say "Echo cancellation"
mkdir -p "$PW_CONF_DIR"
install -m 0644 "$PROJECT/kellogg-aec.conf" "$PW_CONF_DIR/kellogg-aec.conf"
ok "installed to $PW_CONF_DIR/kellogg-aec.conf"

systemctl --user restart pipewire pipewire-pulse 2>/dev/null || \
    warn "could not restart pipewire - is it running in this session?"
sleep 2

if "$PROJECT/venv/bin/python" "$PROJECT/src/audio.py" devices 2>/dev/null \
        | grep -q "echo-cancel"; then
    ok "echo-cancel devices are present"
else
    warn "no echo-cancel devices yet."
    warn "check:  journalctl --user -u pipewire -n 40"
    warn "the usual cause is a missing WebRTC plugin:"
    warn "    sudo apt install libspa-0.2-libcamera pipewire-audio"
fi

# --------------------------------------------------------------------------
say "Audio device detection"
if "$PROJECT/venv/bin/python" "$PROJECT/src/audio.py" detect; then
    ok "audio.json written"
else
    warn "detection failed - the USB sound card may be unplugged"
fi

# --------------------------------------------------------------------------
say "Printer permissions"
if [[ -e /dev/usb/lp0 ]]; then
    if [[ -w /dev/usb/lp0 ]]; then
        ok "/dev/usb/lp0 is writable"
    else
        warn "/dev/usb/lp0 exists but you cannot write to it"
        warn "    sudo usermod -aG lp $USER"
        warn "then log out and back in - group changes need a new session"
    fi
else
    warn "no /dev/usb/lp0 - is the printer powered from its 9 V supply?"
fi

# --------------------------------------------------------------------------
say "Credentials"
if [[ -f "$PROJECT/.env" ]]; then
    ok ".env present"
    if git -C "$PROJECT" check-ignore -q .env 2>/dev/null; then
        ok ".env is gitignored"
    else
        warn ".env is NOT gitignored and this repo is PUBLIC."
        warn "Add it before committing anything:  echo .env >> .gitignore"
    fi
else
    warn "no .env - the phone will run in null telephony mode"
fi

# --------------------------------------------------------------------------
say "The service"
mkdir -p "$USER_SYSTEMD"
install -m 0644 "$PROJECT/kellogg.service" "$USER_SYSTEMD/kellogg.service"
systemctl --user daemon-reload
ok "unit installed"

if loginctl show-user "$USER" 2>/dev/null | grep -q "Linger=yes"; then
    ok "lingering is enabled"
else
    warn "lingering is OFF. The phone will die when you log out."
    warn "    sudo loginctl enable-linger $USER"
fi

systemctl --user enable kellogg >/dev/null
ok "enabled at boot"

# --------------------------------------------------------------------------
say "Done"
cat <<EOF

  Start it:    systemctl --user start kellogg
  Watch it:    journalctl --user -u kellogg -f
  Stop it:     systemctl --user stop kellogg

  Before trusting the audio, run these two and read what they tell you:

      python src/audio.py level    set the MAX4466 trimpot
      python src/audio.py echo     must PASS before the operator is usable

EOF
