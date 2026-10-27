#!/usr/bin/env bash
# Install the session collector as a systemd user timer (every 2 minutes) on this machine.
# The bot key is read from ~/.config/eridani-collector/token; nothing here prints it.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
share="${XDG_DATA_HOME:-$HOME/.local/share}/eridani-collector"
config="${XDG_CONFIG_HOME:-$HOME/.config}/eridani-collector"
units="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

mkdir -p "$share" "$config" "$units"
# A copy, so switching branches in this repository never breaks the timer.
install -m 0755 "$here/collector.py" "$share/collector.py"
install -m 0644 "$here/eridani-collector.service" "$here/eridani-collector.timer" "$units/"

if [ ! -s "$config/token" ]; then
  echo "Save a bot key with sessions:write and sessions:read to $config/token (chmod 600), then rerun."
  echo "Create it in Eridani: Settings → Integrations → Connected agents."
  exit 1
fi
chmod 600 "$config/token"

systemctl --user daemon-reload
systemctl --user enable --now eridani-collector.timer
systemctl --user list-timers eridani-collector.timer --no-pager
