#!/bin/bash
# Install (or remove with --uninstall) the weekday launchd job on macOS.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HOME/Library/LaunchAgents/com.deskfund.daily.plist"

if [[ "${1:-}" == "--uninstall" ]]; then
  launchctl unload "$DEST" 2>/dev/null || true
  rm -f "$DEST"
  echo "removed $DEST"
  exit 0
fi

mkdir -p "$DIR/logs" "$HOME/Library/LaunchAgents"
chmod +x "$DIR/schedule/run_daily.sh"
sed "s#__DESKFUND_DIR__#$DIR#g" "$DIR/schedule/com.deskfund.daily.plist" > "$DEST"
launchctl unload "$DEST" 2>/dev/null || true
launchctl load "$DEST"
echo "installed $DEST (weekdays 9:45). Logs: $DIR/logs/"
