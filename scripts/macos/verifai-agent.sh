#!/bin/zsh
# Keep VerifAI running on this Mac: started at login, restarted if it stops.
#
#   scripts/macos/verifai-agent.sh install    # start now and at every login
#   scripts/macos/verifai-agent.sh status
#   scripts/macos/verifai-agent.sh uninstall
#
# It is a per-user LaunchAgent, so it runs while you are logged in and the
# Mac is awake. Sleep pauses it; waking resumes it. No sudo is needed.
set -euo pipefail

LABEL="app.verifai.server"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGS="$HOME/Library/Logs/VerifAI"
UV="$(command -v uv || echo /opt/homebrew/bin/uv)"
DOMAIN="gui/$(id -u)"

case "${1:-status}" in
  install)
    mkdir -p "$LOGS" "$(dirname "$PLIST")"
    cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>ProgramArguments</key>
  <array>
    <string>$UV</string><string>run</string><string>--no-sync</string><string>verifai</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$LOGS/server.log</string>
  <key>StandardErrorPath</key><string>$LOGS/server.log</string>
</dict>
</plist>
PLIST
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    launchctl bootstrap "$DOMAIN" "$PLIST"
    echo "Installed. VerifAI starts at login and restarts if it stops. Logs: $LOGS/server.log"
    ;;
  uninstall)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    if [ -f "$PLIST" ]; then mv "$PLIST" "$PLIST.disabled"; fi
    echo "Stopped. The agent file was renamed to $PLIST.disabled."
    ;;
  status)
    launchctl print "$DOMAIN/$LABEL" 2>/dev/null | grep -E "state =|pid =|last exit" || echo "Not installed."
    ;;
  *)
    echo "usage: $0 install|status|uninstall" >&2
    exit 2
    ;;
esac
