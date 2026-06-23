#!/usr/bin/env bash
# Author: Chun Kang <kurapa@kurapa.com>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SCRIPT_DIR/subtitle_sync.sh"
DEST_DIR="$HOME/bin"
DEST="$DEST_DIR/subtitle_sync"

if [[ ! -f "$SOURCE" ]]; then
  echo "error: $SOURCE not found" >&2
  exit 1
fi

# subtitle_sync needs a python3 interpreter (>=3.8). Force-install python3 via
# the system package manager when it is missing. Everything else (pip,
# faster-whisper, ffmpeg) is bootstrapped by the script itself at first run.
ensure_python3() {
  if command -v python3 >/dev/null 2>&1; then
    return
  fi
  echo "python3 not found; attempting to install it..."
  local sudo=""
  if [[ "$(id -u)" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
    sudo="sudo"
  fi
  if command -v dnf >/dev/null 2>&1; then
    $sudo dnf install -y python3
  elif command -v yum >/dev/null 2>&1; then
    $sudo yum install -y python3                       # CentOS 7 / RHEL 7
  elif command -v apt-get >/dev/null 2>&1; then
    $sudo apt-get update && $sudo apt-get install -y python3
  elif command -v zypper >/dev/null 2>&1; then
    $sudo zypper install -y python3
  elif command -v brew >/dev/null 2>&1; then
    brew install python
  else
    echo "error: no supported package manager found; install python3 manually" >&2
    exit 1
  fi
}

ensure_python3

mkdir -p "$DEST_DIR"
install -m 0755 "$SOURCE" "$DEST"
echo "installed: $DEST"
