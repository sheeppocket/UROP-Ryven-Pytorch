#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

case "$OSTYPE" in
    msys*|cygwin*)
        exec powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File launchers/windows/_install.ps1
        ;;
    *)
        if [[ -x .venv/bin/python ]]; then
            exec .venv/bin/python launchers/install.py
        fi
        if command -v python3.10 >/dev/null 2>&1; then
            exec python3.10 launchers/install.py
        fi
        ;;
esac

printf '%s\n' 'Install 64-bit Python 3.10 first.' >&2
exit 1
