#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

case "$OSTYPE" in
    msys*|cygwin*)
        if [[ -x .venv/Scripts/python.exe ]]; then
            exec .venv/Scripts/python.exe launchers/install.py
        fi
        if py -3.10 --version >/dev/null 2>&1; then
            exec py -3.10 launchers/install.py
        fi
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

printf '%s\n' 'Install 64-bit Python 3.10 first. On Windows, include the Python Launcher.' >&2
exit 1
