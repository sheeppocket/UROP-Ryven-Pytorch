#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

case "$OSTYPE" in
    msys*|cygwin*) python=.venv/Scripts/python.exe ;;
    *) python=.venv/bin/python ;;
esac

if [[ ! -x "$python" ]]; then
    printf '%s\n' 'Run bash setup.sh from the repository root first.' >&2
    exit 1
fi

exec "$python" pytorch/open_ryven.py "$@"
