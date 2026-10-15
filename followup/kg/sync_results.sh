#!/bin/bash
# Copy per-fold MIMIC-IV metrics from the A40 and drop patient counts before they reach the repository.
# Usage: followup/kg/sync_results.sh [host]
set -euo pipefail
HOST=${1:-main.agentic-research.michel.coder}
KG=$(cd "$(dirname "$0")" && pwd)
DEST=$KG/../results/mimic24
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
ssh -o ConnectTimeout=90 "$HOST" 'cd ~/mimic-iv/derived/results && tar cf - MIMIC*/fold*.json' < /dev/null | tar xf - -C "$STAGE"
"$KG/../.venv/bin/python" "$KG/sanitise_results.py" "$STAGE"
if grep -rl pred_class_counts "$STAGE" >/dev/null; then echo 'counts remain; not copied' >&2; exit 1; fi
mkdir -p "$DEST" && cp -R "$STAGE"/. "$DEST"/
echo "synced $(find "$STAGE" -name 'fold*.json' | wc -l) fold files to $DEST"
