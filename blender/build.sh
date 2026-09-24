#!/usr/bin/env bash
# Build one or all parts: ./blender/build.sh [part ...]
set -euo pipefail
cd "$(dirname "$0")/.."
parts=("$@")
[ ${#parts[@]} -eq 0 ] && parts=($(ls blender/parts/*.py | xargs -n1 basename | sed 's/\.py$//'))
for p in "${parts[@]}"; do
  echo ">> building $p"
  blender -b --factory-startup -P "blender/parts/$p.py" 2>&1 | grep -E "Error|Traceback|built" || true
done
