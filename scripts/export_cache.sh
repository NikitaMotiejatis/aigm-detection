#!/usr/bin/env bash
# Machine A: copy the processed cache to the external disk for machine B.
#
#   scripts/export_cache.sh /mnt/e/aigm-detection
#
# Writes the checksum manifest splits/CACHE.sha256 if it does not exist yet (commit it), copies
# data/processed/ to <target>/processed/ and checks the copy against the manifest.
# Also copies data/raw/ to <target>/raw/ for convenience.
# On machine B: copy <disk>/processed/ to data/processed/ and run `uv run aigm data verify`.
set -euo pipefail

target="${1:?usage: scripts/export_cache.sh <directory on the external disk>}"
processed="${PROCESSED_DIR:-data/processed}"
raw="${RAW_DIR:-data/raw}"
manifest="${MANIFEST:-splits/CACHE.sha256}"

if [ ! -f "$manifest" ]; then
    uv run aigm data manifest --manifest "$manifest"
else
    uv run aigm data verify --manifest "$manifest"
fi
manifest_abs="$(realpath "$manifest")"
mkdir -p "$target/processed"
mkdir -p "$target/raw"
rsync -rt --delete --info=progress2 "$processed/" "$target/processed/"
rsync -rt --delete --info=progress2 "$raw/" "$target/raw/"

echo "checking the copy ..."
(cd "$target/processed" && sha256sum --quiet -c "$manifest_abs")
echo "OK: $target/processed matches $manifest"
