#!/usr/bin/env bash
# Download the datasets into $DATA_DIR (default: data/raw).
# Downloads resume if interrupted; finished datasets are skipped.
#
#   scripts/download_data.sh fakemusiccaps   # 12.9 GB, generated clips of 5 text-to-music models
#   scripts/download_data.sh musiccaps       # real clips for FakeMusicCaps, cut from YouTube
#   scripts/download_data.sh fma-small       # 7.7 GB, 8,000 real 30 s clips
#   scripts/download_data.sh fma-medium      # 23.8 GB, 25,000 real 30 s clips
#   scripts/download_data.sh echoes          # 8.6 GB, 12 generators, test set only
#   scripts/download_data.sh core            # fakemusiccaps + musiccaps + fma-small + echoes
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/raw}"
mkdir -p "$DATA_DIR"

fetch() { # url, target file
    curl -L --fail --retry 5 -C - -o "$2" "$1"
}

extract() { # zip file, target dir
    echo "extracting $1 ..."
    python3 -m zipfile -e "$1" "$2"
}

fakemusiccaps() {
    local dir="$DATA_DIR/fakemusiccaps" zip="$DATA_DIR/FakeMusicCaps.zip"
    [ -d "$dir/MusicGen_medium" ] && { echo "fakemusiccaps: already present"; return; }
    fetch "https://zenodo.org/records/15063698/files/FakeMusicCaps.zip" "$zip"
    echo "db418dc95ab7dc378a55f29d6021fd66  $zip" | md5sum -c -
    mkdir -p "$dir" && extract "$zip" "$dir"
    rm -rf "$dir/__MACOSX" "$zip"
}

musiccaps() {
    uv run scripts/fetch_musiccaps.py --out "$DATA_DIR/musiccaps"
}

fma() { # small | medium
    local dir="$DATA_DIR/fma_$1" zip="$DATA_DIR/fma_$1.zip"
    [ -d "$dir" ] && { echo "fma_$1: already present"; return; }
    fetch "https://os.unil.cloud.switch.ch/fma/fma_$1.zip" "$zip"
    extract "$zip" "$DATA_DIR" && rm "$zip"
    if [ ! -d "$DATA_DIR/fma_metadata" ]; then
        fetch "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip" "$DATA_DIR/fma_metadata.zip"
        extract "$DATA_DIR/fma_metadata.zip" "$DATA_DIR" && rm "$DATA_DIR/fma_metadata.zip"
    fi
}

echoes() {
    local dir="$DATA_DIR/echoes" zip="$DATA_DIR/Echoes.zip"
    [ -d "$dir" ] && { echo "echoes: already present"; return; }
    fetch "https://huggingface.co/datasets/Octavian97/Echoes/resolve/main/Echoes.zip" "$zip"
    mkdir -p "$dir" && extract "$zip" "$dir" && rm "$zip"
}

case "${1:-}" in
    fakemusiccaps) fakemusiccaps ;;
    musiccaps)     musiccaps ;;
    fma-small)     fma small ;;
    fma-medium)    fma medium ;;
    echoes)        echoes ;;
    core)          fakemusiccaps; musiccaps; fma small; echoes ;;
    *) sed -n '2,11p' "$0"; exit 1 ;;
esac
