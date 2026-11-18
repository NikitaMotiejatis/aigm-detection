# aigm-detection

Code for the course work "Detecting AI-Generated Music Using Neural Networks"
(Vilnius University, Faculty of Mathematics and Informatics, Software Engineering, 4th year).

This guide covers three things: setting up a machine, downloading the datasets, and running the
experiments. The thesis text itself lives outside this repository (Overleaf / `thesis/` folder).

**Status (2026-11-17):** setup, dataset download and the data layer (audit, conversion to the
processed cache, splits, checksums) work; see section 4. Training and the experiments are not
implemented yet; section 5 describes what is planned so that both of us build towards the same
interface.

## 1. Machine setup

Everything runs inside WSL 2 (Ubuntu 24.04) on Windows 11 with an NVIDIA GPU.
Two machines are used: an RTX 4070 SUPER (12 GB) and an RTX 3070 Laptop (8 GB).

1. **Windows side**
   - Current NVIDIA driver (the Windows driver also serves WSL; do not install a Linux driver).
   - WSL 2 with Ubuntu 24.04: `wsl --install -d Ubuntu-24.04`
   - Docker, one of:
     - Docker Desktop with *Settings → Resources → WSL integration* enabled for the Ubuntu
       distro (used on the 8 GB machine), or
     - the native Docker engine installed inside Ubuntu together with the NVIDIA Container
       Toolkit (used on the 12 GB machine).

2. **Inside Ubuntu**

   ```bash
   sudo apt update && sudo apt install -y git curl
   curl -LsSf https://astral.sh/uv/install.sh | sh      # Python and dependency manager
   ```

3. **Check the GPU** — both commands must print the name of your card:

   ```bash
   nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
   docker run --rm --gpus all ubuntu:24.04 nvidia-smi --query-gpu=name --format=csv,noheader
   ```

4. **Get the code.** Keep the repository in the Linux home directory, not under `/mnt/c`
   (file access across the Windows boundary is several times slower).

   ```bash
   cd ~ && git clone git@github.com:NikitaMotiejatis/aigm-detection.git && cd aigm-detection
   ```

5. **Network (Tailscale).** Both machines join one tailnet so that B reaches A's MLflow server
   under the fixed name `aigm-a`. Install inside Ubuntu (WSL 2 on Ubuntu 24.04 runs systemd,
   which `tailscaled` needs):

   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up --hostname=aigm-a     # on A; use --hostname=aigm-b on B
   tailscale status                        # both machines must be listed
   ```

   MagicDNS must be on (the default) so that `aigm-a` resolves. If Tailscale inside WSL gives
   trouble, install it on Windows instead and set `networkingMode=mirrored` under `[wsl2]` in
   `%UserProfile%\.wslconfig`. On A, stop Windows from sleeping while plugged in
   (*Settings → System → Power*, or `powercfg /change standby-timeout-ac 0` in PowerShell).

   To access MLflow server from browser that is outside WSL, Tailscale should be installed on 
   Windows as well. 

   MLflow server is accessible via this url: [http:/aigm-a:5000](http:/aigm-a:5000)

## 2. Datasets

All data goes to `data/raw/` (ignored by git). Set `DATA_DIR=/some/path` to store it elsewhere.
Downloads resume when interrupted and skip datasets that are already present.

| Dataset | Command | Download | Role |
|---|---|---|---|
| FakeMusicCaps | `scripts/download_data.sh fakemusiccaps` | 12.9 GB | Generated class: 5 text-to-music models × 5,521 ten-second clips |
| MusicCaps | `scripts/download_data.sh musiccaps` | ~9 GB | Real class for FakeMusicCaps (the same 5,521 clips, human-made) |
| FMA small | `scripts/download_data.sh fma-small` | 7.7 GB | Real music (8,000 clips of 30 s) for zero-shot training |
| Echoes | `scripts/download_data.sh echoes` | 8.6 GB | Unseen-generator test set: 12 generators, 4,468 tracks |
| FMA medium | `scripts/download_data.sh fma-medium` | 23.8 GB | Optional, 25,000 real clips |

`scripts/download_data.sh core` downloads the first four (about 40 GB; allow roughly twice that
while archives are being extracted).

Things to know:

- **FakeMusicCaps contains no real audio.** The archive has five folders, one per generator
  (`MusicGen_medium`, `musicldm`, `audioldm2`, `stable_audio_open`, `mustango`); files are named
  `<YouTube id>.wav` after the MusicCaps clip whose caption was used as the prompt.
- **MusicCaps is cut from YouTube**, because the dataset is published only as a list of video IDs
  and timestamps. Some videos no longer exist, so fewer than 5,521 clips will arrive; the missing
  ones are listed in `data/raw/musiccaps/failed.csv`. Expect the download to take a few hours.
  Re-running the command only fetches what is missing. We must both use the same set of real
  clips, so the download is done on one machine only (the 12 GB one); the other machine receives
  the converted data on an external disk and does not run the download scripts.
- **Licences:** FakeMusicCaps is CC BY-NC 4.0, Echoes is CC BY-SA 4.0, FMA tracks carry
  per-track Creative Commons licences. Research use only; do not redistribute the audio.
- SONICS and M6, which appear in the literature, are not used: SONICS requires downloading
  48,000 full songs from YouTube, and M6 is available only on request.

Resulting layout:

```
data/raw/
├── fakemusiccaps/{MusicGen_medium,musicldm,audioldm2,stable_audio_open,mustango}/<ytid>.wav
├── musiccaps/<ytid>.wav, musiccaps-public.csv, failed.csv
├── fma_small/<000-155>/<track id>.mp3
├── fma_metadata/
└── echoes/
```

## 3. Thesis

The thesis is a LaTeX project (class `VUMIFPSkursinis.cls`, entry point `main.tex`).

- **Overleaf:** upload the folder and set *Menu → Compiler → XeLaTeX*.
- **Locally:** install TeX Live and the VS Code extension *LaTeX Workshop*; the project's
  `latexmkrc` selects XeLaTeX and Biber. Build from the command line with `latexmk main.tex`.

## 4. Development

```bash
uv sync                # creates .venv with all dependencies (PyTorch included, several GB)
uv run aigm --help     # the command-line entry point
make lint test         # ruff and pytest (synthetic audio, no datasets needed)
```

### Data layer

Settings are in `configs/data.yaml`; any of them can be overridden as `key=value` after a
command. Models never read the raw files: every source is converted once into a cache of
16 kHz mono 10-second clips (`data/processed/`, about 27 GB) that all later steps read.

```bash
uv run aigm audit            # statistics and mean spectra of the raw audio -> results/
uv run aigm prepare all      # machine A only: build the cache (refuses to replace an existing one)
uv run aigm make-splits      # machine A only: write splits/*.csv and splits/SPLITS.md
uv run aigm data manifest    # machine A only: write splits/CACHE.sha256
uv run aigm data verify      # either machine: compare data/processed/ with splits/CACHE.sha256
scripts/export_cache.sh /mnt/e/aigm-detection   # machine A: copy the cache to the external disk
```

Conversion: decode, mix to mono, resample to 16 kHz, cut into 10 s clips, apply one low-pass
(flat to 7.0 kHz, fully attenuated from 7.4 kHz) to every clip of every source, scale to
-20 dBFS RMS, store as 16-bit. FakeMusicCaps and MusicCaps give one clip per file, FMA small
three per track, Echoes up to six per track.

**Machine B** does not download or convert anything. Copy `processed/` from the external disk
to `data/processed/` and run `uv run aigm data verify`; it must print `OK`.

The split files in `splits/` are part of the repository and are never regenerated;
`splits/SPLITS.md` lists what is in them.

### Tracking server (machine A)

One MLflow server, on A, holds all runs, artifacts and the model registry (`compose.yaml`).
Its data is stored in `mlflow-data/` (SQLite database and artifact folder, not in git).

```bash
make mlflow-up      # start it; restarts by itself after a crash or reboot of A
make mlflow-down    # stop it (runs are kept)
```

The UI and API are at `http://localhost:5000` on A and at `http://aigm-a:5000` from B over
Tailscale. Code finds the server through `MLFLOW_TRACKING_URI`, e.g.
`export MLFLOW_TRACKING_URI=http://aigm-a:5000` on B. No login is configured: port 5000 must
stay reachable only from the tailnet. The server accepts only the host names `localhost`,
`aigm-a` (with or without the tailnet domain) and tailnet addresses (`100.*`); add others to
`--allowed-hosts` in `compose.yaml`.

## 5. Experiments (planned, not implemented yet)

| # | Experiment | Data |
|---|---|---|
| 1 | Supervised baselines in-domain: ResNet18, MobileNet, small CNN on spectrograms | FakeMusicCaps + MusicCaps |
| 2 | Cross-generator: train on one generator, test on the other four | FakeMusicCaps + MusicCaps |
| 3 | MusicDET, zero-shot (real music only) and class-conditional | FMA / MusicCaps for training, FakeMusicCaps for testing |
| 4 | Cross-dataset: all trained models evaluated without retraining | Echoes |
| 5 | Robustness: pitch shift, time stretch, EQ, reverb, noise, 64 kbps MP3/AAC/Opus | test sets of 1–3 |
| 6 | Attribution analysis (Grad-CAM, Integrated Gradients, Occlusion) of failure cases | failures from 2, 4, 5 |

Planned pipeline: every run is a container started with `docker run --gpus all` that logs
parameters, metrics and the model to one MLflow server. The 12 GB machine trains and hosts
MLflow; the 8 GB machine reaches it over Tailscale and runs evaluation and attribution. Commands
for starting the stack and launching runs will be documented here once they exist.
