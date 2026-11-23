# aigm-detection

Code for the course work "Detecting AI-Generated Music Using Neural Networks"
(Vilnius University, Faculty of Mathematics and Informatics, Software Engineering, 4th year).

This guide covers three things: setting up a machine, downloading the datasets, and running the
experiments. The thesis text itself lives outside this repository (Overleaf / `thesis/` folder).

**Status (2026-11-19):** setup, dataset download, the data layer (audit, conversion to the
processed cache, splits, checksums) and the shared MLflow tracking server on the Proxmox host
work; see section 4. Training and the experiments are not implemented yet; section 5 describes
what is planned so that both of us build towards the same interface.

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

5. **Network (Tailscale).** Both machines join one tailnet so that they reach the MLflow server
   (section 4, *Tracking server*) under the fixed name `aigm-mlflow`. Install inside Ubuntu
   (WSL 2 on Ubuntu 24.04 runs systemd, which `tailscaled` needs):

   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up --hostname=aigm-a     # on A; use --hostname=aigm-b on B
   tailscale status                        # aigm-a, aigm-b and aigm-mlflow must be listed
   ```

   MagicDNS must be on (the default) so that the names resolve. If Tailscale inside WSL gives
   trouble, install it on Windows instead and set `networkingMode=mirrored` under `[wsl2]` in
   `%UserProfile%\.wslconfig`. To open the MLflow UI in a Windows browser, Tailscale must be
   installed on Windows as well; with it only inside WSL, `curl` works but the browser does not.

   **Check the WSL network MTU.** If `eth0` in WSL has MTU 1280, Tailscale's encrypted packets do
   not fit through it. Small requests such as `/health` and the UI still work, but uploads to
   MLflow hang (a training run sits idle with almost no CPU use). Check and fix it:

   ```bash
   ip link show eth0                               # look for "mtu 1500"
   sudo ip link set dev eth0 mtu 1500              # if it is lower
   ping -c 3 -M do -s 1252 aigm-mlflow             # must get replies
   ```

   To keep the setting after a WSL restart, add this line to the existing `[boot]` section of
   `/etc/wsl.conf` (do not add a second `[boot]`), then run `wsl --shutdown` from PowerShell:

   ```ini
   command = ip link set dev eth0 mtu 1500
   ```

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

### Training and evaluation

```bash
uv run aigm train --config configs/smoke.yaml     # small CNN, 64 synthetic clips, CPU, no dataset
uv run aigm train --config configs/e1_indomain.yaml seed=1
uv run aigm train --config configs/e1_indomain.yaml model.name=mobilenet_v2
uv run aigm train --config configs/e1_indomain.yaml model.name=smallcnn model.pretrained=false
uv run aigm eval --run <mlflow run id> --testset fmc_test
```

Set `MLFLOW_TRACKING_URI` first, otherwise runs go to a local MLflow store. Test sets are named in
`configs/testsets.yaml`. Scores are log-odds of "generated"; clip scores are averaged per track
before metrics are computed, and accuracy and F1 use the threshold of the validation EER. A run
started from a working tree with uncommitted changes is tagged `dirty=true`.

### Tracking server

One MLflow server holds all runs, artifacts and the model registry. It runs in an unprivileged
LXC container (`aigm-mlflow`, Debian 12) on a Proxmox host that is always on, so it does not
depend on A or B being awake. Its files are in `deploy/mlflow/` (a copy of `/opt/mlflow` in the
container); its data is on the container's mount point `/srv/mlflow` (SQLite database and
artifact folder), which is included in Proxmox backups.

Code finds the server through `MLFLOW_TRACKING_URI`; set it on both machines (e.g. in
`~/.bashrc`), with `<tailnet>` replaced by the tailnet's domain shown by `tailscale status`:

```bash
export MLFLOW_TRACKING_URI=https://aigm-mlflow.<tailnet>.ts.net
curl -s $MLFLOW_TRACKING_URI/health      # prints OK
```

The same URL opens the UI in a browser. No login is configured: MLflow listens on `localhost`
inside the container only, and `tailscale serve` publishes it to the tailnet with HTTPS, so it is
not reachable from the host's LAN.

**Maintenance** (inside the container: `pct enter <id>` on the Proxmox host, or
`ssh root@aigm-mlflow` through Tailscale SSH):

```bash
cd /opt/mlflow
docker compose up -d --build     # start or update; restarts by itself after a crash or reboot
docker compose logs --tail=50    # if the UI or /health stops answering
tailscale serve status           # must show https://aigm-mlflow.<tailnet>.ts.net -> 127.0.0.1:5000
```

**Old CPU.** The Proxmox host has an Intel Core 2 Duo (x86-64-v1, no SSE4.2/AVX). The official
MLflow image crashes on it at start-up with exit code 132 (illegal instruction), because current
numpy and all pyarrow wheels use newer instructions. `deploy/mlflow/Dockerfile` therefore pins
`numpy<2.4` and removes pyarrow, which the tracking server does not need. If the container ever
shows `Restarting (132)` again, a newly loaded library has the same problem. When MLflow is
upgraded, change the version in `uv.lock` and in the `FROM` line together.

**Container setup** (once, on the Proxmox host): unprivileged container with
`features: nesting=1,keyctl=1` (for Docker), `/dev/net/tun` passed in (`pct set <id> --dev0
/dev/net/tun`, for Tailscale), a mount point `mp0: …,mp=/srv/mlflow,backup=1` and
`onboot: 1`. Inside: Docker (`get.docker.com`), Tailscale (`tailscale up --hostname=aigm-mlflow
--ssh`, key expiry disabled in the admin console), `tailscale serve --bg 5000` (needs MagicDNS
and HTTPS certificates enabled for the tailnet), then the files from `deploy/mlflow/` in
`/opt/mlflow`.

**Local server.** The `compose.yaml` in the repository root starts a separate MLflow server on
the machine it runs on (data in `mlflow-data/`), for testing without the shared server:

```bash
make mlflow-up      # http://localhost:5000; set MLFLOW_TRACKING_URI=http://localhost:5000
make mlflow-down    # stop it (runs are kept)
```

Runs logged there stay on that machine and are not part of the shared results.

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
parameters, metrics and the model to one MLflow server. The 12 GB machine trains; the 8 GB
machine runs evaluation and attribution; both reach MLflow on the Proxmox host over Tailscale. Commands
for starting the stack and launching runs will be documented here once they exist.
