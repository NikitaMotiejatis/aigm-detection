# aigm-detection

Code for the course work "Detecting AI-Generated Music Using Neural Networks"
(Vilnius University, Faculty of Mathematics and Informatics, Software Engineering, 4th year).

This guide covers three things: setting up a machine, downloading the datasets, and running the
experiments. The thesis text itself lives outside this repository (Overleaf / `thesis/` folder).

**Status (2026-11-16):** setup and dataset download work, and the Python package skeleton
exists (section 4). The training pipeline and the experiments are not implemented yet; section 5
describes what is planned so that both of us build towards the same interface.

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
uv run aigm --help     # the command-line entry point; it has no subcommands yet
make lint test         # ruff and pytest
```

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
