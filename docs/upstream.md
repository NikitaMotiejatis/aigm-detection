# Upstream code and published results

Read on 2026-11-23 for PLAN.md WP0 step 3. "n/s" means the source does not state the value.
Numbers are copied from the papers and are comparison targets, not results of this project.

## Sources

| Id | Work | Code | Licence |
|---|---|---|---|
| Benchmark | Li et al., *Explainable Detection of Machine Generated Music and Early Systematic Evaluation*, [arXiv 2412.13421](https://arxiv.org/abs/2412.13421) | [myxp-lyp/Detecting-Machine-Generated-Music-with-Explainability](https://github.com/myxp-lyp/Detecting-Machine-Generated-Music-with-Explainability-A-Challenge-and-Systematic-Evaluation) | none published |
| FakeMusicCaps | Comanducci et al., [arXiv 2409.10684](https://arxiv.org/abs/2409.10684) | [polimi-ispl/FakeMusicCaps](https://github.com/polimi-ispl/FakeMusicCaps) | CC BY-NC 4.0 |
| Deezer | Afchar et al., *AI-Generated Music Detection and its Challenges*, [arXiv 2501.10111](https://arxiv.org/abs/2501.10111) | [deezer/deepfake-detector](https://github.com/deezer/deepfake-detector) | CC BY-NC 4.0 |
| MusicDET | Han et al., *MusicDET: Zero-Shot AI-Generated Music Detection*, [arXiv 2605.18072](https://arxiv.org/abs/2605.18072) | [Chaolei98/MusicDET](https://github.com/Chaolei98/MusicDET) | MIT |

The benchmark repository has no licence, so its code is not copied here. Our models are written
from the described architecture on top of torchvision.

## Models to reproduce

| Setting | ResNet18 (Benchmark) | MobileNet (Benchmark) | Small CNN (Deezer) | MusicDET |
|---|---|---|---|---|
| Task | real vs generated | real vs generated | real vs autoencoder output | real vs generated, zero-shot or class-conditional |
| Input length | whole file (10 s) | whole file (10 s) | `audio_slice` 0.78, unit n/s | 4 s (random crop or zero-pad) |
| Sample rate | paper: 16 kHz; code: native (`sr=None`) | paper: 16 kHz; code: native (`sr=None`) | 44.1 kHz | 16 kHz |
| Features | log-mel (dB, `ref=max`), resized to 224 × 224 | same as ResNet18 | amplitude spectrogram | log energy STFT |
| `n_fft` / hop | librosa defaults: 2048 / 512 | librosa defaults: 2048 / 512 | 2048 / 512 | 512 / 160 |
| Mel bins, band | librosa defaults: 128, 0 to sr/2 | librosa defaults: 128, 0 to sr/2 | no mel filterbank | no mel filterbank |
| Architecture | torchvision ResNet18, 1-channel `conv1` | torchvision MobileNetV2 in code, 1-channel first layer | 6 conv layers [16…512], kernel 3, pooling 2, average pooling, dense 128 | SpecNF: 2 conv layers, multi-band Glow |
| Pretraining | ImageNet | ImageNet | none | none |
| Optimiser | Adam | Adam | Adam | Adam, weight decay 5e-4 |
| Learning rate | 1e-3 in code; paper lists 1e-3, 1e-4, 1e-5 | same as ResNet18 | 1e-3, reduced on plateau | 5e-4 |
| Epochs | 10 | 10 | up to 200, early stopping | 10 |
| Batch size | 64 | 64 | 50 | 64 (README example: 16) |

The shipped benchmark script has ResNet commented out of its model list, so the ResNet18 settings
are those of its class definition and the paper's Table 1.

## Published numbers we compare against

| Model | Dataset and task | Published result | Where |
|---|---|---|---|
| ResNet18 | FakeMusicCaps, binary | accuracy 0.924, F1 0.924 | Benchmark, Table 2 |
| MobileNet | FakeMusicCaps, binary | accuracy 0.968, F1 0.968 | Benchmark, Table 2 |
| Small CNN | Deezer's autoencoder task, amplitude spectrogram | accuracy 99.8 % (different task, not comparable to E1) | Deezer, Table I |
| ResNet18 + Spec | FakeMusicCaps, 6-class attribution | balanced accuracy 1.00 (secondary, different task) | FakeMusicCaps, Table I |
| MusicDET, zero-shot | FakeMusicCaps | mean EER 4.51 % (MusicGen 5.64, MusicLDM 6.55, AudioLDM2 2.36, Stable Audio Open 3.82, Mustango 4.18) | MusicDET, Table 1 |
| MusicDET, class-conditional | FakeMusicCaps | mean EER 0.89 % (1.67, 0.15, 0.22, 2.40, 0.04, same order) | MusicDET, Table 1 |

## Fixed by this reading

**Decision 4 (features).** E1 uses `n_fft=2048`, `hop=512`, 128 mel bins and a 224 × 224 input,
matching the benchmark's librosa defaults. The band stops at 7 kHz instead of 8 kHz (decision 2).

**E1 acceptance threshold (proposed).** E1 passes if test accuracy and F1 are no more than
0.03 below the published value for ResNet18 (0.894) and MobileNet (0.938). Higher values pass and
go to E7. The small CNN has no comparable published number and is reported without a gate.

The comparison is approximate, because our setup differs on purpose:

- splits are by YouTube id, while the benchmark splits randomly, which can place a real clip and
  its generated versions on different sides;
- the benchmark samples 800 clips per generator in code, while we use all of them with a balanced
  sampler;
- we use 4,461 real clips rather than 5,521, a common low-pass, RMS normalisation, AdamW with a
  cosine schedule, per-clip standardisation and ±6 dB gain.

## MusicDET details for E3

- **Feature extractor:** log energy spectrogram (`n_fft=512`, `hop=160`, `win=512`), then two
  convolution layers (1→16→32 channels, stride 2). SpecAugment during training only. The XLS-R and
  MERT options in the repository are other baselines and are not needed for `spec-nf`.
- **Flow:** `MultiBandGlowC`, two frequency bands, K=2 steps per band (ActNorm, invertible 1×1 LU
  convolution, affine coupling), L=1, 256 hidden channels, and one global flow over the joined
  bands.
- **Prior radius:** the paper and `config.py` use R=5. The class default of R=1 and K=8 in
  `glow_model.py` is overridden by `build_model`.
- **Zero-shot:** trained on real music only, with prior N(+R, I). Low likelihood means generated.
- **Class-conditional:** the same network, trained on both classes. The label only selects the
  prior: real N(+R, I), generated N(−R, I). Inference uses the likelihood under the real prior.
- **Licence and reuse:** MIT. The modules can be imported, but they depend on `transformers` and
  a vendored `sonics` package for parts E3 does not use. Port `SpecNF` and the Glow modules into
  `models/musicdet.py`, keep the MIT notice, and check one published number before our runs.
