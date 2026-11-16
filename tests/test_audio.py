import numpy as np
import pytest
import soundfile as sf

from aigm.audio import lowpass
from aigm.audio.io import decode
from aigm.audio.resample import resample, to_mono
from aigm.data.prepare import convert_file, window_starts
from tests.conftest import write_mp3


def level_db(x: np.ndarray, freq: float, rate: int) -> float:
    """Amplitude of the component at `freq` in dB relative to full scale, edges excluded."""
    x = x[rate // 2 : -rate // 2].astype(np.float64) / 32768.0
    t = np.arange(len(x)) / rate
    amplitude = 2 * abs(np.mean(x * np.exp(-2j * np.pi * freq * t)))
    return 20 * np.log10(max(amplitude, 1e-12))


def converted_sine(tmp_path, cfg, freq: float, rate: int) -> np.ndarray:
    t = np.arange(rate * 10) / rate
    path = tmp_path / f"sine_{freq}_{rate}.wav"
    sf.write(path, 0.5 * np.sin(2 * np.pi * freq * t), rate, subtype="FLOAT")
    clips, _, _ = convert_file(path, cfg.audio)
    return clips[0]


@pytest.fixture
def plain(cfg):
    """Settings without RMS normalisation, so that levels can be compared with the input."""
    cfg = cfg.copy()
    cfg.audio.normalise_rms = False
    return cfg


def test_top_of_band_is_removed_for_both_origins(tmp_path, plain):
    # 7.8 kHz lies where a 44.1 kHz original (through soxr) and a 16 kHz original (untouched)
    # would differ; after the common low-pass neither has anything left there.
    for rate in (44100, 16000):
        assert level_db(converted_sine(tmp_path, plain, 7800, rate), 7800, 16000) < -80


def test_passband_is_identical_for_both_origins(tmp_path, plain):
    levels = [level_db(converted_sine(tmp_path, plain, f, rate), f, 16000)
              for f in (1000, 6900) for rate in (44100, 16000)]  # fmt: skip
    expected = 20 * np.log10(0.5)
    assert np.allclose(levels, expected, atol=0.02)


def test_lowpass_design(cfg):
    lp = cfg.audio.lowpass
    taps = lowpass.design(16000, lp.passband_hz, lp.stopband_hz, lp.attenuation_db)
    assert len(taps) % 2 == 1 and np.allclose(taps, taps[::-1])
    response = np.abs(np.fft.rfft(taps, 1 << 16))
    freqs = np.fft.rfftfreq(1 << 16, 1 / 16000)
    assert np.all(np.abs(20 * np.log10(response[freqs <= lp.passband_hz])) < 0.01)
    assert np.all(20 * np.log10(response[freqs >= lp.stopband_hz]) < -lp.attenuation_db + 1)


def test_rms_normalisation(tmp_path, cfg):
    assert cfg.audio.normalise_rms
    for amplitude in (0.5, 0.01):
        path = tmp_path / f"level_{amplitude}.wav"
        rng = np.random.default_rng(0)
        sf.write(path, amplitude * rng.uniform(-1, 1, 160000), 16000, subtype="FLOAT")
        clip = convert_file(path, cfg.audio)[0][0] / 32767.0
        assert 20 * np.log10(np.sqrt(np.mean(clip**2))) == pytest.approx(-20, abs=0.05)


def test_silence_stays_silent(tmp_path, cfg):
    sf.write(tmp_path / "silence.wav", np.zeros(160000), 16000)
    assert not convert_file(tmp_path / "silence.wav", cfg.audio)[0].any()


def test_conversion_is_bit_identical(raw_dir, cfg):
    for rel, count in [("musiccaps/yt00_-AbCdE.wav", 1), ("fma_small/000/000001.mp3", 3)]:
        first = convert_file(raw_dir / rel, cfg.audio, count)
        second = convert_file(raw_dir / rel, cfg.audio, count)
        assert first[0].shape == (count, 160000) and first[0].dtype == np.int16
        assert np.array_equal(first[0], second[0]) and first[1:] == second[1:]


def test_mp3_decoding(tmp_path):
    x = 0.4 * np.sin(2 * np.pi * 440 * np.arange(48000 * 2) / 48000)
    write_mp3(tmp_path / "a.mp3", np.stack([x, x], axis=1), 48000)
    data, rate = decode(tmp_path / "a.mp3")
    assert rate == 48000 and data.shape[1] == 2 and data.dtype == np.float32
    assert len(data) == 96000
    # the encoder lowers the level slightly, so the comparison is loose
    assert np.sqrt(np.mean(to_mono(data) ** 2)) == pytest.approx(0.4 / np.sqrt(2), rel=0.1)


def test_resample_skips_matching_rate():
    x = np.arange(10.0)
    assert resample(x, 16000, 16000) is x
    assert len(resample(np.zeros(44100), 44100, 16000)) == 16000


def test_window_starts():
    clip = 160000
    assert window_starts(100000, clip, 6) == [0]
    assert window_starts(clip, clip, 6) == [0]
    assert window_starts(3 * clip, clip, 3) == [0, clip, 2 * clip]
    assert window_starts(3 * clip - 320, clip, 3) == [0, clip - 160, 2 * clip - 320]
    assert window_starts(int(2.5 * clip), clip, 6) == [0, clip // 2 * 3]
    starts = window_starts(30 * clip, clip, 6)
    assert len(starts) == 6 and starts[0] == 0 and starts[-1] == 29 * clip
