import numpy as np
import pandas as pd
import pytest

from aigm.metrics import at_threshold, average_per_track, bootstrap_ci, eer, summarise


def test_eer_perfect_separation():
    value, threshold = eer([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert value == 0
    assert threshold == 0.8


def test_eer_hand_computed():
    # at threshold 0.4: one of two real clips accepted, one of two generated clips missed
    value, threshold = eer([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert value == 0.5
    assert threshold == 0.4


def test_eer_inverted_scores():
    assert eer([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])[0] == 1


def test_eer_one_class_is_undefined():
    assert np.isnan(eer([1, 1], [0.1, 0.2])[0])


def test_at_threshold_hand_computed():
    # tp=2, fn=1, fp=1, tn=1
    m = at_threshold([1, 1, 1, 0, 0], [0.9, 0.8, 0.1, 0.7, 0.2], 0.5)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["tpr"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 2)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert m["f1"] == pytest.approx(4 / 6)


def test_summarise_uses_given_threshold():
    m = summarise([0, 1], [0.2, 0.8], threshold=0.9)
    assert m["auc"] == 1 and m["eer"] == 0
    assert m["threshold"] == 0.9 and m["tpr"] == 0


def test_average_per_track():
    clips = pd.DataFrame({"track_id": ["a", "a", "b"], "group_id": ["g", "g", "h"],
                          "generator": ["x", "x", "real"], "label": [1, 1, 0],
                          "score": [1.0, 3.0, 5.0]})  # fmt: skip
    tracks = average_per_track(clips)
    assert tracks["score"].tolist() == [2.0, 5.0]


def test_average_per_track_rejects_mixed_labels():
    clips = pd.DataFrame({"track_id": ["a", "a"], "group_id": ["g", "g"],
                          "generator": ["x", "real"], "label": [1, 0],
                          "score": [1.0, 3.0]})  # fmt: skip
    with pytest.raises(ValueError):
        average_per_track(clips)


def test_bootstrap_resamples_whole_groups():
    # within each group the classes are perfectly separated, so any resample of groups has EER 0;
    # resampling single clips could not produce this for a group-wise offset
    labels = np.array([0, 1] * 20)
    groups = np.repeat(np.arange(20), 2)
    scores = labels.astype(float) + groups * 0.001
    assert bootstrap_ci(labels, scores, groups, lambda y, s: eer(y, s)[0], n_boot=50) == (0, 0)


def test_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 200)
    scores = labels + rng.normal(0, 1, len(labels))
    low, high = bootstrap_ci(labels, scores, np.arange(len(labels)),
                             lambda y, s: eer(y, s)[0], n_boot=200)  # fmt: skip
    assert low < eer(labels, scores)[0] < high
