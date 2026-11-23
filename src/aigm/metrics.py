"""Detection metrics (decision 6, WP2 step 5). Label 1 = generated; higher score = generated."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def _two_classes(labels: np.ndarray) -> bool:
    return len(np.unique(labels)) == 2


def eer(labels, scores) -> tuple[float, float]:
    """Equal error rate and its threshold; a clip is called generated when score >= threshold."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if not _two_classes(labels):
        return float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    fpr, fnr, thresholds = fpr[1:], 1 - tpr[1:], thresholds[1:]  # first point is "reject all"
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thresholds[i])


def _ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def at_threshold(labels, scores, threshold: float) -> dict[str, float]:
    """Accuracy, balanced accuracy, F1, TPR and FPR. Undefined values are NaN."""
    labels, predicted = np.asarray(labels) == 1, np.asarray(scores) >= threshold
    tp, fp = int((labels & predicted).sum()), int((~labels & predicted).sum())
    fn, tn = int((labels & ~predicted).sum()), int((~labels & ~predicted).sum())
    tpr, tnr = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    return {
        "accuracy": _ratio(tp + tn, len(labels)),
        "balanced_accuracy": (tpr + tnr) / 2,
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "tpr": tpr,
        "fpr": _ratio(fp, fp + tn),
    }


def summarise(labels, scores, threshold: float | None = None) -> dict[str, float]:
    """All metrics; threshold-based ones use `threshold`, by default this set's EER threshold."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    value, eer_threshold = eer(labels, scores)
    used = eer_threshold if threshold is None else threshold
    auc = float(roc_auc_score(labels, scores)) if _two_classes(labels) else float("nan")
    return {"eer": value, "eer_threshold": eer_threshold, "auc": auc, "threshold": used,
            **at_threshold(labels, scores, used)}  # fmt: skip


def average_per_track(clips: pd.DataFrame) -> pd.DataFrame:
    """One row per track with the mean of its clip scores (windowed sources, WP2 step 4)."""
    if clips.groupby("track_id")["label"].nunique().max() > 1:
        raise ValueError("a track has clips with different labels")
    return clips.groupby("track_id", sort=False, as_index=False).agg(
        group_id=("group_id", "first"), generator=("generator", "first"),
        label=("label", "first"), score=("score", "mean"),
    )  # fmt: skip


def bootstrap_ci(
    labels,
    scores,
    groups,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval, resampling whole groups. One-class resamples are left out."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    codes, unique = pd.factorize(np.asarray(groups))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        take = np.repeat(np.arange(len(labels)), counts[codes])
        if _two_classes(labels[take]):
            values.append(metric(labels[take], scores[take]))
    if not values:
        return float("nan"), float("nan")
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)
