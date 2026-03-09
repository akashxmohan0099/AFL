"""
AFL Prediction Pipeline — Shared Metrics
==========================================
Canonical implementations of Brier score, BSS, hit rates, calibration,
and threshold metric aggregation. All metric code should import from here.
"""

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def brier(probs, actuals):
    """Brier score: mean squared error between predicted probs and binary outcomes."""
    p = np.asarray(probs, dtype=float)
    a = np.asarray(actuals, dtype=float)
    mask = np.isfinite(p) & np.isfinite(a)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean((p[mask] - a[mask]) ** 2))


def bss(probs, actuals):
    """Brier Skill Score: 1 - Brier / climatological_Brier."""
    a = np.asarray(actuals, dtype=float)
    mask = np.isfinite(np.asarray(probs, dtype=float)) & np.isfinite(a)
    if mask.sum() == 0:
        return np.nan
    base = a[mask].mean() * (1 - a[mask].mean())
    if base == 0:
        return 0.0
    return float(1 - brier(probs, actuals) / base)


def hit_rate_at_confidence(probs, actuals, threshold):
    """Accuracy among predictions where P >= threshold.

    Returns (hit_rate, n_confident).
    """
    p = np.asarray(probs, dtype=float)
    a = np.asarray(actuals, dtype=float)
    mask = p >= threshold
    if mask.sum() == 0:
        return np.nan, 0
    return float(a[mask].mean()), int(mask.sum())


def calibration_curve(probs, actuals, n_bins=10):
    """Equal-width calibration curve.

    Returns list of dicts with bin_lower, bin_upper, predicted_mean,
    observed_mean, count.
    """
    p = np.asarray(probs, dtype=float)
    a = np.asarray(actuals, dtype=float)
    bins = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() > 0:
            bins.append({
                "bin_lower": round(lo, 2),
                "bin_upper": round(hi, 2),
                "predicted_mean": round(float(p[mask].mean()), 4),
                "observed_mean": round(float(a[mask].mean()), 4),
                "count": int(mask.sum()),
            })
    return bins


def expected_calibration_error(probs, actuals, n_bins=10):
    """Weighted absolute difference between predicted and observed rates."""
    curve = calibration_curve(probs, actuals, n_bins)
    total = sum(b["count"] for b in curve)
    if total == 0:
        return np.nan
    ece = sum(abs(b["predicted_mean"] - b["observed_mean"]) * b["count"] for b in curve) / total
    return round(ece, 4)


def log_loss_binary(probs, actuals, eps=1e-15):
    """Binary log loss with clipping for numerical stability."""
    p = np.asarray(probs, dtype=float)
    a = np.asarray(actuals, dtype=float)
    mask = np.isfinite(p) & np.isfinite(a)
    if mask.sum() == 0:
        return np.nan
    p = np.clip(p[mask], eps, 1.0 - eps)
    a = a[mask]
    return float(np.mean(-a * np.log(p) - (1 - a) * np.log(1 - p)))


# ---------------------------------------------------------------------------
# Composite threshold metrics
# ---------------------------------------------------------------------------

def compute_threshold_metrics(
    probs,
    actuals,
    label=None,
    n_bins=None,
    min_bucket_size=None,
    min_n=1,
):
    """Compute full metrics for a single probability threshold.

    Returns dict with n, base_rate, brier_score, bss, accuracy,
    hit_rate_p60/p70/p80, log_loss, calibration_ece, calibration_curve.
    Returns None if no valid data.
    """
    if n_bins is None:
        n_bins = getattr(config, "CALIBRATION_N_BUCKETS", 10)
    if min_bucket_size is None:
        min_bucket_size = 1

    p = np.asarray(probs, dtype=float)
    a = np.asarray(actuals, dtype=float)
    valid = np.isfinite(p) & np.isfinite(a)
    p, a = p[valid], a[valid]
    n = len(p)

    if n < max(0, int(min_n)):
        return None

    bs = brier(p, a)
    skill = bss(p, a)
    ll = log_loss_binary(p, a)
    base_rate = float(a.mean())
    acc = float(((p >= 0.5) == a).mean())

    hr60, n60 = hit_rate_at_confidence(p, a, 0.60)
    hr70, n70 = hit_rate_at_confidence(p, a, 0.70)
    hr80, n80 = hit_rate_at_confidence(p, a, 0.80)

    curve = [
        bucket
        for bucket in calibration_curve(p, a, n_bins=n_bins)
        if bucket["count"] >= min_bucket_size
    ]
    total_curve_n = sum(bucket["count"] for bucket in curve)
    if total_curve_n > 0:
        ece = round(
            sum(abs(bucket["predicted_mean"] - bucket["observed_mean"]) * bucket["count"] for bucket in curve)
            / total_curve_n,
            4,
        )
    else:
        ece = np.nan

    return {
        "n": n,
        "base_rate": round(base_rate, 4),
        "brier_score": round(bs, 4),
        "bss": round(skill, 4),
        "log_loss": round(ll, 4),
        "accuracy": round(acc, 4),
        "hit_rate_p60": round(hr60, 4) if not np.isnan(hr60) else None,
        "n_confident_p60": n60,
        "hit_rate_p70": round(hr70, 4) if not np.isnan(hr70) else None,
        "n_confident_p70": n70,
        "hit_rate_p80": round(hr80, 4) if not np.isnan(hr80) else None,
        "n_confident_p80": n80,
        "calibration_ece": ece,
        "calibration_curve": curve,
    }


# ---------------------------------------------------------------------------
# Multi-target Brier aggregation (used by pattern_discovery / weight_optimization)
# ---------------------------------------------------------------------------

def compute_all_brier(merged, prefix=""):
    """Compute Brier/BSS for all standard thresholds + MAEs.

    Expects merged DataFrame with columns: actual_goals, actual_disposals,
    actual_marks, and p_* probability columns.
    """
    results = {}

    # Goals
    for thresh, label in [(1, "1plus_goals"), (2, "2plus_goals"), (3, "3plus_goals")]:
        col = f"p_{label}"
        fallback = "p_scorer" if thresh == 1 else None
        pcol = col if col in merged.columns else fallback
        if pcol is None or pcol not in merged.columns:
            continue
        p = merged[pcol].values.astype(float)
        a = (merged["actual_goals"] >= thresh).astype(int).values.astype(float)
        results[f"{prefix}{label}"] = {"brier": round(brier(p, a), 4), "bss": round(bss(p, a), 4)}

    # Disposals
    for thresh in [10, 15, 20, 25, 30]:
        col = f"p_{thresh}plus_disp"
        if col not in merged.columns:
            continue
        p = merged[col].values.astype(float)
        a = (merged["actual_disposals"] >= thresh).astype(int).values.astype(float)
        results[f"{prefix}{thresh}plus_disp"] = {"brier": round(brier(p, a), 4), "bss": round(bss(p, a), 4)}

    # Marks
    for thresh in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        col = f"p_{thresh}plus_mk"
        if col not in merged.columns:
            continue
        p = merged[col].values.astype(float)
        a = (merged["actual_marks"] >= thresh).astype(int).values.astype(float)
        results[f"{prefix}{thresh}plus_mk"] = {"brier": round(brier(p, a), 4), "bss": round(bss(p, a), 4)}

    # MAE
    for pred_col, act_col, key in [
        ("predicted_goals", "actual_goals", "goals_mae"),
        ("predicted_disposals", "actual_disposals", "disp_mae"),
        ("predicted_marks", "actual_marks", "marks_mae"),
    ]:
        if pred_col in merged.columns and act_col in merged.columns:
            v = merged[pred_col].notna() & merged[act_col].notna()
            if v.sum() > 0:
                results[f"{prefix}{key}"] = round(float(np.abs(
                    merged.loc[v, pred_col] - merged.loc[v, act_col]
                ).mean()), 4)

    return results


# ---------------------------------------------------------------------------
# Sample weight computation (simplified, for experiments only)
# ---------------------------------------------------------------------------

def compute_sample_weights(df, era_weights):
    """Compute sample weights from era weights (no dynamic boost).

    For production sequential mode, use features.add_dynamic_sample_weights()
    instead — it adds current-season boost and within-season recency.
    """
    weights = np.ones(len(df), dtype=float)
    for (y_lo, y_hi), w in era_weights.items():
        mask = (df["year"] >= y_lo) & (df["year"] <= y_hi)
        weights[mask] = w
    return weights


# ---------------------------------------------------------------------------
# Feature matrix loading utility
# ---------------------------------------------------------------------------

def load_feature_matrix():
    """Load feature matrix and feature column list.

    Returns (feature_df, feature_cols).
    """
    import json

    feat_path = config.FEATURES_DIR / "feature_matrix.parquet"
    feat_cols_path = config.FEATURES_DIR / "feature_columns.json"

    if not feat_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {feat_path}. Run --features first.")
    if not feat_cols_path.exists():
        raise FileNotFoundError(f"Feature columns not found: {feat_cols_path}. Run --features first.")

    feature_df = pd.read_parquet(feat_path)
    with open(feat_cols_path) as f:
        feature_cols = json.load(f)

    return feature_df, feature_cols
