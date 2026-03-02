"""
Investigate the Goals 1+ hit rate drop from ~78% to 69.6%.
Compare old (NaN→0) vs new (NaN preserved for HistGBT) scorer behavior.
"""

import numpy as np
import pandas as pd
import json
from sklearn.metrics import brier_score_loss, roc_auc_score

import config
from model import AFLScoringModel, _prepare_features
from store import LearningStore
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor


# ── Load data ─────────────────────────────────────────────────────────

feat_df = pd.read_parquet(config.FEATURES_DIR / "feature_matrix.parquet")
with open(config.FEATURES_DIR / "feature_columns.json") as f:
    feature_cols = json.load(f)

YEAR = 2025

# We'll test on multiple rounds to get a solid comparison
test_rounds = sorted(feat_df[feat_df["year"] == YEAR]["round_number"].dropna().unique())

print("=" * 70)
print("INVESTIGATION: Goals 1+ Hit Rate Drop")
print("=" * 70)

# ── Run both approaches on every round ────────────────────────────────

old_all_p = []  # P(scorer) from old approach (NaN→0)
new_all_p = []  # P(scorer) from new approach (NaN preserved)
all_actual = []  # actual scored 1+

hist_params = config.HIST_GBT_PARAMS_BACKTEST

for rnd in test_rounds:
    rnd_int = int(rnd)

    train_mask = (
        (feat_df["year"] < YEAR)
        | ((feat_df["year"] == YEAR) & (feat_df["round_number"] < rnd))
    )
    train_df = feat_df[train_mask].copy()
    test_mask = (feat_df["year"] == YEAR) & (feat_df["round_number"] == rnd)
    test_df = feat_df[test_mask].copy()

    if len(train_df) < 50 or test_df.empty:
        continue

    y_goals_train = train_df["GL"].values
    y_goals_test = test_df["GL"].values
    y_is_scorer_train = (y_goals_train >= 1).astype(int)
    y_is_scorer_test = (y_goals_test >= 1).astype(int)
    weights = train_df["sample_weight"].values if "sample_weight" in train_df.columns else None

    # --- OLD approach: NaN→0 for everything ---
    X_train_old = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X_test_old = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

    clf_old = HistGradientBoostingClassifier(**hist_params)
    clf_old.fit(X_train_old, y_is_scorer_train, sample_weight=weights)
    p_old = clf_old.predict_proba(X_test_old)[:, 1]

    # --- NEW approach: NaN preserved for HistGBT ---
    X_train_raw, X_train_clean, _ = _prepare_features(
        train_df, feature_cols, scaler=None
    )
    X_test_raw, X_test_clean, _ = _prepare_features(
        test_df, feature_cols, scaler=None
    )

    clf_new = HistGradientBoostingClassifier(**hist_params)
    clf_new.fit(X_train_raw, y_is_scorer_train, sample_weight=weights)
    p_new = clf_new.predict_proba(X_test_raw)[:, 1]

    old_all_p.extend(p_old)
    new_all_p.extend(p_new)
    all_actual.extend(y_is_scorer_test)

    if rnd_int % 7 == 1:
        print(f"  R{rnd_int:02d} done ({len(test_df)} players)")

old_all_p = np.array(old_all_p)
new_all_p = np.array(new_all_p)
all_actual = np.array(all_actual)

print(f"\nTotal predictions: {len(all_actual)}")
print(f"Actual scorer rate: {all_actual.mean():.4f}")


# ══════════════════════════════════════════════════════════════════════
# 1. P(scorer) distribution comparison
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1. P(SCORER) DISTRIBUTION — OLD vs NEW")
print("=" * 70)

# Histogram bins
bins = np.arange(0, 1.05, 0.05)
bin_labels = [f"{b:.2f}" for b in bins[:-1]]

old_hist, _ = np.histogram(old_all_p, bins=bins)
new_hist, _ = np.histogram(new_all_p, bins=bins)

print(f"\n{'Bin':<10s} {'Old count':>10s} {'New count':>10s} {'Δ':>8s}")
print("-" * 40)
for i in range(len(bin_labels)):
    lo = bins[i]
    hi = bins[i + 1]
    label = f"{lo:.2f}-{hi:.2f}"
    delta = new_hist[i] - old_hist[i]
    if old_hist[i] > 0 or new_hist[i] > 0:
        print(f"  {label:<10s} {old_hist[i]:>8d}   {new_hist[i]:>8d}   {delta:>+6d}")

print(f"\nSummary statistics:")
print(f"  {'':20s} {'Old':>10s} {'New':>10s}")
print(f"  {'Mean P(scorer)':<20s} {old_all_p.mean():>10.4f} {new_all_p.mean():>10.4f}")
print(f"  {'Median P(scorer)':<20s} {np.median(old_all_p):>10.4f} {np.median(new_all_p):>10.4f}")
print(f"  {'Std P(scorer)':<20s} {old_all_p.std():>10.4f} {new_all_p.std():>10.4f}")
print(f"  {'P >= 0.50 count':<20s} {(old_all_p >= 0.50).sum():>10d} {(new_all_p >= 0.50).sum():>10d}")
print(f"  {'P >= 0.40 count':<20s} {(old_all_p >= 0.40).sum():>10d} {(new_all_p >= 0.40).sum():>10d}")
print(f"  {'P >= 0.30 count':<20s} {(old_all_p >= 0.30).sum():>10d} {(new_all_p >= 0.30).sum():>10d}")
print(f"  {'P >= 0.70 count':<20s} {(old_all_p >= 0.70).sum():>10d} {(new_all_p >= 0.70).sum():>10d}")


# ══════════════════════════════════════════════════════════════════════
# 2. Decision boundary analysis
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("2. DECISION BOUNDARY SHIFT")
print("=" * 70)

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
    old_above = old_all_p >= threshold
    new_above = new_all_p >= threshold
    n_old = old_above.sum()
    n_new = new_above.sum()
    delta_pct = (n_new - n_old) / n_old * 100 if n_old > 0 else 0

    if n_old > 0:
        old_hr = all_actual[old_above].mean()
    else:
        old_hr = float("nan")
    if n_new > 0:
        new_hr = all_actual[new_above].mean()
    else:
        new_hr = float("nan")

    print(f"  P>={threshold:.2f}:  Old n={n_old:5d} HR={old_hr:.3f}   "
          f"New n={n_new:5d} HR={new_hr:.3f}   "
          f"Δn={n_new-n_old:+5d} ({delta_pct:+.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# 3. Hit rates at various thresholds
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3. HIT RATES AT VARIOUS THRESHOLDS")
print("=" * 70)

print(f"\n  {'Threshold':<12s} {'Old HR':>8s} {'Old n':>7s} {'New HR':>8s} {'New n':>7s} {'Δ HR':>8s}")
print("  " + "-" * 55)

for t in [0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]:
    old_m = old_all_p >= t
    new_m = new_all_p >= t
    n_old = old_m.sum()
    n_new = new_m.sum()
    hr_old = all_actual[old_m].mean() if n_old > 0 else float("nan")
    hr_new = all_actual[new_m].mean() if n_new > 0 else float("nan")
    delta = hr_new - hr_old if not (np.isnan(hr_old) or np.isnan(hr_new)) else float("nan")
    delta_str = f"{delta:+.3f}" if not np.isnan(delta) else "N/A"
    print(f"  P>={t:<5.2f}    {hr_old:>7.3f} {n_old:>7d}  {hr_new:>7.3f} {n_new:>7d}  {delta_str:>8s}")


# ══════════════════════════════════════════════════════════════════════
# 4. Brier score comparison (threshold-independent)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4. BRIER SCORE & AUC COMPARISON (threshold-independent)")
print("=" * 70)

old_brier = brier_score_loss(all_actual, old_all_p)
new_brier = brier_score_loss(all_actual, new_all_p)
old_auc = roc_auc_score(all_actual, old_all_p)
new_auc = roc_auc_score(all_actual, new_all_p)

# Log loss
old_clip = np.clip(old_all_p, 1e-15, 1 - 1e-15)
new_clip = np.clip(new_all_p, 1e-15, 1 - 1e-15)
old_ll = float(np.mean(-all_actual * np.log(old_clip) - (1 - all_actual) * np.log(1 - old_clip)))
new_ll = float(np.mean(-all_actual * np.log(new_clip) - (1 - all_actual) * np.log(1 - new_clip)))

print(f"\n  {'Metric':<20s} {'Old (NaN→0)':>12s} {'New (NaN pres)':>14s} {'Δ':>10s} {'Better?':>8s}")
print("  " + "-" * 65)
print(f"  {'Brier Score':<20s} {old_brier:>12.4f} {new_brier:>14.4f} {new_brier-old_brier:>+10.4f} {'NEW' if new_brier < old_brier else 'OLD':>8s}")
print(f"  {'AUC':<20s} {old_auc:>12.4f} {new_auc:>14.4f} {new_auc-old_auc:>+10.4f} {'NEW' if new_auc > old_auc else 'OLD':>8s}")
print(f"  {'Log Loss':<20s} {old_ll:>12.4f} {new_ll:>14.4f} {new_ll-old_ll:>+10.4f} {'NEW' if new_ll < old_ll else 'OLD':>8s}")


# ══════════════════════════════════════════════════════════════════════
# 5. Calibration comparison
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("5. CALIBRATION COMPARISON")
print("=" * 70)

bin_edges = np.arange(0, 1.01, 0.10)
print(f"\n  {'Bin':<12s} {'Old pred':>9s} {'Old act':>8s} {'Old gap':>8s} │ {'New pred':>9s} {'New act':>8s} {'New gap':>8s}")
print("  " + "-" * 72)

old_ece_num = 0.0
new_ece_num = 0.0
total_n = 0

for i in range(len(bin_edges) - 1):
    lo, hi = bin_edges[i], bin_edges[i + 1]

    old_m = (old_all_p >= lo) & (old_all_p < hi) if i < len(bin_edges) - 2 else (old_all_p >= lo)
    new_m = (new_all_p >= lo) & (new_all_p < hi) if i < len(bin_edges) - 2 else (new_all_p >= lo)

    n_old = old_m.sum()
    n_new = new_m.sum()

    if n_old > 0:
        old_pred = old_all_p[old_m].mean()
        old_act = all_actual[old_m].mean()
        old_gap = abs(old_pred - old_act)
    else:
        old_pred = old_act = old_gap = float("nan")

    if n_new > 0:
        new_pred = new_all_p[new_m].mean()
        new_act = all_actual[new_m].mean()
        new_gap = abs(new_pred - new_act)
    else:
        new_pred = new_act = new_gap = float("nan")

    label = f"{int(lo*100)}-{int(hi*100)}%"

    def fmt(v):
        return f"{v:.4f}" if not np.isnan(v) else "—"

    print(f"  {label:<12s} {fmt(old_pred):>9s} {fmt(old_act):>8s} {fmt(old_gap):>8s} │ "
          f"{fmt(new_pred):>9s} {fmt(new_act):>8s} {fmt(new_gap):>8s}")

    if n_old > 0:
        old_ece_num += old_gap * n_old
    if n_new > 0:
        new_ece_num += new_gap * n_new
    total_n += max(n_old, n_new)

# ECE
old_total = sum(1 for p in old_all_p if True)
new_total = sum(1 for p in new_all_p if True)
print(f"\n  Old ECE: {old_ece_num / old_total:.4f}")
print(f"  New ECE: {new_ece_num / new_total:.4f}")


# ══════════════════════════════════════════════════════════════════════
# 6. ASCII histograms
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("6. P(SCORER) HISTOGRAMS")
print("=" * 70)

hist_bins = np.arange(0, 1.05, 0.10)
old_h, _ = np.histogram(old_all_p, bins=hist_bins)
new_h, _ = np.histogram(new_all_p, bins=hist_bins)
max_count = max(max(old_h), max(new_h))
scale = 50 / max_count

print(f"\n  OLD (NaN→0):")
for i in range(len(hist_bins) - 1):
    lo, hi = hist_bins[i], hist_bins[i + 1]
    bar = "█" * int(old_h[i] * scale)
    print(f"  {lo:.1f}-{hi:.1f} │{bar} {old_h[i]}")

print(f"\n  NEW (NaN preserved):")
for i in range(len(hist_bins) - 1):
    lo, hi = hist_bins[i], hist_bins[i + 1]
    bar = "█" * int(new_h[i] * scale)
    print(f"  {lo:.1f}-{hi:.1f} │{bar} {new_h[i]}")


# ══════════════════════════════════════════════════════════════════════
# 7. Per-player probability shift analysis
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("7. PROBABILITY SHIFT ANALYSIS")
print("=" * 70)

delta_p = new_all_p - old_all_p
print(f"\n  Mean shift:   {delta_p.mean():+.4f}")
print(f"  Median shift: {np.median(delta_p):+.4f}")
print(f"  Std of shift: {delta_p.std():.4f}")
print(f"  Max increase: {delta_p.max():+.4f}")
print(f"  Max decrease: {delta_p.min():+.4f}")

# How many players crossed the 0.50 boundary?
crossed_up = ((old_all_p < 0.50) & (new_all_p >= 0.50)).sum()
crossed_down = ((old_all_p >= 0.50) & (new_all_p < 0.50)).sum()
print(f"\n  Crossed P=0.50 upward:   {crossed_up}")
print(f"  Crossed P=0.50 downward: {crossed_down}")
print(f"  Net change at P>=0.50:   {crossed_up - crossed_down:+d}")

# Shift by scorer bucket
print(f"\n  Shift by old P(scorer) bucket:")
for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
    m = (old_all_p >= lo) & (old_all_p < hi)
    if m.sum() > 0:
        print(f"    P={lo:.1f}-{hi:.1f}: mean shift {delta_p[m].mean():+.4f} "
              f"(n={m.sum()}, {(delta_p[m] > 0).sum()} up, {(delta_p[m] < 0).sum()} down)")


# ══════════════════════════════════════════════════════════════════════
# DIAGNOSIS
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

# Check if the denominator change explains the hit rate drop
old_above_50 = old_all_p >= 0.50
new_above_50 = new_all_p >= 0.50
old_hr_50 = all_actual[old_above_50].mean() if old_above_50.sum() > 0 else 0
new_hr_50 = all_actual[new_above_50].mean() if new_above_50.sum() > 0 else 0

print(f"\n  At P>=0.50:")
print(f"    Old: {old_above_50.sum()} predictions, hit rate = {old_hr_50:.3f}")
print(f"    New: {new_above_50.sum()} predictions, hit rate = {new_hr_50:.3f}")
print(f"    Δ predictions: {new_above_50.sum() - old_above_50.sum():+d}")
print(f"    Δ hit rate: {new_hr_50 - old_hr_50:+.3f}")

if new_above_50.sum() > old_above_50.sum():
    # More predictions above 0.50 → diluted hit rate
    extra_mask = new_above_50 & ~old_above_50
    extra_hr = all_actual[extra_mask].mean()
    print(f"\n  The {extra_mask.sum()} NEW predictions above 0.50 have hit rate = {extra_hr:.3f}")
    print(f"  These marginal predictions dilute the overall hit rate.")
elif new_above_50.sum() < old_above_50.sum():
    lost_mask = old_above_50 & ~new_above_50
    lost_hr = all_actual[lost_mask].mean()
    print(f"\n  The {lost_mask.sum()} predictions that DROPPED below 0.50 had hit rate = {lost_hr:.3f}")
    if lost_hr > new_hr_50:
        print(f"  These were GOOD predictions (HR > current). Model became more conservative.")
    else:
        print(f"  These were WEAK predictions (HR < current). Model correctly pruned them.")

# Find the threshold where new model matches old 78% hit rate
print(f"\n  Finding threshold where NEW model achieves ~78% hit rate:")
for t in np.arange(0.50, 0.95, 0.01):
    m = new_all_p >= t
    if m.sum() < 20:
        continue
    hr = all_actual[m].mean()
    if abs(hr - 0.78) < 0.015:
        print(f"    P>={t:.2f}: HR={hr:.3f} (n={m.sum()})")
        break

print()
