#!/usr/bin/env python3
"""
AFL Pattern Discovery & Validation System
==========================================
Multi-season backtest analysis across 2021-2025.

Analyses:
  A. Feature stability across years
  B. Hyperparameter (blend ratio) sensitivity
  C. Feature subset testing
  D. Era weighting sensitivity
  E. Model architecture (Poisson vs GBT) stability
  F. Validation of best config
"""

import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
import config
from metrics import brier, bss, compute_sample_weights, compute_all_brier, load_feature_matrix
from model import (
    AFLScoringModel, AFLDisposalModel, AFLMarksModel,
    _prepare_features, _predict_with_compatible_input,
)

YEARS = [2021, 2022, 2023, 2024, 2025]
BLEND_RATIOS = [
    {"poisson": 0.9, "gbt": 0.1},
    {"poisson": 0.8, "gbt": 0.2},
    {"poisson": 0.7, "gbt": 0.3},
    {"poisson": 0.6, "gbt": 0.4},
    {"poisson": 0.5, "gbt": 0.5},
]
ERA_WEIGHT_CONFIGS = {
    "current": {(2015, 2019): 0.4, (2020, 2022): 0.7, (2023, 2024): 0.9, (2025, 2026): 1.0},
    "heavy_recency": {(2015, 2019): 0.2, (2020, 2022): 0.5, (2023, 2024): 0.8, (2025, 2026): 1.0},
    "flat": {(2015, 2019): 0.6, (2020, 2022): 0.8, (2023, 2024): 0.9, (2025, 2026): 1.0},
}


# ---------------------------------------------------------------------------
# Fast train-predict (no round-by-round sequential, single train/predict)
# ---------------------------------------------------------------------------

def fast_train_predict(feature_df, test_year, feature_cols, ensemble_weights=None,
                       sample_weights_col="sample_weight", return_components=False):
    """Train on data before test_year, predict test_year. Return merged df with predictions + actuals."""
    ew = ensemble_weights or config.ENSEMBLE_WEIGHTS

    train_df = feature_df[feature_df["year"] < test_year].copy()
    test_df = feature_df[feature_df["year"] == test_year].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        return None

    # --- Scoring model ---
    scoring = AFLScoringModel(ensemble_weights=ew)
    scoring.train_backtest(train_df, feature_cols)

    # --- Disposal model ---
    disp = AFLDisposalModel(
        distribution=config.DISPOSAL_DISTRIBUTION,
        ensemble_weights=ew,
    )
    disp.train_backtest(train_df, feature_cols)

    # --- Marks model ---
    marks = AFLMarksModel(
        distribution=getattr(config, "MARKS_DISTRIBUTION", "gaussian"),
        ensemble_weights=ew,
    )
    marks.train_backtest(train_df, feature_cols)

    # Predict (no store → no calibration)
    goal_preds = scoring.predict_distributions(test_df, store=None, feature_cols=feature_cols)
    disp_preds = disp.predict_distributions(test_df, store=None, feature_cols=feature_cols)
    mark_preds = marks.predict_distributions(test_df, store=None, feature_cols=feature_cols)

    # Build merged result
    join_cols = ["player", "team", "match_id"]
    result = test_df[["player", "team", "match_id", "GL", "DI", "MK"]].copy()
    result = result.rename(columns={"GL": "actual_goals", "DI": "actual_disposals", "MK": "actual_marks"})

    # Merge goal predictions
    g_cols = [c for c in goal_preds.columns if c.startswith("p_") or c.startswith("predicted_") or c.startswith("lambda_")]
    g_merge = goal_preds[["player", "team", "match_id"] + [c for c in g_cols if c in goal_preds.columns]]
    result = result.merge(g_merge, on=join_cols, how="left")

    # Merge disposal predictions
    d_cols = [c for c in disp_preds.columns if c.startswith("p_") or c == "predicted_disposals" or c == "lambda_disposals"]
    d_merge = disp_preds[["player", "team", "match_id"] + [c for c in d_cols if c in disp_preds.columns]]
    result = result.merge(d_merge, on=join_cols, how="left", suffixes=("", "_d"))

    # Merge marks predictions
    m_cols = [c for c in mark_preds.columns if c.startswith("p_") or c == "predicted_marks" or c == "lambda_marks"]
    m_merge = mark_preds[["player", "team", "match_id"] + [c for c in m_cols if c in mark_preds.columns]]
    result = result.merge(m_merge, on=join_cols, how="left", suffixes=("", "_m"))

    components = None
    if return_components:
        components = {
            "scoring": scoring,
            "disposal": disp,
            "marks": marks,
        }

    return result, components


def fast_predict_with_weights(feature_df, test_year, feature_cols, ensemble_weights):
    """Fast train-predict with specific ensemble weights. Returns Brier dict."""
    result = fast_train_predict(feature_df, test_year, feature_cols,
                                ensemble_weights=ensemble_weights)
    if result is None:
        return {}
    merged, _ = result
    return compute_all_brier(merged)


# ---------------------------------------------------------------------------
# Step 2A: Feature Stability Analysis
# ---------------------------------------------------------------------------

def analyze_feature_stability(feature_df, feature_cols):
    """For each year, train GBT and rank features by importance."""
    print("\n" + "=" * 72)
    print("  STEP 2A: FEATURE STABILITY ANALYSIS")
    print("=" * 72)

    year_importances = {}

    for year in YEARS:
        train_df = feature_df[feature_df["year"] < year].copy()
        test_df = feature_df[feature_df["year"] == year].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        y_train = train_df["GL"].values
        w_train = train_df["sample_weight"].values if "sample_weight" in train_df.columns else None

        X_raw, X_clean, X_scaled = _prepare_features(train_df, feature_cols, scaler=StandardScaler(), fit_scaler=True)

        # Train GBT for goals
        gbt = HistGradientBoostingRegressor(**config.HIST_GBT_PARAMS_BACKTEST)
        gbt.fit(X_raw, y_train, sample_weight=w_train)

        # Also train for disposals and marks
        gbt_di = HistGradientBoostingRegressor(**config.DISPOSAL_GBT_PARAMS_BACKTEST)
        gbt_di.fit(X_raw, train_df["DI"].values, sample_weight=w_train)

        gbt_mk = HistGradientBoostingRegressor(**config.MARKS_GBT_PARAMS_BACKTEST)
        gbt_mk.fit(X_raw, train_df["MK"].values, sample_weight=w_train)

        # Combined importance: average across 3 targets
        imp_gl = gbt.feature_importances_ if hasattr(gbt, "feature_importances_") else np.zeros(len(feature_cols))
        imp_di = gbt_di.feature_importances_ if hasattr(gbt_di, "feature_importances_") else np.zeros(len(feature_cols))
        imp_mk = gbt_mk.feature_importances_ if hasattr(gbt_mk, "feature_importances_") else np.zeros(len(feature_cols))

        combined = (imp_gl + imp_di + imp_mk) / 3.0
        ranks = np.argsort(combined)[::-1]

        year_importances[year] = {
            "importance": combined,
            "ranks": ranks,
            "rank_map": {feature_cols[r]: i + 1 for i, r in enumerate(ranks)},
        }
        print(f"  {year}: trained on {len(train_df)} rows, top feature = {feature_cols[ranks[0]]}")

    # Analyze cross-year stability
    all_features = feature_cols
    stability = {}
    for feat in all_features:
        ranks_across_years = []
        for year in YEARS:
            if year in year_importances:
                ranks_across_years.append(year_importances[year]["rank_map"].get(feat, len(feature_cols)))
        if ranks_across_years:
            stability[feat] = {
                "ranks": ranks_across_years,
                "mean_rank": np.mean(ranks_across_years),
                "std_rank": np.std(ranks_across_years),
                "min_rank": min(ranks_across_years),
                "max_rank": max(ranks_across_years),
                "in_top30_count": sum(1 for r in ranks_across_years if r <= 30),
                "in_top50_count": sum(1 for r in ranks_across_years if r <= 50),
                "in_bottom50_count": sum(1 for r in ranks_across_years if r > len(feature_cols) - 50),
            }

    # Sort by mean rank (most stable = lowest mean rank)
    sorted_stable = sorted(stability.items(), key=lambda x: x[1]["mean_rank"])
    # Sort by instability (highest std_rank)
    sorted_unstable = sorted(stability.items(), key=lambda x: -x[1]["std_rank"])

    # Strong patterns: top 30 in ALL 5 years
    strong = [f for f, s in stability.items() if s["in_top30_count"] == len(YEARS)]
    # Weak patterns: top 30 in only 1-2 years
    weak = [f for f, s in stability.items() if s["in_top30_count"] <= 2 and s["min_rank"] <= 30]

    print(f"\n  Strong patterns (top 30 in all {len(YEARS)} years): {len(strong)}")
    print(f"  Weak patterns (top 30 in ≤2 years): {len(weak)}")

    # Print top 20 most stable
    print(f"\n  Top 20 most stable features:")
    print(f"  {'Feature':45s} {'Mean':>6s} {'Std':>6s} " + "  ".join(f"{y}" for y in YEARS))
    for feat, s in sorted_stable[:20]:
        rank_str = "  ".join(f"{r:4d}" for r in s["ranks"])
        print(f"  {feat:45s} {s['mean_rank']:6.1f} {s['std_rank']:6.1f} {rank_str}")

    # Print top 20 most unstable
    print(f"\n  Top 20 most unstable features:")
    print(f"  {'Feature':45s} {'Mean':>6s} {'Std':>6s} " + "  ".join(f"{y}" for y in YEARS))
    for feat, s in sorted_unstable[:20]:
        rank_str = "  ".join(f"{r:4d}" for r in s["ranks"])
        print(f"  {feat:45s} {s['mean_rank']:6.1f} {s['std_rank']:6.1f} {rank_str}")

    return {
        "year_importances": {y: {"rank_map": v["rank_map"]} for y, v in year_importances.items()},
        "stability": {f: {k: v for k, v in s.items() if k != "ranks"} for f, s in stability.items()},
        "strong_patterns": strong,
        "weak_patterns": weak,
        "top20_stable": [(f, s) for f, s in sorted_stable[:20]],
        "top20_unstable": [(f, s) for f, s in sorted_unstable[:20]],
    }


# ---------------------------------------------------------------------------
# Step 2B: Blend Ratio Sensitivity
# ---------------------------------------------------------------------------

def analyze_blend_ratios(feature_df, feature_cols):
    """Test 5 Poisson/GBT blend ratios across all years."""
    print("\n" + "=" * 72)
    print("  STEP 2B: BLEND RATIO SENSITIVITY")
    print("=" * 72)

    results = {}
    for year in YEARS:
        results[year] = {}
        for ratio in BLEND_RATIOS:
            label = f"{int(ratio['poisson']*100)}/{int(ratio['gbt']*100)}"
            t0 = time.time()
            metrics = fast_predict_with_weights(feature_df, year, feature_cols, ratio)
            elapsed = time.time() - t0
            results[year][label] = metrics
            g1 = metrics.get("1plus_goals", {}).get("brier", "N/A")
            d20 = metrics.get("20plus_disp", {}).get("brier", "N/A")
            print(f"  {year} {label:5s}  GL1+={g1}  DI20+={d20}  ({elapsed:.1f}s)")

    # Find best ratio on average across all years
    avg_by_ratio = {}
    for ratio in BLEND_RATIOS:
        label = f"{int(ratio['poisson']*100)}/{int(ratio['gbt']*100)}"
        gl1_avg = np.mean([results[y][label].get("1plus_goals", {}).get("brier", np.nan) for y in YEARS])
        d20_avg = np.mean([results[y][label].get("20plus_disp", {}).get("brier", np.nan) for y in YEARS])
        combined = gl1_avg + d20_avg  # simple sum for ranking
        avg_by_ratio[label] = {"gl1_brier_avg": round(gl1_avg, 4), "d20_brier_avg": round(d20_avg, 4),
                               "combined": round(combined, 4)}

    best_ratio = min(avg_by_ratio, key=lambda k: avg_by_ratio[k]["combined"])
    print(f"\n  Best ratio on average: {best_ratio}")
    print(f"  {'Ratio':7s} {'GL1+ avg':>10s} {'D20+ avg':>10s} {'Combined':>10s}")
    for label, m in sorted(avg_by_ratio.items(), key=lambda x: x[1]["combined"]):
        marker = " <-- best" if label == best_ratio else ""
        print(f"  {label:7s} {m['gl1_brier_avg']:10.4f} {m['d20_brier_avg']:10.4f} {m['combined']:10.4f}{marker}")

    return {"per_year": results, "averages": avg_by_ratio, "best_ratio": best_ratio}


# ---------------------------------------------------------------------------
# Step 2C: Feature Subset Testing
# ---------------------------------------------------------------------------

def analyze_feature_subsets(feature_df, feature_cols, stability_data):
    """Test feature subsets on 2025 backtest."""
    print("\n" + "=" * 72)
    print("  STEP 2C: FEATURE SUBSET TESTING (2025)")
    print("=" * 72)

    stab = stability_data["stability"]

    # Config 1: All features
    config1 = feature_cols

    # Config 2: Features ranking top 50 in at least 4 of 5 years
    config2 = [f for f in feature_cols if f in stab and stab[f]["in_top50_count"] >= 4]

    # Config 3: Features ranking top 30 in at least 3 of 5 years
    config3 = [f for f in feature_cols if f in stab and stab[f]["in_top30_count"] >= 3]

    # Config 4: Remove features in bottom 50 in ANY year
    config4 = [f for f in feature_cols if f in stab and stab[f]["in_bottom50_count"] == 0]

    configs = {
        "all_features": config1,
        "stable_top50_4yr": config2,
        "very_stable_top30_3yr": config3,
        "prune_bottom50": config4,
    }

    results = {}
    for name, feat_subset in configs.items():
        t0 = time.time()
        # Validate all features exist
        valid_feats = [f for f in feat_subset if f in feature_df.columns]
        metrics = fast_predict_with_weights(feature_df, 2025, valid_feats, config.ENSEMBLE_WEIGHTS)
        elapsed = time.time() - t0
        results[name] = {"n_features": len(valid_feats), "metrics": metrics}
        g1 = metrics.get("1plus_goals", {}).get("brier", "N/A")
        d20 = metrics.get("20plus_disp", {}).get("brier", "N/A")
        m3 = metrics.get("3plus_mk", {}).get("brier", "N/A")
        print(f"  {name:25s}  n={len(valid_feats):3d}  GL1+={g1}  DI20+={d20}  MK3+={m3}  ({elapsed:.1f}s)")

    # Determine best config
    best = min(results, key=lambda k: results[k]["metrics"].get("1plus_goals", {}).get("brier", 99))
    print(f"\n  Best feature subset: {best} ({results[best]['n_features']} features)")

    return results


# ---------------------------------------------------------------------------
# Step 2D: Era Weighting Sensitivity
# ---------------------------------------------------------------------------

def analyze_era_weights(feature_df, feature_cols):
    """Test different era weighting schemes."""
    print("\n" + "=" * 72)
    print("  STEP 2D: ERA WEIGHTING SENSITIVITY")
    print("=" * 72)

    results = {}
    for config_name, era_weights in ERA_WEIGHT_CONFIGS.items():
        results[config_name] = {}
        for year in YEARS:
            train_df = feature_df[feature_df["year"] < year].copy()
            test_df = feature_df[feature_df["year"] == year].copy()
            if len(train_df) == 0 or len(test_df) == 0:
                continue

            # Recompute sample weights with this era config
            weights = compute_sample_weights(train_df, era_weights)
            train_df = train_df.copy()
            train_df["sample_weight"] = weights

            # Recombine for fast_train_predict
            combined = pd.concat([train_df, test_df], ignore_index=True)
            metrics = fast_predict_with_weights(combined, year, feature_cols, config.ENSEMBLE_WEIGHTS)
            results[config_name][year] = metrics

        # Average Brier across years
        gl1_vals = [results[config_name][y].get("1plus_goals", {}).get("brier", np.nan) for y in YEARS if y in results[config_name]]
        d20_vals = [results[config_name][y].get("20plus_disp", {}).get("brier", np.nan) for y in YEARS if y in results[config_name]]
        avg_gl1 = np.nanmean(gl1_vals)
        avg_d20 = np.nanmean(d20_vals)
        results[config_name]["avg_gl1"] = round(avg_gl1, 4)
        results[config_name]["avg_d20"] = round(avg_d20, 4)
        print(f"  {config_name:15s}  avg GL1+={avg_gl1:.4f}  avg DI20+={avg_d20:.4f}")

    best = min(results, key=lambda k: results[k].get("avg_gl1", 99) + results[k].get("avg_d20", 99))
    print(f"\n  Best era weighting: {best}")

    return results


# ---------------------------------------------------------------------------
# Step 2E: Model Architecture Stability
# ---------------------------------------------------------------------------

def analyze_architecture(feature_df, feature_cols):
    """Analyze Poisson vs GBT component correlation and accuracy."""
    print("\n" + "=" * 72)
    print("  STEP 2E: MODEL ARCHITECTURE STABILITY")
    print("=" * 72)

    results = {}
    for year in YEARS:
        train_df = feature_df[feature_df["year"] < year].copy()
        test_df = feature_df[feature_df["year"] == year].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        y_train = train_df["GL"].values
        w_train = train_df["sample_weight"].values if "sample_weight" in train_df.columns else None

        scaler = StandardScaler()
        X_train_raw, X_train_clean, X_train_scaled = _prepare_features(train_df, feature_cols, scaler=scaler, fit_scaler=True)
        X_test_raw, X_test_clean, X_test_scaled = _prepare_features(test_df, feature_cols, scaler=scaler)

        # Train Poisson
        poi = PoissonRegressor(alpha=config.POISSON_PARAMS["alpha"], max_iter=1000)
        poi.fit(X_train_scaled, y_train, sample_weight=w_train)
        pred_poi = poi.predict(X_test_scaled)

        # Train GBT
        gbt = HistGradientBoostingRegressor(**config.HIST_GBT_PARAMS_BACKTEST)
        gbt.fit(X_train_raw, y_train, sample_weight=w_train)
        X_test_clean_np = X_test_raw.fillna(0) if hasattr(X_test_raw, "fillna") else X_test_raw
        pred_gbt = _predict_with_compatible_input(gbt, X_test_raw, X_test_clean_np)

        actual = test_df["GL"].values

        # Correlation
        corr = float(np.corrcoef(pred_poi, pred_gbt)[0, 1])

        # MAE per component
        mae_poi = float(mean_absolute_error(actual, pred_poi))
        mae_gbt = float(mean_absolute_error(actual, pred_gbt))

        # When they disagree most (top 10% disagreement)
        disagreement = np.abs(pred_poi - pred_gbt)
        thresh = np.percentile(disagreement, 90)
        high_disagree = disagreement >= thresh
        mae_poi_disagree = float(mean_absolute_error(actual[high_disagree], pred_poi[high_disagree]))
        mae_gbt_disagree = float(mean_absolute_error(actual[high_disagree], pred_gbt[high_disagree]))
        gbt_wins_disagree = mae_gbt_disagree < mae_poi_disagree

        results[year] = {
            "poisson_gbt_corr": round(corr, 4),
            "poisson_mae": round(mae_poi, 4),
            "gbt_mae": round(mae_gbt, 4),
            "poisson_mae_high_disagree": round(mae_poi_disagree, 4),
            "gbt_mae_high_disagree": round(mae_gbt_disagree, 4),
            "gbt_wins_when_disagree": gbt_wins_disagree,
        }
        winner = "GBT" if gbt_wins_disagree else "Poisson"
        print(f"  {year}: corr={corr:.3f}  POI_MAE={mae_poi:.3f}  GBT_MAE={mae_gbt:.3f}  "
              f"High-disagree winner: {winner}")

    return results


# ---------------------------------------------------------------------------
# Step 3: Validation
# ---------------------------------------------------------------------------

def validate_best_config(feature_df, feature_cols, best_blend, best_features, best_era):
    """Apply best config to 2023-2025 and compare to default."""
    print("\n" + "=" * 72)
    print("  STEP 3: VALIDATION (Best vs Default)")
    print("=" * 72)

    val_years = [2023, 2024, 2025]
    results = {"default": {}, "best": {}}

    for year in val_years:
        # Default config
        metrics_default = fast_predict_with_weights(feature_df, year, feature_cols, config.ENSEMBLE_WEIGHTS)
        results["default"][year] = metrics_default

        # Best config: apply best era weights + best blend + best features
        # Recompute sample weights
        era_w = ERA_WEIGHT_CONFIGS.get(best_era, ERA_WEIGHT_CONFIGS["current"])
        train_df = feature_df[feature_df["year"] < year].copy()
        test_df = feature_df[feature_df["year"] == year].copy()
        weights = compute_sample_weights(train_df, era_w)
        train_df["sample_weight"] = weights
        combined = pd.concat([train_df, test_df], ignore_index=True)

        # Parse best blend
        parts = best_blend.split("/")
        blend_w = {"poisson": int(parts[0]) / 100, "gbt": int(parts[1]) / 100}

        # Use best features
        valid_feats = [f for f in best_features if f in feature_df.columns]
        metrics_best = fast_predict_with_weights(combined, year, valid_feats, blend_w)
        results["best"][year] = metrics_best

        g1_def = metrics_default.get("1plus_goals", {}).get("brier", np.nan)
        g1_best = metrics_best.get("1plus_goals", {}).get("brier", np.nan)
        d20_def = metrics_default.get("20plus_disp", {}).get("brier", np.nan)
        d20_best = metrics_best.get("20plus_disp", {}).get("brier", np.nan)
        g1_delta = g1_best - g1_def if not np.isnan(g1_def) and not np.isnan(g1_best) else np.nan
        d20_delta = d20_best - d20_def if not np.isnan(d20_def) and not np.isnan(d20_best) else np.nan
        print(f"  {year}: GL1+ default={g1_def:.4f} best={g1_best:.4f} delta={g1_delta:+.4f}  "
              f"DI20+ default={d20_def:.4f} best={d20_best:.4f} delta={d20_delta:+.4f}")

    # Count improvements
    improve_count = 0
    for year in val_years:
        g1_def = results["default"][year].get("1plus_goals", {}).get("brier", np.nan)
        g1_best = results["best"][year].get("1plus_goals", {}).get("brier", np.nan)
        if not np.isnan(g1_def) and not np.isnan(g1_best) and g1_best <= g1_def:
            improve_count += 1

    print(f"\n  Best config improves/matches {improve_count}/{len(val_years)} years")
    if improve_count >= len(val_years) - 1:
        print("  → Validated: consistent improvement across years")
    else:
        print("  → Possible overfitting: improvement not consistent")

    return results


# ---------------------------------------------------------------------------
# Step 4: Report
# ---------------------------------------------------------------------------

def print_report(stability_data, blend_data, subset_data, era_data, arch_data, validation_data, feature_cols):
    """Print summary tables."""
    print("\n")
    print("=" * 80)
    print("  PATTERN DISCOVERY SUMMARY REPORT")
    print("=" * 80)

    # Table 1: Feature stability
    print("\n--- TABLE 1: Feature Stability (Top 20 by cross-year consistency) ---")
    print(f"  {'#':>3s} {'Feature':45s} {'Mean':>6s} {'Std':>5s} " +
          "  ".join(f"{y}" for y in YEARS))
    for i, (feat, s) in enumerate(stability_data["top20_stable"][:20], 1):
        ranks = s.get("ranks", [])
        rank_str = "  ".join(f"{r:4d}" for r in ranks)
        print(f"  {i:3d} {feat:45s} {s['mean_rank']:6.1f} {s['std_rank']:5.1f} {rank_str}")

    # Table 2: Best hyperparameter config
    print("\n--- TABLE 2: Best Hyperparameter Config ---")
    best_blend = blend_data["best_ratio"]
    best_era = min(era_data, key=lambda k: era_data[k].get("avg_gl1", 99) + era_data[k].get("avg_d20", 99))
    best_subset = min(subset_data, key=lambda k: subset_data[k]["metrics"].get("1plus_goals", {}).get("brier", 99))

    print(f"  Best blend ratio:     {best_blend}")
    print(f"  Best era weighting:   {best_era}")
    print(f"  Best feature subset:  {best_subset} ({subset_data[best_subset]['n_features']} features)")
    print()

    print(f"  {'Year':>6s} ", end="")
    for ratio_label in sorted(blend_data["averages"]):
        print(f" {ratio_label:>8s}", end="")
    print()
    for year in YEARS:
        print(f"  {year:6d} ", end="")
        for ratio_label in sorted(blend_data["per_year"][year]):
            val = blend_data["per_year"][year][ratio_label].get("1plus_goals", {}).get("brier", np.nan)
            print(f" {val:8.4f}", end="")
        print()

    # Table 3: Winning config vs default
    print("\n--- TABLE 3: Winning Config vs Default ---")
    val_years = [2023, 2024, 2025]
    targets = ["1plus_goals", "2plus_goals", "3plus_goals", "15plus_disp", "20plus_disp",
               "25plus_disp", "30plus_disp", "2plus_mk", "3plus_mk", "4plus_mk", "5plus_mk"]
    print(f"  {'Target':15s}", end="")
    for year in val_years:
        print(f"  {year} def  {year} best  delta", end="")
    print()
    for tgt in targets:
        print(f"  {tgt:15s}", end="")
        for year in val_years:
            d = validation_data["default"].get(year, {}).get(tgt, {}).get("brier", np.nan)
            b = validation_data["best"].get(year, {}).get(tgt, {}).get("brier", np.nan)
            delta = b - d if not np.isnan(b) and not np.isnan(d) else np.nan
            d_s = f"{d:.4f}" if not np.isnan(d) else "  N/A "
            b_s = f"{b:.4f}" if not np.isnan(b) else "  N/A "
            delta_s = f"{delta:+.4f}" if not np.isnan(delta) else "  N/A "
            print(f"  {d_s}  {b_s} {delta_s}", end="")
        print()

    # Table 4: Features recommended for removal
    print("\n--- TABLE 4: Features Recommended for Removal ---")
    print("  (Bottom 50 in 2+ years)")
    stab = stability_data["stability"]
    removal = [(f, s) for f, s in stab.items() if s.get("in_bottom50_count", 0) >= 2]
    removal.sort(key=lambda x: -x[1]["in_bottom50_count"])
    for feat, s in removal[:20]:
        print(f"  {feat:45s}  bottom50_in={s['in_bottom50_count']} years  mean_rank={s['mean_rank']:.0f}")
    if len(removal) > 20:
        print(f"  ... and {len(removal) - 20} more")
    print(f"\n  Total features recommended for removal: {len(removal)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print("=" * 72)
    print("  AFL PATTERN DISCOVERY & VALIDATION SYSTEM")
    print("  Multi-season analysis: 2021-2025")
    print("=" * 72)

    # Load feature matrix
    print("\nLoading feature matrix...")
    feature_df, feature_cols = load_feature_matrix()
    print(f"  {len(feature_df)} rows, {len(feature_cols)} features")

    # Step 2A: Feature stability
    stability_data = analyze_feature_stability(feature_df, feature_cols)

    # Step 2B: Blend ratio sensitivity
    blend_data = analyze_blend_ratios(feature_df, feature_cols)

    # Step 2C: Feature subset testing
    subset_data = analyze_feature_subsets(feature_df, feature_cols, stability_data)

    # Step 2D: Era weighting sensitivity
    era_data = analyze_era_weights(feature_df, feature_cols)

    # Step 2E: Model architecture stability
    arch_data = analyze_architecture(feature_df, feature_cols)

    # Step 3: Validation
    best_blend = blend_data["best_ratio"]
    best_era = min(era_data, key=lambda k: era_data[k].get("avg_gl1", 99) + era_data[k].get("avg_d20", 99))
    best_subset_name = min(subset_data, key=lambda k: subset_data[k]["metrics"].get("1plus_goals", {}).get("brier", 99))
    # Get features for best subset
    stab = stability_data["stability"]
    if best_subset_name == "all_features":
        best_features = feature_cols
    elif best_subset_name == "stable_top50_4yr":
        best_features = [f for f in feature_cols if f in stab and stab[f]["in_top50_count"] >= 4]
    elif best_subset_name == "very_stable_top30_3yr":
        best_features = [f for f in feature_cols if f in stab and stab[f]["in_top30_count"] >= 3]
    elif best_subset_name == "prune_bottom50":
        best_features = [f for f in feature_cols if f in stab and stab[f]["in_bottom50_count"] == 0]
    else:
        best_features = feature_cols

    validation_data = validate_best_config(feature_df, feature_cols, best_blend, best_features, best_era)

    # Step 4: Report
    print_report(stability_data, blend_data, subset_data, era_data, arch_data, validation_data, feature_cols)

    # Save JSON
    experiment = {
        "label": "pattern_discovery_2021_2025",
        "years": YEARS,
        "n_features": len(feature_cols),
        "n_rows": len(feature_df),
        "step2a_stability": {
            "strong_patterns": stability_data["strong_patterns"],
            "weak_patterns": stability_data["weak_patterns"],
            "top20_stable": [(f, {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in s.items()})
                            for f, s in stability_data["top20_stable"]],
            "top20_unstable": [(f, {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in s.items()})
                              for f, s in stability_data["top20_unstable"]],
        },
        "step2b_blend_ratios": {
            "per_year": blend_data["per_year"],
            "averages": blend_data["averages"],
            "best_ratio": blend_data["best_ratio"],
        },
        "step2c_feature_subsets": {
            name: {"n_features": d["n_features"], "metrics": d["metrics"]}
            for name, d in subset_data.items()
        },
        "step2d_era_weights": {
            name: {k: v for k, v in d.items() if k not in YEARS}
            for name, d in era_data.items()
        },
        "step2e_architecture": arch_data,
        "step3_validation": {
            "best_config": {
                "blend": best_blend,
                "era": best_era,
                "feature_subset": best_subset_name,
                "n_features": len(best_features),
            },
            "results": validation_data,
        },
        "step4_removal_candidates": [
            f for f, s in stability_data["stability"].items()
            if s.get("in_bottom50_count", 0) >= 2
        ],
    }

    out_path = config.EXPERIMENTS_DIR / "pattern_discovery_2021_2025.json"
    config.ensure_dirs()
    with open(out_path, "w") as f:
        json.dump(experiment, f, indent=2, default=str)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Saved: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
