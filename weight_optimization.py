#!/usr/bin/env python3
"""
AFL Weight Optimization System
===============================
Exhaustive multi-season sweep across 7 dimensions to discover optimal
model weights. Trains models once per year, then varies prediction-time
parameters for dimensions A-D and G. Only dimensions E and F require
retraining.

Dimensions:
  A. Goals blend ratio (Poisson/GBT)
  B. Disposals blend ratio
  C. Marks blend ratio
  D. Scorer exponent + mark-taker params
  E. Era weight schemes (requires retrain)
  F. Elo parameters + hybrid params (requires retrain)
  G. Gaussian disposal tail parameters

Output: data/experiments/weight_optimization_2021_2025.json
"""

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

try:
    from contextlib import nullcontext
except ImportError:
    @contextmanager
    def nullcontext():
        yield

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from model import (
    AFLScoringModel, AFLDisposalModel, AFLMarksModel,
    AFLGameWinnerModel, EloSystem,
    _prepare_features, _predict_with_compatible_input,
    _predict_proba_with_compatible_input,
)
from metrics import brier, bss, compute_sample_weights, load_feature_matrix
from pattern_discovery import fast_train_predict
from features import add_dynamic_sample_weights

YEARS = [2021, 2022, 2023, 2024, 2025]

# ---------------------------------------------------------------------------
# Blend ratio grids (Dim A-C)
# ---------------------------------------------------------------------------
BLEND_RATIOS = [
    {"poisson": 0.9, "gbt": 0.1},
    {"poisson": 0.8, "gbt": 0.2},
    {"poisson": 0.7, "gbt": 0.3},
    {"poisson": 0.6, "gbt": 0.4},
    {"poisson": 0.5, "gbt": 0.5},
    {"poisson": 0.4, "gbt": 0.6},
    {"poisson": 0.3, "gbt": 0.7},
    {"poisson": 0.2, "gbt": 0.8},
]

# ---------------------------------------------------------------------------
# Era weight schemes (Dim E)
# ---------------------------------------------------------------------------
ERA_WEIGHT_CONFIGS = {
    "current": {(2015, 2019): 0.4, (2020, 2022): 0.7, (2023, 2024): 0.9, (2025, 2026): 1.0},
    "heavy_recency": {(2015, 2019): 0.2, (2020, 2022): 0.5, (2023, 2024): 0.8, (2025, 2026): 1.0},
    "flat": {(2015, 2019): 0.8, (2020, 2022): 0.9, (2023, 2024): 0.95, (2025, 2026): 1.0},
    "exclude_covid": {(2015, 2019): 0.5, (2020, 2020): 0.1, (2021, 2022): 0.6, (2023, 2024): 0.9, (2025, 2026): 1.0},
    "recent_only": {(2015, 2019): 0.1, (2020, 2022): 0.3, (2023, 2024): 1.0, (2025, 2026): 1.0},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def config_override(**kwargs):
    """Temporarily override config globals, auto-restore on exit."""
    originals = {}
    for key, value in kwargs.items():
        originals[key] = getattr(config, key)
        setattr(config, key, value)
    try:
        yield
    finally:
        for key, value in originals.items():
            setattr(config, key, value)


def ratio_label(r):
    return f"{int(r['poisson']*100)}/{int(r['gbt']*100)}"


def compute_full_metrics(merged):
    """Compute Brier/BSS for ALL thresholds + MAEs."""
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
        results[label] = {"brier": round(brier(p, a), 6), "bss": round(bss(p, a), 6)}

    # Disposals — expanded thresholds
    for thresh in [10, 15, 20, 25, 30]:
        col = f"p_{thresh}plus_disp"
        if col not in merged.columns:
            continue
        p = merged[col].values.astype(float)
        a = (merged["actual_disposals"] >= thresh).astype(int).values.astype(float)
        results[f"{thresh}plus_disp"] = {"brier": round(brier(p, a), 6), "bss": round(bss(p, a), 6)}

    # Marks — expanded thresholds
    for thresh in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        col = f"p_{thresh}plus_mk"
        if col not in merged.columns:
            continue
        p = merged[col].values.astype(float)
        a = (merged["actual_marks"] >= thresh).astype(int).values.astype(float)
        results[f"{thresh}plus_mk"] = {"brier": round(brier(p, a), 6), "bss": round(bss(p, a), 6)}

    # MAEs
    for pred_col, act_col, key in [
        ("predicted_goals", "actual_goals", "goals_mae"),
        ("predicted_disposals", "actual_disposals", "disp_mae"),
        ("predicted_marks", "actual_marks", "marks_mae"),
    ]:
        if pred_col in merged.columns and act_col in merged.columns:
            v = merged[pred_col].notna() & merged[act_col].notna()
            results[key] = round(float(np.abs(
                merged.loc[v, pred_col] - merged.loc[v, act_col]
            ).mean()), 6)

    return results


def train_models_for_year(feature_df, year, feature_cols, ensemble_weights=None):
    """Train scoring/disposal/marks models on data before `year`.

    Returns (models_dict, test_df) where models_dict has keys
    'scoring', 'disposal', 'marks'.
    """
    ew = ensemble_weights or config.ENSEMBLE_WEIGHTS
    train_df = feature_df[feature_df["year"] < year].copy()
    test_df = feature_df[feature_df["year"] == year].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        return None, None

    scoring = AFLScoringModel(ensemble_weights=ew)
    scoring.train_backtest(train_df, feature_cols)

    disp = AFLDisposalModel(
        distribution=config.DISPOSAL_DISTRIBUTION,
        ensemble_weights=ew,
    )
    disp.train_backtest(train_df, feature_cols)

    marks = AFLMarksModel(
        distribution=getattr(config, "MARKS_DISTRIBUTION", "gaussian"),
        ensemble_weights=ew,
    )
    marks.train_backtest(train_df, feature_cols)

    models = {"scoring": scoring, "disposal": disp, "marks": marks}
    return models, test_df


def predict_and_merge(models, test_df, feature_cols):
    """Run predict_distributions on all 3 models, merge with actuals."""
    scoring = models["scoring"]
    disp = models["disposal"]
    marks = models["marks"]

    goal_preds = scoring.predict_distributions(test_df, store=None, feature_cols=feature_cols)
    disp_preds = disp.predict_distributions(test_df, store=None, feature_cols=feature_cols)
    mark_preds = marks.predict_distributions(test_df, store=None, feature_cols=feature_cols)

    join_cols = ["player", "team", "match_id"]
    result = test_df[["player", "team", "match_id", "GL", "DI", "MK"]].copy()
    result = result.rename(columns={"GL": "actual_goals", "DI": "actual_disposals", "MK": "actual_marks"})

    g_cols = [c for c in goal_preds.columns if c.startswith("p_") or c.startswith("predicted_") or c.startswith("lambda_")]
    g_merge = goal_preds[join_cols + [c for c in g_cols if c in goal_preds.columns]]
    result = result.merge(g_merge, on=join_cols, how="left")

    d_cols = [c for c in disp_preds.columns if c.startswith("p_") or c == "predicted_disposals" or c == "lambda_disposals"]
    d_merge = disp_preds[join_cols + [c for c in d_cols if c in disp_preds.columns]]
    result = result.merge(d_merge, on=join_cols, how="left", suffixes=("", "_d"))

    m_cols = [c for c in mark_preds.columns if c.startswith("p_") or c == "predicted_marks" or c == "lambda_marks"]
    m_merge = mark_preds[join_cols + [c for c in m_cols if c in mark_preds.columns]]
    result = result.merge(m_merge, on=join_cols, how="left", suffixes=("", "_m"))

    return result


# ---------------------------------------------------------------------------
# Step 2A-C: Per-Target Blend Ratio Sweeps
# ---------------------------------------------------------------------------

def sweep_blend_ratios(feature_df, feature_cols):
    """Sweep blend ratios independently for goals, disposals, marks.

    Train once per year, then override ensemble_weights per target before predict.
    """
    print("\n" + "=" * 72)
    print("  DIM A-C: PER-TARGET BLEND RATIO SWEEPS")
    print("=" * 72)

    results = {"goals": {}, "disposals": {}, "marks": {}}

    for year in YEARS:
        t0 = time.time()
        # Train with default weights
        models, test_df = train_models_for_year(feature_df, year, feature_cols)
        if models is None:
            continue
        train_time = time.time() - t0
        print(f"\n  {year}: trained in {train_time:.1f}s, {len(test_df)} test rows")

        # -- Dim A: Goals blend --
        for ratio in BLEND_RATIOS:
            label = ratio_label(ratio)
            # Override scoring model weights, keep others at default
            models["scoring"].ensemble_weights = ratio
            models["disposal"].ensemble_weights = config.ENSEMBLE_WEIGHTS.copy()
            models["marks"].ensemble_weights = config.ENSEMBLE_WEIGHTS.copy()
            merged = predict_and_merge(models, test_df, feature_cols)
            metrics = compute_full_metrics(merged)
            results["goals"].setdefault(label, {})[year] = metrics
            g1_bss = metrics.get("1plus_goals", {}).get("bss", "N/A")
            print(f"    Goals {label}: 1+ BSS={g1_bss}")

        # -- Dim B: Disposals blend --
        for ratio in BLEND_RATIOS:
            label = ratio_label(ratio)
            models["scoring"].ensemble_weights = config.ENSEMBLE_WEIGHTS.copy()
            models["disposal"].ensemble_weights = ratio
            models["marks"].ensemble_weights = config.ENSEMBLE_WEIGHTS.copy()
            merged = predict_and_merge(models, test_df, feature_cols)
            metrics = compute_full_metrics(merged)
            results["disposals"].setdefault(label, {})[year] = metrics

        # -- Dim C: Marks blend --
        for ratio in BLEND_RATIOS:
            label = ratio_label(ratio)
            models["scoring"].ensemble_weights = config.ENSEMBLE_WEIGHTS.copy()
            models["disposal"].ensemble_weights = config.ENSEMBLE_WEIGHTS.copy()
            models["marks"].ensemble_weights = ratio
            merged = predict_and_merge(models, test_df, feature_cols)
            metrics = compute_full_metrics(merged)
            results["marks"].setdefault(label, {})[year] = metrics

        # Reset all to default
        for m in models.values():
            m.ensemble_weights = config.ENSEMBLE_WEIGHTS.copy()

    # Print summary
    for target in ["goals", "disposals", "marks"]:
        print(f"\n  --- {target.upper()} blend summary ---")
        # Determine key metric for this target
        if target == "goals":
            key_metrics = ["1plus_goals", "2plus_goals", "3plus_goals"]
        elif target == "disposals":
            key_metrics = ["15plus_disp", "20plus_disp", "25plus_disp", "30plus_disp"]
        else:
            key_metrics = ["3plus_mk", "5plus_mk", "7plus_mk"]

        for km in key_metrics:
            print(f"  {km}:")
            for label in [ratio_label(r) for r in BLEND_RATIOS]:
                vals = [results[target][label].get(y, {}).get(km, {}).get("brier", np.nan) for y in YEARS]
                avg = np.nanmean(vals)
                print(f"    {label}: avg Brier={avg:.6f}  per-year={[round(v, 4) if not np.isnan(v) else 'N/A' for v in vals]}")

    return results


# ---------------------------------------------------------------------------
# Step 2D: Scorer Exponent + Mark-Taker Parameters
# ---------------------------------------------------------------------------

def sweep_scorer_and_marks_params(feature_df, feature_cols):
    """Sweep scorer exponent (goals) and mark-taker blend/threshold (marks)."""
    print("\n" + "=" * 72)
    print("  DIM D: SCORER EXPONENT + MARK-TAKER PARAMS")
    print("=" * 72)

    scorer_alphas = [0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    mt_blends = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    mt_thresholds = [3, 4, 5, 6, 7]

    results = {"scorer_alpha": {}, "mt_blend": {}, "mt_threshold": {}}

    for year in YEARS:
        models, test_df = train_models_for_year(feature_df, year, feature_cols)
        if models is None:
            continue
        print(f"\n  {year}:")

        scoring = models["scoring"]
        marks_model = models["marks"]

        # Scorer exponent sweep: modify scorer_prob before computing predictions
        fc = feature_cols
        X_raw, X_clean, X_scaled = _prepare_features(test_df, fc, scaler=scoring.scaler)

        from scipy.stats import poisson as poisson_dist

        # Get raw scorer components once
        _, scorer_prob_base, lambda_if_scorer = scoring._ensemble_predict(
            X_raw, X_scaled, "goals", df=test_df, X_clean=X_clean
        )

        for alpha in scorer_alphas:
            # Apply exponent
            scorer_prob_mod = np.clip(scorer_prob_base ** alpha, 0, 1)
            pred_goals_mod = scorer_prob_mod * lambda_if_scorer

            # Build merged with modified goals only
            merged = predict_and_merge(models, test_df, feature_cols)
            # Override goal predictions with alpha-modified versions (vectorized)
            merged["predicted_goals"] = np.round(pred_goals_mod, 4)

            sp = np.clip(scorer_prob_mod, 0, 1)
            raw_mean = np.maximum(lambda_if_scorer, 1.0)
            mu = np.maximum(raw_mean - 1.0, 0.001)
            p0 = 1.0 - sp
            p1 = sp * poisson_dist.pmf(0, mu)
            p2 = sp * poisson_dist.pmf(1, mu)
            merged["p_1plus_goals"] = np.round(1.0 - p0, 4)
            merged["p_2plus_goals"] = np.round(np.maximum(1.0 - p0 - p1, 0), 4)
            merged["p_3plus_goals"] = np.round(np.maximum(1.0 - p0 - p1 - p2, 0), 4)

            metrics = compute_full_metrics(merged)
            results["scorer_alpha"].setdefault(alpha, {})[year] = metrics
            g1 = metrics.get("1plus_goals", {}).get("bss", "N/A")
            print(f"    scorer_alpha={alpha}: 1+ BSS={g1}")

        # Mark-taker blend sweep (prediction-time only)
        for blend_val in mt_blends:
            marks_model._mark_taker_blend = blend_val
            merged = predict_and_merge(models, test_df, feature_cols)
            metrics = compute_full_metrics(merged)
            results["mt_blend"].setdefault(blend_val, {})[year] = metrics
        # Restore default
        marks_model._mark_taker_blend = getattr(config, "MARKS_TAKER_BLEND", 0.3)

        # Mark-taker threshold sweep (requires retraining classifier)
        train_df = feature_df[feature_df["year"] < year].copy()
        for thresh in mt_thresholds:
            marks_retrain = AFLMarksModel(
                distribution=getattr(config, "MARKS_DISTRIBUTION", "gaussian"),
            )
            marks_retrain._mark_taker_threshold = thresh
            marks_retrain.train_backtest(train_df, feature_cols)
            models_copy = {**models, "marks": marks_retrain}
            merged = predict_and_merge(models_copy, test_df, feature_cols)
            metrics = compute_full_metrics(merged)
            results["mt_threshold"].setdefault(thresh, {})[year] = metrics
            m5 = metrics.get("5plus_mk", {}).get("bss", "N/A")
            print(f"    mt_threshold={thresh}: 5+ BSS={m5}")

    # Print summary
    print("\n  --- Scorer alpha summary ---")
    for alpha in scorer_alphas:
        vals = [results["scorer_alpha"].get(alpha, {}).get(y, {}).get("1plus_goals", {}).get("brier", np.nan) for y in YEARS]
        print(f"    alpha={alpha}: avg 1+ Brier={np.nanmean(vals):.6f}")

    print("\n  --- Mark-taker blend summary ---")
    for blend_val in mt_blends:
        vals = [results["mt_blend"].get(blend_val, {}).get(y, {}).get("5plus_mk", {}).get("brier", np.nan) for y in YEARS]
        print(f"    blend={blend_val}: avg 5+ Brier={np.nanmean(vals):.6f}")

    print("\n  --- Mark-taker threshold summary ---")
    for thresh in mt_thresholds:
        vals = [results["mt_threshold"].get(thresh, {}).get(y, {}).get("5plus_mk", {}).get("brier", np.nan) for y in YEARS]
        print(f"    threshold={thresh}: avg 5+ Brier={np.nanmean(vals):.6f}")

    return results


# ---------------------------------------------------------------------------
# Step 2E: Era Weight Schemes (requires retraining)
# ---------------------------------------------------------------------------

def sweep_era_weights(feature_df, feature_cols):
    """Sweep era weight schemes — requires retraining for each scheme."""
    print("\n" + "=" * 72)
    print("  DIM E: ERA WEIGHT SCHEMES")
    print("=" * 72)

    results = {}

    for scheme_name, era_weights in ERA_WEIGHT_CONFIGS.items():
        results[scheme_name] = {}
        print(f"\n  Scheme: {scheme_name}")

        for year in YEARS:
            train_df = feature_df[feature_df["year"] < year].copy()
            test_df = feature_df[feature_df["year"] == year].copy()
            if len(train_df) == 0 or len(test_df) == 0:
                continue

            # Recompute sample weights with this era config
            weights = compute_sample_weights(train_df, era_weights)
            train_df["sample_weight"] = weights

            # Recombine for training
            combined = pd.concat([train_df, test_df], ignore_index=True)
            res = fast_train_predict(combined, year, feature_cols)
            if res is None:
                continue
            merged, _ = res
            metrics = compute_full_metrics(merged)
            results[scheme_name][year] = metrics

            g1 = metrics.get("1plus_goals", {}).get("bss", "N/A")
            d20 = metrics.get("20plus_disp", {}).get("bss", "N/A")
            print(f"    {year}: 1+ BSS={g1}  20+ BSS={d20}")

    # Print summary
    print("\n  --- Era weight summary ---")
    for scheme_name in ERA_WEIGHT_CONFIGS:
        g1_vals = [results[scheme_name].get(y, {}).get("1plus_goals", {}).get("brier", np.nan) for y in YEARS]
        d20_vals = [results[scheme_name].get(y, {}).get("20plus_disp", {}).get("brier", np.nan) for y in YEARS]
        print(f"    {scheme_name:15s}: avg GL1+ Brier={np.nanmean(g1_vals):.6f}  avg DI20+ Brier={np.nanmean(d20_vals):.6f}")

    return results


# ---------------------------------------------------------------------------
# Step 2F: Elo Parameters + Hybrid Params
# ---------------------------------------------------------------------------

def sweep_elo_and_hybrid(feature_df, feature_cols):
    """Sweep Elo grid and hybrid winner params."""
    print("\n" + "=" * 72)
    print("  DIM F: ELO PARAMETERS + HYBRID PARAMS")
    print("=" * 72)

    # Load team_matches for game winner model
    tm_path = config.BASE_STORE_DIR / "team_matches.parquet"
    if not tm_path.exists():
        print("  WARNING: team_matches.parquet not found, skipping Elo sweep")
        return {"elo": {}, "hybrid": {}}
    team_match_df = pd.read_parquet(tm_path)

    k_factors = [8, 12, 20, 30]
    home_advs = [10, 16, 22, 30]
    season_regs = [0.2, 0.3, 0.4, 0.5]

    elo_results = {}
    print("\n  Elo grid sweep...")

    for year in YEARS:
        train_tm = team_match_df[team_match_df["year"] < year].copy()
        test_tm = team_match_df[team_match_df["year"] == year].copy()
        if len(train_tm) == 0 or len(test_tm) == 0:
            continue

        for k in k_factors:
            for ha in home_advs:
                for sr in season_regs:
                    combo_key = f"k={k}_ha={ha}_sr={sr}"
                    elo_params = {
                        "k_factor": k,
                        "home_advantage": ha,
                        "season_regression": sr,
                    }
                    gw = AFLGameWinnerModel(elo_params=elo_params)
                    try:
                        gw.train_backtest(train_tm)
                        # Predict on test year
                        full_tm = pd.concat([train_tm, test_tm], ignore_index=True)
                        preds = gw.predict_with_margin(full_tm)
                        # Filter to test year matches
                        test_match_ids = set(test_tm["match_id"].unique())
                        preds = preds[preds["match_id"].isin(test_match_ids)]

                        if len(preds) == 0:
                            continue

                        # Compute accuracy and Brier
                        test_home = test_tm[test_tm["is_home"]].copy()
                        test_home["home_win_actual"] = (test_home["margin"] > 0).astype(int)
                        eval_df = preds.merge(
                            test_home[["match_id", "home_win_actual"]],
                            on="match_id", how="inner"
                        )
                        if len(eval_df) == 0:
                            continue

                        acc = float((eval_df["home_win_prob"] > 0.5).astype(int).eq(eval_df["home_win_actual"]).mean())
                        brier_val = float(brier(eval_df["home_win_prob"].values, eval_df["home_win_actual"].values))
                        try:
                            auc_val = float(roc_auc_score(eval_df["home_win_actual"], eval_df["home_win_prob"]))
                        except Exception:
                            auc_val = np.nan

                        elo_results.setdefault(combo_key, {})[year] = {
                            "accuracy": round(acc, 4),
                            "brier": round(brier_val, 6),
                            "auc": round(auc_val, 4) if not np.isnan(auc_val) else None,
                            "n_games": len(eval_df),
                        }
                    except Exception as e:
                        pass  # Some combos may fail

        print(f"  {year}: tested {len(k_factors)*len(home_advs)*len(season_regs)} Elo combos")

    # Hybrid params sweep (prediction-time only, uses default Elo)
    hybrid_results = {}
    alphas = [0.05, 0.1, 0.15, 0.2, 0.3]
    betas = [0.5, 0.6, 0.7, 0.8, 0.9]
    biases = [-0.05, -0.02, 0.0, 0.02, 0.05]

    print("\n  Hybrid params sweep...")
    for year in YEARS:
        train_tm = team_match_df[team_match_df["year"] < year].copy()
        test_tm = team_match_df[team_match_df["year"] == year].copy()
        if len(train_tm) == 0 or len(test_tm) == 0:
            continue

        # Train once with default Elo
        gw = AFLGameWinnerModel()
        gw.train_backtest(train_tm)
        full_tm = pd.concat([train_tm, test_tm], ignore_index=True)

        test_match_ids = set(test_tm["match_id"].unique())
        test_home = test_tm[test_tm["is_home"]].copy()
        test_home["home_win_actual"] = (test_home["margin"] > 0).astype(int)

        for a in alphas:
            for b in betas:
                for bias in biases:
                    combo_key = f"a={a}_b={b}_bias={bias}"
                    # Override hybrid params at prediction time
                    gw.hybrid_alpha = a
                    gw.hybrid_beta = b
                    gw.hybrid_bias = bias
                    gw.hybrid_enabled = True

                    try:
                        preds = gw.predict_with_margin(full_tm)
                        preds = preds[preds["match_id"].isin(test_match_ids)]
                        eval_df = preds.merge(
                            test_home[["match_id", "home_win_actual"]],
                            on="match_id", how="inner"
                        )
                        if len(eval_df) == 0:
                            continue

                        acc = float((eval_df["home_win_prob"] > 0.5).astype(int).eq(eval_df["home_win_actual"]).mean())
                        brier_val = float(brier(eval_df["home_win_prob"].values, eval_df["home_win_actual"].values))

                        hybrid_results.setdefault(combo_key, {})[year] = {
                            "accuracy": round(acc, 4),
                            "brier": round(brier_val, 6),
                        }
                    except Exception:
                        pass

        print(f"  {year}: tested {len(alphas)*len(betas)*len(biases)} hybrid combos")

    # Find best Elo combo
    if elo_results:
        best_elo = min(elo_results, key=lambda k: np.nanmean(
            [elo_results[k].get(y, {}).get("brier", np.nan) for y in YEARS]
        ))
        best_elo_brier = np.nanmean([elo_results[best_elo].get(y, {}).get("brier", np.nan) for y in YEARS])
        print(f"\n  Best Elo combo: {best_elo} (avg Brier={best_elo_brier:.6f})")

    if hybrid_results:
        best_hybrid = min(hybrid_results, key=lambda k: np.nanmean(
            [hybrid_results[k].get(y, {}).get("brier", np.nan) for y in YEARS]
        ))
        best_hybrid_brier = np.nanmean([hybrid_results[best_hybrid].get(y, {}).get("brier", np.nan) for y in YEARS])
        print(f"  Best hybrid combo: {best_hybrid} (avg Brier={best_hybrid_brier:.6f})")

    return {"elo": elo_results, "hybrid": hybrid_results}


# ---------------------------------------------------------------------------
# Step 2G: Gaussian Disposal Tail Parameters
# ---------------------------------------------------------------------------

def sweep_gaussian_tail(feature_df, feature_cols):
    """Sweep Gaussian disposal tail params (prediction-time only)."""
    print("\n" + "=" * 72)
    print("  DIM G: GAUSSIAN DISPOSAL TAIL PARAMETERS")
    print("=" * 72)

    std_multipliers = [0.8, 1.0, 1.2, 1.4, 1.6]
    skew_alphas = [0.0, 1.0, 2.0, 3.0]
    scale_30s = [1.0, 1.1, 1.2, 1.3, 1.5]
    cap_30s = [0.30, 0.35, 0.40, 0.45, 0.50]

    results_grid1 = {}  # std_mult × skew_alpha
    results_grid2 = {}  # scale_30 × cap_30

    for year in YEARS:
        models, test_df = train_models_for_year(feature_df, year, feature_cols)
        if models is None:
            continue
        print(f"\n  {year}:")

        # Sub-grid 1: std_multiplier × skew_alpha
        for sm in std_multipliers:
            for sa in skew_alphas:
                combo = f"sm={sm}_sa={sa}"
                with config_override(
                    DISPOSAL_UPPER_TAIL_STD_MULTIPLIER=sm,
                    DISPOSAL_UPPER_TAIL_SKEW_ALPHA=sa,
                ):
                    merged = predict_and_merge(models, test_df, feature_cols)
                    metrics = compute_full_metrics(merged)
                    results_grid1.setdefault(combo, {})[year] = metrics

        # Sub-grid 2: scale_30 × cap_30
        for sc in scale_30s:
            for cap in cap_30s:
                combo = f"sc={sc}_cap={cap}"
                with config_override(
                    DISPOSAL_30PLUS_PROB_SCALE=sc,
                    DISPOSAL_30PLUS_PROB_CAP=cap,
                ):
                    merged = predict_and_merge(models, test_df, feature_cols)
                    metrics = compute_full_metrics(merged)
                    results_grid2.setdefault(combo, {})[year] = metrics

        print(f"    Tested {len(std_multipliers)*len(skew_alphas)} + {len(scale_30s)*len(cap_30s)} combos")

    # Print summary for grid 1 (affects 25+ and 30+)
    print("\n  --- Grid 1: std_mult × skew_alpha (25+/30+ impact) ---")
    for km in ["25plus_disp", "30plus_disp"]:
        print(f"  {km}:")
        best_combo = None
        best_val = np.inf
        for combo in sorted(results_grid1):
            vals = [results_grid1[combo].get(y, {}).get(km, {}).get("brier", np.nan) for y in YEARS]
            avg = np.nanmean(vals)
            if avg < best_val:
                best_val = avg
                best_combo = combo
            print(f"    {combo:20s}: avg Brier={avg:.6f}")
        print(f"    BEST: {best_combo} ({best_val:.6f})")

    # Print summary for grid 2 (affects 30+ only)
    print("\n  --- Grid 2: scale_30 × cap_30 (30+ impact) ---")
    best_combo = None
    best_val = np.inf
    for combo in sorted(results_grid2):
        vals = [results_grid2[combo].get(y, {}).get("30plus_disp", {}).get("brier", np.nan) for y in YEARS]
        avg = np.nanmean(vals)
        if avg < best_val:
            best_val = avg
            best_combo = combo
        print(f"    {combo:20s}: avg 30+ Brier={avg:.6f}")
    print(f"    BEST: {best_combo} ({best_val:.6f})")

    return {"grid1": results_grid1, "grid2": results_grid2}


# ---------------------------------------------------------------------------
# Step 3: Analysis — find best per dimension
# ---------------------------------------------------------------------------

def analyze_dimension(dim_results, key_metric, label=""):
    """Rank all variations by average Brier across years for key_metric.

    Returns dict with best_value, impact, stability info.
    """
    rankings = {}
    for variation, year_data in dim_results.items():
        brier_vals = []
        for y in YEARS:
            m = year_data.get(y, {})
            if isinstance(m, dict) and key_metric in m:
                val = m[key_metric]
                if isinstance(val, dict):
                    brier_vals.append(val.get("brier", np.nan))
                else:
                    brier_vals.append(val)
            else:
                brier_vals.append(np.nan)
        avg = np.nanmean(brier_vals)
        std = np.nanstd(brier_vals)
        rankings[variation] = {
            "avg_brier": round(avg, 6),
            "std_brier": round(std, 6),
            "per_year": {y: round(v, 6) if not np.isnan(v) else None for y, v in zip(YEARS, brier_vals)},
        }

    if not rankings:
        return {"best": None, "impact": 0, "rankings": {}}

    sorted_rankings = sorted(rankings.items(), key=lambda x: x[1]["avg_brier"])
    best = sorted_rankings[0][0]
    worst = sorted_rankings[-1][0]
    impact = rankings[worst]["avg_brier"] - rankings[best]["avg_brier"]

    return {
        "best": best,
        "best_avg_brier": rankings[best]["avg_brier"],
        "worst": worst,
        "worst_avg_brier": rankings[worst]["avg_brier"],
        "impact": round(impact, 6),
        "stability_std": rankings[best]["std_brier"],
        "rankings": rankings,
    }


# ---------------------------------------------------------------------------
# Step 4-5: Combined Config Validation + Interaction Testing
# ---------------------------------------------------------------------------

def build_best_config(analysis_results):
    """Extract best values from each dimension analysis into a combined config."""
    best = {}

    # Dim A: best goals blend
    if "dim_a" in analysis_results:
        best["goals_blend"] = analysis_results["dim_a"].get("best")

    # Dim B: best disposals blend
    if "dim_b" in analysis_results:
        best["disp_blend"] = analysis_results["dim_b"].get("best")

    # Dim C: best marks blend
    if "dim_c" in analysis_results:
        best["marks_blend"] = analysis_results["dim_c"].get("best")

    # Dim D: best scorer alpha, mt_blend, mt_threshold
    if "dim_d_alpha" in analysis_results:
        best["scorer_alpha"] = analysis_results["dim_d_alpha"].get("best")
    if "dim_d_mt_blend" in analysis_results:
        best["mt_blend"] = analysis_results["dim_d_mt_blend"].get("best")
    if "dim_d_mt_threshold" in analysis_results:
        best["mt_threshold"] = analysis_results["dim_d_mt_threshold"].get("best")

    # Dim E: best era scheme
    if "dim_e" in analysis_results:
        best["era_scheme"] = analysis_results["dim_e"].get("best")

    # Dim F: best elo, best hybrid
    if "dim_f_elo" in analysis_results:
        best["elo_params"] = analysis_results["dim_f_elo"].get("best")
    if "dim_f_hybrid" in analysis_results:
        best["hybrid_params"] = analysis_results["dim_f_hybrid"].get("best")

    # Dim G: best gaussian params
    if "dim_g_grid1" in analysis_results:
        best["gaussian_grid1"] = analysis_results["dim_g_grid1"].get("best")
    if "dim_g_grid2" in analysis_results:
        best["gaussian_grid2"] = analysis_results["dim_g_grid2"].get("best")

    return best


def validate_combined(feature_df, feature_cols, best_config, default_metrics):
    """Run combined best config vs default on 2023-2025."""
    print("\n" + "=" * 72)
    print("  STEP 4: COMBINED CONFIG VALIDATION")
    print("=" * 72)

    val_years = [2023, 2024, 2025]
    combined_results = {}

    for year in val_years:
        # Parse best blend ratios
        goals_ew = config.ENSEMBLE_WEIGHTS.copy()
        disp_ew = config.ENSEMBLE_WEIGHTS.copy()
        marks_ew = config.ENSEMBLE_WEIGHTS.copy()

        if best_config.get("goals_blend"):
            parts = best_config["goals_blend"].split("/")
            goals_ew = {"poisson": int(parts[0]) / 100, "gbt": int(parts[1]) / 100}
        if best_config.get("disp_blend"):
            parts = best_config["disp_blend"].split("/")
            disp_ew = {"poisson": int(parts[0]) / 100, "gbt": int(parts[1]) / 100}
        if best_config.get("marks_blend"):
            parts = best_config["marks_blend"].split("/")
            marks_ew = {"poisson": int(parts[0]) / 100, "gbt": int(parts[1]) / 100}

        # Apply era weights if best differs from current
        era_name = best_config.get("era_scheme", "current")
        era_weights = ERA_WEIGHT_CONFIGS.get(era_name, ERA_WEIGHT_CONFIGS["current"])
        train_df = feature_df[feature_df["year"] < year].copy()
        test_df = feature_df[feature_df["year"] == year].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        weights = compute_sample_weights(train_df, era_weights)
        train_df["sample_weight"] = weights
        combined_df = pd.concat([train_df, test_df], ignore_index=True)

        # Apply Gaussian tail overrides
        overrides = {}
        g1 = best_config.get("gaussian_grid1", "")
        if g1 and "sm=" in str(g1):
            parts = str(g1).split("_")
            for p in parts:
                if p.startswith("sm="):
                    overrides["DISPOSAL_UPPER_TAIL_STD_MULTIPLIER"] = float(p.split("=")[1])
                elif p.startswith("sa="):
                    overrides["DISPOSAL_UPPER_TAIL_SKEW_ALPHA"] = float(p.split("=")[1])

        g2 = best_config.get("gaussian_grid2", "")
        if g2 and "sc=" in str(g2):
            parts = str(g2).split("_")
            for p in parts:
                if p.startswith("sc="):
                    overrides["DISPOSAL_30PLUS_PROB_SCALE"] = float(p.split("=")[1])
                elif p.startswith("cap="):
                    overrides["DISPOSAL_30PLUS_PROB_CAP"] = float(p.split("=")[1])

        with config_override(**overrides) if overrides else nullcontext():
            # Train with specific blend ratios per model
            scoring = AFLScoringModel(ensemble_weights=goals_ew)
            scoring.train_backtest(train_df, feature_cols)

            disp = AFLDisposalModel(
                distribution=config.DISPOSAL_DISTRIBUTION,
                ensemble_weights=disp_ew,
            )
            disp.train_backtest(train_df, feature_cols)

            # Apply mark-taker threshold/blend if specified
            mt_thresh = best_config.get("mt_threshold", config.MARKS_TAKER_THRESHOLD)
            mt_blend = best_config.get("mt_blend", config.MARKS_TAKER_BLEND)
            marks_model = AFLMarksModel(
                distribution=getattr(config, "MARKS_DISTRIBUTION", "gaussian"),
                ensemble_weights=marks_ew,
            )
            marks_model._mark_taker_threshold = mt_thresh if mt_thresh else config.MARKS_TAKER_THRESHOLD
            marks_model._mark_taker_blend = mt_blend if mt_blend is not None else config.MARKS_TAKER_BLEND
            marks_model.train_backtest(train_df, feature_cols)

            models = {"scoring": scoring, "disposal": disp, "marks": marks_model}
            merged = predict_and_merge(models, test_df, feature_cols)

        metrics = compute_full_metrics(merged)
        combined_results[year] = metrics
        print(f"  {year}: combined config metrics computed")

    # Compare combined vs default
    print("\n  --- Combined vs Default ---")
    all_targets = [
        "1plus_goals", "2plus_goals", "3plus_goals",
        "15plus_disp", "20plus_disp", "25plus_disp", "30plus_disp",
        "3plus_mk", "5plus_mk", "7plus_mk",
    ]
    print(f"  {'Target':15s}", end="")
    for year in val_years:
        print(f"  {year} def   {year} best  delta ", end="")
    print()
    for tgt in all_targets:
        print(f"  {tgt:15s}", end="")
        for year in val_years:
            d = default_metrics.get(year, {}).get(tgt, {})
            b = combined_results.get(year, {}).get(tgt, {})
            d_val = d.get("brier", np.nan) if isinstance(d, dict) else np.nan
            b_val = b.get("brier", np.nan) if isinstance(b, dict) else np.nan
            delta = b_val - d_val if not np.isnan(b_val) and not np.isnan(d_val) else np.nan
            d_s = f"{d_val:.4f}" if not np.isnan(d_val) else "  N/A "
            b_s = f"{b_val:.4f}" if not np.isnan(b_val) else "  N/A "
            delta_s = f"{delta:+.4f}" if not np.isnan(delta) else "  N/A "
            print(f"  {d_s}  {b_s} {delta_s}", end="")
        print()

    return combined_results


def interaction_test(feature_df, feature_cols, best_config, default_metrics):
    """Test combined config vs each dimension applied independently."""
    print("\n" + "=" * 72)
    print("  STEP 5: INTERACTION TESTING")
    print("=" * 72)

    val_years = [2023, 2024, 2025]
    configs_to_test = {}

    # Individual best configs
    if best_config.get("goals_blend"):
        configs_to_test["best_goals_blend_only"] = {"goals_blend": best_config["goals_blend"]}
    if best_config.get("disp_blend"):
        configs_to_test["best_disp_blend_only"] = {"disp_blend": best_config["disp_blend"]}
    if best_config.get("era_scheme") and best_config["era_scheme"] != "current":
        configs_to_test["best_era_only"] = {"era_scheme": best_config["era_scheme"]}
    if best_config.get("gaussian_grid1"):
        configs_to_test["best_gaussian_only"] = {"gaussian_grid1": best_config["gaussian_grid1"]}

    interaction_results = {}

    for config_name, partial_config in configs_to_test.items():
        interaction_results[config_name] = {}
        for year in val_years:
            train_df = feature_df[feature_df["year"] < year].copy()
            test_df = feature_df[feature_df["year"] == year].copy()
            if len(train_df) == 0 or len(test_df) == 0:
                continue

            # Apply only the partial config dimensions
            goals_ew = config.ENSEMBLE_WEIGHTS.copy()
            disp_ew = config.ENSEMBLE_WEIGHTS.copy()
            marks_ew = config.ENSEMBLE_WEIGHTS.copy()

            if "goals_blend" in partial_config:
                parts = partial_config["goals_blend"].split("/")
                goals_ew = {"poisson": int(parts[0]) / 100, "gbt": int(parts[1]) / 100}
            if "disp_blend" in partial_config:
                parts = partial_config["disp_blend"].split("/")
                disp_ew = {"poisson": int(parts[0]) / 100, "gbt": int(parts[1]) / 100}

            era_name = partial_config.get("era_scheme", "current")
            era_w = ERA_WEIGHT_CONFIGS.get(era_name, ERA_WEIGHT_CONFIGS["current"])
            weights = compute_sample_weights(train_df, era_w)
            train_df["sample_weight"] = weights

            overrides = {}
            g1 = partial_config.get("gaussian_grid1", "")
            if g1 and "sm=" in str(g1):
                parts_g = str(g1).split("_")
                for p in parts_g:
                    if p.startswith("sm="):
                        overrides["DISPOSAL_UPPER_TAIL_STD_MULTIPLIER"] = float(p.split("=")[1])
                    elif p.startswith("sa="):
                        overrides["DISPOSAL_UPPER_TAIL_SKEW_ALPHA"] = float(p.split("=")[1])

            with config_override(**overrides) if overrides else nullcontext():
                scoring = AFLScoringModel(ensemble_weights=goals_ew)
                scoring.train_backtest(train_df, feature_cols)
                disp = AFLDisposalModel(distribution=config.DISPOSAL_DISTRIBUTION, ensemble_weights=disp_ew)
                disp.train_backtest(train_df, feature_cols)
                marks_model = AFLMarksModel(distribution=getattr(config, "MARKS_DISTRIBUTION", "gaussian"), ensemble_weights=marks_ew)
                marks_model.train_backtest(train_df, feature_cols)
                models = {"scoring": scoring, "disposal": disp, "marks": marks_model}
                merged = predict_and_merge(models, test_df, feature_cols)

            metrics = compute_full_metrics(merged)
            interaction_results[config_name][year] = metrics

        print(f"  {config_name}: computed")

    return interaction_results


# ---------------------------------------------------------------------------
# Step 6: Sequential Validation (shell out to pipeline.py)
# ---------------------------------------------------------------------------

def run_sequential_validation(best_config):
    """Run full sequential validation for top config."""
    print("\n" + "=" * 72)
    print("  STEP 6: FULL SEQUENTIAL VALIDATION")
    print("  (This step must be run manually via pipeline.py --sequential)")
    print("=" * 72)
    print("\n  Recommended command:")
    print("    python pipeline.py --sequential --year 2025 --reset-calibration")
    print("\n  Apply these config changes before running:")
    for key, val in best_config.items():
        print(f"    {key}: {val}")
    print("\n  Compare Brier scores against default sequential runs.")
    return {"note": "Manual step — run pipeline.py --sequential with best config"}


# ---------------------------------------------------------------------------
# Step 7: Report & Save
# ---------------------------------------------------------------------------

def print_full_report(all_results, analysis, best_config, combined_results,
                      interaction_results, default_metrics):
    """Print comprehensive report."""
    print("\n")
    print("=" * 80)
    print("  WEIGHT OPTIMIZATION — FULL REPORT")
    print("=" * 80)

    # Table 1: Best value per dimension with Brier improvement
    print("\n--- TABLE 1: Best Value Per Dimension ---")
    print(f"  {'Dimension':25s} {'Best Value':25s} {'Avg Brier':>10s} {'Impact':>10s} {'Stability':>10s}")
    for dim_name, dim_analysis in sorted(analysis.items()):
        if dim_analysis.get("best") is None:
            continue
        print(f"  {dim_name:25s} {str(dim_analysis['best']):25s} "
              f"{dim_analysis.get('best_avg_brier', 0):10.6f} "
              f"{dim_analysis.get('impact', 0):10.6f} "
              f"{dim_analysis.get('stability_std', 0):10.6f}")

    # Table 2: Impact ranking
    print("\n--- TABLE 2: Impact Ranking (which dimensions matter most) ---")
    ranked = sorted(analysis.items(), key=lambda x: -x[1].get("impact", 0))
    for i, (dim_name, dim_analysis) in enumerate(ranked, 1):
        if dim_analysis.get("impact", 0) == 0:
            continue
        print(f"  {i}. {dim_name:25s} impact={dim_analysis.get('impact', 0):.6f}")

    # Table 3: Cross-year stability
    print("\n--- TABLE 3: Cross-Year Stability ---")
    for dim_name, dim_analysis in sorted(analysis.items()):
        best_var = dim_analysis.get("best")
        if best_var is None:
            continue
        rankings = dim_analysis.get("rankings", {})
        if best_var in rankings:
            per_year = rankings[best_var].get("per_year", {})
            vals_str = "  ".join(f"{y}={v}" for y, v in per_year.items() if v is not None)
            print(f"  {dim_name:25s}: {vals_str}")

    # Table 4: Combined config vs default
    if combined_results and default_metrics:
        print("\n--- TABLE 4: Combined Config vs Default ---")
        val_years = [2023, 2024, 2025]
        all_targets = [
            "1plus_goals", "2plus_goals", "3plus_goals",
            "15plus_disp", "20plus_disp", "25plus_disp", "30plus_disp",
            "3plus_mk", "5plus_mk", "7plus_mk",
        ]
        for tgt in all_targets:
            deltas = []
            for year in val_years:
                d = default_metrics.get(year, {}).get(tgt, {})
                b = combined_results.get(year, {}).get(tgt, {})
                d_val = d.get("brier", np.nan) if isinstance(d, dict) else np.nan
                b_val = b.get("brier", np.nan) if isinstance(b, dict) else np.nan
                if not np.isnan(d_val) and not np.isnan(b_val):
                    deltas.append(b_val - d_val)
            if deltas:
                avg_delta = np.mean(deltas)
                direction = "IMPROVED" if avg_delta < 0 else "REGRESSED"
                print(f"  {tgt:15s}: avg delta={avg_delta:+.6f} ({direction})")

    # Table 5: Interaction effects
    if interaction_results:
        print("\n--- TABLE 5: Interaction Effects ---")
        for config_name, year_results in interaction_results.items():
            for tgt in ["1plus_goals", "20plus_disp", "5plus_mk"]:
                vals = [year_results.get(y, {}).get(tgt, {}).get("brier", np.nan)
                        for y in [2023, 2024, 2025]]
                avg = np.nanmean(vals)
                print(f"  {config_name:30s} {tgt:15s}: avg Brier={avg:.6f}")

    # Table 6: Final recommended config
    print("\n--- TABLE 6: Final Recommended Config ---")
    for key, val in best_config.items():
        print(f"  {key:25s}: {val}")


def save_results(all_results, analysis, best_config, combined_results,
                 interaction_results, default_metrics):
    """Save everything to JSON."""
    config.ensure_dirs()

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return str(obj)

    experiment = {
        "label": "weight_optimization_2021_2025",
        "years": YEARS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_config": best_config,
        "analysis": analysis,
        "combined_validation": combined_results,
        "interaction_results": interaction_results,
        "default_metrics": default_metrics,
        "raw_results": {
            k: v for k, v in all_results.items()
            if k not in ("dim_f",)  # Elo results can be huge
        },
    }

    out_path = config.EXPERIMENTS_DIR / "weight_optimization_2021_2025.json"
    with open(out_path, "w") as f:
        json.dump(experiment, f, indent=2, default=convert)

    print(f"\n  Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print("=" * 72)
    print("  AFL WEIGHT OPTIMIZATION SYSTEM")
    print("  Exhaustive multi-season sweep: 2021-2025")
    print("=" * 72)

    # --- Step 1: Load data ---
    print("\nLoading feature matrix...")
    feature_df, feature_cols = load_feature_matrix()
    print(f"  {len(feature_df)} rows, {len(feature_cols)} features")

    # Compute default metrics for comparison
    print("\nComputing default metrics...")
    default_metrics = {}
    for year in YEARS:
        res = fast_train_predict(feature_df, year, feature_cols)
        if res is not None:
            merged, _ = res
            default_metrics[year] = compute_full_metrics(merged)

    all_results = {}
    analysis = {}

    # --- Step 2A-C: Blend ratio sweeps ---
    blend_results = sweep_blend_ratios(feature_df, feature_cols)
    all_results["dim_abc"] = blend_results

    # Analyze each target's blend
    analysis["dim_a"] = analyze_dimension(blend_results["goals"], "1plus_goals", "Goals blend")
    analysis["dim_b"] = analyze_dimension(blend_results["disposals"], "20plus_disp", "Disp blend")
    analysis["dim_c"] = analyze_dimension(blend_results["marks"], "5plus_mk", "Marks blend")

    # --- Step 2D: Scorer + marks params ---
    scorer_marks_results = sweep_scorer_and_marks_params(feature_df, feature_cols)
    all_results["dim_d"] = scorer_marks_results

    analysis["dim_d_alpha"] = analyze_dimension(scorer_marks_results["scorer_alpha"], "1plus_goals", "Scorer alpha")
    analysis["dim_d_mt_blend"] = analyze_dimension(scorer_marks_results["mt_blend"], "5plus_mk", "MT blend")
    analysis["dim_d_mt_threshold"] = analyze_dimension(scorer_marks_results["mt_threshold"], "5plus_mk", "MT threshold")

    # --- Step 2E: Era weights ---
    era_results = sweep_era_weights(feature_df, feature_cols)
    all_results["dim_e"] = era_results

    # Analyze on composite metric (avg of GL1+ and DI20+ Brier)
    analysis["dim_e"] = analyze_dimension(era_results, "1plus_goals", "Era weights")

    # --- Step 2F: Elo + hybrid ---
    elo_hybrid_results = sweep_elo_and_hybrid(feature_df, feature_cols)
    all_results["dim_f"] = elo_hybrid_results

    if elo_hybrid_results["elo"]:
        analysis["dim_f_elo"] = analyze_dimension(elo_hybrid_results["elo"], "brier", "Elo params")
    if elo_hybrid_results["hybrid"]:
        analysis["dim_f_hybrid"] = analyze_dimension(elo_hybrid_results["hybrid"], "brier", "Hybrid params")

    # --- Step 2G: Gaussian tail ---
    gaussian_results = sweep_gaussian_tail(feature_df, feature_cols)
    all_results["dim_g"] = gaussian_results

    analysis["dim_g_grid1"] = analyze_dimension(gaussian_results["grid1"], "25plus_disp", "Gaussian grid1")
    analysis["dim_g_grid2"] = analyze_dimension(gaussian_results["grid2"], "30plus_disp", "Gaussian grid2")

    # --- Step 3: Build best config ---
    best_config = build_best_config(analysis)
    print(f"\n  Best config: {best_config}")

    # --- Step 4: Combined validation ---
    combined_results = validate_combined(feature_df, feature_cols, best_config, default_metrics)

    # --- Step 5: Interaction testing ---
    interaction_results = interaction_test(feature_df, feature_cols, best_config, default_metrics)

    # --- Step 6: Sequential validation (manual) ---
    seq_info = run_sequential_validation(best_config)

    # --- Step 7: Report & Save ---
    print_full_report(all_results, analysis, best_config, combined_results,
                      interaction_results, default_metrics)

    out_path = save_results(all_results, analysis, best_config, combined_results,
                            interaction_results, default_metrics)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 72)


if __name__ == "__main__":
    main()
