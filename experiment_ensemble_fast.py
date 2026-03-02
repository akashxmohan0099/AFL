"""
Multi-Model Ensemble vs Single-Model Comparison (Fast Version)
==============================================================
Skips game winner model (bottleneck) to focus on core scoring + disposal metrics.
Runs 2025 sequential backtest with both:
  A) Single-model baseline (HistGBT only)
  B) Multi-model ensemble (HistGBT + GBT + ExtraTrees, averaged)

Compares: Goals Brier (1+/2+/3+), Disposal Brier (15+/20+/25+/30+),
Scorer AUC, MAE.
"""

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score

import config
from analysis import _compute_threshold_metrics, _extract_threshold_data
from model import AFLDisposalModel, AFLScoringModel

warnings.filterwarnings("ignore")

EXPERIMENT_DIR = config.DATA_DIR / "experiments"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_YEAR = 2025


def load_data():
    """Load feature matrix and feature columns."""
    feature_df = pd.read_parquet("data/features/feature_matrix.parquet")
    with open("data/features/feature_columns.json") as f:
        feature_cols = json.load(f)
    feature_df = feature_df.sort_values(["date", "match_id", "player"]).reset_index(drop=True)
    return feature_df, feature_cols


def run_condition(feature_df, feature_cols, condition_name, use_ensemble, rounds):
    """Run sequential backtest for one condition (scoring + disposal only)."""
    config.MULTI_MODEL_ENSEMBLE = use_ensemble

    results = []
    total_start = time.time()

    for rnd in rounds:
        rnd_start = time.time()

        train_mask = (feature_df["year"] < BACKTEST_YEAR) | (
            (feature_df["year"] == BACKTEST_YEAR) & (feature_df["round_number"] < rnd)
        )
        test_mask = (
            (feature_df["year"] == BACKTEST_YEAR) & (feature_df["round_number"] == rnd)
        )
        train_df = feature_df[train_mask]
        test_df = feature_df[test_mask]

        if len(test_df) == 0 or len(train_df) < 50:
            continue

        # --- Scoring model ---
        scoring_model = AFLScoringModel()
        scoring_model.train_backtest(train_df, feature_cols)
        scoring_preds = scoring_model.predict_distributions(test_df, store=None, feature_cols=feature_cols)

        # --- Disposal model ---
        disposal_model = AFLDisposalModel(distribution=config.DISPOSAL_DISTRIBUTION)
        disposal_model.train_backtest(train_df, feature_cols)
        disposal_preds = disposal_model.predict_distributions(test_df, store=None, feature_cols=feature_cols)

        # --- Scoring metrics ---
        merged_scoring = scoring_preds.copy()
        merged_scoring["actual_goals"] = test_df["GL"].values

        thresholds = _extract_threshold_data(merged_scoring)
        round_result = {
            "round": int(rnd),
            "n_players": len(test_df),
        }

        # Goals MAE
        round_result["goals_mae"] = float(np.mean(np.abs(
            merged_scoring["predicted_goals"].values - merged_scoring["actual_goals"].values
        )))

        # Scorer AUC
        actual_scored = (test_df["GL"].values >= 1).astype(int)
        if "p_scorer" in merged_scoring.columns:
            try:
                round_result["scorer_auc"] = float(roc_auc_score(
                    actual_scored, merged_scoring["p_scorer"].values
                ))
            except ValueError:
                round_result["scorer_auc"] = None
        else:
            round_result["scorer_auc"] = None

        # Goals Brier scores
        for key in ["1plus_goals", "2plus_goals", "3plus_goals"]:
            if key in thresholds:
                m = _compute_threshold_metrics(*thresholds[key])
                round_result[f"brier_{key}"] = m["brier_score"] if m else None
                round_result[f"logloss_{key}"] = m["log_loss"] if m else None
            else:
                round_result[f"brier_{key}"] = None
                round_result[f"logloss_{key}"] = None

        # --- Disposal metrics ---
        merged_disp = disposal_preds.copy()
        merged_disp["actual_disposals"] = test_df["DI"].values

        round_result["disp_mae"] = float(np.mean(np.abs(
            merged_disp["predicted_disposals"].values - merged_disp["actual_disposals"].values
        )))

        actual_disp = merged_disp["actual_disposals"].values
        for t in [15, 20, 25, 30]:
            col = f"p_{t}plus_disp"
            if col in merged_disp.columns:
                m = _compute_threshold_metrics(
                    merged_disp[col].values.astype(float),
                    (actual_disp >= t).astype(int),
                )
                round_result[f"brier_{t}plus_disp"] = m["brier_score"] if m else None
                round_result[f"logloss_{t}plus_disp"] = m["log_loss"] if m else None
            else:
                round_result[f"brier_{t}plus_disp"] = None
                round_result[f"logloss_{t}plus_disp"] = None

        results.append(round_result)

        elapsed = time.time() - rnd_start
        b1 = round_result.get("brier_1plus_goals")
        auc = round_result.get("scorer_auc")
        b20 = round_result.get("brier_20plus_disp")
        _f = lambda v: f"{v:.4f}" if v is not None else "N/A"
        print(f"  {condition_name} R{int(rnd):02d}  n={len(test_df):<4d}"
              f" B1+={_f(b1)} AUC={_f(auc)} B20+d={_f(b20)}"
              f" ({elapsed:.1f}s)", flush=True)

    total_elapsed = time.time() - total_start
    print(f"  {condition_name} total: {total_elapsed:.0f}s\n", flush=True)
    return results


def compare_conditions(results_a, results_b, label_a, label_b):
    """Print comparison table and run paired t-tests."""
    metric_keys = set()
    for r in results_a + results_b:
        for k, v in r.items():
            if k not in ("round", "n_players") and v is not None:
                metric_keys.add(k)
    metric_keys = sorted(metric_keys)

    print(f"\n{'='*110}")
    print(f"COMPARISON: {label_a} vs {label_b}")
    print(f"{'='*110}")

    print(f"\n{'Metric':>30}  {label_a:>12}  {label_b:>12}  {'Diff':>10}  {'p-value':>10}  {'Winner':>10}")
    print(f"{'─'*30}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}")

    winners = {}
    for key in metric_keys:
        vals_a = [r.get(key) for r in results_a if r.get(key) is not None]
        vals_b = [r.get(key) for r in results_b if r.get(key) is not None]

        if not vals_a or not vals_b:
            continue

        mean_a = np.mean(vals_a)
        mean_b = np.mean(vals_b)
        diff = mean_b - mean_a

        paired_a, paired_b = [], []
        for ra, rb in zip(results_a, results_b):
            va = ra.get(key)
            vb = rb.get(key)
            if va is not None and vb is not None:
                paired_a.append(va)
                paired_b.append(vb)

        p_val = None
        if len(paired_a) >= 2:
            _, p_val = ttest_rel(paired_a, paired_b)

        higher_better = key in ("scorer_auc",)
        if higher_better:
            winner = label_b if mean_b > mean_a else label_a
        else:
            winner = label_b if mean_b < mean_a else label_a

        sig = ""
        if p_val is not None:
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""

        p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
        winners[key] = (winner, p_val, sig)

        print(f"  {key:>28}  {mean_a:12.4f}  {mean_b:12.4f}  {diff:+10.4f}  {p_str:>10}  {winner:>8} {sig}")

    return winners


def main():
    print("=" * 110, flush=True)
    print("MULTI-MODEL ENSEMBLE vs SINGLE-MODEL COMPARISON (Fast — no game winner)", flush=True)
    print("HistGBT-base only  vs  HistGBT-base + HistGBT-deep + HistGBT-wide (averaged)", flush=True)
    print(f"2025 Sequential Backtest", flush=True)
    print("=" * 110, flush=True)

    feature_df, feature_cols = load_data()
    print(f"\n  Feature matrix: {feature_df.shape}", flush=True)
    print(f"  Features: {len(feature_cols)}", flush=True)

    season = feature_df[feature_df["year"] == BACKTEST_YEAR]
    all_rounds = sorted(season["round_number"].dropna().unique())
    rounds = [r for r in all_rounds if r <= 24]
    print(f"  Rounds: {[int(r) for r in rounds]}\n", flush=True)

    # --- Condition A: Single-model baseline ---
    print("─" * 80, flush=True)
    print("CONDITION A: Single-Model (HistGBT only)", flush=True)
    print("─" * 80, flush=True)
    results_single = run_condition(
        feature_df, feature_cols,
        "Single", use_ensemble=False, rounds=rounds
    )

    # --- Condition B: Multi-model ensemble ---
    print("─" * 80, flush=True)
    print("CONDITION B: Multi-Model Ensemble (HistGBT-base + HistGBT-deep + HistGBT-wide)", flush=True)
    print("─" * 80, flush=True)
    results_ensemble = run_condition(
        feature_df, feature_cols,
        "Ensemble", use_ensemble=True, rounds=rounds
    )

    # --- Comparison ---
    winners = compare_conditions(results_single, results_ensemble, "Single", "Ensemble")

    # Per-round detail table for key metrics
    print(f"\n{'='*110}")
    print("PER-ROUND BRIER SCORES (Goals 1+)")
    print(f"{'='*110}")
    print(f"{'Round':>5}  {'Single':>10}  {'Ensemble':>10}  {'Diff':>10}")
    for rs, re in zip(results_single, results_ensemble):
        bs = rs.get("brier_1plus_goals")
        be = re.get("brier_1plus_goals")
        if bs is not None and be is not None:
            print(f"  {rs['round']:3d}  {bs:10.4f}  {be:10.4f}  {be-bs:+10.4f}")

    # Save results
    rows = []
    for cond, results in [("single", results_single), ("ensemble", results_ensemble)]:
        for r in results:
            row = {"condition": cond}
            row.update(r)
            rows.append(row)
    results_df = pd.DataFrame(rows)
    out_path = EXPERIMENT_DIR / "ensemble_comparison_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved to {out_path}")

    # --- Verdict ---
    print(f"\n{'='*110}")
    print("VERDICT")
    print(f"{'='*110}")

    primary_metrics = [
        "brier_1plus_goals", "brier_2plus_goals", "brier_3plus_goals",
        "brier_15plus_disp", "brier_20plus_disp", "brier_25plus_disp", "brier_30plus_disp",
        "scorer_auc",
    ]
    ensemble_wins = sum(1 for k in primary_metrics
                        if k in winners and winners[k][0] == "Ensemble")
    single_wins = sum(1 for k in primary_metrics
                      if k in winners and winners[k][0] == "Single")
    sig_ensemble = sum(1 for k in primary_metrics
                       if k in winners and winners[k][0] == "Ensemble" and winners[k][2])
    sig_single = sum(1 for k in primary_metrics
                     if k in winners and winners[k][0] == "Single" and winners[k][2])

    print(f"\n  Primary metrics ({len(primary_metrics)} total):")
    print(f"    Ensemble wins: {ensemble_wins} ({sig_ensemble} significant)")
    print(f"    Single wins:   {single_wins} ({sig_single} significant)")

    if ensemble_wins > single_wins:
        print(f"\n  >>> ENSEMBLE IS BETTER. Recommend keeping MULTI_MODEL_ENSEMBLE = True")
    elif single_wins > ensemble_wins:
        print(f"\n  >>> SINGLE MODEL IS BETTER. Recommend reverting MULTI_MODEL_ENSEMBLE = False")
    else:
        print(f"\n  >>> TIE. Keep ensemble if training time is acceptable.")


if __name__ == "__main__":
    main()
