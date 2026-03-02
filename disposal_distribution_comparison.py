"""
Disposal Distribution Comparison
=================================
Runs the 2025 sequential backtest with three distribution options:
  1. Poisson (current baseline)
  2. Gaussian (Normal)
  3. Negative Binomial

Compares Brier scores, log loss, and calibration at each disposal threshold.
"""
import sys
import os
import json
import time
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

import config
from store import LearningStore
from analysis import generate_season_report

# We'll import and call the pipeline functions directly
import pipeline


def run_backtest_with_distribution(dist_name, year=2025):
    """Run 2025 sequential backtest with a specific disposal distribution."""
    print(f"\n{'#' * 80}")
    print(f"#  Running 2025 backtest with distribution: {dist_name.upper()}")
    print(f"{'#' * 80}")

    # Clear sequential learning state so each run starts fresh
    seq_dir = config.SEQUENTIAL_DIR
    for subdir in ["predictions", "outcomes", "diagnostics", "calibration",
                    "game_predictions", "archetypes", "concessions"]:
        d = seq_dir / subdir
        if d.exists():
            for f in d.glob(f"*_{year}_*.parquet"):
                f.unlink()
    # Clear calibration state
    cal_file = seq_dir / "calibration" / "calibration_state.parquet"
    if cal_file.exists():
        cal_file.unlink()

    # Run the sequential backtest
    class Args:
        year = 2025

    start = time.time()
    pipeline.cmd_sequential(Args(), disposal_distribution=dist_name)
    elapsed = time.time() - start

    # Generate report
    store = LearningStore(base_dir=seq_dir)
    report = generate_season_report(store, year)

    # Save report
    report_path = seq_dir / f"season_report_{year}_{dist_name}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  [{dist_name}] Completed in {elapsed:.1f}s")
    print(f"  Report saved: {report_path}")

    return report


def compare_reports(reports):
    """Print comparison table across all distributions."""
    distributions = list(reports.keys())
    col_w = 14

    print(f"\n{'=' * 90}")
    print(f"  DISPOSAL DISTRIBUTION COMPARISON — 2025 Sequential Backtest")
    print(f"{'=' * 90}")

    # Header
    header = f"  {'Threshold':<18s}"
    for d in distributions:
        header += f" {d:>{col_w}s}"
    header += f" {'Best':>{col_w}s}"

    # Brier Scores
    print(f"\n  BRIER SCORES (lower = better):")
    print(header)
    print("-" * (20 + (col_w + 1) * (len(distributions) + 1)))

    thresholds = ["10plus_disp", "15plus_disp", "20plus_disp", "25plus_disp", "30plus_disp"]
    brier_wins = {d: 0 for d in distributions}

    for thresh in thresholds:
        row = f"  {thresh:<18s}"
        values = {}
        for d in distributions:
            te = reports[d].get("threshold_evaluation", {})
            if thresh in te:
                val = te[thresh].get("brier_score", float("nan"))
                values[d] = val
                row += f" {val:>{col_w}.4f}"
            else:
                values[d] = float("nan")
                row += f" {'N/A':>{col_w}s}"

        # Find best
        valid = {k: v for k, v in values.items() if not np.isnan(v)}
        if valid:
            best = min(valid, key=valid.get)
            brier_wins[best] += 1
            row += f" {best:>{col_w}s}"
        else:
            row += f" {'—':>{col_w}s}"
        print(row)

    # Log Loss
    print(f"\n  LOG LOSS (lower = better):")
    print(header)
    print("-" * (20 + (col_w + 1) * (len(distributions) + 1)))

    logloss_wins = {d: 0 for d in distributions}
    for thresh in thresholds:
        row = f"  {thresh:<18s}"
        values = {}
        for d in distributions:
            te = reports[d].get("threshold_evaluation", {})
            if thresh in te:
                val = te[thresh].get("log_loss", float("nan"))
                values[d] = val
                row += f" {val:>{col_w}.4f}"
            else:
                values[d] = float("nan")
                row += f" {'N/A':>{col_w}s}"
        valid = {k: v for k, v in values.items() if not np.isnan(v)}
        if valid:
            best = min(valid, key=valid.get)
            logloss_wins[best] += 1
            row += f" {best:>{col_w}s}"
        else:
            row += f" {'—':>{col_w}s}"
        print(row)

    # Calibration curves (predicted vs observed) for key thresholds
    print(f"\n  CALIBRATION DETAIL (predicted_mean → observed_mean):")
    for thresh in ["15plus_disp", "20plus_disp", "25plus_disp"]:
        print(f"\n  --- {thresh} ---")
        # Get calibration data for each distribution
        for d in distributions:
            te = reports[d].get("threshold_evaluation", {})
            if thresh in te:
                cal = te[thresh].get("calibration_curve", [])
                if cal:
                    parts = []
                    for bucket in cal:
                        pred = bucket["predicted_mean"]
                        obs = bucket["observed_mean"]
                        n = bucket["count"]
                        diff = obs - pred
                        parts.append(f"  p={pred:.2f}→o={obs:.2f} (Δ={diff:+.2f}, n={n})")
                    # Print first 5 buckets and last 3
                    label = f"    {d:<10s}: "
                    if len(parts) <= 6:
                        print(label + " | ".join(parts))
                    else:
                        print(label + " | ".join(parts[:3]))
                        print(f"    {'':10s}  " + " | ".join(parts[-3:]))

    # Summary
    print(f"\n{'=' * 90}")
    print(f"  SUMMARY")
    print(f"{'=' * 90}")
    print(f"  Brier score wins:  ", end="")
    for d in distributions:
        print(f"{d}: {brier_wins[d]}/5  ", end="")
    print()
    print(f"  Log loss wins:     ", end="")
    for d in distributions:
        print(f"{d}: {logloss_wins[d]}/5  ", end="")
    print()

    # Determine overall winner
    total_wins = {d: brier_wins[d] + logloss_wins[d] for d in distributions}
    winner = max(total_wins, key=total_wins.get)
    print(f"\n  Overall winner: {winner.upper()} ({total_wins[winner]}/10 threshold-metric wins)")

    # Print deltas vs Poisson baseline
    if "poisson" in reports:
        print(f"\n  IMPROVEMENT vs POISSON BASELINE:")
        print(f"  {'Threshold':<18s} {'Gaussian Δ':>14s} {'NegBin Δ':>14s}")
        print("-" * 50)
        for thresh in thresholds:
            te_p = reports["poisson"].get("threshold_evaluation", {}).get(thresh, {})
            row = f"  {thresh:<18s}"
            for d in ["gaussian", "negbin"]:
                te_d = reports.get(d, {}).get("threshold_evaluation", {}).get(thresh, {})
                bp = te_p.get("brier_score", float("nan"))
                bd = te_d.get("brier_score", float("nan"))
                if not np.isnan(bp) and not np.isnan(bd):
                    delta = bd - bp
                    marker = " ✓" if delta < -0.0005 else (" ✗" if delta > 0.0005 else "  ")
                    row += f" {delta:>+12.4f}{marker}"
                else:
                    row += f" {'N/A':>14s}"
            print(row)

    return winner


if __name__ == "__main__":
    distributions = ["poisson", "gaussian", "negbin"]
    reports = {}

    for dist in distributions:
        reports[dist] = run_backtest_with_distribution(dist, year=2025)

    winner = compare_reports(reports)
    print(f"\nDone. Recommended distribution: {winner}")
