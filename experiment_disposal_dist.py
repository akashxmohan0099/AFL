"""
Disposal Distribution Comparison Experiment
============================================
Compares Poisson, Gaussian, and Negative Binomial distributions for
disposal threshold probability estimation through a 2025 sequential backtest.

The underlying regression model (PoissonRegressor + HistGBT ensemble) is
identical across all three — only the CDF used to compute P(X >= k) differs.

Output:
  - Per-round Brier scores, log loss, hit rates for each threshold
  - Season-aggregate comparison table
  - Winner determination per threshold
  - Results saved to data/experiments/disposal_dist_results.csv
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

import config
from analysis import _compute_threshold_metrics, _extract_threshold_data
from model import AFLDisposalModel, _prepare_features

warnings.filterwarnings("ignore")

EXPERIMENT_DIR = config.DATA_DIR / "experiments"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

DISTRIBUTIONS = ["poisson", "gaussian", "negbin"]
THRESHOLDS = config.DISPOSAL_THRESHOLDS  # [10, 15, 20, 25, 30]
BACKTEST_YEAR = 2025


def load_data():
    """Load feature matrix and feature columns."""
    feature_df = pd.read_parquet("data/features/feature_matrix.parquet")
    with open("data/features/feature_columns.json") as f:
        feature_cols = json.load(f)
    feature_df = feature_df.sort_values(["date", "match_id", "player"]).reset_index(drop=True)
    return feature_df, feature_cols


def run_backtest_for_distribution(feature_df, feature_cols, dist_name, rounds):
    """Run sequential backtest for a single distribution variant.

    The regression models are identical; only the probability CDF differs.
    Returns per-round metrics for each disposal threshold.
    """
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

        # Train disposal model with this distribution
        model = AFLDisposalModel(distribution=dist_name)
        model.train_backtest(train_df, feature_cols)

        # Predict (no calibration store — raw comparison)
        preds = model.predict_distributions(test_df, store=None, feature_cols=feature_cols)

        # Build merged DataFrame with actuals
        merged = preds.copy()
        merged["actual_disposals"] = test_df["DI"].values

        # MAE of point predictions
        mae = float(np.mean(np.abs(
            merged["predicted_disposals"].values - merged["actual_disposals"].values
        )))

        # Compute Brier/log-loss per threshold
        actual_disp = merged["actual_disposals"].values
        round_result = {
            "round": int(rnd),
            "n_players": len(test_df),
            "mae": round(mae, 4),
        }

        for t in THRESHOLDS:
            col = f"p_{t}plus_disp"
            if col not in merged.columns:
                continue

            pred_probs = merged[col].values.astype(float)
            actual_binary = (actual_disp >= t).astype(int)

            metrics = _compute_threshold_metrics(pred_probs, actual_binary)
            if metrics is not None:
                round_result[f"brier_{t}plus"] = metrics["brier_score"]
                round_result[f"logloss_{t}plus"] = metrics["log_loss"]
                round_result[f"base_rate_{t}plus"] = metrics["base_rate"]

                # Hit rate: among predictions with P >= 0.5, what fraction actually hit?
                high_conf = pred_probs >= 0.5
                if high_conf.sum() > 0:
                    hit_rate = float(actual_binary[high_conf].mean())
                    round_result[f"hitrate_{t}plus"] = round(hit_rate, 4)
                else:
                    round_result[f"hitrate_{t}plus"] = None
            else:
                round_result[f"brier_{t}plus"] = None
                round_result[f"logloss_{t}plus"] = None
                round_result[f"base_rate_{t}plus"] = None
                round_result[f"hitrate_{t}plus"] = None

        elapsed = time.time() - rnd_start
        results.append(round_result)

        # Progress line
        b15 = round_result.get("brier_15plus", "N/A")
        b20 = round_result.get("brier_20plus", "N/A")
        b25 = round_result.get("brier_25plus", "N/A")
        b_str = lambda v: f"{v:.4f}" if isinstance(v, float) else "N/A"
        print(f"  {dist_name:8s} R{int(rnd):02d}  n={len(test_df):<4d} MAE={mae:.2f}  "
              f"B15+={b_str(b15)}  B20+={b_str(b20)}  B25+={b_str(b25)}  ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start
    print(f"  {dist_name} total: {total_elapsed:.0f}s\n")
    return results


def aggregate_results(all_results):
    """Compute season-aggregate metrics per distribution per threshold."""
    agg = {}
    for dist_name, results in all_results.items():
        agg[dist_name] = {"mae": np.mean([r["mae"] for r in results if r["mae"] is not None])}
        for t in THRESHOLDS:
            key_b = f"brier_{t}plus"
            key_l = f"logloss_{t}plus"
            key_h = f"hitrate_{t}plus"
            key_br = f"base_rate_{t}plus"

            brier_vals = [r[key_b] for r in results if r.get(key_b) is not None]
            logloss_vals = [r[key_l] for r in results if r.get(key_l) is not None]
            hitrate_vals = [r[key_h] for r in results if r.get(key_h) is not None]
            baserate_vals = [r[key_br] for r in results if r.get(key_br) is not None]

            agg[dist_name][key_b] = np.mean(brier_vals) if brier_vals else None
            agg[dist_name][key_l] = np.mean(logloss_vals) if logloss_vals else None
            agg[dist_name][key_h] = np.mean(hitrate_vals) if hitrate_vals else None
            agg[dist_name][key_br] = np.mean(baserate_vals) if baserate_vals else None

    return agg


def paired_t_tests(all_results, baseline="poisson"):
    """Run paired t-tests comparing each distribution against baseline."""
    tests = {}
    baseline_results = all_results[baseline]

    for dist_name, results in all_results.items():
        if dist_name == baseline:
            continue
        tests[dist_name] = {}
        for t in THRESHOLDS:
            key = f"brier_{t}plus"
            vals_x = [r.get(key) for r in results]
            vals_base = [r.get(key) for r in baseline_results]

            # Align and filter Nones
            pairs = [(a, b) for a, b in zip(vals_x, vals_base)
                     if a is not None and b is not None]
            if len(pairs) >= 2:
                x_arr = [p[0] for p in pairs]
                b_arr = [p[1] for p in pairs]
                t_stat, p_val = ttest_rel(x_arr, b_arr)
                diff = np.mean(x_arr) - np.mean(b_arr)
                tests[dist_name][f"{t}plus"] = {
                    "t_stat": round(t_stat, 4),
                    "p_val": round(p_val, 4),
                    "mean_diff": round(diff, 6),
                    "direction": "better" if diff < 0 else "worse",
                }

    return tests


def print_comparison_table(agg, tests):
    """Print the main comparison table."""
    print("\n" + "=" * 100)
    print("SEASON AGGREGATE COMPARISON")
    print("=" * 100)

    # MAE (same for all since regression is identical)
    print(f"\nPoint prediction MAE (identical regression model):")
    for dist_name in DISTRIBUTIONS:
        print(f"  {dist_name:10s}: {agg[dist_name]['mae']:.4f}")

    # Per-threshold comparison
    for t in THRESHOLDS:
        print(f"\n{'─' * 80}")
        print(f"  {t}+ DISPOSALS")
        print(f"{'─' * 80}")

        br_key = f"base_rate_{t}plus"
        base_rate = agg["poisson"].get(br_key)
        if base_rate is not None:
            print(f"  Base rate: {base_rate:.3f} ({base_rate*100:.1f}% of players reach {t}+ disposals)")

        print(f"\n  {'Distribution':>12}  {'Brier':>10}  {'LogLoss':>10}  {'HitRate@50%':>12}  {'vs Poisson':>15}")
        print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*15}")

        best_brier = None
        best_dist = None

        for dist_name in DISTRIBUTIONS:
            b_key = f"brier_{t}plus"
            l_key = f"logloss_{t}plus"
            h_key = f"hitrate_{t}plus"

            brier = agg[dist_name].get(b_key)
            logloss = agg[dist_name].get(l_key)
            hitrate = agg[dist_name].get(h_key)

            if brier is not None and (best_brier is None or brier < best_brier):
                best_brier = brier
                best_dist = dist_name

            b_str = f"{brier:.4f}" if brier is not None else "N/A"
            l_str = f"{logloss:.4f}" if logloss is not None else "N/A"
            h_str = f"{hitrate:.3f}" if hitrate is not None else "N/A"

            if dist_name == "poisson":
                sig_str = "(baseline)"
            else:
                test_data = tests.get(dist_name, {}).get(f"{t}plus")
                if test_data:
                    p = test_data["p_val"]
                    d = test_data["direction"]
                    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
                    sig_str = f"p={p:.3f} {d} {stars}"
                else:
                    sig_str = ""

            print(f"  {dist_name:>12}  {b_str:>10}  {l_str:>10}  {h_str:>12}  {sig_str:>15}")

        if best_dist:
            print(f"  >>> WINNER: {best_dist}")


def determine_winner(agg):
    """Determine overall winner across all thresholds."""
    # Count wins per distribution
    wins = {d: 0 for d in DISTRIBUTIONS}
    total_brier_improvement = {d: 0.0 for d in DISTRIBUTIONS}

    for t in THRESHOLDS:
        key = f"brier_{t}plus"
        scores = {d: agg[d].get(key) for d in DISTRIBUTIONS}
        valid = {d: s for d, s in scores.items() if s is not None}
        if valid:
            best = min(valid, key=valid.get)
            wins[best] += 1
            # Track improvement over poisson
            if scores.get("poisson") is not None:
                for d in DISTRIBUTIONS:
                    if d != "poisson" and scores.get(d) is not None:
                        total_brier_improvement[d] += scores["poisson"] - scores[d]

    print("\n" + "=" * 100)
    print("OVERALL VERDICT")
    print("=" * 100)
    print(f"\n  Brier score wins by threshold (out of {len(THRESHOLDS)}):")
    for d in DISTRIBUTIONS:
        print(f"    {d:10s}: {wins[d]} wins")

    overall_winner = max(wins, key=wins.get)
    print(f"\n  Overall winner: {overall_winner}")
    print(f"  Total Brier improvement over Poisson:")
    for d in ["gaussian", "negbin"]:
        imp = total_brier_improvement[d]
        print(f"    {d:10s}: {imp:+.6f} ({'better' if imp > 0 else 'worse'})")

    return overall_winner


def main():
    print("=" * 100)
    print("DISPOSAL DISTRIBUTION COMPARISON EXPERIMENT")
    print(f"Poisson vs Gaussian vs Negative Binomial — {BACKTEST_YEAR} Sequential Backtest")
    print("=" * 100)

    # Load data
    print("\nLoading data...")
    feature_df, feature_cols = load_data()
    print(f"  Feature matrix: {feature_df.shape}")
    print(f"  Features: {len(feature_cols)}")

    season_df = feature_df[feature_df["year"] == BACKTEST_YEAR]
    all_rounds = sorted(season_df["round_number"].dropna().unique())
    rounds = [r for r in all_rounds if r <= 24]  # Regular season only
    print(f"  {BACKTEST_YEAR} rounds: {[int(r) for r in rounds]}")
    print(f"  Thresholds: {THRESHOLDS}")

    # Run backtest for each distribution
    all_results = {}
    for dist_name in DISTRIBUTIONS:
        print(f"\n{'─' * 80}")
        print(f"CONDITION: {dist_name.upper()}")
        print(f"{'─' * 80}")
        all_results[dist_name] = run_backtest_for_distribution(
            feature_df, feature_cols, dist_name, rounds
        )

    # Aggregate and compare
    agg = aggregate_results(all_results)
    tests = paired_t_tests(all_results, baseline="poisson")
    print_comparison_table(agg, tests)
    winner = determine_winner(agg)

    # Save detailed per-round results
    rows = []
    for dist_name, results in all_results.items():
        for r in results:
            row = {"distribution": dist_name}
            row.update(r)
            rows.append(row)
    results_df = pd.DataFrame(rows)
    out_path = EXPERIMENT_DIR / "disposal_dist_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Detailed results saved to {out_path}")

    # Save summary table
    summary_rows = []
    for dist_name in DISTRIBUTIONS:
        row = {"distribution": dist_name, "mae": agg[dist_name]["mae"]}
        for t in THRESHOLDS:
            row[f"brier_{t}plus"] = agg[dist_name].get(f"brier_{t}plus")
            row[f"logloss_{t}plus"] = agg[dist_name].get(f"logloss_{t}plus")
            row[f"hitrate_{t}plus"] = agg[dist_name].get(f"hitrate_{t}plus")
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = EXPERIMENT_DIR / "disposal_dist_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary saved to {summary_path}")

    # Print recommendation
    print(f"\n{'=' * 100}")
    print(f"RECOMMENDATION")
    print(f"{'=' * 100}")
    current = config.DISPOSAL_DISTRIBUTION
    print(f"  Current config: DISPOSAL_DISTRIBUTION = '{current}'")
    print(f"  Experiment winner: '{winner}'")
    if winner == current:
        print(f"  -> Current setting is already optimal. No change needed.")
    else:
        print(f"  -> Consider changing config.DISPOSAL_DISTRIBUTION to '{winner}'")


if __name__ == "__main__":
    main()
