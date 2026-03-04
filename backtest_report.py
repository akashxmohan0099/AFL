"""
AFL Prediction Pipeline — Comprehensive Backtest Report
========================================================
Loads the latest sequential run's predictions/outcomes and computes
full metrics across all model types. Saves experiment JSON.

Usage:
    python3 backtest_report.py [--year 2025] [--run-id <id>]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from store import LearningStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def brier_score(probs, actuals):
    """Mean squared error between predicted probabilities and binary outcomes."""
    p = np.asarray(probs, dtype=float)
    a = np.asarray(actuals, dtype=float)
    mask = np.isfinite(p) & np.isfinite(a)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean((p[mask] - a[mask]) ** 2))


def brier_skill_score(probs, actuals):
    """BSS = 1 - Brier / Brier_climatology."""
    a = np.asarray(actuals, dtype=float)
    mask = np.isfinite(np.asarray(probs, dtype=float)) & np.isfinite(a)
    if mask.sum() == 0:
        return np.nan
    base_rate = a[mask].mean()
    brier_clim = base_rate * (1 - base_rate)
    bs = brier_score(probs, actuals)
    if brier_clim == 0:
        return 0.0
    return float(1 - bs / brier_clim)


def hit_rate_at_confidence(probs, actuals, threshold):
    """Accuracy among predictions where P >= threshold."""
    p = np.asarray(probs, dtype=float)
    a = np.asarray(actuals, dtype=float)
    mask = p >= threshold
    if mask.sum() == 0:
        return np.nan, 0
    acc = float(a[mask].mean())
    return acc, int(mask.sum())


def calibration_curve(probs, actuals, n_bins=10):
    """Simple equal-width calibration curve."""
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_rounds(store, year, subdir):
    """Load and concatenate all round parquets for a year from the latest run."""
    round_tuples = store.list_rounds(subdir=subdir, year=year)
    if not round_tuples:
        return pd.DataFrame()
    frames = []
    for yr, rnd in sorted(round_tuples):
        try:
            if subdir == "predictions":
                df = store.load_predictions(year=yr, round_num=rnd)
            elif subdir == "outcomes":
                df = store.load_outcomes(year=yr, round_num=rnd)
            elif subdir == "game_predictions":
                df = store.load_game_predictions(year=yr, round_num=rnd)
            else:
                continue
            if df is not None and not df.empty:
                df["round_number"] = rnd
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Threshold metrics
# ---------------------------------------------------------------------------

def compute_threshold_metrics(probs, actuals, label):
    """Compute Brier, BSS, hit rates, calibration for a single threshold."""
    p = np.asarray(probs, dtype=float)
    a = np.asarray(actuals, dtype=float)
    valid = np.isfinite(p) & np.isfinite(a)
    p, a = p[valid], a[valid]
    n = len(p)

    if n == 0:
        return None

    bs = brier_score(p, a)
    bss = brier_skill_score(p, a)
    base_rate = float(a.mean())
    acc = float(((p >= 0.5) == a).mean())

    hr60, n60 = hit_rate_at_confidence(p, a, 0.60)
    hr70, n70 = hit_rate_at_confidence(p, a, 0.70)
    hr80, n80 = hit_rate_at_confidence(p, a, 0.80)

    curve = calibration_curve(p, a)
    ece = expected_calibration_error(p, a)

    return {
        "n": n,
        "base_rate": round(base_rate, 4),
        "brier_score": round(bs, 4),
        "bss": round(bss, 4),
        "accuracy": round(acc, 4),
        "hit_rate_p60": round(hr60, 4) if hr60 is not np.nan else None,
        "n_confident_p60": n60,
        "hit_rate_p70": round(hr70, 4) if hr70 is not np.nan else None,
        "n_confident_p70": n70,
        "hit_rate_p80": round(hr80, 4) if hr80 is not np.nan else None,
        "n_confident_p80": n80,
        "calibration_ece": ece,
        "calibration_curve": curve,
    }


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def report_goals(merged):
    """Section B: Goals (1+, 2+, 3+)."""
    results = {}
    thresholds = [
        ("1plus_goals", "p_1plus_goals", "p_scorer", 1),
        ("2plus_goals", "p_2plus_goals", None, 2),
        ("3plus_goals", "p_3plus_goals", None, 3),
    ]
    for label, col, fallback_col, thresh in thresholds:
        pcol = col if col in merged.columns else fallback_col
        if pcol is None or pcol not in merged.columns:
            continue
        actuals = (merged["actual_goals"] >= thresh).astype(int)
        metrics = compute_threshold_metrics(merged[pcol], actuals, label)
        if metrics:
            results[label] = metrics

    # MAE
    if "predicted_goals" in merged.columns:
        valid = merged["predicted_goals"].notna() & merged["actual_goals"].notna()
        mae = float(np.abs(merged.loc[valid, "predicted_goals"] - merged.loc[valid, "actual_goals"]).mean())
        baseline_mae = float(np.abs(merged.loc[valid, "actual_goals"].mean() - merged.loc[valid, "actual_goals"]).mean())
        results["mae"] = {
            "goals_mae": round(mae, 4),
            "baseline_mae": round(baseline_mae, 4),
            "improvement_pct": round((1 - mae / baseline_mae) * 100, 1) if baseline_mae > 0 else 0,
        }

    return results


def report_disposals(merged):
    """Section C: Disposals (15+, 20+, 25+, 30+)."""
    results = {}
    for thresh in [10, 15, 20, 25, 30]:
        col = f"p_{thresh}plus_disp"
        if col not in merged.columns:
            continue
        actuals = (merged["actual_disposals"] >= thresh).astype(int)
        metrics = compute_threshold_metrics(merged[col], actuals, f"{thresh}plus_disp")
        if metrics:
            results[f"{thresh}plus_disp"] = metrics

    # MAE
    if "predicted_disposals" in merged.columns and "actual_disposals" in merged.columns:
        valid = merged["predicted_disposals"].notna() & merged["actual_disposals"].notna()
        if valid.sum() > 0:
            mae = float(np.abs(merged.loc[valid, "predicted_disposals"] - merged.loc[valid, "actual_disposals"]).mean())
            baseline_mae = float(np.abs(merged.loc[valid, "actual_disposals"].mean() - merged.loc[valid, "actual_disposals"]).mean())
            results["mae"] = {
                "disposals_mae": round(mae, 4),
                "baseline_mae": round(baseline_mae, 4),
                "improvement_pct": round((1 - mae / baseline_mae) * 100, 1) if baseline_mae > 0 else 0,
            }

    return results


def report_marks(merged):
    """Section D: Marks (3+, 5+, 7+)."""
    results = {}
    for thresh in [3, 5, 7]:
        col = f"p_{thresh}plus_mk"
        if col not in merged.columns:
            continue
        actuals = (merged["actual_marks"] >= thresh).astype(int)
        metrics = compute_threshold_metrics(merged[col], actuals, f"{thresh}plus_mk")
        if metrics:
            results[f"{thresh}plus_mk"] = metrics

    # MAE
    if "predicted_marks" in merged.columns and "actual_marks" in merged.columns:
        valid = merged["predicted_marks"].notna() & merged["actual_marks"].notna()
        if valid.sum() > 0:
            mae = float(np.abs(merged.loc[valid, "predicted_marks"] - merged.loc[valid, "actual_marks"]).mean())
            baseline_mae = float(np.abs(merged.loc[valid, "actual_marks"].mean() - merged.loc[valid, "actual_marks"]).mean())
            results["mae"] = {
                "marks_mae": round(mae, 4),
                "baseline_mae": round(baseline_mae, 4),
                "improvement_pct": round((1 - mae / baseline_mae) * 100, 1) if baseline_mae > 0 else 0,
            }

    return results


def report_game_winner(game_preds, team_match_df, year):
    """Section A: Game winner metrics."""
    if game_preds.empty:
        return {}

    results = {}

    # Merge with actual results
    actual = team_match_df[team_match_df["year"] == year].copy()
    if actual.empty or "home_win_prob" not in game_preds.columns:
        return {}

    # Build match-level actual outcomes
    if "match_id" in game_preds.columns and "match_id" in actual.columns:
        # Get home team results
        home = actual[actual["is_home"] == True][["match_id", "team", "score", "opponent"]].copy()
        away = actual[actual["is_home"] == False][["match_id", "team", "score"]].copy()

        if home.empty or away.empty:
            return {}

        match_results = home.merge(
            away[["match_id", "score"]].rename(columns={"score": "away_score"}),
            on="match_id", how="inner"
        ).rename(columns={"score": "home_score"})
        match_results["home_won"] = (match_results["home_score"] > match_results["away_score"]).astype(int)
        match_results["actual_margin"] = match_results["home_score"] - match_results["away_score"]

        gp = game_preds.drop_duplicates(subset=["match_id"]).copy()
        merged_gw = gp.merge(match_results[["match_id", "home_won", "actual_margin"]], on="match_id", how="inner")

        if merged_gw.empty:
            return {}

        n = len(merged_gw)
        probs = merged_gw["home_win_prob"].values
        actual_hw = merged_gw["home_won"].values

        # Accuracy
        predicted_winner = (probs >= 0.5).astype(int)
        accuracy = float((predicted_winner == actual_hw).mean())

        # Brier
        bs = brier_score(probs, actual_hw)

        # Margin MAE
        if "predicted_margin" in merged_gw.columns:
            margin_mae = float(np.abs(merged_gw["predicted_margin"] - merged_gw["actual_margin"]).mean())
        else:
            margin_mae = None

        results = {
            "n_games": n,
            "accuracy": round(accuracy, 4),
            "brier_score": round(bs, 4),
            "margin_mae": round(margin_mae, 2) if margin_mae is not None else None,
        }

        # Simulated flat betting ROI
        # Bet on predicted winner when confidence >= 55%
        confident = np.abs(probs - 0.5) >= 0.05
        if confident.sum() > 0:
            bets = confident.sum()
            wins = ((predicted_winner == actual_hw) & confident).sum()
            # Assume fair odds (~1.91 for 52.5% implied)
            roi = (wins * 1.91 - bets) / bets * 100
            results["flat_bet_roi_pct"] = round(float(roi), 1)
            results["flat_bet_n"] = int(bets)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Comprehensive backtest report")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--label", type=str, default="6_comprehensive_v4.0")
    args = parser.parse_args()

    year = args.year
    store = LearningStore(base_dir=config.SEQUENTIAL_DIR, run_id=args.run_id)

    # Find latest run
    runs = store.list_runs(year=year, subdir="predictions")
    if not runs:
        print(f"No sequential runs found for {year}")
        sys.exit(1)

    run_id = args.run_id or runs[-1]
    store = LearningStore(base_dir=config.SEQUENTIAL_DIR, run_id=run_id)
    print(f"Loading run: {run_id}")

    # Load data
    predictions = load_all_rounds(store, year, "predictions")
    outcomes = load_all_rounds(store, year, "outcomes")
    game_preds = load_all_rounds(store, year, "game_predictions")

    if predictions.empty or outcomes.empty:
        print("No predictions or outcomes found")
        sys.exit(1)

    # Merge predictions + outcomes
    join_cols = ["player", "team"]
    if "match_id" in predictions.columns and "match_id" in outcomes.columns:
        join_cols.append("match_id")
    if "round_number" in predictions.columns and "round_number" in outcomes.columns:
        join_cols.append("round_number")

    merged = predictions.merge(outcomes, on=join_cols, how="inner")
    n_rounds = merged["round_number"].nunique() if "round_number" in merged.columns else 0
    n_preds = len(merged)

    print(f"Merged: {n_preds} predictions across {n_rounds} rounds")
    print(f"Columns: {sorted(merged.columns.tolist())}\n")

    # --- Reports ---
    print("=" * 70)
    print(f"  COMPREHENSIVE BACKTEST REPORT — {year}")
    print("=" * 70)

    # A. Game Winner
    print("\n--- A. GAME WINNER ---")
    tm_path = config.BASE_STORE_DIR / "team_matches.parquet"
    team_match_df = pd.read_parquet(tm_path) if tm_path.exists() else pd.DataFrame()
    gw_results = report_game_winner(game_preds, team_match_df, year)
    if gw_results:
        print(f"  Games: {gw_results.get('n_games', 'N/A')}")
        print(f"  Accuracy: {gw_results.get('accuracy', 'N/A')}")
        print(f"  Brier: {gw_results.get('brier_score', 'N/A')}")
        if gw_results.get("margin_mae"):
            print(f"  Margin MAE: {gw_results['margin_mae']} pts")
        if gw_results.get("flat_bet_roi_pct") is not None:
            print(f"  Flat bet ROI: {gw_results['flat_bet_roi_pct']}% ({gw_results['flat_bet_n']} bets)")
    else:
        print("  No game winner data available")

    # B. Goals
    print("\n--- B. GOALS ---")
    goal_results = report_goals(merged)
    for label in ["1plus_goals", "2plus_goals", "3plus_goals"]:
        if label in goal_results:
            m = goal_results[label]
            print(f"  {label}: Brier={m['brier_score']:.4f}  BSS={m['bss']:.1%}  "
                  f"Acc={m['accuracy']:.1%}  BaseRate={m['base_rate']:.1%}  n={m['n']}")
            if m.get("hit_rate_p70") is not None:
                print(f"    P>=70%: {m['hit_rate_p70']:.1%} hit rate ({m['n_confident_p70']} calls)")
    if "mae" in goal_results:
        m = goal_results["mae"]
        print(f"  Goals MAE: {m['goals_mae']:.4f} (baseline {m['baseline_mae']:.4f}, "
              f"{m['improvement_pct']:+.1f}% improvement)")

    # C. Disposals
    print("\n--- C. DISPOSALS ---")
    disp_results = report_disposals(merged)
    for label in ["10plus_disp", "15plus_disp", "20plus_disp", "25plus_disp", "30plus_disp"]:
        if label in disp_results:
            m = disp_results[label]
            print(f"  {label}: Brier={m['brier_score']:.4f}  BSS={m['bss']:.1%}  "
                  f"Acc={m['accuracy']:.1%}  BaseRate={m['base_rate']:.1%}  n={m['n']}")
            if m.get("hit_rate_p70") is not None:
                print(f"    P>=70%: {m['hit_rate_p70']:.1%} hit rate ({m['n_confident_p70']} calls)")
    if "mae" in disp_results:
        m = disp_results["mae"]
        print(f"  Disposals MAE: {m['disposals_mae']:.4f} (baseline {m['baseline_mae']:.4f}, "
              f"{m['improvement_pct']:+.1f}% improvement)")

    # D. Marks
    print("\n--- D. MARKS ---")
    marks_results = report_marks(merged)
    for label in ["3plus_mk", "5plus_mk", "7plus_mk"]:
        if label in marks_results:
            m = marks_results[label]
            print(f"  {label}: Brier={m['brier_score']:.4f}  BSS={m['bss']:.1%}  "
                  f"Acc={m['accuracy']:.1%}  BaseRate={m['base_rate']:.1%}  n={m['n']}")
            if m.get("hit_rate_p70") is not None:
                print(f"    P>=70%: {m['hit_rate_p70']:.1%} hit rate ({m['n_confident_p70']} calls)")
    if "mae" in marks_results:
        m = marks_results["mae"]
        print(f"  Marks MAE: {m['marks_mae']:.4f} (baseline {m['baseline_mae']:.4f}, "
              f"{m['improvement_pct']:+.1f}% improvement)")

    # E. Summary
    print("\n--- E. SUMMARY ---")
    print(f"  Total predictions: {n_preds}")
    print(f"  Rounds: {n_rounds}")
    print(f"  Run ID: {run_id}")

    # Count bettable calls (P >= 0.6 on any threshold)
    bettable = 0
    for col in merged.columns:
        if col.startswith("p_") and col.endswith(("_goals", "_disp", "_mk")):
            bettable += int((merged[col] >= 0.6).sum())
    if bettable > 0:
        print(f"  Bettable calls (P>=60%): {bettable}")

    # F. Save experiment JSON
    experiment = {
        "label": args.label,
        "season": year,
        "run_id": run_id,
        "n_predictions": n_preds,
        "n_rounds": n_rounds,
        "game_winner": gw_results,
        "thresholds": {},
        "disposal_thresholds": {},
        "marks_thresholds": {},
        "mae": goal_results.get("mae", {}),
        "disposal_mae": disp_results.get("mae", {}),
        "marks_mae": marks_results.get("mae", {}),
    }

    # Goal thresholds
    for k in ["1plus_goals", "2plus_goals", "3plus_goals"]:
        if k in goal_results:
            experiment["thresholds"][k] = goal_results[k]

    # Disposal thresholds
    for k in ["10plus_disp", "15plus_disp", "20plus_disp", "25plus_disp", "30plus_disp"]:
        if k in disp_results:
            experiment["disposal_thresholds"][k] = disp_results[k]

    # Marks thresholds
    for k in ["3plus_mk", "5plus_mk", "7plus_mk"]:
        if k in marks_results:
            experiment["marks_thresholds"][k] = marks_results[k]

    out_path = config.EXPERIMENTS_DIR / f"{args.label}.json"
    config.ensure_dirs()
    with open(out_path, "w") as f:
        json.dump(experiment, f, indent=2, default=str)
    print(f"\n  Saved experiment: {out_path}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
