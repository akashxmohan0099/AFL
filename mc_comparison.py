#!/usr/bin/env python3
"""
Monte Carlo vs Direct Model — Sequential Backtest Comparison
=============================================================
Replays the 2025 sequential backtest producing both direct model and
Monte Carlo predictions for every player in every round, then compares
Brier scores across all targets with calibration analysis.

Usage:
    python mc_comparison.py [--year 2025] [--n-sims 10000]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

import config
from model import (
    AFLScoringModel, AFLDisposalModel, AFLMarksModel,
    AFLGameWinnerModel, MonteCarloSimulator, _prepare_features,
)


def run_comparison(year: int = 2025, n_sims: int = 10000):
    """Run sequential backtest with both direct and MC predictions."""

    from metrics import load_feature_matrix
    tm_path = config.BASE_STORE_DIR / "team_matches.parquet"

    feature_df, feature_cols = load_feature_matrix()

    team_match_df = pd.read_parquet(tm_path)
    team_match_df["date"] = pd.to_datetime(team_match_df["date"])

    rounds = sorted(feature_df[feature_df["year"] == year]["round_number"].unique())

    print(f"\nSequential MC Comparison — {year}")
    print(f"  Rounds: {len(rounds)}, Sims: {n_sims:,}")
    print("=" * 80)

    # Accumulators: list of dicts per player-round
    all_direct = []
    all_mc = []
    all_actuals = []
    all_margins = []  # for blowout analysis

    total_start = time.time()

    for rnd in rounds:
        rnd_int = int(rnd)
        rnd_start = time.time()

        # 1. SPLIT
        train_mask = (
            (feature_df["year"] < year)
            | ((feature_df["year"] == year) & (feature_df["round_number"] < rnd))
        )
        train_df = feature_df[train_mask].copy()
        test_df = feature_df[
            (feature_df["year"] == year) & (feature_df["round_number"] == rnd)
        ].copy()

        if len(train_df) < 50 or test_df.empty:
            continue

        # 2. TRAIN
        scoring_model = AFLScoringModel()
        scoring_model.train_backtest(train_df, feature_cols)

        disposal_model = AFLDisposalModel(distribution=config.DISPOSAL_DISTRIBUTION)
        disposal_model.train_backtest(train_df, feature_cols)

        # Game winner model
        game_preds = pd.DataFrame()
        winner_model = AFLGameWinnerModel()
        tm_train = team_match_df[
            (team_match_df["year"] < year)
            | ((team_match_df["year"] == year) & (team_match_df["round_number"] < rnd))
        ].copy()
        tm_round = team_match_df[
            (team_match_df["year"] == year) & (team_match_df["round_number"] == rnd)
        ].copy()

        if len(tm_train) >= 20 and not tm_round.empty:
            try:
                winner_model.train_backtest(tm_train)
                game_preds = winner_model.predict_with_margin(
                    pd.concat([tm_train, tm_round], ignore_index=True)
                )
                # Filter to this round's matches
                round_match_ids = set(tm_round["match_id"].unique())
                game_preds = game_preds[game_preds["match_id"].isin(round_match_ids)]
            except Exception:
                pass

        # 3. PREDICT — direct model
        scoring_preds = scoring_model.predict_distributions(test_df, feature_cols=feature_cols)
        disposal_preds = disposal_model.predict_distributions(test_df, feature_cols=feature_cols)

        # Merge
        join_cols = ["player", "team", "match_id"]
        disp_cols = [c for c in disposal_preds.columns
                     if c not in scoring_preds.columns or c in join_cols]
        merged = scoring_preds.merge(disposal_preds[disp_cols], on=join_cols, how="left")

        # Add is_home for MC
        if "is_home" in test_df.columns:
            merged["is_home"] = test_df["is_home"].values

        # 4. MC SIMULATION
        mc_sim = MonteCarloSimulator(
            scoring_model=scoring_model,
            disposal_model=disposal_model,
        )
        mc_sim.estimate_correlation_factors(train_df, team_match_df)
        mc_results = mc_sim.simulate_round(merged, game_preds_df=game_preds, n_sims=n_sims)

        # 5. COLLECT
        n_players = len(test_df)

        # Direct model predictions
        direct_row = {
            "p_1plus_goals": merged["p_1plus_goals"].values if "p_1plus_goals" in merged.columns
                else merged.get("p_scorer", pd.Series(np.nan, index=merged.index)).values,
            "p_2plus_goals": merged["p_2plus_goals"].values if "p_2plus_goals" in merged.columns
                else np.full(n_players, np.nan),
            "p_3plus_goals": merged["p_3plus_goals"].values if "p_3plus_goals" in merged.columns
                else np.full(n_players, np.nan),
            "p_15plus_disp": merged["p_15plus_disp"].values if "p_15plus_disp" in merged.columns
                else np.full(n_players, np.nan),
            "p_20plus_disp": merged["p_20plus_disp"].values if "p_20plus_disp" in merged.columns
                else np.full(n_players, np.nan),
            "p_25plus_disp": merged["p_25plus_disp"].values if "p_25plus_disp" in merged.columns
                else np.full(n_players, np.nan),
            "p_30plus_disp": merged["p_30plus_disp"].values if "p_30plus_disp" in merged.columns
                else np.full(n_players, np.nan),
        }
        all_direct.append(direct_row)

        # MC predictions
        mc_row = {
            "p_1plus_goals": mc_results["mc_p_1plus_goals"].values,
            "p_2plus_goals": mc_results["mc_p_2plus_goals"].values,
            "p_3plus_goals": mc_results["mc_p_3plus_goals"].values,
            "p_15plus_disp": mc_results["mc_p_15plus_disp"].values,
            "p_20plus_disp": mc_results["mc_p_20plus_disp"].values,
            "p_25plus_disp": mc_results["mc_p_25plus_disp"].values,
            "p_30plus_disp": mc_results["mc_p_30plus_disp"].values,
        }
        all_mc.append(mc_row)

        # Actuals
        actual_row = {
            "GL": test_df["GL"].values,
            "DI": test_df["DI"].values,
            "match_id": test_df["match_id"].values,
            "team": test_df["team"].values,
        }
        all_actuals.append(actual_row)

        # Team margins for blowout analysis
        if not tm_round.empty:
            margin_map = tm_round.set_index(["match_id", "team"])["margin"].to_dict()
            margins = np.array([
                margin_map.get((mid, t), 0)
                for mid, t in zip(test_df["match_id"].values, test_df["team"].values)
            ])
        else:
            margins = np.zeros(n_players)
        all_margins.append(margins)

        elapsed = time.time() - rnd_start
        print(f"  R{rnd_int:02d}  n={n_players:<4d} ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start
    print(f"\n  Total: {total_elapsed:.1f}s")

    # ── ANALYSIS ──────────────────────────────────────────────────────────────
    # Concatenate all rounds
    def concat_field(lst, field):
        return np.concatenate([d[field] for d in lst])

    targets = [
        ("Goals 1+", "p_1plus_goals", "GL", 1),
        ("Goals 2+", "p_2plus_goals", "GL", 2),
        ("Goals 3+", "p_3plus_goals", "GL", 3),
        ("Disp 15+", "p_15plus_disp", "DI", 15),
        ("Disp 20+", "p_20plus_disp", "DI", 20),
        ("Disp 25+", "p_25plus_disp", "DI", 25),
        ("Disp 30+", "p_30plus_disp", "DI", 30),
    ]

    all_margins_flat = np.concatenate(all_margins)

    print(f"\n{'=' * 80}")
    print(f"  BRIER SCORE COMPARISON — Direct Model vs Monte Carlo")
    print(f"{'=' * 80}")
    print(f"{'Target':<12} {'Direct':>10} {'MC':>10} {'Delta':>10} {'Winner':>10} "
          f"{'BSS_D':>8} {'BSS_MC':>8}")
    print("-" * 80)

    results = {}
    mc_wins = 0
    direct_wins = 0

    for name, pred_col, actual_col, threshold in targets:
        d_preds = concat_field(all_direct, pred_col)
        mc_preds = concat_field(all_mc, pred_col)
        actuals = (concat_field(all_actuals, actual_col) >= threshold).astype(int)

        # Skip NaN
        valid = ~(np.isnan(d_preds) | np.isnan(mc_preds))
        d_preds = d_preds[valid]
        mc_preds = mc_preds[valid]
        actuals = actuals[valid]

        d_brier = brier_score_loss(actuals, np.clip(d_preds, 0, 1))
        mc_brier = brier_score_loss(actuals, np.clip(mc_preds, 0, 1))
        delta = mc_brier - d_brier
        base_rate = actuals.mean()
        naive_brier = base_rate * (1 - base_rate)
        d_bss = 1 - d_brier / naive_brier if naive_brier > 0 else 0
        mc_bss = 1 - mc_brier / naive_brier if naive_brier > 0 else 0

        winner = "MC" if mc_brier < d_brier else "Direct"
        if mc_brier < d_brier:
            mc_wins += 1
        else:
            direct_wins += 1

        print(f"{name:<12} {d_brier:>10.4f} {mc_brier:>10.4f} {delta:>+10.4f} "
              f"{winner:>10} {d_bss:>7.1%} {mc_bss:>7.1%}")

        results[name] = {
            "direct_brier": round(float(d_brier), 6),
            "mc_brier": round(float(mc_brier), 6),
            "delta": round(float(delta), 6),
            "winner": winner,
            "direct_bss": round(float(d_bss), 4),
            "mc_bss": round(float(mc_bss), 4),
            "n_samples": int(len(actuals)),
            "base_rate": round(float(base_rate), 4),
        }

    print(f"\n  MC wins: {mc_wins}/{len(targets)}, Direct wins: {direct_wins}/{len(targets)}")

    # ── BLOWOUT ANALYSIS ──────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  BLOWOUT GAME CALIBRATION (|margin| >= 40)")
    print(f"{'=' * 80}")

    blowout_mask = np.abs(all_margins_flat) >= 40
    close_mask = np.abs(all_margins_flat) <= 12
    n_blowout = blowout_mask.sum()
    n_close = close_mask.sum()

    print(f"  Blowout games: {n_blowout} player-rows, Close games: {n_close} player-rows\n")
    print(f"{'Target':<12} {'Subset':<10} {'Direct':>10} {'MC':>10} {'Delta':>10}")
    print("-" * 55)

    for name, pred_col, actual_col, threshold in targets:
        d_preds = concat_field(all_direct, pred_col)
        mc_preds = concat_field(all_mc, pred_col)
        actuals = (concat_field(all_actuals, actual_col) >= threshold).astype(int)
        valid = ~(np.isnan(d_preds) | np.isnan(mc_preds))

        for subset_name, mask in [("Blowout", blowout_mask & valid),
                                   ("Close", close_mask & valid)]:
            if mask.sum() < 30:
                continue
            d_b = brier_score_loss(actuals[mask], np.clip(d_preds[mask], 0, 1))
            mc_b = brier_score_loss(actuals[mask], np.clip(mc_preds[mask], 0, 1))
            delta = mc_b - d_b
            print(f"{name:<12} {subset_name:<10} {d_b:>10.4f} {mc_b:>10.4f} {delta:>+10.4f}")

        results.setdefault(f"{name}_blowout", {})
        results.setdefault(f"{name}_close", {})
        for subset_name, mask in [("blowout", blowout_mask & valid),
                                   ("close", close_mask & valid)]:
            if mask.sum() < 30:
                continue
            d_b = brier_score_loss(actuals[mask], np.clip(d_preds[mask], 0, 1))
            mc_b = brier_score_loss(actuals[mask], np.clip(mc_preds[mask], 0, 1))
            results[f"{name}_{subset_name}"] = {
                "direct_brier": round(float(d_b), 6),
                "mc_brier": round(float(mc_b), 6),
                "n": int(mask.sum()),
            }

    # ── CALIBRATION BY PROBABILITY BIN ────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  CALIBRATION BY PROBABILITY BIN — Goals 1+")
    print(f"{'=' * 80}")

    d_preds_g1 = concat_field(all_direct, "p_1plus_goals")
    mc_preds_g1 = concat_field(all_mc, "p_1plus_goals")
    actuals_g1 = (concat_field(all_actuals, "GL") >= 1).astype(int)
    valid_g1 = ~(np.isnan(d_preds_g1) | np.isnan(mc_preds_g1))

    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]

    print(f"{'Bin':<12} {'N':>6} {'Actual':>8} {'Direct':>8} {'D_Gap':>8} "
          f"{'MC':>8} {'MC_Gap':>8} {'Better':>8}")
    print("-" * 75)

    cal_results = []
    for lo, hi in bins:
        d_mask = valid_g1 & (d_preds_g1 >= lo) & (d_preds_g1 < hi)
        mc_mask = valid_g1 & (mc_preds_g1 >= lo) & (mc_preds_g1 < hi)

        # Use direct bins for both to keep consistent grouping
        mask = d_mask
        if mask.sum() < 20:
            continue

        actual_rate = actuals_g1[mask].mean()
        d_mean = d_preds_g1[mask].mean()
        mc_mean = mc_preds_g1[mask].mean()
        d_gap = abs(d_mean - actual_rate)
        mc_gap = abs(mc_mean - actual_rate)
        better = "MC" if mc_gap < d_gap else "Direct"

        label = f"[{lo:.0%}-{hi:.0%})"
        print(f"{label:<12} {mask.sum():>6} {actual_rate:>7.1%} {d_mean:>7.1%} "
              f"{d_gap:>7.3f} {mc_mean:>7.1%} {mc_gap:>7.3f} {better:>8}")

        cal_results.append({
            "bin": label, "n": int(mask.sum()),
            "actual": round(float(actual_rate), 4),
            "direct_mean": round(float(d_mean), 4),
            "mc_mean": round(float(mc_mean), 4),
            "direct_gap": round(float(d_gap), 4),
            "mc_gap": round(float(mc_gap), 4),
        })

    results["calibration_bins_goals_1plus"] = cal_results

    # ── CALIBRATION BY BIN — Disposals 20+ ────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  CALIBRATION BY PROBABILITY BIN — Disposals 20+")
    print(f"{'=' * 80}")

    d_preds_d20 = concat_field(all_direct, "p_20plus_disp")
    mc_preds_d20 = concat_field(all_mc, "p_20plus_disp")
    actuals_d20 = (concat_field(all_actuals, "DI") >= 20).astype(int)
    valid_d20 = ~(np.isnan(d_preds_d20) | np.isnan(mc_preds_d20))

    print(f"{'Bin':<12} {'N':>6} {'Actual':>8} {'Direct':>8} {'D_Gap':>8} "
          f"{'MC':>8} {'MC_Gap':>8} {'Better':>8}")
    print("-" * 75)

    cal_results_d20 = []
    for lo, hi in bins:
        mask = valid_d20 & (d_preds_d20 >= lo) & (d_preds_d20 < hi)
        if mask.sum() < 20:
            continue

        actual_rate = actuals_d20[mask].mean()
        d_mean = d_preds_d20[mask].mean()
        mc_mean = mc_preds_d20[mask].mean()
        d_gap = abs(d_mean - actual_rate)
        mc_gap = abs(mc_mean - actual_rate)
        better = "MC" if mc_gap < d_gap else "Direct"

        label = f"[{lo:.0%}-{hi:.0%})"
        print(f"{label:<12} {mask.sum():>6} {actual_rate:>7.1%} {d_mean:>7.1%} "
              f"{d_gap:>7.3f} {mc_mean:>7.1%} {mc_gap:>7.3f} {better:>8}")

        cal_results_d20.append({
            "bin": label, "n": int(mask.sum()),
            "actual": round(float(actual_rate), 4),
            "direct_mean": round(float(d_mean), 4),
            "mc_mean": round(float(mc_mean), 4),
            "direct_gap": round(float(d_gap), 4),
            "mc_gap": round(float(mc_gap), 4),
        })

    results["calibration_bins_disp_20plus"] = cal_results_d20

    # ── SAVE ──────────────────────────────────────────────────────────────────
    out_path = config.EXPERIMENTS_DIR / "mc_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results → {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="MC vs Direct Model Comparison")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--n-sims", type=int, default=10000, dest="n_sims")
    args = parser.parse_args()
    run_comparison(year=args.year, n_sims=args.n_sims)


if __name__ == "__main__":
    main()
