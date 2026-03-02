"""
Build BASELINE v2.0 report from sequential backtest results.
Computes all metrics and saves to baseline_v2.json.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import poisson as poisson_dist
from sklearn.metrics import (
    brier_score_loss, log_loss, mean_absolute_error, roc_auc_score,
)

import config
from store import LearningStore

YEAR = 2025
store = LearningStore(base_dir=config.SEQUENTIAL_DIR)

# ── Load all predictions, outcomes, game predictions ──────────────────

all_preds = store.load_predictions(year=YEAR)
all_outcomes = store.load_outcomes(year=YEAR)
all_game_preds = store.load_game_predictions(year=YEAR)

# Merge predictions + outcomes
merged = all_preds.merge(
    all_outcomes, on=["player", "team", "match_id"], how="inner"
)
print(f"Total player-match predictions: {len(merged)}")

# Load team match data for game actuals
tm = pd.read_parquet(config.BASE_STORE_DIR / "team_matches.parquet")
tm25_home = tm[(tm["year"] == YEAR) & (tm["is_home"])].copy()

# Load feature matrix for sample_weight
feat_df = pd.read_parquet(config.FEATURES_DIR / "feature_matrix.parquet")
feat_25 = feat_df[feat_df["year"] == YEAR].copy()

# ── Helpers ───────────────────────────────────────────────────────────

def safe(v, decimals=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), decimals)


def compute_threshold_metrics(pred_probs, actual_binary, name):
    """Brier, log loss, hit rate at P>=0.50, base rate."""
    mask = np.isfinite(pred_probs) & np.isfinite(actual_binary)
    p = np.asarray(pred_probs[mask], dtype=float)
    y = np.asarray(actual_binary[mask], dtype=float)
    n = len(p)
    if n < 10:
        return None

    brier = float(np.mean((p - y) ** 2))
    p_clip = np.clip(p, 1e-15, 1 - 1e-15)
    ll = float(np.mean(-y * np.log(p_clip) - (1 - y) * np.log(1 - p_clip)))
    base_rate = float(y.mean())

    # Hit rate at P>=0.50: of predictions where we said >=50%, how often correct?
    confident_mask = p >= 0.50
    if confident_mask.sum() > 0:
        hit_rate_50 = float(y[confident_mask].mean())
        n_confident = int(confident_mask.sum())
    else:
        hit_rate_50 = None
        n_confident = 0

    return {
        "brier_score": safe(brier),
        "log_loss": safe(ll),
        "hit_rate_p50": safe(hit_rate_50),
        "n_confident_p50": n_confident,
        "n": n,
        "base_rate": safe(base_rate),
    }


# ── 1. Brier scores for all thresholds ───────────────────────────────

print("\n" + "=" * 70)
print("BASELINE v2.0 — COMPREHENSIVE METRICS")
print("=" * 70)

actual_goals = merged["actual_goals"].values
actual_disp = merged["actual_disposals"].values
pred_goals = merged["predicted_goals"].values

thresholds = {}

# Goal thresholds
# 1+ goals
p_1plus = merged["p_scorer"].values.astype(float)
y_1plus = (actual_goals >= 1).astype(int)
thresholds["1plus_goals"] = compute_threshold_metrics(p_1plus, y_1plus, "1+ Goals")

# 2+ goals: P(>=2) = 1 - P(0) - P(1)
p_2plus = 1.0 - merged["p_goals_0"].values.astype(float) - merged["p_goals_1"].values.astype(float)
p_2plus = np.clip(p_2plus, 0, 1)
y_2plus = (actual_goals >= 2).astype(int)
thresholds["2plus_goals"] = compute_threshold_metrics(p_2plus, y_2plus, "2+ Goals")

# 3+ goals: P(>=3) = P(>=2) - P(2)
p_3plus = p_2plus - merged["p_goals_2"].values.astype(float)
p_3plus = np.clip(p_3plus, 0, 1)
y_3plus = (actual_goals >= 3).astype(int)
thresholds["3plus_goals"] = compute_threshold_metrics(p_3plus, y_3plus, "3+ Goals")

# Disposal thresholds
for t in [15, 20, 25, 30]:
    col = f"p_{t}plus_disp"
    if col in merged.columns:
        p_disp = merged[col].values.astype(float)
        y_disp = (actual_disp >= t).astype(int)
        thresholds[f"{t}plus_disp"] = compute_threshold_metrics(p_disp, y_disp, f"{t}+ Disp")

print("\n1. BRIER SCORES & LOG LOSS")
print("-" * 70)
print(f"  {'Threshold':<16s} {'Brier':>8s} {'LogLoss':>8s} {'HitRate@50':>11s} {'n@50':>6s} {'BaseRate':>9s}")
for name, m in thresholds.items():
    if m is None:
        continue
    hr = f"{m['hit_rate_p50']:.3f}" if m['hit_rate_p50'] is not None else "N/A"
    print(f"  {name:<16s} {m['brier_score']:>8.4f} {m['log_loss']:>8.4f} {hr:>11s} {m['n_confident_p50']:>6d} {m['base_rate']:>9.4f}")

# ── 4. Scorer AUC ────────────────────────────────────────────────────

print("\n4. SCORER AUC")
print("-" * 70)
scorer_auc = roc_auc_score(y_1plus, p_1plus)
print(f"  Overall Scorer AUC: {scorer_auc:.4f}")

# Per-round AUC
round_col = "round"
round_aucs = {}
for rnd in sorted(merged[round_col].unique()):
    mask = merged[round_col] == rnd
    y_r = (actual_goals[mask] >= 1).astype(int)
    p_r = p_1plus[mask]
    if len(np.unique(y_r)) > 1:
        round_aucs[int(rnd)] = safe(roc_auc_score(y_r, p_r))

# ── 5. Overall MAE and weighted MAE ─────────────────────────────────

print("\n5. MAE METRICS")
print("-" * 70)

goals_mae = mean_absolute_error(actual_goals, pred_goals)
goals_rmse = float(np.sqrt(np.mean((actual_goals - pred_goals) ** 2)))
behinds_mae = mean_absolute_error(
    merged["actual_behinds"].values, merged["predicted_behinds"].values
)
disp_mae = mean_absolute_error(actual_disp, merged["predicted_disposals"].values)

# Weighted MAE: weight by sample_weight from feature matrix
weights = None
if "sample_weight" in feat_25.columns:
    weight_lookup = feat_25.set_index(["player", "team", "match_id"])["sample_weight"]
    merged_key = merged.set_index(["player", "team", "match_id"])
    joined = merged_key.join(weight_lookup, how="left")
    weights = joined["sample_weight"].fillna(1.0).values
    weighted_goals_mae = float(
        np.average(np.abs(actual_goals - pred_goals), weights=weights)
    )
else:
    weighted_goals_mae = goals_mae

# Baseline MAE
if "career_goal_avg" in merged.columns:
    baseline = merged["career_goal_avg"].fillna(0).values
    baseline_mae = mean_absolute_error(actual_goals, baseline)
    improvement_pct = (baseline_mae - goals_mae) / baseline_mae * 100
else:
    baseline_mae = None
    improvement_pct = None

print(f"  Goals MAE:          {goals_mae:.4f}")
print(f"  Goals RMSE:         {goals_rmse:.4f}")
print(f"  Weighted Goals MAE: {weighted_goals_mae:.4f}")
print(f"  Behinds MAE:        {behinds_mae:.4f}")
print(f"  Disposals MAE:      {disp_mae:.4f}")
if baseline_mae:
    print(f"  Baseline MAE:       {baseline_mae:.4f}")
    print(f"  Improvement:        {improvement_pct:+.1f}%")

# ── 6. Game winner accuracy and margin MAE ───────────────────────────

print("\n6. GAME WINNER")
print("-" * 70)

game_metrics = {}
if not all_game_preds.empty:
    # Merge game predictions with actuals
    game_merged = all_game_preds.merge(
        tm25_home[["match_id", "team", "opponent", "margin", "score", "opp_score"]],
        left_on=["match_id", "home_team"],
        right_on=["match_id", "team"],
        how="inner",
    )

    if not game_merged.empty:
        actual_home_win = (game_merged["margin"] > 0).astype(int)
        pred_home_win = (game_merged["home_win_prob"] > 0.5).astype(int)
        accuracy = float((pred_home_win == actual_home_win).mean())
        n_games = len(game_merged)

        # Margin MAE
        margin_mae = mean_absolute_error(
            game_merged["margin"].values, game_merged["predicted_margin"].values
        )

        # AUC
        try:
            game_auc = roc_auc_score(actual_home_win, game_merged["home_win_prob"])
        except ValueError:
            game_auc = None

        # Confident predictions (prob >= 0.65)
        confident_mask = (game_merged["home_win_prob"] >= 0.65) | (game_merged["home_win_prob"] <= 0.35)
        if confident_mask.sum() > 0:
            confident_acc = float(
                (pred_home_win[confident_mask] == actual_home_win[confident_mask]).mean()
            )
            n_confident = int(confident_mask.sum())
        else:
            confident_acc = None
            n_confident = 0

        game_metrics = {
            "accuracy": safe(accuracy),
            "margin_mae": safe(margin_mae, 1),
            "auc": safe(game_auc),
            "n_games": n_games,
            "confident_accuracy": safe(confident_acc),
            "n_confident_games": n_confident,
        }
        print(f"  Accuracy:           {accuracy:.3f} ({n_games} games)")
        print(f"  Margin MAE:         {margin_mae:.1f} points")
        if game_auc:
            print(f"  AUC:                {game_auc:.4f}")
        if confident_acc:
            print(f"  Confident acc (65%+): {confident_acc:.3f} ({n_confident} games)")
    else:
        print("  No matching game results found")
        game_metrics = {"note": "no matching data"}
else:
    print("  No game predictions found")
    game_metrics = {"note": "no game predictions"}

# ── 7. Learning effect ────────────────────────────────────────────────

print("\n7. LEARNING EFFECT")
print("-" * 70)

round_maes = {}
for rnd in sorted(merged[round_col].unique()):
    mask = merged[round_col] == rnd
    round_maes[int(rnd)] = safe(mean_absolute_error(actual_goals[mask], pred_goals[mask]))

rounds_list = sorted(round_maes.keys())
n_rounds = len(rounds_list)
half = n_rounds // 2
first_half_rounds = rounds_list[:half]
second_half_rounds = rounds_list[half:]

first_half_mae = float(np.mean([round_maes[r] for r in first_half_rounds]))
second_half_mae = float(np.mean([round_maes[r] for r in second_half_rounds]))
learning_effect = (first_half_mae - second_half_mae) / first_half_mae * 100

print(f"  Rounds: {n_rounds}")
print(f"  First-half MAE  (R{first_half_rounds[0]}-R{first_half_rounds[-1]}):  {first_half_mae:.4f}")
print(f"  Second-half MAE (R{second_half_rounds[0]}-R{second_half_rounds[-1]}): {second_half_mae:.4f}")
print(f"  Learning effect: {learning_effect:+.1f}%")

# ── 8. Calibration table ─────────────────────────────────────────────

print("\n8. CALIBRATION TABLE — P(1+ Goals)")
print("-" * 70)

cal_bins = []
bin_edges = np.arange(0, 1.01, 0.10)
for i in range(len(bin_edges) - 1):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    mask = (p_1plus >= lo) & (p_1plus < hi) if i < len(bin_edges) - 2 else (p_1plus >= lo) & (p_1plus <= hi)
    n = int(mask.sum())
    if n == 0:
        cal_bins.append({
            "bin": f"{lo:.0%}-{hi:.0%}",
            "predicted_mean": None,
            "actual_rate": None,
            "n": 0,
        })
        continue
    pred_mean = float(p_1plus[mask].mean())
    actual_rate = float(y_1plus[mask].mean())
    cal_bins.append({
        "bin": f"{int(lo*100)}-{int(hi*100)}%",
        "predicted_mean": safe(pred_mean),
        "actual_rate": safe(actual_rate),
        "n": n,
        "gap": safe(abs(pred_mean - actual_rate)),
    })

print(f"  {'Bin':<12s} {'Predicted':>10s} {'Actual':>8s} {'Gap':>6s} {'Count':>7s}")
for b in cal_bins:
    if b["n"] == 0:
        print(f"  {b['bin']:<12s} {'—':>10s} {'—':>8s} {'—':>6s} {b['n']:>7d}")
    else:
        print(f"  {b['bin']:<12s} {b['predicted_mean']:>10.4f} {b['actual_rate']:>8.4f} {b['gap']:>6.4f} {b['n']:>7d}")

# Overall ECE
ece_vals = [b["gap"] * b["n"] for b in cal_bins if b["n"] > 0 and b["gap"] is not None]
total_n = sum(b["n"] for b in cal_bins if b["n"] > 0)
ece = sum(ece_vals) / total_n if total_n > 0 else None
print(f"\n  ECE (weighted): {ece:.4f}")

# ── 9. 3-game spot check ─────────────────────────────────────────────

print("\n9. SPOT CHECK — 3 Games with Confident Prediction Accuracy")
print("-" * 70)

spot_checks = [
    {"round": 2, "home": "Adelaide", "away": "St Kilda"},
    {"round": 13, "home": "Brisbane Lions", "away": "Essendon"},
    {"round": 22, "home": "Fremantle", "away": "Carlton"},
]

spot_results = []
for game in spot_checks:
    rnd = game["round"]
    home = game["home"]
    away = game["away"]

    # Get player predictions for this match
    mask = (
        (merged[round_col] == rnd)
        & (merged["team"].isin([home, away]))
        & (merged["opponent"].isin([home, away]))
    )
    game_preds_df = merged[mask].copy()

    if game_preds_df.empty:
        print(f"\n  R{rnd} {home} vs {away}: NO DATA")
        spot_results.append({"round": rnd, "home": home, "away": away, "note": "no data"})
        continue

    # Game result
    tm_game = tm25_home[
        (tm25_home["round_number"] == rnd)
        & (tm25_home["team"] == home)
    ]
    if not tm_game.empty:
        actual_margin = int(tm_game.iloc[0]["margin"])
        actual_score = f"{int(tm_game.iloc[0]['score'])}-{int(tm_game.iloc[0]['opp_score'])}"
    else:
        actual_margin = None
        actual_score = "?"

    n_players = len(game_preds_df)
    game_goals_mae = mean_absolute_error(
        game_preds_df["actual_goals"].values,
        game_preds_df["predicted_goals"].values,
    )

    # Confident predictions: P(scorer) >= 0.70 or P(scorer) <= 0.10
    p_sc = game_preds_df["p_scorer"].values
    ag = game_preds_df["actual_goals"].values

    confident_scorer = p_sc >= 0.70
    confident_non_scorer = p_sc <= 0.10
    confident_any = confident_scorer | confident_non_scorer

    if confident_any.sum() > 0:
        # For scorers: did they score >= 1?
        # For non-scorers: did they score == 0?
        correct = np.zeros(len(p_sc), dtype=bool)
        correct[confident_scorer] = ag[confident_scorer] >= 1
        correct[confident_non_scorer] = ag[confident_non_scorer] == 0
        confident_acc = float(correct[confident_any].mean())
        n_conf = int(confident_any.sum())
    else:
        confident_acc = None
        n_conf = 0

    winner_str = home if actual_margin and actual_margin > 0 else away

    result = {
        "round": rnd,
        "home": home,
        "away": away,
        "actual_result": actual_score,
        "actual_margin": actual_margin,
        "winner": winner_str,
        "n_players": n_players,
        "goals_mae": safe(game_goals_mae),
        "confident_predictions": n_conf,
        "confident_accuracy": safe(confident_acc),
    }
    spot_results.append(result)

    print(f"\n  R{rnd} {home} vs {away}  ({actual_score}, margin={actual_margin:+d})")
    print(f"    Players: {n_players}  Goals MAE: {game_goals_mae:.3f}")
    print(f"    Confident predictions (P>=0.70 or P<=0.10): {n_conf}")
    if confident_acc is not None:
        print(f"    Confident accuracy: {confident_acc:.3f}")

    # Top predictions for this game
    game_preds_df = game_preds_df.sort_values("p_scorer", ascending=False)
    print(f"    Top 5 scorer predictions:")
    for _, row in game_preds_df.head(5).iterrows():
        hit = "✓" if row["actual_goals"] >= 1 else "✗"
        print(f"      {row['player']:<25s} {row['team']:<18s} "
              f"P={row['p_scorer']:.2f} pred={row['predicted_goals']:.1f} "
              f"actual={int(row['actual_goals'])} {hit}")

# ── Game winner spot check ────────────────────────────────────────────

print("\n  Game Winner Predictions for spot check games:")
for game in spot_checks:
    rnd = game["round"]
    gp_rnd = store.load_game_predictions(year=YEAR, round_num=rnd)
    if gp_rnd.empty:
        print(f"    R{rnd}: No game prediction")
        continue
    gp_match = gp_rnd[
        (gp_rnd["home_team"] == game["home"]) | (gp_rnd["away_team"] == game["home"])
    ]
    if gp_match.empty:
        print(f"    R{rnd}: Game not found in predictions")
        continue
    row = gp_match.iloc[0]
    tm_game = tm25_home[
        (tm25_home["round_number"] == rnd) & (tm25_home["team"] == game["home"])
    ]
    actual_winner = game["home"] if (not tm_game.empty and tm_game.iloc[0]["margin"] > 0) else game["away"]
    hit = "✓" if row["predicted_winner"] == actual_winner else "✗"
    print(f"    R{rnd:02d} {row['home_team']:<18s} vs {row['away_team']:<18s} "
          f"P(home)={row['home_win_prob']:.2f}  margin={row['predicted_margin']:+.0f}  "
          f"pred={row['predicted_winner']}  actual={actual_winner} {hit}")


# ══════════════════════════════════════════════════════════════════════
# BUILD JSON
# ══════════════════════════════════════════════════════════════════════

baseline = {
    "version": "2.0",
    "label": "BASELINE v2.0 — post feature pruning, career leakage fix, opponent concession fix, NaN handling, mixture CIs",
    "season": YEAR,
    "n_predictions": len(merged),
    "n_rounds": n_rounds,

    "thresholds": thresholds,

    "scorer_auc": {
        "overall": safe(scorer_auc),
        "per_round": round_aucs,
    },

    "mae": {
        "goals": safe(goals_mae),
        "goals_rmse": safe(goals_rmse),
        "goals_weighted": safe(weighted_goals_mae),
        "behinds": safe(behinds_mae),
        "disposals": safe(disp_mae),
        "baseline_goals": safe(baseline_mae),
        "improvement_pct": safe(improvement_pct, 1),
    },

    "game_winner": game_metrics,

    "learning_effect": {
        "first_half_mae": safe(first_half_mae),
        "second_half_mae": safe(second_half_mae),
        "learning_pct": safe(learning_effect, 1),
        "first_half_rounds": f"R{first_half_rounds[0]}-R{first_half_rounds[-1]}",
        "second_half_rounds": f"R{second_half_rounds[0]}-R{second_half_rounds[-1]}",
    },

    "calibration_1plus_goals": cal_bins,
    "calibration_ece": safe(ece),

    "round_detail": {
        r: {"mae": round_maes.get(r), "auc": round_aucs.get(r)}
        for r in rounds_list
    },

    "spot_checks": spot_results,
}

out_path = Path("baseline_v2.json")
with open(out_path, "w") as f:
    json.dump(baseline, f, indent=2, default=str)

print(f"\n{'=' * 70}")
print(f"Saved to {out_path.resolve()}")
print(f"{'=' * 70}")
