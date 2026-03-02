"""
Build BASELINE v2.1 (with odds) report from sequential backtest results.
Compare against v2.0 and save to baseline_v2.1_with_odds.json.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, roc_auc_score

import config
from store import LearningStore

YEAR = 2025
store = LearningStore(base_dir=config.SEQUENTIAL_DIR)

# ── Load all predictions, outcomes, game predictions ──────────────────

all_preds = store.load_predictions(year=YEAR)
all_outcomes = store.load_outcomes(year=YEAR)
all_game_preds = store.load_game_predictions(year=YEAR)

merged = all_preds.merge(
    all_outcomes, on=["player", "team", "match_id"], how="inner"
)
print(f"Total player-match predictions: {len(merged)}")

tm = pd.read_parquet(config.BASE_STORE_DIR / "team_matches.parquet")
tm25_home = tm[(tm["year"] == YEAR) & (tm["is_home"])].copy()

feat_df = pd.read_parquet(config.FEATURES_DIR / "feature_matrix.parquet")
feat_25 = feat_df[feat_df["year"] == YEAR].copy()


def safe(v, decimals=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), decimals)


def compute_threshold_metrics(pred_probs, actual_binary):
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

    result = {
        "brier_score": safe(brier),
        "log_loss": safe(ll),
        "n": n,
        "base_rate": safe(base_rate),
    }

    # Hit rates at P>=0.50 and P>=0.70
    for thresh, label in [(0.50, "p50"), (0.70, "p70")]:
        mask_t = p >= thresh
        if mask_t.sum() > 0:
            result[f"hit_rate_{label}"] = safe(float(y[mask_t].mean()))
            result[f"n_confident_{label}"] = int(mask_t.sum())
        else:
            result[f"hit_rate_{label}"] = None
            result[f"n_confident_{label}"] = 0

    return result


# ── Compute all metrics ───────────────────────────────────────────────

actual_goals = merged["actual_goals"].values
actual_disp = merged["actual_disposals"].values
pred_goals = merged["predicted_goals"].values
round_col = "round"

# Thresholds
thresholds = {}

p_1plus = merged["p_scorer"].values.astype(float)
y_1plus = (actual_goals >= 1).astype(int)
thresholds["1plus_goals"] = compute_threshold_metrics(p_1plus, y_1plus)

p_2plus = 1.0 - merged["p_goals_0"].values.astype(float) - merged["p_goals_1"].values.astype(float)
p_2plus = np.clip(p_2plus, 0, 1)
y_2plus = (actual_goals >= 2).astype(int)
thresholds["2plus_goals"] = compute_threshold_metrics(p_2plus, y_2plus)

p_3plus = p_2plus - merged["p_goals_2"].values.astype(float)
p_3plus = np.clip(p_3plus, 0, 1)
y_3plus = (actual_goals >= 3).astype(int)
thresholds["3plus_goals"] = compute_threshold_metrics(p_3plus, y_3plus)

for t in [15, 20, 25, 30]:
    col = f"p_{t}plus_disp"
    if col in merged.columns:
        p_disp = merged[col].values.astype(float)
        y_disp = (actual_disp >= t).astype(int)
        thresholds[f"{t}plus_disp"] = compute_threshold_metrics(p_disp, y_disp)

# Scorer AUC
scorer_auc = roc_auc_score(y_1plus, p_1plus)
round_aucs = {}
for rnd in sorted(merged[round_col].unique()):
    mask = merged[round_col] == rnd
    y_r = (actual_goals[mask] >= 1).astype(int)
    p_r = p_1plus[mask]
    if len(np.unique(y_r)) > 1:
        round_aucs[int(rnd)] = safe(roc_auc_score(y_r, p_r))

# MAE
goals_mae = mean_absolute_error(actual_goals, pred_goals)
goals_rmse = float(np.sqrt(np.mean((actual_goals - pred_goals) ** 2)))
behinds_mae = mean_absolute_error(merged["actual_behinds"].values, merged["predicted_behinds"].values)
disp_mae = mean_absolute_error(actual_disp, merged["predicted_disposals"].values)

weights = None
if "sample_weight" in feat_25.columns:
    weight_lookup = feat_25.set_index(["player", "team", "match_id"])["sample_weight"]
    merged_key = merged.set_index(["player", "team", "match_id"])
    joined = merged_key.join(weight_lookup, how="left")
    weights = joined["sample_weight"].fillna(1.0).values
    weighted_goals_mae = float(np.average(np.abs(actual_goals - pred_goals), weights=weights))
else:
    weighted_goals_mae = goals_mae

if "career_goal_avg" in merged.columns:
    baseline = merged["career_goal_avg"].fillna(0).values
    baseline_mae = mean_absolute_error(actual_goals, baseline)
    improvement_pct = (baseline_mae - goals_mae) / baseline_mae * 100
else:
    baseline_mae = None
    improvement_pct = None

# Game winner
game_metrics = {}
if not all_game_preds.empty:
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
        margin_mae = mean_absolute_error(game_merged["margin"].values, game_merged["predicted_margin"].values)
        try:
            game_auc = roc_auc_score(actual_home_win, game_merged["home_win_prob"])
        except ValueError:
            game_auc = None

        confident_mask = (game_merged["home_win_prob"] >= 0.65) | (game_merged["home_win_prob"] <= 0.35)
        if confident_mask.sum() > 0:
            confident_acc = float((pred_home_win[confident_mask] == actual_home_win[confident_mask]).mean())
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

# Learning effect
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

# Calibration
cal_bins = []
bin_edges = np.arange(0, 1.01, 0.10)
for i in range(len(bin_edges) - 1):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    mask = (p_1plus >= lo) & (p_1plus < hi) if i < len(bin_edges) - 2 else (p_1plus >= lo) & (p_1plus <= hi)
    n = int(mask.sum())
    if n == 0:
        cal_bins.append({"bin": f"{int(lo*100)}-{int(hi*100)}%", "predicted_mean": None, "actual_rate": None, "n": 0})
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

ece_vals = [b["gap"] * b["n"] for b in cal_bins if b["n"] > 0 and b["gap"] is not None]
total_n = sum(b["n"] for b in cal_bins if b["n"] > 0)
ece = sum(ece_vals) / total_n if total_n > 0 else None

# ══════════════════════════════════════════════════════════════════════
# BUILD JSON
# ══════════════════════════════════════════════════════════════════════

baseline = {
    "version": "2.1",
    "label": "BASELINE v2.1 — v2.0 + market odds features (175 features)",
    "season": YEAR,
    "n_predictions": len(merged),
    "n_rounds": n_rounds,
    "n_features": 175,

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
}

out_path = Path("baseline_v2.1_with_odds.json")
with open(out_path, "w") as f:
    json.dump(baseline, f, indent=2, default=str)
print(f"\nSaved to {out_path}")

# ══════════════════════════════════════════════════════════════════════
# COMPARISON vs v2.0
# ══════════════════════════════════════════════════════════════════════

with open("baseline_v2.json") as f:
    v20 = json.load(f)

v21 = baseline

print("\n" + "=" * 80)
print("COMPARISON: v2.0 (167 features) vs v2.1 (175 features, +odds)")
print("=" * 80)

# 1. Brier scores
print("\n1. BRIER SCORES & LOG LOSS")
print("-" * 80)
print(f"  {'Threshold':<16s} {'Brier v2.0':>10s} {'Brier v2.1':>10s} {'Delta':>8s} {'LL v2.0':>8s} {'LL v2.1':>8s} {'Delta':>8s}")

for name in ["1plus_goals", "2plus_goals", "3plus_goals", "15plus_disp", "20plus_disp", "25plus_disp", "30plus_disp"]:
    old = v20["thresholds"].get(name, {})
    new = v21["thresholds"].get(name, {})
    if not old or not new:
        continue
    ob, nb = old.get("brier_score"), new.get("brier_score")
    ol, nl = old.get("log_loss"), new.get("log_loss")
    if ob is not None and nb is not None:
        db = nb - ob
        dl = nl - ol if ol and nl else 0
        arrow_b = "better" if db < -0.001 else ("worse" if db > 0.001 else "same")
        print(f"  {name:<16s} {ob:>10.4f} {nb:>10.4f} {db:>+8.4f} {ol:>8.4f} {nl:>8.4f} {dl:>+8.4f}  ({arrow_b})")

# 2. Hit rates
print("\n2. HIT RATES AT P>=0.50 AND P>=0.70")
print("-" * 80)
print(f"  {'Threshold':<16s} {'HR50 v2.0':>10s} {'HR50 v2.1':>10s} {'Delta':>8s} {'HR70 v2.1':>10s} {'n@50':>6s} {'n@70':>6s}")

for name in ["1plus_goals", "2plus_goals", "3plus_goals", "15plus_disp", "20plus_disp", "25plus_disp"]:
    old = v20["thresholds"].get(name, {})
    new = v21["thresholds"].get(name, {})
    if not old or not new:
        continue
    old_hr50 = old.get("hit_rate_p50")
    new_hr50 = new.get("hit_rate_p50")
    new_hr70 = new.get("hit_rate_p70")
    n50 = new.get("n_confident_p50", 0)
    n70 = new.get("n_confident_p70", 0)
    old_str = f"{old_hr50:.4f}" if old_hr50 is not None else "N/A"
    new50_str = f"{new_hr50:.4f}" if new_hr50 is not None else "N/A"
    new70_str = f"{new_hr70:.4f}" if new_hr70 is not None else "N/A"
    delta_str = f"{new_hr50 - old_hr50:+.4f}" if (old_hr50 is not None and new_hr50 is not None) else "N/A"
    print(f"  {name:<16s} {old_str:>10s} {new50_str:>10s} {delta_str:>8s} {new70_str:>10s} {n50:>6d} {n70:>6d}")

# 3. Scorer AUC
print("\n3. SCORER AUC")
print("-" * 80)
old_auc = v20["scorer_auc"]["overall"]
new_auc = v21["scorer_auc"]["overall"]
print(f"  v2.0: {old_auc:.4f}    v2.1: {new_auc:.4f}    delta: {new_auc - old_auc:+.4f}")

# 4. Game winner
print("\n4. GAME WINNER")
print("-" * 80)
og = v20.get("game_winner", {})
ng = v21.get("game_winner", {})
for key, label in [("accuracy", "Accuracy"), ("margin_mae", "Margin MAE"), ("auc", "AUC"),
                    ("confident_accuracy", "Confident Acc"), ("n_confident_games", "N Confident")]:
    ov = og.get(key)
    nv = ng.get(key)
    if ov is not None and nv is not None:
        delta = nv - ov
        print(f"  {label:<20s} v2.0: {ov:>8}    v2.1: {nv:>8}    delta: {delta:>+8}")

# 5. Learning effect
print("\n5. LEARNING EFFECT")
print("-" * 80)
ol = v20.get("learning_effect", {})
nl = v21.get("learning_effect", {})
print(f"  v2.0: {ol.get('learning_pct', '?')}%    v2.1: {nl.get('learning_pct', '?')}%")
print(f"  v2.0 first/second half MAE: {ol.get('first_half_mae')}/{ol.get('second_half_mae')}")
print(f"  v2.1 first/second half MAE: {nl.get('first_half_mae')}/{nl.get('second_half_mae')}")

# 6. Goals MAE
print("\n6. MAE METRICS")
print("-" * 80)
om = v20.get("mae", {})
nm = v21.get("mae", {})
for key, label in [("goals", "Goals MAE"), ("goals_rmse", "Goals RMSE"), ("goals_weighted", "Weighted Goals MAE"),
                    ("behinds", "Behinds MAE"), ("disposals", "Disposals MAE"), ("improvement_pct", "vs Career Avg")]:
    ov = om.get(key)
    nv = nm.get(key)
    if ov is not None and nv is not None:
        delta = nv - ov
        unit = "%" if "pct" in key else ""
        print(f"  {label:<22s} v2.0: {ov:>8}{unit}    v2.1: {nv:>8}{unit}    delta: {delta:>+8.4f}{unit}")

print("\n" + "=" * 80)
print("CALIBRATION ECE")
print(f"  v2.0: {v20.get('calibration_ece')}    v2.1: {v21.get('calibration_ece')}")
print("=" * 80)
