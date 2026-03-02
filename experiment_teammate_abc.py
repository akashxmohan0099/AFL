"""
A/B/C Experiment: Teammate Lineup Features
===========================================
Tests two teammate-awareness approaches against the current baseline
using a 2024 sequential backtest (scoring model only).

Conditions:
  C (Baseline):  Current 181 features
  A (+combo):    181 + 2 = 183 features (combination hash + combo goal avg)
  B (+quality):  181 + 4 = 185 features (mid/fwd quality present, key mid absent, opp def quality)

Output:
  - Per-round Brier scores (1+, 2+, 3+) for all 3 conditions
  - Season-aggregate Brier means with paired t-test A-vs-C and B-vs-C
  - GBT feature importances — do new features crack top 30?
  - Coverage stats for each new feature
  - Final verdict
"""

import json
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from sklearn.inspection import permutation_importance

import config
from analysis import _compute_threshold_metrics, _extract_threshold_data
from model import AFLScoringModel, _prepare_features

warnings.filterwarnings("ignore")

EXPERIMENT_DIR = config.DATA_DIR / "experiments"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------

def load_data():
    """Load feature matrix and baseline feature columns."""
    feature_df = pd.read_parquet("data/features/feature_matrix.parquet")
    with open("data/features/feature_columns.json") as f:
        baseline_feature_cols = json.load(f)

    # Ensure sort order for temporal correctness
    feature_df = feature_df.sort_values(["date", "match_id", "player"]).reset_index(drop=True)

    return feature_df, baseline_feature_cols


# ---------------------------------------------------------------------------
# Approach A: Combination encoding (+2 features)
# ---------------------------------------------------------------------------

def build_approach_a_features(df):
    """Add combination encoding features: teammate_combo_hash, teammate_combo_goal_avg.

    For each (match_id, team), builds a sorted tuple of teammate player_ids
    (excluding self), hashes it, and computes historical goal average of that
    exact combination.

    Returns (df_with_features, list_of_new_feature_names).
    """
    df = df.copy()

    # Step 1: Build roster string per (match_id, team) — sorted player_ids joined
    roster_df = (
        df.groupby(["match_id", "team"], observed=True)["player_id"]
        .apply(lambda x: ",".join(sorted(x.astype(str).tolist())))
        .reset_index()
        .rename(columns={"player_id": "_roster_str"})
    )
    df = df.merge(roster_df, on=["match_id", "team"], how="left")

    # Step 2: For each player, remove self from roster string and hash
    def _remove_self_and_hash(row):
        roster_parts = row["_roster_str"].split(",")
        pid = str(row["player_id"])
        teammates = [p for p in roster_parts if p != pid]
        combo_str = ",".join(teammates)
        return zlib.crc32(combo_str.encode())

    df["teammate_combo_hash"] = df.apply(_remove_self_and_hash, axis=1)

    # Step 3: Team goals per (match_id, team) for historical combo average
    team_gl = (
        df.groupby(["match_id", "team"], observed=True)["GL"]
        .sum()
        .reset_index()
        .rename(columns={"GL": "_team_gl"})
    )
    df = df.merge(team_gl, on=["match_id", "team"], how="left")

    # Step 4: Temporal expanding mean of team goals per combo_hash
    # Need one row per (match_id, combo_hash) with date for sorting
    combo_hist = df[["match_id", "team", "date", "teammate_combo_hash", "_team_gl"]].drop_duplicates(
        subset=["match_id", "team", "teammate_combo_hash"]
    ).sort_values("date")

    combo_hist["teammate_combo_goal_avg"] = (
        combo_hist.groupby("teammate_combo_hash", observed=True)["_team_gl"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )

    # Merge back to player level
    df = df.merge(
        combo_hist[["match_id", "team", "teammate_combo_hash", "teammate_combo_goal_avg"]].drop_duplicates(),
        on=["match_id", "team", "teammate_combo_hash"],
        how="left",
        suffixes=("", "_y"),
    )
    # Clean up any _y cols from merge
    df = df.drop(columns=[c for c in df.columns if c.endswith("_y")], errors="ignore")

    # Drop temp columns
    df = df.drop(columns=["_roster_str", "_team_gl"], errors="ignore")

    new_cols = ["teammate_combo_hash", "teammate_combo_goal_avg"]
    return df, new_cols


# ---------------------------------------------------------------------------
# Approach B: Continuous quality scores (+4 features)
# ---------------------------------------------------------------------------

def build_approach_b_features(df):
    """Add continuous quality score features for selected lineup.

    New features:
      - team_mid_quality_present: sum of midfielder quality scores for this match's lineup (minus self)
      - team_fwd_quality_present: sum of forward quality scores for this match's lineup (minus self)
      - key_mid_absent: binary — is a top-3 historical mid NOT in this match's lineup?
      - opp_def_quality_present: sum of opponent defender quality for this match's opponent lineup

    Returns (df_with_features, list_of_new_feature_names).
    """
    df = df.sort_values(["player", "team", "date"]).copy()

    # Cumulative stats (temporal, shift(1) to avoid leakage)
    pg = df.groupby(["player", "team"], observed=True)
    df["_cum_di_avg"] = pg["DI"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["_cum_cl_avg"] = pg["CL"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["_cum_if_avg"] = pg["IF"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["_cum_gl_avg"] = pg["GL"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["_cum_mi_avg"] = pg["MI"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["_cum_rb_avg"] = pg["RB"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["_cum_one_pct_avg"] = pg["one_pct"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    # Role flags (same thresholds as features.py)
    df["_is_mid"] = (
        (df["_cum_di_avg"].fillna(0) >= config.MIDFIELDER_DI_THRESHOLD)
        & (df["_cum_cl_avg"].fillna(0) >= config.MIDFIELDER_CL_THRESHOLD)
    ).astype(int)

    df["_is_fwd"] = (
        (df["_cum_gl_avg"].fillna(0) >= config.KEY_FORWARD_GL_THRESHOLD)
        | (
            (df["_cum_gl_avg"].fillna(0) >= config.SMALL_FORWARD_GL_THRESHOLD)
            & (df["_cum_if_avg"].fillna(0) >= config.SMALL_FORWARD_IF_THRESHOLD)
        )
    ).astype(int)

    df["_is_def"] = (
        (df["_cum_rb_avg"].fillna(0) >= config.KEY_DEFENDER_RB_THRESHOLD)
        & (df["_cum_one_pct_avg"].fillna(0) >= config.KEY_DEFENDER_ONE_PCT_THRESHOLD)
    ).astype(int)

    # Quality scores per player
    df["_mid_quality"] = (
        df["_cum_di_avg"].fillna(0)
        + df["_cum_cl_avg"].fillna(0) * 2
        + df["_cum_if_avg"].fillna(0)
    ) * df["_is_mid"]

    df["_fwd_quality"] = (
        df["_cum_gl_avg"].fillna(0) * 2
        + df["_cum_mi_avg"].fillna(0)
        + df["_cum_if_avg"].fillna(0) * 0.5
    ) * df["_is_fwd"]

    df["_def_quality"] = (
        df["_cum_rb_avg"].fillna(0)
        + df["_cum_one_pct_avg"].fillna(0)
        + df["_cum_if_avg"].fillna(0) * 0.5
    ) * df["_is_def"]

    # --- Aggregate per (match_id, team) ---
    team_agg = (
        df.groupby(["match_id", "team"], observed=True)
        .agg(
            _team_mid_quality_sum=("_mid_quality", "sum"),
            _team_fwd_quality_sum=("_fwd_quality", "sum"),
            _team_player_count=("player", "count"),
        )
        .reset_index()
    )
    df = df.merge(team_agg, on=["match_id", "team"], how="left")

    # Subtract self-contribution (same pattern as add_enabler_features)
    df["team_mid_quality_present"] = df["_team_mid_quality_sum"] - df["_mid_quality"]
    df["team_fwd_quality_present"] = df["_team_fwd_quality_sum"] - df["_fwd_quality"]

    # --- key_mid_absent: is any of the team's top-3 mids NOT in this lineup? ---
    # For each (team, date), identify the top-3 mids by cumulative DI avg
    # across all historical appearances. Then check if all 3 are in this match.
    # We compute this per (match_id, team).

    # Get unique player-level mid quality per team at each match date
    # Use the _cum_di_avg already computed (which is temporal)
    mid_players = df[df["_is_mid"] == 1][["match_id", "team", "player", "_cum_di_avg", "date"]].copy()

    # For each team, rank players by cum_di_avg at each match date
    # We need: for each (match_id, team), who are the historical top-3 mids?
    # This requires knowing the team's historical roster, not just this match.
    # Simplification: use ALL players who have played for this team up to this date,
    # ranked by their latest _cum_di_avg.

    # Instead, simpler approach: among this match's midfielders, compute
    # "are all of the team's best mids playing?" by checking if the sum of
    # mid quality is >= the team's rolling max mid quality.
    # Even simpler: for each (team, match), count how many of the team's
    # top-3 mids (by expanding DI avg) are present in the lineup.

    # Build expanding top-3 per team
    # Get all mids per team across all matches (sorted by date)
    team_mid_hist = (
        df[["match_id", "team", "player", "date", "_cum_di_avg", "_is_mid"]]
        .drop_duplicates(subset=["match_id", "player"])
        .sort_values("date")
    )

    def _key_mid_absent_for_team(team_df):
        """For each match of a team, check if top-3 historical mids are in the lineup."""
        result = []
        seen_mids = {}  # player -> best _cum_di_avg seen so far

        for match_id, match_grp in team_df.groupby("match_id", sort=False):
            # Current lineup players
            lineup_players = set(match_grp["player"].values)

            # Top-3 mids from history (before this match)
            if len(seen_mids) >= 1:
                top3 = sorted(seen_mids.items(), key=lambda x: -x[1])[:3]
                top3_players = {p for p, _ in top3}
                absent_count = len(top3_players - lineup_players)
                key_absent = int(absent_count > 0)
            else:
                key_absent = np.nan  # no history yet

            for _, row in match_grp.iterrows():
                result.append({"match_id": match_id, "player": row["player"], "key_mid_absent": key_absent})

            # Update seen_mids with this match's midfielders
            for _, row in match_grp.iterrows():
                if row["_is_mid"] == 1 and not np.isnan(row["_cum_di_avg"]):
                    player = row["player"]
                    if player not in seen_mids or row["_cum_di_avg"] > seen_mids[player]:
                        seen_mids[player] = row["_cum_di_avg"]

        return pd.DataFrame(result)

    # This is slow but correct — apply per team
    key_mid_parts = []
    for team_name, team_grp in team_mid_hist.groupby("team", observed=True):
        # Sort by date within team
        team_grp = team_grp.sort_values("date")
        part = _key_mid_absent_for_team(team_grp)
        if not part.empty:
            part["team"] = team_name
            key_mid_parts.append(part)

    if key_mid_parts:
        key_mid_df = pd.concat(key_mid_parts, ignore_index=True)
        df = df.merge(key_mid_df, on=["match_id", "team", "player"], how="left")
    else:
        df["key_mid_absent"] = np.nan

    # --- opp_def_quality_present: opponent's selected defenders' quality ---
    opp_def_agg = (
        df.groupby(["match_id", "team"], observed=True)["_def_quality"]
        .sum()
        .reset_index()
        .rename(columns={"team": "opponent", "_def_quality": "opp_def_quality_present"})
    )
    df = df.merge(opp_def_agg, on=["match_id", "opponent"], how="left")

    # Drop temp columns
    temp_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=temp_cols, errors="ignore")

    new_cols = ["team_mid_quality_present", "team_fwd_quality_present",
                "key_mid_absent", "opp_def_quality_present"]
    return df, new_cols


# ---------------------------------------------------------------------------
# Sequential backtest loop
# ---------------------------------------------------------------------------

def run_backtest(feature_df, feature_cols, condition_name, rounds):
    """Run 2024 sequential backtest for scoring model only.

    Returns list of dicts: [{round, brier_1plus, brier_2plus, brier_3plus, n_players}, ...]
    Also returns the model from the last round and its test data for feature importance.
    """
    results = []
    model = None
    last_test_df = None
    total_start = time.time()

    for rnd in rounds:
        rnd_start = time.time()

        # Train/test split (same as pipeline.py sequential backtest)
        train_mask = (feature_df["year"] < 2024) | (
            (feature_df["year"] == 2024) & (feature_df["round_number"] < rnd)
        )
        test_mask = (feature_df["year"] == 2024) & (feature_df["round_number"] == rnd)

        train_df = feature_df[train_mask]
        test_df = feature_df[test_mask]

        if len(test_df) == 0:
            continue

        last_test_df = test_df

        # Train scoring model
        model = AFLScoringModel()
        model.train_backtest(train_df, feature_cols)

        # Predict distributions
        preds = model.predict_distributions(test_df, store=None, feature_cols=feature_cols)

        # Merge actuals for Brier score computation
        outcomes = test_df[["player", "team", "match_id"]].copy()
        outcomes["actual_goals"] = test_df["GL"].values
        merged = preds.merge(outcomes, on=["player", "team", "match_id"])

        # Compute Brier scores
        thresholds = _extract_threshold_data(merged)
        brier_1 = None
        brier_2 = None
        brier_3 = None

        if "1plus_goals" in thresholds:
            metrics = _compute_threshold_metrics(*thresholds["1plus_goals"])
            brier_1 = metrics["brier_score"] if metrics else None
        if "2plus_goals" in thresholds:
            metrics = _compute_threshold_metrics(*thresholds["2plus_goals"])
            brier_2 = metrics["brier_score"] if metrics else None
        if "3plus_goals" in thresholds:
            metrics = _compute_threshold_metrics(*thresholds["3plus_goals"])
            brier_3 = metrics["brier_score"] if metrics else None

        elapsed = time.time() - rnd_start
        results.append({
            "round": int(rnd),
            "brier_1plus": brier_1,
            "brier_2plus": brier_2,
            "brier_3plus": brier_3,
            "n_players": len(test_df),
        })
        print(f"  {condition_name} Round {int(rnd):2d}: "
              f"B1+={brier_1:.4f} B2+={brier_2:.4f} B3+={brier_3:.4f} "
              f"n={len(test_df)} ({elapsed:.1f}s)" if brier_1 else
              f"  {condition_name} Round {int(rnd):2d}: skipped (insufficient data)")

    total_elapsed = time.time() - total_start
    print(f"  {condition_name} total: {total_elapsed:.0f}s")

    return results, model, last_test_df


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------

def extract_importances(model, feature_cols, label, test_df, new_features=None):
    """Extract and print top-30 feature importances using permutation importance.

    HistGBT lacks feature_importances_, so we use permutation importance
    on the last round's test set for both goals GBT and scorer classifier.
    """
    print(f"\n  Feature importance ({label}):")

    X_raw, _, _ = _prepare_features(test_df, feature_cols, scaler=model.scaler)
    y_goals = test_df["GL"].values

    # Goals GBT (stage 2) — permutation importance on scorers
    scorer_mask = y_goals >= 1
    if scorer_mask.sum() > 10:
        X_scorers = X_raw[scorer_mask]
        y_scorers = y_goals[scorer_mask]
        perm_result = permutation_importance(
            model.goals_gbt, X_scorers, y_scorers,
            n_repeats=10, random_state=42, scoring="neg_mean_squared_error"
        )
        importances = perm_result.importances_mean
        top_idx = np.argsort(importances)[::-1][:30]
        print(f"    Goals GBT top-30 (permutation importance on last round scorers):")
        new_in_top30 = []
        for rank, idx in enumerate(top_idx, 1):
            name = feature_cols[idx]
            imp = importances[idx]
            marker = " ***" if new_features and name in new_features else ""
            print(f"      {rank:2d}. {name:45s} {imp:.4f}{marker}")
            if new_features and name in new_features:
                new_in_top30.append((name, rank, imp))

        if new_features:
            if new_in_top30:
                print(f"    >> New features in top 30: {new_in_top30}")
            else:
                for feat in new_features:
                    if feat in feature_cols:
                        feat_idx = feature_cols.index(feat)
                        feat_rank = int(np.where(np.argsort(importances)[::-1] == feat_idx)[0][0]) + 1
                        feat_imp = importances[feat_idx]
                        print(f"    >> {feat}: rank {feat_rank}, importance {feat_imp:.6f}")
                    else:
                        print(f"    >> {feat}: not in feature_cols")

    # Scorer classifier (stage 1) — permutation importance
    y_scorer_binary = (y_goals >= 1).astype(int)
    perm_clf = permutation_importance(
        model.scorer_clf, X_raw, y_scorer_binary,
        n_repeats=10, random_state=42, scoring="roc_auc"
    )
    importances_clf = perm_clf.importances_mean
    top_idx_clf = np.argsort(importances_clf)[::-1][:10]
    print(f"    Scorer classifier top-10 (permutation importance):")
    for rank, idx in enumerate(top_idx_clf, 1):
        name = feature_cols[idx]
        imp = importances_clf[idx]
        marker = " ***" if new_features and name in new_features else ""
        print(f"      {rank:2d}. {name:45s} {imp:.4f}{marker}")


# ---------------------------------------------------------------------------
# Coverage computation
# ---------------------------------------------------------------------------

def compute_coverage(df, feature_cols_new, label):
    """Compute and print coverage stats for new features."""
    print(f"\n  Coverage ({label}):")
    season_2024 = df[df["year"] == 2024]

    for col in feature_cols_new:
        if col in df.columns:
            all_cov = 1.0 - (df[col].isna().sum() / len(df))
            s24_cov = 1.0 - (season_2024[col].isna().sum() / len(season_2024))
            print(f"    {col:35s} all={all_cov*100:.1f}%  2024={s24_cov*100:.1f}%")
        else:
            print(f"    {col:35s} NOT FOUND IN DF")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TEAMMATE FEATURE EXPERIMENT (A/B/C)")
    print("=" * 70)

    # Load shared data
    print("\nLoading data...")
    feature_df, baseline_feature_cols = load_data()
    print(f"  Feature matrix: {feature_df.shape}")
    print(f"  Baseline features: {len(baseline_feature_cols)}")

    # Determine 2024 rounds (regular season: 1-24)
    season_2024 = feature_df[feature_df["year"] == 2024]
    all_rounds = sorted(season_2024["round_number"].dropna().unique())
    rounds = [r for r in all_rounds if r <= 24]
    print(f"  2024 rounds for backtest: {[int(r) for r in rounds]}")

    # -----------------------------------------------------------------------
    # Condition C: Baseline
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CONDITION C: Baseline ({} features)".format(len(baseline_feature_cols)))
    print("-" * 70)
    results_c, model_c, test_c = run_backtest(
        feature_df, baseline_feature_cols, "C", rounds
    )

    # -----------------------------------------------------------------------
    # Condition A: Combination encoding
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CONDITION A: Combination encoding (+2 features)")
    print("-" * 70)
    print("  Building Approach A features...")
    t0 = time.time()
    df_a, new_cols_a = build_approach_a_features(feature_df)
    print(f"  Done in {time.time()-t0:.1f}s")

    feature_cols_a = baseline_feature_cols + new_cols_a
    print(f"  Total features: {len(feature_cols_a)}")

    compute_coverage(df_a, new_cols_a, "Approach A")

    results_a, model_a, test_a = run_backtest(
        df_a, feature_cols_a, "A", rounds
    )

    # -----------------------------------------------------------------------
    # Condition B: Continuous quality scores
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CONDITION B: Continuous quality scores (+4 features)")
    print("-" * 70)
    print("  Building Approach B features...")
    t0 = time.time()
    df_b, new_cols_b = build_approach_b_features(feature_df)
    print(f"  Done in {time.time()-t0:.1f}s")

    feature_cols_b = baseline_feature_cols + new_cols_b
    print(f"  Total features: {len(feature_cols_b)}")

    compute_coverage(df_b, new_cols_b, "Approach B")

    results_b, model_b, test_b = run_backtest(
        df_b, feature_cols_b, "B", rounds
    )

    # -----------------------------------------------------------------------
    # Results summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Per-round table
    print(f"\n{'Round':>5}  {'C_B1+':>8}  {'A_B1+':>8}  {'B_B1+':>8}  "
          f"{'C_B2+':>8}  {'A_B2+':>8}  {'B_B2+':>8}  "
          f"{'C_B3+':>8}  {'A_B3+':>8}  {'B_B3+':>8}")
    print("-" * 95)

    for rc, ra, rb in zip(results_c, results_a, results_b):
        def _fmt(v):
            return f"{v:.4f}" if v is not None else "   N/A"
        print(f"{rc['round']:5d}  {_fmt(rc['brier_1plus']):>8}  {_fmt(ra['brier_1plus']):>8}  {_fmt(rb['brier_1plus']):>8}  "
              f"{_fmt(rc['brier_2plus']):>8}  {_fmt(ra['brier_2plus']):>8}  {_fmt(rb['brier_2plus']):>8}  "
              f"{_fmt(rc['brier_3plus']):>8}  {_fmt(ra['brier_3plus']):>8}  {_fmt(rb['brier_3plus']):>8}")

    # Season aggregates
    def _mean_brier(results, key):
        vals = [r[key] for r in results if r[key] is not None]
        return np.mean(vals) if vals else None

    print(f"\nSeason aggregate Brier scores:")
    print(f"{'':>15}  {'Brier 1+':>10}  {'Brier 2+':>10}  {'Brier 3+':>10}")
    for label, res in [("C (Baseline)", results_c), ("A (+combo)", results_a), ("B (+quality)", results_b)]:
        b1 = _mean_brier(res, "brier_1plus")
        b2 = _mean_brier(res, "brier_2plus")
        b3 = _mean_brier(res, "brier_3plus")
        print(f"{label:>15}  {b1:10.4f}  {b2:10.4f}  {b3:10.4f}")

    # Paired t-tests
    print(f"\nPaired t-tests (per-round Brier 1+ goals):")
    for label, res_x in [("A vs C", results_a), ("B vs C", results_b)]:
        brier_x = [r["brier_1plus"] for r in res_x if r["brier_1plus"] is not None]
        brier_c = [r["brier_1plus"] for r in results_c if r["brier_1plus"] is not None]
        # Align lengths
        n = min(len(brier_x), len(brier_c))
        brier_x, brier_c = brier_x[:n], brier_c[:n]
        if n >= 2:
            t_stat, p_val = ttest_rel(brier_x, brier_c)
            direction = "better" if t_stat < 0 else "worse"
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
            print(f"  {label}: t={t_stat:+.4f}, p={p_val:.4f} ({direction}) {sig}")
            diff = np.mean(brier_x) - np.mean(brier_c)
            print(f"    Mean diff: {diff:+.6f} ({'improvement' if diff < 0 else 'degradation'})")
        else:
            print(f"  {label}: insufficient rounds for t-test")

    # Brier 2+ t-tests
    print(f"\nPaired t-tests (per-round Brier 2+ goals):")
    for label, res_x in [("A vs C", results_a), ("B vs C", results_b)]:
        brier_x = [r["brier_2plus"] for r in res_x if r["brier_2plus"] is not None]
        brier_c = [r["brier_2plus"] for r in results_c if r["brier_2plus"] is not None]
        n = min(len(brier_x), len(brier_c))
        brier_x, brier_c = brier_x[:n], brier_c[:n]
        if n >= 2:
            t_stat, p_val = ttest_rel(brier_x, brier_c)
            direction = "better" if t_stat < 0 else "worse"
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
            print(f"  {label}: t={t_stat:+.4f}, p={p_val:.4f} ({direction}) {sig}")

    # Feature importances
    print("\n" + "-" * 70)
    print("FEATURE IMPORTANCES")
    print("-" * 70)
    extract_importances(model_c, baseline_feature_cols, "Baseline C", test_c)
    extract_importances(model_a, feature_cols_a, "Approach A", test_a, new_features=new_cols_a)
    extract_importances(model_b, feature_cols_b, "Approach B", test_b, new_features=new_cols_b)

    # -----------------------------------------------------------------------
    # Save results to CSV
    # -----------------------------------------------------------------------
    rows = []
    for rc, ra, rb in zip(results_c, results_a, results_b):
        rows.append({
            "round": rc["round"],
            "n_players": rc["n_players"],
            "C_brier_1plus": rc["brier_1plus"],
            "C_brier_2plus": rc["brier_2plus"],
            "C_brier_3plus": rc["brier_3plus"],
            "A_brier_1plus": ra["brier_1plus"],
            "A_brier_2plus": ra["brier_2plus"],
            "A_brier_3plus": ra["brier_3plus"],
            "B_brier_1plus": rb["brier_1plus"],
            "B_brier_2plus": rb["brier_2plus"],
            "B_brier_3plus": rb["brier_3plus"],
        })
    results_df = pd.DataFrame(rows)
    out_path = EXPERIMENT_DIR / "teammate_abc_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # -----------------------------------------------------------------------
    # Final verdict
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    b1_c = _mean_brier(results_c, "brier_1plus")
    b1_a = _mean_brier(results_a, "brier_1plus")
    b1_b = _mean_brier(results_b, "brier_1plus")

    # T-tests for verdict
    brier_a_vals = [r["brier_1plus"] for r in results_a if r["brier_1plus"] is not None]
    brier_b_vals = [r["brier_1plus"] for r in results_b if r["brier_1plus"] is not None]
    brier_c_vals = [r["brier_1plus"] for r in results_c if r["brier_1plus"] is not None]

    n_a = min(len(brier_a_vals), len(brier_c_vals))
    n_b = min(len(brier_b_vals), len(brier_c_vals))

    _, p_a = ttest_rel(brier_a_vals[:n_a], brier_c_vals[:n_a]) if n_a >= 2 else (0, 1)
    _, p_b = ttest_rel(brier_b_vals[:n_b], brier_c_vals[:n_b]) if n_b >= 2 else (0, 1)

    a_better = b1_a < b1_c
    b_better = b1_b < b1_c

    print(f"  Approach A: {'improves' if a_better else 'does not improve'} on baseline "
          f"(Brier 1+: {b1_a:.4f} vs {b1_c:.4f}, p={p_a:.4f})")
    print(f"  Approach B: {'improves' if b_better else 'does not improve'} on baseline "
          f"(Brier 1+: {b1_b:.4f} vs {b1_c:.4f}, p={p_b:.4f})")

    if a_better and p_a < 0.05:
        print("  >> Approach A shows SIGNIFICANT improvement. Recommend integration.")
    elif a_better:
        print("  >> Approach A shows directional improvement but not significant.")
    else:
        print("  >> Approach A does not improve predictions.")

    if b_better and p_b < 0.05:
        print("  >> Approach B shows SIGNIFICANT improvement. Recommend integration.")
    elif b_better:
        print("  >> Approach B shows directional improvement but not significant.")
    else:
        print("  >> Approach B does not improve predictions.")

    if not (a_better or b_better):
        print("  >> Neither approach improves the baseline. Teammate lineup features")
        print("     may not add signal beyond existing teammate_scoring_avg.")


if __name__ == "__main__":
    main()
