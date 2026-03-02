"""
Feature Importance Analysis — SHAP + Permutation Importance
============================================================
Trains models on 2015-2024, computes SHAP and permutation importance
on 2024 validation data. Reports top/bottom 30 features for each model.
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import config
from model import AFLScoringModel, AFLDisposalModel
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data():
    """Load feature matrix and feature columns."""
    feat_path = config.FEATURES_DIR / "feature_matrix.parquet"
    df = pd.read_parquet(feat_path)
    with open(config.FEATURES_DIR / "feature_columns.json") as f:
        feature_cols = json.load(f)
    return df, feature_cols


def prepare_splits(df, feature_cols, target_col="GL"):
    """Split into train (2015-2023) and val (2024), clean features."""
    # Filter out did_not_play
    df = df[df["did_not_play"] == False].copy()
    # Need minimum matches
    df = df[df["career_games_pre"] >= config.MIN_PLAYER_MATCHES].copy()

    train = df[df["year"] < 2024].copy()
    val = df[df["year"] == 2024].copy()

    # Ensure feature cols exist
    valid_cols = [c for c in feature_cols if c in df.columns]

    # Reset index so positional and label indexing align
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    X_train = train[valid_cols].fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)
    X_val = val[valid_cols].fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)
    y_train = train[target_col].values
    y_val = val[target_col].values

    return X_train, X_val, y_train, y_val, valid_cols, train, val


def print_table(rows, headers, title, width=None):
    """Print a formatted table."""
    if width is None:
        width = [max(len(str(r[i])) for r in [headers] + rows) + 2 for i in range(len(headers))]

    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    header_str = "".join(str(headers[i]).rjust(width[i]) for i in range(len(headers)))
    print(header_str)
    print("-" * sum(width))

    for row in rows:
        print("".join(str(row[i]).rjust(width[i]) for i in range(len(row))))


def shap_importance(shap_values, feature_names):
    """Compute mean |SHAP| per feature, return sorted DataFrame."""
    if hasattr(shap_values, 'values'):
        vals = shap_values.values
    else:
        vals = shap_values

    mean_abs = np.mean(np.abs(vals), axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "shap_importance": mean_abs,
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)
    df["shap_rank"] = range(1, len(df) + 1)
    return df


def perm_importance(model, X, y, scoring, n_repeats=10):
    """Run permutation importance, return sorted DataFrame."""
    result = permutation_importance(
        model, X, y, scoring=scoring, n_repeats=n_repeats,
        random_state=42, n_jobs=-1
    )
    df = pd.DataFrame({
        "feature": X.columns.tolist(),
        "perm_importance": result.importances_mean,
        "perm_std": result.importances_std,
    }).sort_values("perm_importance", ascending=False).reset_index(drop=True)
    df["perm_rank"] = range(1, len(df) + 1)
    return df


def print_top_bottom(merged, importance_col, rank_col, n=30, model_name="Model"):
    """Print top N and bottom N features by importance."""
    sorted_df = merged.sort_values(importance_col, ascending=False).reset_index(drop=True)

    # Top N
    top = sorted_df.head(n)
    rows = []
    for i, r in top.iterrows():
        rows.append([
            f"{i+1:>3}",
            f"{r['feature']:<45s}",
            f"{r['shap_importance']:>10.6f}",
            f"{r['shap_rank']:>6.0f}",
            f"{r['perm_importance']:>12.6f}",
            f"{r['perm_rank']:>6.0f}",
        ])
    headers = ["#", "Feature", "SHAP |val|", "S.Rank", "Perm Imp", "P.Rank"]
    print(f"\n{'=' * 95}")
    print(f"  TOP {n} FEATURES — {model_name}")
    print(f"{'=' * 95}")
    print(f"{'#':>3s}  {'Feature':<45s} {'SHAP |val|':>10s} {'S.Rank':>6s} {'Perm Imp':>12s} {'P.Rank':>6s}")
    print("-" * 95)
    for row in rows:
        print(f"{row[0]}  {row[1]} {row[2]} {row[3]} {row[4]} {row[5]}")

    # Bottom N
    bottom = sorted_df.tail(n).iloc[::-1]  # worst first
    rows = []
    for _, r in bottom.iterrows():
        rank_in_sorted = int(r.name) + 1 if hasattr(r, 'name') else 0
        rows.append([
            f"{r['feature']:<45s}",
            f"{r['shap_importance']:>10.6f}",
            f"{r['shap_rank']:>6.0f}",
            f"{r['perm_importance']:>12.6f}",
            f"{r['perm_rank']:>6.0f}",
        ])
    print(f"\n{'=' * 95}")
    print(f"  BOTTOM {n} FEATURES — {model_name}")
    print(f"{'=' * 95}")
    print(f"  {'Feature':<45s} {'SHAP |val|':>10s} {'S.Rank':>6s} {'Perm Imp':>12s} {'P.Rank':>6s}")
    print("-" * 95)
    for row in rows:
        print(f"  {row[0]} {row[1]} {row[2]} {row[3]} {row[4]}")


# ---------------------------------------------------------------------------
# Step 1 & 2: Scoring Model — Scorer Classifier + Goals GBT
# ---------------------------------------------------------------------------

def analyze_scoring_model(df, feature_cols):
    """Full SHAP + permutation analysis for both scoring sub-models."""
    import shap

    print("\n" + "#" * 80)
    print("#  STEP 1 & 2: SCORING MODEL ANALYSIS")
    print("#" * 80)

    X_train, X_val, y_train_gl, y_val_gl, valid_cols, train_df, val_df = \
        prepare_splits(df, feature_cols, target_col="GL")

    # ---- Train scorer classifier (binary: scored >= 1 goal) ----
    print("\n[1a] Training scorer classifier (GradientBoostingClassifier)...")
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    scorer_clf = GradientBoostingClassifier(
        **config.GBT_PARAMS, random_state=42
    )
    y_train_binary = (y_train_gl >= 1).astype(int)
    y_val_binary = (y_val_gl >= 1).astype(int)
    scorer_clf.fit(X_train, y_train_binary)

    from sklearn.metrics import roc_auc_score, accuracy_score
    val_proba = scorer_clf.predict_proba(X_val)[:, 1]
    val_pred = scorer_clf.predict(X_val)
    print(f"  Scorer AUC:  {roc_auc_score(y_val_binary, val_proba):.4f}")
    print(f"  Scorer Acc:  {accuracy_score(y_val_binary, val_pred):.4f}")
    print(f"  Train: {len(X_train)} rows, Val: {len(X_val)} rows")

    # ---- Train goals GBT (on scorers only) ----
    print("\n[1b] Training goals GBT (GradientBoostingRegressor, scorers only)...")
    scorer_mask_train = y_train_gl >= 1
    scorer_mask_val = y_val_gl >= 1
    X_train_scorers = X_train[scorer_mask_train].reset_index(drop=True)
    y_train_scorers = y_train_gl[scorer_mask_train]
    X_val_scorers = X_val[scorer_mask_val].reset_index(drop=True)
    y_val_scorers = y_val_gl[scorer_mask_val]

    goals_gbt = GradientBoostingRegressor(
        **config.GBT_PARAMS, random_state=42
    )
    goals_gbt.fit(X_train_scorers, y_train_scorers)

    from sklearn.metrics import mean_absolute_error
    goals_pred = goals_gbt.predict(X_val_scorers)
    print(f"  Goals MAE (scorers): {mean_absolute_error(y_val_scorers, goals_pred):.4f}")
    print(f"  Train scorers: {len(X_train_scorers)}, Val scorers: {len(X_val_scorers)}")

    # ---- SHAP for scorer classifier ----
    print("\n[1c] Computing SHAP values for scorer classifier...")
    sample_size = min(5000, len(X_val))
    X_val_sample = X_val.sample(n=sample_size, random_state=42)
    y_val_sample_binary = y_val_binary[X_val_sample.index]

    explainer_scorer = shap.TreeExplainer(scorer_clf)
    shap_vals_scorer = explainer_scorer.shap_values(X_val_sample)
    # For binary classifier, shap_values returns [class0, class1] — use class1
    if isinstance(shap_vals_scorer, list):
        shap_vals_scorer = shap_vals_scorer[1]
    shap_scorer_df = shap_importance(shap_vals_scorer, valid_cols)

    # ---- SHAP for goals GBT ----
    print("[1d] Computing SHAP values for goals GBT...")
    sample_scorers = min(5000, len(X_val_scorers))
    X_val_scorers_sample = X_val_scorers.sample(n=sample_scorers, random_state=42)

    explainer_goals = shap.TreeExplainer(goals_gbt)
    shap_vals_goals = explainer_goals.shap_values(X_val_scorers_sample)
    shap_goals_df = shap_importance(shap_vals_goals, valid_cols)

    # ---- Permutation importance for scorer classifier ----
    print("[2a] Computing permutation importance for scorer classifier...")
    perm_scorer_df = perm_importance(scorer_clf, X_val_sample, y_val_sample_binary,
                                      scoring="roc_auc", n_repeats=10)

    # ---- Permutation importance for goals GBT ----
    print("[2b] Computing permutation importance for goals GBT...")
    y_val_scorers_sample = y_val_scorers[X_val_scorers_sample.index]
    perm_goals_df = perm_importance(goals_gbt, X_val_scorers_sample, y_val_scorers_sample,
                                     scoring="neg_mean_absolute_error", n_repeats=10)

    # ---- Merge and report: Scorer Classifier ----
    merged_scorer = shap_scorer_df.merge(perm_scorer_df, on="feature")
    print_top_bottom(merged_scorer, "shap_importance", "shap_rank",
                     n=30, model_name="Scorer Classifier (P(goals>=1))")

    # ---- Merge and report: Goals GBT ----
    merged_goals = shap_goals_df.merge(perm_goals_df, on="feature")
    print_top_bottom(merged_goals, "shap_importance", "shap_rank",
                     n=30, model_name="Goals GBT (E[goals|scorer])")

    return merged_scorer, merged_goals


# ---------------------------------------------------------------------------
# Step 3: Disposal Model
# ---------------------------------------------------------------------------

def analyze_disposal_model(df, feature_cols):
    """Full SHAP + permutation analysis for disposal GBT."""
    import shap

    print("\n" + "#" * 80)
    print("#  STEP 3: DISPOSAL MODEL ANALYSIS")
    print("#" * 80)

    X_train, X_val, y_train_di, y_val_di, valid_cols, train_df, val_df = \
        prepare_splits(df, feature_cols, target_col="DI")

    # ---- Train disposal GBT ----
    print("\n[3a] Training disposal GBT (GradientBoostingRegressor)...")
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error

    disp_gbt = GradientBoostingRegressor(
        **config.GBT_PARAMS, random_state=42
    )
    disp_gbt.fit(X_train, y_train_di)

    disp_pred = disp_gbt.predict(X_val)
    print(f"  Disposal MAE: {mean_absolute_error(y_val_di, disp_pred):.4f}")
    print(f"  Train: {len(X_train)} rows, Val: {len(X_val)} rows")

    # ---- SHAP ----
    print("\n[3b] Computing SHAP values for disposal GBT...")
    sample_size = min(5000, len(X_val))
    X_val_sample = X_val.sample(n=sample_size, random_state=42)
    y_val_sample = y_val_di[X_val_sample.index]

    explainer = shap.TreeExplainer(disp_gbt)
    shap_vals = explainer.shap_values(X_val_sample)
    shap_disp_df = shap_importance(shap_vals, valid_cols)

    # ---- Permutation importance ----
    print("[3c] Computing permutation importance for disposal GBT...")
    perm_disp_df = perm_importance(disp_gbt, X_val_sample, y_val_sample,
                                    scoring="neg_mean_absolute_error", n_repeats=10)

    # ---- Merge and report ----
    merged_disp = shap_disp_df.merge(perm_disp_df, on="feature")
    print_top_bottom(merged_disp, "shap_importance", "shap_rank",
                     n=30, model_name="Disposal GBT (E[disposals])")

    return merged_disp


# ---------------------------------------------------------------------------
# Step 4: Cross-model summary of unimportant features
# ---------------------------------------------------------------------------

def cross_model_summary(merged_scorer, merged_goals, merged_disp, feature_cols):
    """Identify features that are consistently unimportant across all models."""

    print("\n" + "#" * 80)
    print("#  STEP 4: CROSS-MODEL UNIMPORTANT FEATURE SUMMARY")
    print("#" * 80)

    n_features = len(feature_cols)

    # Define "near-zero" thresholds
    # SHAP: bottom quartile AND perm importance <= 0 (or near zero)
    shap_threshold_pct = 0.75  # bottom 25%
    perm_threshold = 0.0005  # near-zero permutation importance

    # For each model, identify features with both SHAP rank in bottom 25% and perm_importance near zero
    def get_unimportant(merged, model_name):
        shap_rank_threshold = int(n_features * shap_threshold_pct)
        low_shap = set(merged[merged["shap_rank"] > shap_rank_threshold]["feature"])
        low_perm = set(merged[merged["perm_importance"] < perm_threshold]["feature"])
        both_low = low_shap & low_perm
        return both_low

    unimportant_scorer = get_unimportant(merged_scorer, "Scorer Clf")
    unimportant_goals = get_unimportant(merged_goals, "Goals GBT")
    unimportant_disp = get_unimportant(merged_disp, "Disposal GBT")

    # Features unimportant in ALL three models
    unimportant_all = unimportant_scorer & unimportant_goals & unimportant_disp

    # Features unimportant in at least 2 models
    from collections import Counter
    count = Counter()
    for feat in feature_cols:
        c = 0
        if feat in unimportant_scorer: c += 1
        if feat in unimportant_goals: c += 1
        if feat in unimportant_disp: c += 1
        count[feat] = c

    unimportant_2plus = {f for f, c in count.items() if c >= 2}

    print(f"\n  Total features: {n_features}")
    print(f"  Threshold: SHAP rank > {int(n_features * shap_threshold_pct)} (bottom 25%)")
    print(f"             AND perm_importance < {perm_threshold}")
    print(f"\n  Unimportant in scorer classifier:  {len(unimportant_scorer)}")
    print(f"  Unimportant in goals GBT:          {len(unimportant_goals)}")
    print(f"  Unimportant in disposal GBT:       {len(unimportant_disp)}")
    print(f"\n  Unimportant in ALL 3 models:       {len(unimportant_all)}")
    print(f"  Unimportant in >= 2 models:        {len(unimportant_2plus)}")

    # Detailed table of features unimportant in all 3
    if unimportant_all:
        # Get their ranks in each model
        rows = []
        for feat in sorted(unimportant_all):
            s_row = merged_scorer[merged_scorer["feature"] == feat].iloc[0]
            g_row = merged_goals[merged_goals["feature"] == feat].iloc[0]
            d_row = merged_disp[merged_disp["feature"] == feat].iloc[0]
            rows.append({
                "feature": feat,
                "scorer_shap_rank": int(s_row["shap_rank"]),
                "scorer_perm": s_row["perm_importance"],
                "goals_shap_rank": int(g_row["shap_rank"]),
                "goals_perm": g_row["perm_importance"],
                "disp_shap_rank": int(d_row["shap_rank"]),
                "disp_perm": d_row["perm_importance"],
                "avg_shap_rank": (s_row["shap_rank"] + g_row["shap_rank"] + d_row["shap_rank"]) / 3,
            })

        rows.sort(key=lambda x: x["avg_shap_rank"], reverse=True)  # worst first

        print(f"\n{'=' * 120}")
        print(f"  FEATURES UNIMPORTANT IN ALL 3 MODELS ({len(rows)} features) — Candidates for Removal")
        print(f"{'=' * 120}")
        print(f"  {'Feature':<40s} {'Scorer':>8s} {'S.Perm':>8s} {'Goals':>8s} {'G.Perm':>8s} {'Disp':>8s} {'D.Perm':>8s} {'AvgRank':>8s}")
        print(f"  {'':40s} {'Rank':>8s} {'':>8s} {'Rank':>8s} {'':>8s} {'Rank':>8s} {'':>8s} {'':>8s}")
        print("-" * 120)
        for r in rows:
            print(f"  {r['feature']:<40s} {r['scorer_shap_rank']:>8d} {r['scorer_perm']:>8.5f}"
                  f" {r['goals_shap_rank']:>8d} {r['goals_perm']:>8.5f}"
                  f" {r['disp_shap_rank']:>8d} {r['disp_perm']:>8.5f}"
                  f" {r['avg_shap_rank']:>8.1f}")

    # Also show features unimportant in 2+ models but not all 3
    only_2 = unimportant_2plus - unimportant_all
    if only_2:
        rows2 = []
        for feat in sorted(only_2):
            s_row = merged_scorer[merged_scorer["feature"] == feat].iloc[0]
            g_row = merged_goals[merged_goals["feature"] == feat].iloc[0]
            d_row = merged_disp[merged_disp["feature"] == feat].iloc[0]
            models = []
            if feat in unimportant_scorer: models.append("Scorer")
            if feat in unimportant_goals: models.append("Goals")
            if feat in unimportant_disp: models.append("Disp")
            rows2.append({
                "feature": feat,
                "models": "+".join(models),
                "scorer_shap_rank": int(s_row["shap_rank"]),
                "scorer_perm": s_row["perm_importance"],
                "goals_shap_rank": int(g_row["shap_rank"]),
                "goals_perm": g_row["perm_importance"],
                "disp_shap_rank": int(d_row["shap_rank"]),
                "disp_perm": d_row["perm_importance"],
            })

        rows2.sort(key=lambda x: x["feature"])

        print(f"\n{'=' * 120}")
        print(f"  FEATURES UNIMPORTANT IN 2 OF 3 MODELS ({len(rows2)} features) — Watch List")
        print(f"{'=' * 120}")
        print(f"  {'Feature':<40s} {'Models':<15s} {'Scorer':>8s} {'S.Perm':>8s} {'Goals':>8s} {'G.Perm':>8s} {'Disp':>8s} {'D.Perm':>8s}")
        print("-" * 120)
        for r in rows2:
            print(f"  {r['feature']:<40s} {r['models']:<15s} {r['scorer_shap_rank']:>8d} {r['scorer_perm']:>8.5f}"
                  f" {r['goals_shap_rank']:>8d} {r['goals_perm']:>8.5f}"
                  f" {r['disp_shap_rank']:>8d} {r['disp_perm']:>8.5f}")

    # Final summary
    print(f"\n{'=' * 80}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total features analyzed:               {n_features}")
    print(f"  Strong removal candidates (all 3):     {len(unimportant_all)}")
    print(f"  Watch list (2 of 3):                   {len(only_2)}")
    print(f"  Total potentially removable:           {len(unimportant_2plus)}")
    print(f"  Features clearly important (>=1 model):{n_features - len(unimportant_2plus)}")
    print(f"\n  NOTE: No features have been removed. Review the tables above before cutting.")

    return unimportant_all, unimportant_2plus


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("  FEATURE IMPORTANCE ANALYSIS")
    print("  SHAP Values + Permutation Importance")
    print("  Training on 2015-2023, evaluating on 2024")
    print("=" * 80)

    df, feature_cols = load_data()
    print(f"\nLoaded {len(df)} rows, {len(feature_cols)} features")
    print(f"Years: {sorted(df['year'].unique())}")

    # Steps 1 & 2: Scoring model
    merged_scorer, merged_goals = analyze_scoring_model(df, feature_cols)

    # Step 3: Disposal model
    merged_disp = analyze_disposal_model(df, feature_cols)

    # Step 4: Cross-model summary
    unimportant_all, unimportant_2plus = cross_model_summary(
        merged_scorer, merged_goals, merged_disp, feature_cols
    )

    print("\nDone.")
