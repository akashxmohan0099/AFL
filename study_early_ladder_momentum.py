"""
Early-season ladder and momentum study (2015-2025).

Runs four analyses:
1) Team-level early ladder (after R6) vs later performance (R7-R24)
2) Player-level spillover by team tier (top 4 / middle / bottom 6)
3) Momentum deep dive for 2024-2025 (4+ win/loss streak effects)
4) SHAP rank check for `team_win_pct_5` across scorer/goals/disposal models
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from pandas.api.types import is_float_dtype

import config

warnings.filterwarnings("ignore", category=FutureWarning)


YEARS = list(range(2015, 2026))


def _tier_from_pos(pos: int) -> str:
    if pos <= 4:
        return "top4"
    if pos <= 12:
        return "middle5_12"
    return "bottom13_18"


def load_data():
    team = pd.read_parquet("data/base/team_matches.parquet").copy()
    feat = pd.read_parquet("data/features/feature_matrix.parquet").copy()
    with open("data/features/feature_columns.json") as f:
        feature_cols = json.load(f)

    team = team[(team["year"].isin(YEARS)) & (~team["is_finals"])].copy()
    team["round_number"] = team["round_number"].astype(int)
    team["date"] = pd.to_datetime(team["date"])

    feat = feat[feat["year"].isin(YEARS)].copy()
    if "did_not_play" in feat.columns:
        feat = feat[feat["did_not_play"] == False].copy()
    feat["round_number"] = feat["round_number"].astype(int)
    feat["date"] = pd.to_datetime(feat["date"])

    # Keep regular-season matches only and align to team table match IDs.
    regular_match_ids = set(team["match_id"].unique())
    feat = feat[feat["match_id"].isin(regular_match_ids)].copy()
    feat = feat[(feat["round_number"] >= 1) & (feat["round_number"] <= 24)].copy()

    # Reconstruct pre-game disposal career average (not stored as a built-in feature).
    feat = feat.sort_values(["player", "date", "match_id"]).reset_index(drop=True)
    g = feat.groupby("player", sort=False)
    di_cumsum = g["DI"].cumsum()
    di_games = g.cumcount()
    feat["career_di_avg_pre"] = np.where(di_games > 0, (di_cumsum - feat["DI"]) / di_games, np.nan)

    return team, feat, feature_cols


def compute_team_ladder_analysis(team: pd.DataFrame):
    rows = []
    for year, sdf in team.groupby("year"):
        sdf = sdf[(sdf["round_number"] >= 1) & (sdf["round_number"] <= 24)].copy()
        early = sdf[sdf["round_number"] <= 6].copy()
        later = sdf[sdf["round_number"] >= 7].copy()

        e = (
            early.groupby("team", as_index=False)
            .agg(
                gp6=("match_id", "count"),
                wins6=("margin", lambda x: int((x > 0).sum())),
                draws6=("margin", lambda x: int((x == 0).sum())),
                points_for6=("score", "sum"),
                points_against6=("opp_score", "sum"),
            )
        )
        e["points6"] = 4 * e["wins6"] + 2 * e["draws6"]
        e["pct6"] = np.where(
            e["points_against6"] > 0,
            100.0 * e["points_for6"] / e["points_against6"],
            np.nan,
        )
        e["early_win_rate"] = np.where(e["gp6"] > 0, e["wins6"] / e["gp6"], np.nan)
        e = e.sort_values(["points6", "pct6", "points_for6"], ascending=False).reset_index(drop=True)
        e["ladder_pos_r6"] = np.arange(1, len(e) + 1)

        l = (
            later.groupby("team", as_index=False)
            .agg(
                later_games=("match_id", "count"),
                later_wins=("margin", lambda x: int((x > 0).sum())),
            )
        )
        l["later_win_rate"] = np.where(l["later_games"] > 0, l["later_wins"] / l["later_games"], np.nan)

        merged = e.merge(l[["team", "later_games", "later_wins", "later_win_rate"]], on="team", how="left")
        merged["later_games"] = merged["later_games"].fillna(0).astype(int)
        merged["later_wins"] = merged["later_wins"].fillna(0).astype(int)
        merged["later_win_rate"] = merged["later_win_rate"].fillna(0.0)
        merged["year"] = int(year)
        merged["tier"] = merged["ladder_pos_r6"].apply(_tier_from_pos)
        merged["win_rate_change_later_minus_early"] = merged["later_win_rate"] - merged["early_win_rate"]
        rows.append(merged)

    df = pd.concat(rows, ignore_index=True)
    pearson = float(df["ladder_pos_r6"].corr(df["later_win_rate"], method="pearson"))
    spearman = float(df["ladder_pos_r6"].corr(df["later_win_rate"], method="spearman"))

    tier_summary = (
        df.groupby("tier", as_index=False)
        .agg(
            teams=("team", "count"),
            early_win_rate=("early_win_rate", "mean"),
            later_win_rate=("later_win_rate", "mean"),
            delta_later_minus_early=("win_rate_change_later_minus_early", "mean"),
        )
    )
    tier_summary["improve_rate"] = [
        float((df[df["tier"] == t]["win_rate_change_later_minus_early"] > 0).mean())
        for t in tier_summary["tier"]
    ]

    return {
        "team_year_tier_table": df,
        "pearson_corr_pos_vs_later_winrate": pearson,
        "spearman_corr_pos_vs_later_winrate": spearman,
        "tier_summary": tier_summary,
    }


def compute_player_spillover(feat: pd.DataFrame, team_tier_df: pd.DataFrame):
    tiers = team_tier_df[["year", "team", "ladder_pos_r6", "tier"]].copy()
    late = feat[(feat["round_number"] >= 7) & (feat["round_number"] <= 24)].copy()
    late = late.merge(tiers, on=["year", "team"], how="inner")

    # Goals spillover (player-game weighted)
    focus = late[late["tier"].isin(["top4", "bottom13_18"])].copy()
    goals_summary = (
        focus.groupby("tier", as_index=False)
        .agg(
            player_games=("player", "count"),
            unique_players=("player", "nunique"),
            avg_goals_r7_r24=("GL", "mean"),
            avg_career_goal_pre=("career_goal_avg_pre", "mean"),
        )
    )
    goals_summary["goal_uplift_abs"] = goals_summary["avg_goals_r7_r24"] - goals_summary["avg_career_goal_pre"]
    goals_summary["goal_uplift_pct"] = np.where(
        goals_summary["avg_career_goal_pre"].abs() > 1e-9,
        100.0 * goals_summary["goal_uplift_abs"] / goals_summary["avg_career_goal_pre"],
        np.nan,
    )

    # Midfielder disposal spillover.
    if "role_midfielder" in late.columns:
        late["is_midfielder"] = late["role_midfielder"].fillna(0).astype(float) >= 0.5
    elif "player_role" in late.columns:
        late["is_midfielder"] = late["player_role"].astype(str).str.lower().eq("midfielder")
    else:
        late["is_midfielder"] = False

    mids = late[(late["is_midfielder"]) & (late["tier"].isin(["top4", "bottom13_18"]))].copy()
    mids = mids[np.isfinite(mids["career_di_avg_pre"])].copy()
    disp_summary = (
        mids.groupby("tier", as_index=False)
        .agg(
            midfielder_games=("player", "count"),
            unique_midfielders=("player", "nunique"),
            avg_di_r7_r24=("DI", "mean"),
            avg_career_di_pre=("career_di_avg_pre", "mean"),
        )
    )
    disp_summary["di_uplift_abs"] = disp_summary["avg_di_r7_r24"] - disp_summary["avg_career_di_pre"]
    disp_summary["di_uplift_pct"] = np.where(
        disp_summary["avg_career_di_pre"].abs() > 1e-9,
        100.0 * disp_summary["di_uplift_abs"] / disp_summary["avg_career_di_pre"],
        np.nan,
    )

    # Control for player talent: goals residual vs career_goal_avg_pre.
    reg_df = late[np.isfinite(late["career_goal_avg_pre"])].copy()
    reg_df = reg_df[reg_df["tier"].isin(["top4", "middle5_12", "bottom13_18"])].copy()

    # Baseline model with individual quality only.
    lm_base = LinearRegression()
    X_base = reg_df[["career_goal_avg_pre"]].values
    y = reg_df["GL"].values
    lm_base.fit(X_base, y)
    reg_df["goal_residual_after_career"] = y - lm_base.predict(X_base)

    resid_summary = (
        reg_df.groupby("tier", as_index=False)["goal_residual_after_career"]
        .mean()
        .rename(columns={"goal_residual_after_career": "avg_residual_goals"})
    )

    # Explicit team-effect model: middle tier as reference.
    reg_df["is_top4"] = (reg_df["tier"] == "top4").astype(int)
    reg_df["is_bottom"] = (reg_df["tier"] == "bottom13_18").astype(int)
    lm_tier = LinearRegression()
    X_tier = reg_df[["career_goal_avg_pre", "is_top4", "is_bottom"]].values
    lm_tier.fit(X_tier, y)

    top4_resid = reg_df.loc[reg_df["tier"] == "top4", "goal_residual_after_career"].values
    mid_resid = reg_df.loc[reg_df["tier"] == "middle5_12", "goal_residual_after_career"].values
    bot_resid = reg_df.loc[reg_df["tier"] == "bottom13_18", "goal_residual_after_career"].values
    t_top4_mid = ttest_ind(top4_resid, mid_resid, equal_var=False, nan_policy="omit")
    t_bot_mid = ttest_ind(bot_resid, mid_resid, equal_var=False, nan_policy="omit")

    return {
        "goals_summary_top4_vs_bottom6": goals_summary,
        "midfielder_disposals_summary_top4_vs_bottom6": disp_summary,
        "residual_goal_summary_by_tier": resid_summary,
        "goal_control_model_coeffs": {
            "intercept": float(lm_tier.intercept_),
            "career_goal_avg_pre_coef": float(lm_tier.coef_[0]),
            "top4_coef_vs_middle": float(lm_tier.coef_[1]),
            "bottom6_coef_vs_middle": float(lm_tier.coef_[2]),
        },
        "residual_tests": {
            "top4_vs_middle_pvalue": float(t_top4_mid.pvalue),
            "bottom6_vs_middle_pvalue": float(t_bot_mid.pvalue),
        },
    }


def compute_momentum_deep_dive(team: pd.DataFrame, feat: pd.DataFrame):
    years = [2024, 2025]
    t = team[(team["year"].isin(years)) & (team["round_number"] <= 24)].copy()

    rolling_rows = []
    events = []
    for (year, team_name), g in t.groupby(["year", "team"]):
        g = g.sort_values(["round_number", "date"]).reset_index(drop=True)
        result = np.sign(g["margin"]).astype(int)  # win=1, draw=0, loss=-1
        win = (result == 1).astype(int)
        rolling5 = pd.Series(win).rolling(5, min_periods=1).mean().values
        for i in range(len(g)):
            rolling_rows.append(
                {
                    "year": int(year),
                    "team": team_name,
                    "round_number": int(g.loc[i, "round_number"]),
                    "rolling5_win_pct": float(rolling5[i]),
                }
            )

        streak = 0
        for i, r in enumerate(result):
            if r == 1:
                streak = streak + 1 if streak > 0 else 1
            elif r == -1:
                streak = streak - 1 if streak < 0 else -1
            else:
                streak = 0

            event_type = None
            if streak == 4:
                event_type = "win_streak_4plus_trigger"
            elif streak == -4:
                event_type = "loss_streak_4plus_trigger"

            if event_type is None:
                continue

            next_games = g.iloc[i + 1 : i + 4].copy()
            if next_games.empty:
                continue
            events.append(
                {
                    "year": int(year),
                    "team": team_name,
                    "event_type": event_type,
                    "trigger_round": int(g.loc[i, "round_number"]),
                    "next_rounds": [int(x) for x in next_games["round_number"].tolist()],
                    "next_match_ids": [int(x) for x in next_games["match_id"].tolist()],
                }
            )

    rolling_df = pd.DataFrame(rolling_rows)
    events_df = pd.DataFrame(events)

    p = feat[(feat["year"].isin(years)) & (feat["round_number"] <= 24)].copy()
    season_player_avg = (
        p.groupby(["year", "player"], observed=False)[["GL", "DI"]]
        .mean()
        .rename(columns={"GL": "player_season_gl_avg", "DI": "player_season_di_avg"})
        .reset_index()
    )

    event_effect_rows = []
    for ev in events:
        mask = (
            (p["year"] == ev["year"])
            & (p["team"] == ev["team"])
            & (p["match_id"].isin(ev["next_match_ids"]))
        )
        subset = p.loc[mask, ["year", "team", "match_id", "player", "GL", "DI"]].copy()
        if subset.empty:
            continue
        subset = subset.merge(season_player_avg, on=["year", "player"], how="left")
        subset["delta_gl_vs_player_season"] = subset["GL"] - subset["player_season_gl_avg"]
        subset["delta_di_vs_player_season"] = subset["DI"] - subset["player_season_di_avg"]
        event_effect_rows.append(
            {
                "year": ev["year"],
                "team": ev["team"],
                "event_type": ev["event_type"],
                "trigger_round": ev["trigger_round"],
                "next_rounds": ev["next_rounds"],
                "n_next_games": len(ev["next_rounds"]),
                "n_player_games": int(len(subset)),
                "n_players": int(subset["player"].nunique()),
                "avg_gl_next3": float(subset["GL"].mean()),
                "avg_di_next3": float(subset["DI"].mean()),
                "avg_delta_gl_vs_player_season": float(subset["delta_gl_vs_player_season"].mean()),
                "avg_delta_di_vs_player_season": float(subset["delta_di_vs_player_season"].mean()),
            }
        )

    event_effects_df = pd.DataFrame(event_effect_rows)
    summary = (
        event_effects_df.groupby(["year", "event_type"], as_index=False)
        .agg(
            events=("team", "count"),
            player_games=("n_player_games", "sum"),
            avg_delta_gl_vs_player_season=("avg_delta_gl_vs_player_season", "mean"),
            avg_delta_di_vs_player_season=("avg_delta_di_vs_player_season", "mean"),
            avg_gl_next3=("avg_gl_next3", "mean"),
            avg_di_next3=("avg_di_next3", "mean"),
        )
    )

    # Representative examples: largest absolute combined shift per year/event type.
    examples = []
    if not event_effects_df.empty:
        temp = event_effects_df.copy()
        temp["abs_shift_score"] = (
            temp["avg_delta_gl_vs_player_season"].abs() + temp["avg_delta_di_vs_player_season"].abs()
        )
        for (year, event_type), g in temp.groupby(["year", "event_type"]):
            best = g.sort_values("abs_shift_score", ascending=False).head(2)
            for _, r in best.iterrows():
                examples.append(
                    {
                        "year": int(year),
                        "event_type": event_type,
                        "team": r["team"],
                        "trigger_round": int(r["trigger_round"]),
                        "next_rounds": r["next_rounds"],
                        "avg_delta_gl_vs_player_season": float(r["avg_delta_gl_vs_player_season"]),
                        "avg_delta_di_vs_player_season": float(r["avg_delta_di_vs_player_season"]),
                        "n_player_games": int(r["n_player_games"]),
                    }
                )

    return {
        "team_rolling5": rolling_df,
        "streak_events": events_df,
        "event_effects": event_effects_df,
        "event_effect_summary": summary,
        "examples": pd.DataFrame(examples),
    }


def compute_team_win_pct_5_shap_rank(feat: pd.DataFrame, feature_cols: list[str]):
    target_feature = "team_win_pct_5"
    if target_feature not in feature_cols or target_feature not in feat.columns:
        return {"exists": False}

    try:
        import shap
    except Exception as e:
        return {"exists": True, "error": f"shap import failed: {e}"}

    df = feat.copy()
    if "career_games_pre" in df.columns:
        df = df[df["career_games_pre"] >= config.MIN_PLAYER_MATCHES].copy()
    train = df[df["year"] < 2024].copy()
    val = df[df["year"] == 2024].copy()
    # Keep this diagnostic lightweight.
    if len(train) > 40000:
        train = train.sample(n=40000, random_state=42)
    if len(val) > 12000:
        val = val.sample(n=12000, random_state=42)

    valid_cols = [c for c in feature_cols if c in df.columns]
    X_train = train[valid_cols].fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)
    X_val = val[valid_cols].fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)
    y_train_gl = train["GL"].values
    y_val_gl = val["GL"].values
    y_train_di = train["DI"].values

    gbt_params = dict(getattr(config, "GBT_PARAMS_BACKTEST", config.GBT_PARAMS))
    scorer = GradientBoostingClassifier(**gbt_params)
    scorer.fit(X_train, (y_train_gl >= 1).astype(int))

    scorer_mask_train = y_train_gl >= 1
    scorer_mask_val = y_val_gl >= 1
    goals = GradientBoostingRegressor(**gbt_params)
    goals.fit(X_train.loc[scorer_mask_train], y_train_gl[scorer_mask_train])

    disp = GradientBoostingRegressor(**gbt_params)
    disp.fit(X_train, y_train_di)

    def _rank_from_shap(model, X: pd.DataFrame, feat_name: str):
        sample_n = min(1500, len(X))
        Xs = X.sample(n=sample_n, random_state=42) if sample_n < len(X) else X
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(Xs)
        if isinstance(vals, list):
            vals = vals[1]
        imp = np.mean(np.abs(vals), axis=0)
        order = np.argsort(-imp)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(imp) + 1)
        idx = X.columns.get_loc(feat_name)
        return {
            "shap_importance": float(imp[idx]),
            "shap_rank": int(ranks[idx]),
            "n_features": int(len(imp)),
            "rank_percentile": float(100.0 * ranks[idx] / len(imp)),
            "sample_size": int(sample_n),
        }

    scorer_rank = _rank_from_shap(scorer, X_val, target_feature)
    X_val_scorers = X_val.loc[scorer_mask_val]
    goals_rank = _rank_from_shap(goals, X_val_scorers, target_feature)
    disp_rank = _rank_from_shap(disp, X_val, target_feature)

    return {
        "exists": True,
        "target_feature": target_feature,
        "scorer_classifier": scorer_rank,
        "goals_gbt": goals_rank,
        "disposals_gbt": disp_rank,
    }


def df_to_records(df: pd.DataFrame, decimals: int = 6):
    if df is None or df.empty:
        return []
    out = df.copy()
    for c in out.columns:
        if is_float_dtype(out[c]):
            out[c] = out[c].round(decimals)
    return json.loads(out.to_json(orient="records"))


def main():
    team, feat, feature_cols = load_data()

    part1 = compute_team_ladder_analysis(team)
    part2 = compute_player_spillover(feat, part1["team_year_tier_table"])
    part3 = compute_momentum_deep_dive(team, feat)
    part4 = compute_team_win_pct_5_shap_rank(feat, feature_cols)

    output = {
        "meta": {
            "years": YEARS,
            "notes": [
                "Regular season only (non-finals); rounds capped at 24 for comparability.",
                "Team tiers based on ladder position after Round 6.",
                "Player-level comparisons are player-game weighted unless stated.",
            ],
        },
        "part1_team_level": {
            "pearson_corr_pos_vs_later_winrate": round(part1["pearson_corr_pos_vs_later_winrate"], 6),
            "spearman_corr_pos_vs_later_winrate": round(part1["spearman_corr_pos_vs_later_winrate"], 6),
            "tier_summary": df_to_records(part1["tier_summary"], decimals=6),
            "team_year_tier_table": df_to_records(part1["team_year_tier_table"], decimals=6),
        },
        "part2_player_spillover": {
            "goals_summary_top4_vs_bottom6": df_to_records(part2["goals_summary_top4_vs_bottom6"], decimals=6),
            "midfielder_disposals_summary_top4_vs_bottom6": df_to_records(
                part2["midfielder_disposals_summary_top4_vs_bottom6"], decimals=6
            ),
            "residual_goal_summary_by_tier": df_to_records(part2["residual_goal_summary_by_tier"], decimals=6),
            "goal_control_model_coeffs": part2["goal_control_model_coeffs"],
            "residual_tests": part2["residual_tests"],
        },
        "part3_momentum_2024_2025": {
            "event_effect_summary": df_to_records(part3["event_effect_summary"], decimals=6),
            "examples": df_to_records(part3["examples"], decimals=6),
            "streak_events": df_to_records(part3["streak_events"], decimals=6),
            "event_effects": df_to_records(part3["event_effects"], decimals=6),
            "team_rolling5": df_to_records(part3["team_rolling5"], decimals=6),
        },
        "part4_model_signal_check": part4,
    }

    out_path = Path("momentum_study_2015_2025.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(out_path.resolve())

    # Concise console summary for quick inspection.
    print("\n=== PART 1 ===")
    print("pearson:", round(part1["pearson_corr_pos_vs_later_winrate"], 4))
    print("spearman:", round(part1["spearman_corr_pos_vs_later_winrate"], 4))
    print(part1["tier_summary"][["tier", "early_win_rate", "later_win_rate", "delta_later_minus_early"]].round(4))

    print("\n=== PART 2 ===")
    print(part2["goals_summary_top4_vs_bottom6"][["tier", "avg_goals_r7_r24", "avg_career_goal_pre", "goal_uplift_abs"]].round(4))
    print(part2["midfielder_disposals_summary_top4_vs_bottom6"][["tier", "avg_di_r7_r24", "avg_career_di_pre", "di_uplift_abs"]].round(4))
    print("control coeffs:", {k: round(v, 6) for k, v in part2["goal_control_model_coeffs"].items()})
    print("residual p-values:", {k: round(v, 6) for k, v in part2["residual_tests"].items()})

    print("\n=== PART 3 ===")
    print(part3["event_effect_summary"][["year", "event_type", "events", "avg_delta_gl_vs_player_season", "avg_delta_di_vs_player_season"]].round(4))
    if not part3["examples"].empty:
        print(part3["examples"][["year", "event_type", "team", "trigger_round", "next_rounds", "avg_delta_gl_vs_player_season", "avg_delta_di_vs_player_season"]].round(4))

    print("\n=== PART 4 ===")
    print(part4)


if __name__ == "__main__":
    main()
