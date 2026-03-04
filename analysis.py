"""
AFL Prediction Pipeline — Round Analysis Engine
=================================================
Generates per-round analysis from predictions, outcomes, and game results.
All output is JSON-safe (native Python types, no numpy/Timestamp objects).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

import config


def _to_native(val):
    """Convert numpy/pandas types to native Python for JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (pd.Timestamp,)):
        return val.isoformat()
    if isinstance(val, (np.ndarray,)):
        return [_to_native(v) for v in val]
    return val


def _safe_float(val, decimals=4):
    """Round and convert to native float, handling NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), decimals)


def generate_round_analysis(year, round_num, predictions, outcomes,
                            game_preds, game_actuals, feature_df, store):
    """Generate comprehensive round analysis.

    Args:
        year: Season year.
        round_num: Round number.
        predictions: DataFrame with predicted goals/behinds/disposals.
        outcomes: DataFrame with actual goals/behinds/disposals.
        game_preds: DataFrame with game-level predictions (or empty).
        game_actuals: DataFrame with actual game results (team_match for this round).
        feature_df: Feature matrix for this round's players.
        store: LearningStore instance.

    Returns:
        JSON-serializable dict with analysis sections.
    """
    analysis = {
        "year": int(year),
        "round": int(round_num),
    }

    # Merge predictions with outcomes
    merged = _merge_pred_outcome(predictions, outcomes)

    analysis["summary"] = _compute_summary(merged)
    analysis["calibration"] = compute_calibration_drift(store, year, round_num)
    analysis["hot_players"] = _find_hot_players(store, year, round_num)
    analysis["cold_players"] = _find_cold_players(store, year, round_num)
    analysis["biggest_misses"] = _biggest_misses(merged)
    analysis["biggest_hits"] = _biggest_hits(merged)
    analysis["weather_impact"] = _weather_impact(merged, feature_df)
    analysis["team_analysis"] = _team_analysis(merged)
    analysis["game_results"] = _game_results(game_preds, game_actuals)
    analysis["model_improvement"] = _model_improvement(store, year, round_num, merged)
    analysis["streaks"] = _streak_summary(store, year, round_num)
    analysis["miss_classification"] = _build_miss_summary(merged, feature_df)
    analysis["archetype_drift"] = _archetype_drift_analysis(merged, feature_df)
    analysis["concession_drift"] = _concession_drift_analysis(merged, feature_df)

    return analysis


def _merge_pred_outcome(predictions, outcomes):
    """Merge prediction and outcome DataFrames on player+team."""
    if predictions.empty or outcomes.empty:
        return pd.DataFrame()

    # Determine join columns
    join_cols = ["player", "team"]
    if "match_id" in predictions.columns and "match_id" in outcomes.columns:
        join_cols.append("match_id")

    pred_cols = [c for c in predictions.columns if c not in outcomes.columns or c in join_cols]
    merged = predictions[pred_cols].merge(outcomes, on=join_cols, how="inner")
    return merged


def _extract_threshold_data(merged):
    """Extract predicted probabilities and actual binary outcomes per threshold.

    Returns dict of {name: (pred_probs, actual_binary)} arrays.
    """
    results = {}

    if merged.empty or "actual_goals" not in merged.columns:
        return results

    actual_goals = merged["actual_goals"].values

    # Goal thresholds
    if "p_1plus_goals" in merged.columns:
        results["1plus_goals"] = (
            merged["p_1plus_goals"].values.astype(float),
            (actual_goals >= 1).astype(int),
        )
    elif "p_scorer" in merged.columns:
        results["1plus_goals"] = (
            merged["p_scorer"].values.astype(float),
            (actual_goals >= 1).astype(int),
        )

    if "p_2plus_goals" in merged.columns:
        results["2plus_goals"] = (
            merged["p_2plus_goals"].values.astype(float),
            (actual_goals >= 2).astype(int),
        )
    elif "p_goals_0" in merged.columns and "p_goals_1" in merged.columns:
        p_2plus = 1.0 - merged["p_goals_0"].values.astype(float) - merged["p_goals_1"].values.astype(float)
        results["2plus_goals"] = (np.clip(p_2plus, 0.0, 1.0), (actual_goals >= 2).astype(int))

    if "p_3plus_goals" in merged.columns:
        results["3plus_goals"] = (
            merged["p_3plus_goals"].values.astype(float),
            (actual_goals >= 3).astype(int),
        )
    elif "p_goals_0" in merged.columns and "p_goals_1" in merged.columns and "p_goals_2" in merged.columns:
        p_2plus = 1.0 - merged["p_goals_0"].values.astype(float) - merged["p_goals_1"].values.astype(float)
        p_3plus = p_2plus - merged["p_goals_2"].values.astype(float)
        results["3plus_goals"] = (np.clip(p_3plus, 0.0, 1.0), (actual_goals >= 3).astype(int))

    # Disposal thresholds
    if "actual_disposals" in merged.columns:
        actual_disp = merged["actual_disposals"].values
        for name, threshold in config.DISPOSAL_THRESHOLDS_EVAL.items():
            col = f"p_{name}"
            if col in merged.columns:
                results[name] = (
                    merged[col].values.astype(float),
                    (actual_disp >= threshold).astype(int),
                )

    return results


def _compute_threshold_metrics(predicted_probs, actual_binary, n_buckets=None):
    """Compute Brier score, log loss, and calibration curve for a threshold.

    Returns dict with brier_score, log_loss, n, base_rate, calibration_curve.
    Returns None if n < 10.
    """
    if n_buckets is None:
        n_buckets = config.CALIBRATION_N_BUCKETS

    n = len(predicted_probs)
    if n < 10:
        return None

    p = np.asarray(predicted_probs, dtype=float)
    y = np.asarray(actual_binary, dtype=float)

    # Brier score
    brier = float(np.mean((p - y) ** 2))

    # Log loss (clip to avoid log(0))
    p_clip = np.clip(p, 1e-15, 1.0 - 1e-15)
    log_loss_val = float(np.mean(-y * np.log(p_clip) - (1 - y) * np.log(1 - p_clip)))

    base_rate = float(np.mean(y))

    # Calibration curve: equal-width bins
    bin_edges = np.linspace(0.0, 1.0, n_buckets + 1)
    cal_curve = []
    for i in range(n_buckets):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        if i < n_buckets - 1:
            mask = (p >= lower) & (p < upper)
        else:
            mask = (p >= lower) & (p <= upper)
        count = int(mask.sum())
        if count < config.CALIBRATION_MIN_BUCKET_SIZE:
            continue
        cal_curve.append({
            "bin_lower": round(float(lower), 2),
            "bin_upper": round(float(upper), 2),
            "predicted_mean": round(float(p[mask].mean()), 4),
            "observed_mean": round(float(y[mask].mean()), 4),
            "count": count,
        })

    return {
        "brier_score": round(brier, 4),
        "log_loss": round(log_loss_val, 4),
        "n": n,
        "base_rate": round(base_rate, 4),
        "calibration_curve": cal_curve,
    }


def _compute_summary(merged):
    """Compute summary metrics for the round."""
    if merged.empty:
        return {}

    summary = {}
    if "predicted_goals" in merged.columns and "actual_goals" in merged.columns:
        pred_gl = merged["predicted_goals"].values
        actual_gl = merged["actual_goals"].values
        summary["goals_mae"] = _safe_float(mean_absolute_error(actual_gl, pred_gl))
        summary["goals_rmse"] = _safe_float(np.sqrt(mean_squared_error(actual_gl, pred_gl)))

        if "career_goal_avg" in merged.columns or "career_goal_avg_pre" in merged.columns:
            _col = "career_goal_avg_pre" if "career_goal_avg_pre" in merged.columns else "career_goal_avg"
            baseline = merged[_col].fillna(0).values
            baseline_mae = mean_absolute_error(actual_gl, baseline)
            summary["baseline_mae"] = _safe_float(baseline_mae)
            if baseline_mae > 0:
                summary["improvement_pct"] = _safe_float(
                    (baseline_mae - summary["goals_mae"]) / baseline_mae * 100, 1
                )

        # Scorer AUC
        if "p_scorer" in merged.columns:
            actual_scored = (actual_gl >= 1).astype(int)
            try:
                summary["scorer_auc"] = _safe_float(
                    roc_auc_score(actual_scored, merged["p_scorer"])
                )
            except ValueError:
                summary["scorer_auc"] = None

        # Calibration ratio
        mean_pred = float(pred_gl.mean())
        mean_actual = float(actual_gl.mean())
        summary["calibration_ratio"] = _safe_float(
            mean_pred / mean_actual if mean_actual > 0 else None
        )

    if "predicted_behinds" in merged.columns and "actual_behinds" in merged.columns:
        summary["behinds_mae"] = _safe_float(mean_absolute_error(
            merged["actual_behinds"].values, merged["predicted_behinds"].values
        ))

    if "predicted_disposals" in merged.columns and "actual_disposals" in merged.columns:
        summary["disposals_mae"] = _safe_float(mean_absolute_error(
            merged["actual_disposals"].values, merged["predicted_disposals"].values
        ))

    summary["n_players"] = int(len(merged))

    # Threshold probability metrics (Brier, log loss per threshold)
    threshold_data = _extract_threshold_data(merged)
    threshold_metrics = {}
    for name, (preds, actuals) in threshold_data.items():
        metrics = _compute_threshold_metrics(preds, actuals)
        if metrics:
            threshold_metrics[name] = {
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
                "n": metrics["n"],
                "base_rate": metrics["base_rate"],
            }
    summary["threshold_metrics"] = threshold_metrics

    return summary


def compute_player_streaks(store, year, up_to_round):
    """Compute per-player scoring and cold streaks from stored outcomes.

    Returns dict keyed by (player, team) with:
        scoring_streak, cold_streak, disposal_form (avg last 3)
    """
    streaks = {}
    recent_disposals = {}  # (player, team) -> list of last 3 disposal counts

    for rnd in range(1, up_to_round + 1):
        outcomes = store.load_outcomes(year, rnd)
        if outcomes.empty:
            continue

        for _, row in outcomes.iterrows():
            key = (str(row["player"]), str(row["team"]))
            goals = int(row.get("actual_goals", 0))

            if key not in streaks:
                streaks[key] = {"scoring_streak": 0, "cold_streak": 0}

            if goals >= 1:
                streaks[key]["scoring_streak"] += 1
                streaks[key]["cold_streak"] = 0
            else:
                streaks[key]["cold_streak"] += 1
                streaks[key]["scoring_streak"] = 0

            # Disposal form
            disp = float(row.get("actual_disposals", 0))
            if key not in recent_disposals:
                recent_disposals[key] = []
            recent_disposals[key].append(disp)
            if len(recent_disposals[key]) > 3:
                recent_disposals[key] = recent_disposals[key][-3:]

    # Add disposal form to streaks
    for key, disps in recent_disposals.items():
        if key in streaks:
            streaks[key]["disposal_form"] = round(np.mean(disps), 1) if disps else 0.0

    return streaks


def compute_calibration_drift(store, year, round_num):
    """Compare predicted vs observed rates per bucket.

    Returns dict with overall_drift, bucket_drifts, trend.
    """
    cal = store.get_calibration_state()
    if cal.empty:
        return {"overall_drift": 0.0, "trend": "stable", "active_buckets": 0}

    active = cal[cal["n_predictions"] > 0].copy()
    if active.empty:
        return {"overall_drift": 0.0, "trend": "stable", "active_buckets": 0}

    # Drift = abs(observed_rate - bucket_midpoint)
    active = active.copy()
    active["drift"] = (active["observed_rate"] - active["probability_bucket"]).abs()
    overall_drift = float(active["drift"].mean())

    # Trend assessment: compare early vs late calibration adjustments
    # Simple heuristic: if mean abs(calibration_adj) is decreasing, improving
    trend = "stable"
    active_adj = cal[cal["calibration_adj"] != 0]
    if len(active_adj) > 0:
        mean_adj = float(active_adj["calibration_adj"].abs().mean())
        if mean_adj < 0.05:
            trend = "improving"
        elif mean_adj > 0.15:
            trend = "degrading"

    bucket_drifts = []
    for _, row in active.head(10).iterrows():
        bucket_drifts.append({
            "target": str(row["target"]),
            "bucket": _safe_float(row["probability_bucket"]),
            "observed_rate": _safe_float(row["observed_rate"]),
            "n_predictions": int(row["n_predictions"]),
            "drift": _safe_float(row["drift"]),
        })

    return {
        "overall_drift": _safe_float(overall_drift),
        "trend": trend,
        "active_buckets": int(len(active)),
        "bucket_details": bucket_drifts,
    }


def _find_hot_players(store, year, round_num):
    """Find players scoring in 3+ consecutive rounds."""
    streaks = compute_player_streaks(store, year, round_num)
    hot = []
    for (player, team), info in streaks.items():
        if info["scoring_streak"] >= 3:
            hot.append({
                "player": player,
                "team": team,
                "streak_length": int(info["scoring_streak"]),
            })
    hot.sort(key=lambda x: x["streak_length"], reverse=True)
    return hot[:20]


def _find_cold_players(store, year, round_num):
    """Find players scoreless for 4+ rounds."""
    streaks = compute_player_streaks(store, year, round_num)
    cold = []
    for (player, team), info in streaks.items():
        if info["cold_streak"] >= 4:
            cold.append({
                "player": player,
                "team": team,
                "rounds_without_goal": int(info["cold_streak"]),
            })
    cold.sort(key=lambda x: x["rounds_without_goal"], reverse=True)
    return cold[:20]


def _biggest_misses(merged):
    """Top 5 predictions by absolute goal error."""
    if merged.empty or "predicted_goals" not in merged.columns:
        return []

    merged = merged.copy()
    merged["abs_error"] = (merged["predicted_goals"] - merged["actual_goals"]).abs()
    top = merged.nlargest(5, "abs_error")

    misses = []
    for _, row in top.iterrows():
        misses.append({
            "player": str(row["player"]),
            "team": str(row["team"]),
            "predicted": _safe_float(row["predicted_goals"], 2),
            "actual": int(row["actual_goals"]),
            "error": _safe_float(row["predicted_goals"] - row["actual_goals"], 2),
        })
    return misses


def _biggest_hits(merged):
    """Top 5 most accurate predictions (lowest absolute error among scorers)."""
    if merged.empty or "predicted_goals" not in merged.columns:
        return []

    merged = merged.copy()
    # Focus on players who scored — accurately predicting 0 goals isn't impressive
    scorers = merged[merged["actual_goals"] >= 1].copy()
    if scorers.empty:
        return []

    scorers["abs_error"] = (scorers["predicted_goals"] - scorers["actual_goals"]).abs()
    top = scorers.nsmallest(5, "abs_error")

    hits = []
    for _, row in top.iterrows():
        hits.append({
            "player": str(row["player"]),
            "team": str(row["team"]),
            "predicted": _safe_float(row["predicted_goals"], 2),
            "actual": int(row["actual_goals"]),
            "error": _safe_float(row["predicted_goals"] - row["actual_goals"], 2),
        })
    return hits


def _weather_impact(merged, feature_df):
    """Assess impact of weather on predictions this round.

    Extracts per-match weather from feature_df, identifies wet/difficult
    matches, and compares goals MAE in weather-affected vs normal matches.
    """
    if feature_df is None or feature_df.empty:
        return {"wet_matches": 0, "note": "No weather data available"}

    if "is_wet" not in feature_df.columns and "weather_difficulty_score" not in feature_df.columns:
        return {"wet_matches": 0, "note": "No weather features in data"}

    if merged.empty or "predicted_goals" not in merged.columns:
        return {"wet_matches": 0, "note": "No predictions to compare"}

    # Extract per-match weather (one row per match)
    weather_cols = ["match_id"]
    for c in ["is_wet", "is_heavy_rain", "precipitation_total", "wind_gusts_max",
              "weather_difficulty_score", "wind_severity", "slippery_conditions"]:
        if c in feature_df.columns:
            weather_cols.append(c)

    match_weather = feature_df[weather_cols].groupby("match_id", observed=True).first().reset_index()

    # Identify wet/difficult matches
    wet_mask = pd.Series(False, index=match_weather.index)
    if "is_wet" in match_weather.columns:
        wet_mask = wet_mask | (match_weather["is_wet"] == 1)
    if "weather_difficulty_score" in match_weather.columns:
        wet_mask = wet_mask | (match_weather["weather_difficulty_score"] > 5)

    total_matches = len(match_weather)
    wet_matches = int(wet_mask.sum())

    avg_difficulty = _safe_float(
        match_weather["weather_difficulty_score"].mean(), 2
    ) if "weather_difficulty_score" in match_weather.columns else None

    result = {
        "wet_matches": wet_matches,
        "total_matches": total_matches,
        "avg_weather_difficulty": avg_difficulty,
    }

    if wet_matches == 0:
        result["note"] = "No wet or difficult-weather matches this round"
        return result

    # Merge weather into predictions via match_id
    if "match_id" not in merged.columns:
        result["note"] = "Cannot link weather to predictions (no match_id)"
        return result

    merged_w = merged.merge(match_weather, on="match_id", how="left")
    # Recompute wet flag on merged data
    merged_wet_mask = pd.Series(False, index=merged_w.index)
    if "is_wet" in merged_w.columns:
        merged_wet_mask = merged_wet_mask | (merged_w["is_wet"] == 1)
    if "weather_difficulty_score" in merged_w.columns:
        merged_wet_mask = merged_wet_mask | (merged_w["weather_difficulty_score"] > 5)

    wet_rows = merged_w[merged_wet_mask]
    dry_rows = merged_w[~merged_wet_mask]

    # Per-match wet details
    wet_match_ids = match_weather.loc[wet_mask, "match_id"].values
    wet_details = []
    for mid in wet_match_ids:
        mw = match_weather[match_weather["match_id"] == mid].iloc[0]
        match_rows = merged_w[merged_w["match_id"] == mid]
        detail = {"match_id": _to_native(mid)}

        # Try to get venue from feature_df
        if "venue" in feature_df.columns:
            venue_vals = feature_df.loc[feature_df["match_id"] == mid, "venue"]
            if not venue_vals.empty:
                detail["venue"] = str(venue_vals.iloc[0])

        if "precipitation_total" in mw.index:
            detail["precipitation"] = _safe_float(mw["precipitation_total"], 1)
        if "weather_difficulty_score" in mw.index:
            detail["weather_difficulty"] = _safe_float(mw["weather_difficulty_score"], 1)
        if "wind_gusts_max" in mw.index:
            detail["wind_gusts"] = _safe_float(mw["wind_gusts_max"], 1)

        if not match_rows.empty and "predicted_goals" in match_rows.columns:
            detail["goals_mae"] = _safe_float(
                mean_absolute_error(match_rows["actual_goals"], match_rows["predicted_goals"])
            )

        wet_details.append(detail)

    result["wet_match_details"] = wet_details

    # Wet vs dry comparison
    wet_vs_dry = {}
    if not wet_rows.empty and "actual_goals" in wet_rows.columns:
        # Goals per team in wet matches
        if "team" in wet_rows.columns:
            wet_team_goals = wet_rows.groupby(["match_id", "team"], observed=True)["actual_goals"].sum()
            wet_vs_dry["wet_goals_per_team"] = _safe_float(wet_team_goals.mean(), 1)
        wet_vs_dry["wet_mae"] = _safe_float(
            mean_absolute_error(wet_rows["actual_goals"], wet_rows["predicted_goals"])
        )

    if not dry_rows.empty and "actual_goals" in dry_rows.columns:
        if "team" in dry_rows.columns:
            dry_team_goals = dry_rows.groupby(["match_id", "team"], observed=True)["actual_goals"].sum()
            wet_vs_dry["dry_goals_per_team"] = _safe_float(dry_team_goals.mean(), 1)
        wet_vs_dry["dry_mae"] = _safe_float(
            mean_absolute_error(dry_rows["actual_goals"], dry_rows["predicted_goals"])
        )

    result["wet_vs_dry"] = wet_vs_dry
    return result


def _team_analysis(merged):
    """Per-team predicted vs actual totals."""
    if merged.empty or "predicted_goals" not in merged.columns:
        return []

    teams = []
    for team, grp in merged.groupby("team", observed=True):
        pred_total = float(grp["predicted_goals"].sum())
        actual_total = int(grp["actual_goals"].sum())
        mae = float((grp["predicted_goals"] - grp["actual_goals"]).abs().mean())

        entry = {
            "team": str(team),
            "predicted_goals": _safe_float(pred_total, 1),
            "actual_goals": int(actual_total),
            "mae": _safe_float(mae),
            "n_players": int(len(grp)),
        }

        # Best and worst predictions within team
        grp = grp.copy()
        grp["signed_error"] = grp["predicted_goals"] - grp["actual_goals"]
        if len(grp) > 0:
            best_idx = grp["signed_error"].abs().idxmin()
            worst_idx = grp["signed_error"].abs().idxmax()
            entry["best_pred"] = str(grp.loc[best_idx, "player"])
            entry["worst_pred"] = str(grp.loc[worst_idx, "player"])

        teams.append(entry)

    teams.sort(key=lambda x: x["mae"])
    return teams


def _game_results(game_preds, game_actuals):
    """Per-match: predicted vs actual winner, margin accuracy."""
    if game_preds is None or game_preds.empty:
        return []

    results = []
    for _, pred in game_preds.iterrows():
        match_id = pred.get("match_id")
        entry = {
            "match_id": _to_native(match_id),
            "home_team": str(pred.get("home_team", "")),
            "away_team": str(pred.get("away_team", "")),
            "home_win_prob": _safe_float(pred.get("home_win_prob", 0.5)),
            "predicted_margin": _safe_float(pred.get("predicted_margin", 0), 1),
            "predicted_winner": str(pred.get("predicted_winner", "")),
        }

        # Match with actuals
        if game_actuals is not None and not game_actuals.empty:
            home_actual = game_actuals[
                (game_actuals["match_id"] == match_id) & (game_actuals["is_home"])
            ]
            if not home_actual.empty:
                actual_margin = int(home_actual.iloc[0]["margin"])
                actual_winner = str(home_actual.iloc[0]["team"]) if actual_margin > 0 else str(home_actual.iloc[0]["opponent"])
                if actual_margin == 0:
                    actual_winner = "Draw"
                entry["actual_margin"] = int(actual_margin)
                entry["actual_winner"] = actual_winner
                entry["correct"] = entry["predicted_winner"] == actual_winner
                entry["margin_error"] = _safe_float(
                    abs(entry.get("predicted_margin", 0) - actual_margin), 1
                )

        results.append(entry)

    return results


def _model_improvement(store, year, round_num, merged):
    """Track cumulative MAE and trend direction."""
    if merged.empty or "predicted_goals" not in merged.columns:
        return {}

    this_mae = float((merged["predicted_goals"] - merged["actual_goals"]).abs().mean())

    # Gather MAEs from prior rounds
    prior_maes = []
    for rnd in range(1, round_num):
        preds = store.load_predictions(year, rnd)
        outs = store.load_outcomes(year, rnd)
        if preds.empty or outs.empty:
            continue
        m = _merge_pred_outcome(preds, outs)
        if not m.empty and "predicted_goals" in m.columns and "actual_goals" in m.columns:
            prior_maes.append(float((m["predicted_goals"] - m["actual_goals"]).abs().mean()))

    all_maes = prior_maes + [this_mae]
    cumulative_mae = _safe_float(np.mean(all_maes))
    running_avg = _safe_float(np.mean(prior_maes)) if prior_maes else None

    # Trend: compare this round to running average
    if running_avg is not None and running_avg > 0:
        if this_mae < running_avg * 0.95:
            trend = "improving"
        elif this_mae > running_avg * 1.05:
            trend = "degrading"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"

    return {
        "this_round_mae": _safe_float(this_mae),
        "cumulative_mae": cumulative_mae,
        "running_avg_mae": running_avg,
        "trend": trend,
        "n_prior_rounds": int(len(prior_maes)),
    }


def _streak_summary(store, year, round_num):
    """Summarize streak changes this round."""
    if round_num < 2:
        return {"new_hot": [], "continued_hot": [], "broken_hot": [],
                "new_cold": [], "note": "First round — no streak data"}

    prev_streaks = compute_player_streaks(store, year, round_num - 1)
    curr_streaks = compute_player_streaks(store, year, round_num)

    new_hot = []
    continued_hot = []
    broken_hot = []
    new_cold = []

    for key, curr in curr_streaks.items():
        prev = prev_streaks.get(key, {"scoring_streak": 0, "cold_streak": 0})

        # Hot streaks
        if curr["scoring_streak"] >= 3 and prev["scoring_streak"] < 3:
            new_hot.append({"player": key[0], "team": key[1],
                           "streak": int(curr["scoring_streak"])})
        elif curr["scoring_streak"] >= 3 and prev["scoring_streak"] >= 3:
            continued_hot.append({"player": key[0], "team": key[1],
                                  "streak": int(curr["scoring_streak"])})

        # Broken hot streaks
        if prev["scoring_streak"] >= 3 and curr["scoring_streak"] == 0:
            broken_hot.append({"player": key[0], "team": key[1],
                              "was_streak": int(prev["scoring_streak"])})

        # New cold streaks (just crossed 4-round threshold)
        if curr["cold_streak"] == 4 and prev["cold_streak"] == 3:
            new_cold.append({"player": key[0], "team": key[1]})

    return {
        "new_hot": new_hot[:10],
        "continued_hot": continued_hot[:10],
        "broken_hot": broken_hot[:10],
        "new_cold": new_cold[:10],
    }


# ------------------------------------------------------------------
# Miss Classification
# ------------------------------------------------------------------

def classify_prediction_misses(merged, feature_df):
    """Classify why significant prediction misses occurred.

    For each player with abs(predicted_goals - actual_goals) >= threshold,
    assigns a primary miss type and list of contributing factors.

    Priority order: opportunity → team_environment → role → matchup → variance

    Returns DataFrame with columns: player, team, miss_type, contributing_factors,
        predicted_goals, actual_goals, abs_error
    """
    if merged.empty or "predicted_goals" not in merged.columns:
        return pd.DataFrame()

    threshold = config.MISS_ERROR_THRESHOLD

    m = merged.copy()
    m["abs_error"] = (m["predicted_goals"] - m["actual_goals"]).abs()
    misses = m[m["abs_error"] >= threshold].copy()

    if misses.empty:
        return pd.DataFrame()

    # Build lookup helpers from feature_df
    has_features = feature_df is not None and not feature_df.empty

    # Per-player pct_played
    pct_lookup = {}
    if has_features and "pct_played" in feature_df.columns:
        for _, row in feature_df[["player", "team", "match_id", "pct_played"]].iterrows():
            pct_lookup[(str(row["player"]), str(row["team"]),
                        row.get("match_id"))] = float(row["pct_played"])

    # Team actual goals this match (from merged outcomes)
    team_match_goals = {}
    if "actual_goals" in m.columns and "team" in m.columns and "match_id" in m.columns:
        for (mid, team), grp in m.groupby(["match_id", "team"], observed=True):
            team_match_goals[(mid, str(team))] = float(grp["actual_goals"].sum())

    # Team rolling average from feature_df
    team_avg_lookup = {}
    if has_features and "team_goals_avg_5" in feature_df.columns:
        for (mid, team), grp in feature_df.groupby(["match_id", "team"], observed=True):
            team_avg_lookup[(mid, str(team))] = float(grp["team_goals_avg_5"].iloc[0])

    # Archetype stats for this round
    arch_gl_stats = {}  # archetype → (mean, std)
    if has_features and "archetype" in feature_df.columns and "GL" in feature_df.columns:
        for arch, grp in feature_df.groupby("archetype", observed=True):
            gl_vals = grp["GL"].values
            arch_gl_stats[int(arch)] = (float(np.mean(gl_vals)), float(np.std(gl_vals)) if len(gl_vals) > 1 else 1.0)

    # Archetype per player
    arch_lookup = {}
    if has_features and "archetype" in feature_df.columns:
        for _, row in feature_df[["player", "team", "match_id", "archetype"]].iterrows():
            arch_lookup[(str(row["player"]), str(row["team"]),
                         row.get("match_id"))] = int(row["archetype"])

    # Opponent concession lookup
    opp_concession_lookup = {}
    if has_features and "opp_arch_gl_conceded_avg_5" in feature_df.columns:
        for _, row in feature_df[["player", "team", "match_id",
                                   "opp_arch_gl_conceded_avg_5"]].iterrows():
            opp_concession_lookup[(str(row["player"]), str(row["team"]),
                                   row.get("match_id"))] = float(row["opp_arch_gl_conceded_avg_5"])

    records = []
    for _, row in misses.iterrows():
        player = str(row["player"])
        team = str(row["team"])
        mid = row.get("match_id")
        pred_gl = float(row["predicted_goals"])
        actual_gl = float(row["actual_goals"])
        abs_err = float(row["abs_error"])

        factors = []

        # 1. Opportunity check
        pct = pct_lookup.get((player, team, mid))
        if pct is not None and pct < config.MISS_OPPORTUNITY_PCT_PLAYED:
            factors.append("opportunity")

        # 2. Team environment check
        team_actual = team_match_goals.get((mid, team))
        team_avg = team_avg_lookup.get((mid, team))
        if team_actual is not None and team_avg is not None and team_avg > 0:
            ratio = team_actual / team_avg
            if ratio < (1 - config.MISS_TEAM_ENV_DEVIATION) or ratio > (1 + config.MISS_TEAM_ENV_DEVIATION):
                factors.append("team_environment")

        # 3. Role check
        arch = arch_lookup.get((player, team, mid))
        if arch is not None and arch in arch_gl_stats:
            arch_mean, arch_std = arch_gl_stats[arch]
            if arch_std > 0 and abs(actual_gl - arch_mean) > 2 * arch_std:
                factors.append("role")

        # 4. Matchup check
        opp_conc = opp_concession_lookup.get((player, team, mid))
        if opp_conc is not None and opp_conc > 0:
            conc_ratio = actual_gl / opp_conc
            if abs(conc_ratio - 1.0) > config.MISS_MATCHUP_CONCESSION_DEV:
                factors.append("matchup")

        # 5. Variance (default)
        if not factors:
            factors.append("variance")

        primary = factors[0]

        records.append({
            "player": player,
            "team": team,
            "match_id": _to_native(mid),
            "predicted_goals": _safe_float(pred_gl, 2),
            "actual_goals": int(actual_gl),
            "abs_error": _safe_float(abs_err, 2),
            "miss_type": primary,
            "contributing_factors": factors,
        })

    return pd.DataFrame(records)


def _build_miss_summary(merged, feature_df):
    """Build miss classification summary for round analysis JSON."""
    miss_df = classify_prediction_misses(merged, feature_df)

    if miss_df.empty:
        return {
            "total_significant_misses": 0,
            "breakdown": {"opportunity": 0, "team_environment": 0, "role": 0,
                          "matchup": 0, "variance": 0},
            "top_misses": [],
        }

    # Breakdown by primary miss type
    breakdown = {}
    for mt in ["opportunity", "team_environment", "role", "matchup", "variance"]:
        breakdown[mt] = int((miss_df["miss_type"] == mt).sum())

    # Top misses by error magnitude
    top = miss_df.nlargest(10, "abs_error")
    top_misses = []
    for _, row in top.iterrows():
        top_misses.append({
            "player": row["player"],
            "team": row["team"],
            "predicted": row["predicted_goals"],
            "actual": row["actual_goals"],
            "error": row["abs_error"],
            "miss_type": row["miss_type"],
            "contributing_factors": row["contributing_factors"],
        })

    return {
        "total_significant_misses": int(len(miss_df)),
        "breakdown": breakdown,
        "top_misses": top_misses,
    }


# ------------------------------------------------------------------
# Archetype & Concession Drift
# ------------------------------------------------------------------

def _archetype_drift_analysis(merged, feature_df):
    """Detect players whose in-game profile doesn't match their archetype.

    Compares each player's actual stat profile this game against their
    archetype's average profile. Flags players whose stats better fit
    a different archetype.
    """
    if feature_df is None or feature_df.empty:
        return {"drift_rate": 0.0, "most_drifted": [], "archetype_stability": {}}

    needed = ["archetype", "GL", "player", "team"]
    if not all(c in feature_df.columns for c in needed):
        return {"drift_rate": 0.0, "most_drifted": [], "archetype_stability": {},
                "note": "Missing required columns"}

    # Stat columns to compare profiles
    stat_cols = [c for c in ["GL", "DI", "MK", "TK"] if c in feature_df.columns]
    if not stat_cols:
        return {"drift_rate": 0.0, "most_drifted": [], "archetype_stability": {}}

    # Compute per-archetype average stat profiles this round
    arch_profiles = {}
    for arch, grp in feature_df.groupby("archetype", observed=True):
        arch = int(arch)
        profile = {}
        for col in stat_cols:
            vals = grp[col].values
            profile[col] = float(np.mean(vals))
        arch_profiles[arch] = profile

    if len(arch_profiles) < 2:
        return {"drift_rate": 0.0, "most_drifted": [], "archetype_stability": {},
                "note": "Fewer than 2 archetypes present"}

    # Soft probabilities
    prob_cols = [f"archetype_prob_{i}" for i in range(config.N_ARCHETYPES)
                 if f"archetype_prob_{i}" in feature_df.columns]

    # For each player, check if their actual stats better fit another archetype
    drifted = []
    arch_correct = {}  # archetype → [correct_count, total]

    for _, row in feature_df.iterrows():
        arch = int(row["archetype"])
        player_stats = {col: float(row[col]) for col in stat_cols}

        # Compute distance to each archetype's profile
        distances = {}
        for a, profile in arch_profiles.items():
            dist = sum((player_stats[c] - profile[c]) ** 2 for c in stat_cols)
            distances[a] = dist

        best_fit = min(distances, key=distances.get)
        is_drifted = best_fit != arch

        if arch not in arch_correct:
            arch_correct[arch] = [0, 0]
        arch_correct[arch][1] += 1
        if not is_drifted:
            arch_correct[arch][0] += 1

        if is_drifted:
            entry = {
                "player": str(row["player"]),
                "team": str(row["team"]),
                "assigned_archetype": arch,
                "best_fit_archetype": int(best_fit),
                "distance_to_assigned": _safe_float(distances[arch], 2),
                "distance_to_best": _safe_float(distances[best_fit], 2),
            }

            # Add max soft probability if available
            if prob_cols:
                probs = [float(row.get(c, 0)) for c in prob_cols]
                entry["max_archetype_prob"] = _safe_float(max(probs), 3)

            drifted.append(entry)

    # Sort by how far off they are
    drifted.sort(key=lambda x: x["distance_to_assigned"] - x["distance_to_best"], reverse=True)

    total_players = len(feature_df)
    drift_rate = len(drifted) / total_players if total_players > 0 else 0.0

    # Stability scores per archetype
    stability = {}
    for arch, (correct, total) in arch_correct.items():
        stability[str(arch)] = {
            "stability": _safe_float(correct / total if total > 0 else 0.0),
            "n_players": total,
        }

    return {
        "drift_rate": _safe_float(drift_rate),
        "n_drifted": len(drifted),
        "most_drifted": drifted[:10],
        "archetype_stability": stability,
    }


def _concession_drift_analysis(merged, feature_df):
    """Track how opponents' actual concessions compare to their profile.

    For each opponent, compare opp_arch_gl_conceded_avg_5 vs actual goals
    conceded to each archetype this round.
    """
    if feature_df is None or feature_df.empty:
        return {"opponent_drifts": []}

    needed = ["opponent", "archetype", "GL", "opp_arch_gl_conceded_avg_5"]
    if not all(c in feature_df.columns for c in needed):
        return {"opponent_drifts": [], "note": "Missing required columns"}

    if merged.empty or "actual_goals" not in merged.columns:
        return {"opponent_drifts": []}

    # Per (opponent, archetype): expected concession vs actual
    drifts = []
    for (opp, arch), grp in feature_df.groupby(["opponent", "archetype"], observed=True):
        expected = float(grp["opp_arch_gl_conceded_avg_5"].mean())
        actual_gl = float(grp["GL"].mean())
        n_players = len(grp)

        if expected > 0 and n_players >= 2:
            delta = actual_gl - expected
            pct_change = delta / expected

            drifts.append({
                "opponent": str(opp),
                "archetype": int(arch),
                "expected_concession": _safe_float(expected),
                "actual_concession": _safe_float(actual_gl),
                "delta": _safe_float(delta),
                "pct_change": _safe_float(pct_change),
                "n_players": n_players,
            })

    # Sort by magnitude of unexpected concessions
    drifts.sort(key=lambda x: abs(x["delta"]), reverse=True)

    # Per-opponent summary
    opp_summary = {}
    for d in drifts:
        opp = d["opponent"]
        if opp not in opp_summary:
            opp_summary[opp] = {"total_delta": 0.0, "n_archetypes": 0}
        opp_summary[opp]["total_delta"] += d["delta"] or 0
        opp_summary[opp]["n_archetypes"] += 1

    opponent_summaries = []
    for opp, info in opp_summary.items():
        opponent_summaries.append({
            "opponent": opp,
            "mean_delta": _safe_float(info["total_delta"] / info["n_archetypes"]
                                       if info["n_archetypes"] > 0 else 0.0),
            "n_archetypes": info["n_archetypes"],
            "overall_direction": "conceded_more" if info["total_delta"] > 0 else "conceded_less",
        })

    opponent_summaries.sort(key=lambda x: abs(x["mean_delta"] or 0), reverse=True)

    return {
        "archetype_concession_drifts": drifts[:15],
        "opponent_summaries": opponent_summaries,
    }


# ------------------------------------------------------------------
# Season Report
# ------------------------------------------------------------------

def generate_season_report(store, year):
    """Generate comprehensive post-season report from all round analyses.

    Reads all analysis JSONs for the year and aggregates into a single
    report covering learning curves, calibration, miss distribution,
    archetype accuracy, weather, game predictions, player leaderboard,
    and streak summaries.
    """
    import json

    # Gather all round analyses
    analyses = []
    for rnd in range(1, 29):
        a = store.load_analysis(year, rnd)
        if a:
            analyses.append(a)

    if not analyses:
        return {"error": f"No analysis data found for {year}"}

    report = {
        "year": year,
        "rounds_analyzed": len(analyses),
    }

    # 0. Threshold evaluation (PRIMARY)
    report["threshold_evaluation"] = _season_threshold_report(store, year)

    # 1. Learning curve
    report["learning_curve"] = _season_learning_curve(analyses)

    # 2. Calibration curve
    report["calibration_curve"] = _season_calibration_curve(store, year)

    # 3. Miss type distribution
    report["miss_type_distribution"] = _season_miss_distribution(analyses)

    # 4. Archetype accuracy
    report["archetype_accuracy"] = _season_archetype_accuracy(store, year)

    # 5. Weather summary
    report["weather_summary"] = _season_weather_summary(analyses)

    # 6. Game winner accuracy
    report["game_winner_accuracy"] = _season_game_winner_accuracy(analyses)

    # 7. Player leaderboard
    report["player_leaderboard"] = _season_player_leaderboard(store, year)

    # 8. Streak summary
    report["streak_summary"] = _season_streak_summary(analyses)

    return report


def _season_threshold_report(store, year):
    """Compute full-season probability calibration metrics per threshold.

    Loads ALL predictions + outcomes for the year, merges, then computes
    Brier score, log loss, base rate, and full calibration curve per threshold.
    This is the PRIMARY evaluation section.
    """
    all_preds = store.load_predictions(year)
    all_outs = store.load_outcomes(year)

    if all_preds.empty or all_outs.empty:
        return {}

    merged = _merge_pred_outcome(all_preds, all_outs)
    if merged.empty:
        return {}

    threshold_data = _extract_threshold_data(merged)
    results = {}
    for name, (preds, actuals) in threshold_data.items():
        metrics = _compute_threshold_metrics(preds, actuals)
        if metrics:
            results[name] = metrics

    return results


def _season_learning_curve(analyses):
    """Round-by-round MAE, AUC, calibration ratio — shows learning trend."""
    curve = []
    for a in analyses:
        summary = a.get("summary", {})
        entry = {
            "round": a.get("round"),
            "goals_mae": summary.get("goals_mae"),
            "scorer_auc": summary.get("scorer_auc"),
            "calibration_ratio": summary.get("calibration_ratio"),
            "n_players": summary.get("n_players"),
        }
        improvement = a.get("model_improvement", {})
        entry["trend"] = improvement.get("trend")
        entry["cumulative_mae"] = improvement.get("cumulative_mae")

        # Per-round Brier scores from threshold_metrics
        tm = summary.get("threshold_metrics", {})
        entry["goals_1plus_brier"] = tm.get("1plus_goals", {}).get("brier_score")
        entry["goals_2plus_brier"] = tm.get("2plus_goals", {}).get("brier_score")
        entry["goals_3plus_brier"] = tm.get("3plus_goals", {}).get("brier_score")
        entry["disp_20plus_brier"] = tm.get("20plus_disp", {}).get("brier_score")

        curve.append(entry)

    # Compute overall trajectory
    maes = [c["goals_mae"] for c in curve if c["goals_mae"] is not None]
    if len(maes) >= 4:
        half = len(maes) // 2
        first_half = np.mean(maes[:half])
        second_half = np.mean(maes[half:])
        learning_effect = (first_half - second_half) / first_half * 100 if first_half > 0 else 0
    else:
        first_half = second_half = learning_effect = None

    # First/second half Brier comparison for 1+ goals
    briers = [c["goals_1plus_brier"] for c in curve if c["goals_1plus_brier"] is not None]
    if len(briers) >= 4:
        bhalf = len(briers) // 2
        first_half_brier = float(np.mean(briers[:bhalf]))
        second_half_brier = float(np.mean(briers[bhalf:]))
    else:
        first_half_brier = second_half_brier = None

    return {
        "rounds": curve,
        "first_half_mae": _safe_float(first_half),
        "second_half_mae": _safe_float(second_half),
        "learning_effect_pct": _safe_float(learning_effect, 1),
        "first_half_brier_1plus": _safe_float(first_half_brier),
        "second_half_brier_1plus": _safe_float(second_half_brier),
    }


def _season_calibration_curve(store, year=None):
    """Per-target bucket: predicted_prob vs observed_rate across full season."""
    cal = store.get_calibration_state(year=year)
    if cal.empty:
        return {"buckets": []}

    active = cal[cal["n_predictions"] > 0].copy()
    if active.empty:
        return {"buckets": []}

    buckets = []
    for _, row in active.iterrows():
        buckets.append({
            "target": str(row["target"]),
            "bucket": _safe_float(row["probability_bucket"]),
            "observed_rate": _safe_float(row["observed_rate"]),
            "n_predictions": int(row["n_predictions"]),
            "calibration_adj": _safe_float(row["calibration_adj"]),
        })

    # Overall calibration error
    if len(active) > 0:
        cal_error = float((active["observed_rate"] - active["probability_bucket"]).abs().mean())
    else:
        cal_error = None

    return {
        "buckets": buckets,
        "mean_absolute_calibration_error": _safe_float(cal_error),
    }


def _season_miss_distribution(analyses):
    """Aggregate miss type counts across all rounds."""
    totals = {"opportunity": 0, "team_environment": 0, "role": 0,
              "matchup": 0, "variance": 0}
    total_misses = 0

    for a in analyses:
        mc = a.get("miss_classification", {})
        total_misses += mc.get("total_significant_misses", 0)
        breakdown = mc.get("breakdown", {})
        for mt in totals:
            totals[mt] += breakdown.get(mt, 0)

    # Percentages
    pcts = {}
    for mt, count in totals.items():
        pcts[mt] = _safe_float(count / total_misses * 100 if total_misses > 0 else 0, 1)

    return {
        "total_significant_misses": total_misses,
        "counts": totals,
        "percentages": pcts,
    }


def _season_archetype_accuracy(store, year):
    """Per-archetype MAE and scorer AUC across the season."""
    # Load all predictions and outcomes for the year
    all_preds = store.load_predictions(year)
    all_outs = store.load_outcomes(year)

    if all_preds.empty or all_outs.empty:
        return {"per_archetype": []}

    # Load archetype assignments
    archetypes = store.load_archetypes()
    if archetypes.empty:
        return {"per_archetype": [], "note": "No archetype data"}

    merged = _merge_pred_outcome(all_preds, all_outs)
    if merged.empty:
        return {"per_archetype": []}

    # Merge with archetypes
    if "player" in archetypes.columns and "archetype" in archetypes.columns:
        arch_lookup = archetypes.drop_duplicates(subset=["player", "team"])[
            ["player", "team", "archetype"]
        ]
        merged = merged.merge(arch_lookup, on=["player", "team"], how="left")

    if "archetype" not in merged.columns:
        return {"per_archetype": [], "note": "Could not merge archetypes"}

    results = []
    for arch, grp in merged.groupby("archetype", observed=True):
        if len(grp) < 5:
            continue

        entry = {"archetype": int(arch), "n_predictions": len(grp)}

        if "predicted_goals" in grp.columns and "actual_goals" in grp.columns:
            entry["goals_mae"] = _safe_float(
                mean_absolute_error(grp["actual_goals"], grp["predicted_goals"])
            )

        if "p_scorer" in grp.columns and "actual_goals" in grp.columns:
            actual_scored = (grp["actual_goals"] >= 1).astype(int)
            try:
                entry["scorer_auc"] = _safe_float(
                    roc_auc_score(actual_scored, grp["p_scorer"])
                )
            except ValueError:
                entry["scorer_auc"] = None

        results.append(entry)

    results.sort(key=lambda x: x.get("goals_mae", 99))
    return {"per_archetype": results}


def _season_weather_summary(analyses):
    """Aggregate wet vs dry accuracy and scoring deviation."""
    total_wet = 0
    total_matches = 0
    wet_maes = []
    dry_maes = []

    for a in analyses:
        wi = a.get("weather_impact", {})
        total_wet += wi.get("wet_matches", 0)
        total_matches += wi.get("total_matches", 0)

        wvd = wi.get("wet_vs_dry", {})
        if "wet_mae" in wvd and wvd["wet_mae"] is not None:
            wet_maes.append(wvd["wet_mae"])
        if "dry_mae" in wvd and wvd["dry_mae"] is not None:
            dry_maes.append(wvd["dry_mae"])

    return {
        "total_wet_matches": total_wet,
        "total_matches": total_matches,
        "wet_match_pct": _safe_float(total_wet / total_matches * 100 if total_matches > 0 else 0, 1),
        "avg_wet_mae": _safe_float(np.mean(wet_maes) if wet_maes else None),
        "avg_dry_mae": _safe_float(np.mean(dry_maes) if dry_maes else None),
    }


def _season_game_winner_accuracy(analyses):
    """Overall win prediction %, margin MAE."""
    correct = 0
    total = 0
    margin_errors = []

    for a in analyses:
        for game in a.get("game_results", []):
            if "correct" in game:
                total += 1
                if game["correct"]:
                    correct += 1
            if "margin_error" in game and game["margin_error"] is not None:
                margin_errors.append(game["margin_error"])

    return {
        "total_games": total,
        "correct_predictions": correct,
        "accuracy_pct": _safe_float(correct / total * 100 if total > 0 else 0, 1),
        "margin_mae": _safe_float(np.mean(margin_errors) if margin_errors else None, 1),
    }


def _season_player_leaderboard(store, year):
    """Best/worst predicted players (min 10 appearances)."""
    all_preds = store.load_predictions(year)
    all_outs = store.load_outcomes(year)

    if all_preds.empty or all_outs.empty:
        return {"best": [], "worst": []}

    merged = _merge_pred_outcome(all_preds, all_outs)
    if merged.empty or "predicted_goals" not in merged.columns:
        return {"best": [], "worst": []}

    merged["abs_error"] = (merged["predicted_goals"] - merged["actual_goals"]).abs()
    merged["signed_error"] = merged["predicted_goals"] - merged["actual_goals"]

    player_stats = merged.groupby(["player", "team"], observed=True).agg(
        n=("abs_error", "count"),
        mae=("abs_error", "mean"),
        bias=("signed_error", "mean"),
    ).reset_index()

    # Min 10 appearances
    frequent = player_stats[player_stats["n"] >= 10].copy()
    if frequent.empty:
        # Fall back to min 5
        frequent = player_stats[player_stats["n"] >= 5].copy()

    if frequent.empty:
        return {"best": [], "worst": []}

    best = []
    for _, row in frequent.nsmallest(10, "mae").iterrows():
        best.append({
            "player": str(row["player"]),
            "team": str(row["team"]),
            "appearances": int(row["n"]),
            "mae": _safe_float(row["mae"]),
            "bias": _safe_float(row["bias"]),
        })

    worst = []
    for _, row in frequent.nlargest(10, "mae").iterrows():
        worst.append({
            "player": str(row["player"]),
            "team": str(row["team"]),
            "appearances": int(row["n"]),
            "mae": _safe_float(row["mae"]),
            "bias": _safe_float(row["bias"]),
        })

    return {"best": best, "worst": worst}


def _season_streak_summary(analyses):
    """Longest hot/cold streaks and most streak breaks from the season."""
    all_hot = []
    all_cold = []
    all_broken = []

    for a in analyses:
        streaks = a.get("streaks", {})
        rnd = a.get("round", 0)

        for h in streaks.get("new_hot", []) + streaks.get("continued_hot", []):
            all_hot.append({**h, "round": rnd})

        for c in streaks.get("new_cold", []):
            all_cold.append({**c, "round": rnd})

        for b in streaks.get("broken_hot", []):
            all_broken.append({**b, "round": rnd})

    # Longest hot streaks
    all_hot.sort(key=lambda x: x.get("streak", 0), reverse=True)
    # Deduplicate by player (keep longest)
    seen = set()
    longest_hot = []
    for h in all_hot:
        key = (h["player"], h["team"])
        if key not in seen:
            seen.add(key)
            longest_hot.append(h)
        if len(longest_hot) >= 10:
            break

    return {
        "longest_hot_streaks": longest_hot,
        "total_streak_breaks": len(all_broken),
        "streak_breaks": all_broken[:10],
        "new_cold_streaks": len(all_cold),
    }
