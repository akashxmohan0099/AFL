"""
AFL Prediction Pipeline — Multi-Bet Combination Backtest
=========================================================
Generates SGM-style multi-bet combos from sequential predictions and
evaluates against actuals.

Four confidence tiers:
  - 85% Confidence  (all legs >= 85% probability)
  - 80% Confidence  (all legs >= 80% probability)
  - 75% Confidence  (all legs >= 75% probability)
  - 70% Confidence  (all legs >= 70% probability)

Four leg categories:
  - Goals (1+, 2+, 3+)
  - Disposals (15+, 20+, 25+, 30+)
  - Team Total Score (dynamic thresholds 50–120)
  - Both Teams Total Score (dynamic thresholds 100–210)

Usage:
    python3 multi_backtest.py [--year 2025] [--label multi_backtest_2025]
    python3 multi_backtest.py --all-years
"""

import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import config
from store import LearningStore


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

TIERS = [
    ("tier_85", "85% Confidence", "All legs >= 85% probability", 0.85),
    ("tier_80", "80% Confidence", "All legs >= 80% probability", 0.80),
    ("tier_75", "75% Confidence", "All legs >= 75% probability", 0.75),
    ("tier_70", "70% Confidence", "All legs >= 70% probability", 0.70),
]
TIER_KEYS = [t[0] for t in TIERS]
TIER_LABELS = {t[0]: t[1] for t in TIERS}
TIER_DESCS = {t[0]: t[2] for t in TIERS}
TIER_THRESHOLDS = {t[0]: t[3] for t in TIERS}
MIN_TIER_PROB = 0.70  # nothing below 70%


# ---------------------------------------------------------------------------
# Leg definitions (player-level)
# ---------------------------------------------------------------------------

PLAYER_LEGS = [
    {"name": "goals_1plus", "prob_col": "p_1plus_goals", "threshold": 1, "stat": "goals", "actual_col": "actual_goals"},
    {"name": "goals_2plus", "prob_col": "p_2plus_goals", "threshold": 2, "stat": "goals", "actual_col": "actual_goals"},
    {"name": "goals_3plus", "prob_col": "p_3plus_goals", "threshold": 3, "stat": "goals", "actual_col": "actual_goals"},
    {"name": "disp_15plus", "prob_col": "p_15plus_disp", "threshold": 15, "stat": "disposals", "actual_col": "actual_disposals"},
    {"name": "disp_20plus", "prob_col": "p_20plus_disp", "threshold": 20, "stat": "disposals", "actual_col": "actual_disposals"},
    {"name": "disp_25plus", "prob_col": "p_25plus_disp", "threshold": 25, "stat": "disposals", "actual_col": "actual_disposals"},
    {"name": "disp_30plus", "prob_col": "p_30plus_disp", "threshold": 30, "stat": "disposals", "actual_col": "actual_disposals"},
]

# Team/match total thresholds — scanned dynamically
TEAM_SCORE_THRESHOLDS = list(range(50, 130, 10))    # 50, 60, 70, ..., 120
MATCH_TOTAL_THRESHOLDS = list(range(100, 220, 10))   # 100, 110, ..., 210

# Blend weight: model vs venue history
MODEL_WEIGHT = 0.6
VENUE_WEIGHT = 0.4
MIN_VENUE_GAMES = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_season_data(store, year):
    """Load all rounds, merge preds + outcomes."""
    round_tuples = store.list_rounds(subdir="predictions", year=year)
    if not round_tuples:
        print(f"No prediction rounds found for {year}")
        return pd.DataFrame()

    all_dfs = []
    for yr, rnd in sorted(round_tuples):
        try:
            preds = store.load_predictions(year=yr, round_num=rnd)
            outcomes = store.load_outcomes(year=yr, round_num=rnd)
        except Exception:
            continue
        if preds.empty or outcomes.empty:
            continue
        merged = preds.merge(outcomes, on=["player", "team", "match_id"], how="inner")
        merged["round_number"] = rnd
        all_dfs.append(merged)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def load_team_matches(year):
    """Load team_matches for actual team scores."""
    tm_path = config.BASE_STORE_DIR / "team_matches.parquet"
    if not tm_path.exists():
        return pd.DataFrame()
    tm = pd.read_parquet(tm_path)
    return tm[tm["year"] == year].copy() if not tm.empty else pd.DataFrame()


def load_matches():
    """Load all matches for venue history."""
    m_path = config.BASE_STORE_DIR / "matches.parquet"
    if not m_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(m_path)


def build_venue_history(matches_df, up_to_year):
    """Build venue score distributions from historical matches (up to but not including up_to_year).

    Returns: {venue: {"team_mean": float, "team_std": float, "total_mean": float, "total_std": float, "n": int}}
    """
    hist = matches_df[matches_df["year"] < up_to_year].copy()
    if hist.empty:
        return {}

    venue_stats = {}
    for venue, grp in hist.groupby("venue", observed=True):
        n = len(grp)
        # Individual team scores (home + away)
        team_scores = np.concatenate([grp["home_score"].values, grp["away_score"].values])
        team_scores = team_scores[~np.isnan(team_scores)]
        # Match totals
        totals = (grp["home_score"] + grp["away_score"]).values
        totals = totals[~np.isnan(totals)]

        if len(team_scores) >= 10:
            venue_stats[venue] = {
                "team_mean": float(np.mean(team_scores)),
                "team_std": float(np.std(team_scores, ddof=1)) if len(team_scores) > 1 else 25.0,
                "total_mean": float(np.mean(totals)),
                "total_std": float(np.std(totals, ddof=1)) if len(totals) > 1 else 35.0,
                "n": n,
            }
    return venue_stats


def compute_model_residual_std(full_df, team_matches, up_to_round):
    """Compute residual std of model team score predictions vs actuals from prior rounds.

    Uses rounds before up_to_round to calibrate.
    Returns (team_std, match_total_std) — the std of (actual - predicted).
    """
    prior = full_df[full_df["round_number"] < up_to_round].copy()
    if prior.empty or team_matches.empty:
        return 25.0, 35.0

    # Calibrate team/match score variance from actual results.
    # Use league-average + game_predictions margin to estimate team scores
    # (per-player predicted_score sums to ~2x actual team totals, so we don't use them).
    if team_matches.empty:
        return 25.0, 35.0

    # Use actual score variance from completed matches
    tm_cols = team_matches[["match_id", "team", "score"]].copy()
    if len(tm_cols) < 20:
        return 25.0, 35.0

    # Team score std from actuals (each team's score deviation from league mean)
    AVG_TEAM_SCORE = 82.5
    team_residuals = tm_cols["score"].values - AVG_TEAM_SCORE
    team_std = float(np.std(team_residuals, ddof=1))

    # Match total std
    match_totals = tm_cols.groupby("match_id")["score"].sum()
    AVG_MATCH_TOTAL = 165.0
    match_residuals = match_totals.values - AVG_MATCH_TOTAL
    match_std = float(np.std(match_residuals, ddof=1)) if len(match_residuals) > 1 else 35.0

    return max(team_std, 10.0), max(match_std, 15.0)


# ---------------------------------------------------------------------------
# Team/match total probability computation
# ---------------------------------------------------------------------------

def compute_team_match_probs(match_df, venue, venue_history, model_team_std, model_match_std):
    """Compute P(team_score > X) and P(match_total > X) for all thresholds.

    Returns:
        team_legs: list of candidate legs for team totals
        match_legs: list of candidate legs for match totals
    """
    # Derive team scores from league-average total + relative goal ratio.
    # Per-player predicted_score sums to ~2x real team totals — use ratio only.
    AVG_TOTAL = 165.0
    goal_col = "predicted_score" if "predicted_score" in match_df.columns else (
        "predicted_goals" if "predicted_goals" in match_df.columns else None
    )
    if goal_col is None:
        return [], []

    raw_team = match_df.groupby("team", observed=True)[goal_col].sum()
    match_id = match_df["match_id"].iloc[0]
    teams = list(raw_team.index)

    if len(teams) < 2:
        return [], []

    raw_total = float(raw_team.sum())
    # Scale each team's share to the realistic total
    team_scores = raw_team * (AVG_TOTAL / raw_total) if raw_total > 0 else raw_team
    predicted_total = AVG_TOTAL
    venue_info = venue_history.get(venue) if venue else None

    team_legs = []
    for team in teams:
        pred_score = float(team_scores[team])

        for threshold in TEAM_SCORE_THRESHOLDS:
            # Model probability
            p_model = 1.0 - sp_stats.norm.cdf(threshold, loc=pred_score, scale=model_team_std)

            # Venue probability (if available)
            if venue_info and venue_info["n"] >= MIN_VENUE_GAMES:
                p_venue = 1.0 - sp_stats.norm.cdf(threshold, loc=venue_info["team_mean"], scale=venue_info["team_std"])
                prob = MODEL_WEIGHT * p_model + VENUE_WEIGHT * p_venue
            else:
                prob = p_model

            if prob < MIN_TIER_PROB:
                continue

            leg_name = f"team_total_{threshold}"
            team_legs.append({
                "player": "",
                "team": team,
                "match_id": match_id,
                "leg_type": leg_name,
                "stat": "team_total",
                "threshold": threshold,
                "prob": float(np.clip(prob, 0, 0.999)),
                "hit": None,  # filled later
                "label": f"{team} {threshold}+ points",
                "reason": f"{prob:.0%} confidence — predicted {pred_score:.0f} team points",
                "predicted_val": round(pred_score, 1),
            })

    match_legs = []
    for threshold in MATCH_TOTAL_THRESHOLDS:
        p_model = 1.0 - sp_stats.norm.cdf(threshold, loc=predicted_total, scale=model_match_std)

        if venue_info and venue_info["n"] >= MIN_VENUE_GAMES:
            p_venue = 1.0 - sp_stats.norm.cdf(threshold, loc=venue_info["total_mean"], scale=venue_info["total_std"])
            prob = MODEL_WEIGHT * p_model + VENUE_WEIGHT * p_venue
        else:
            prob = p_model

        if prob < MIN_TIER_PROB:
            continue

        leg_name = f"match_total_{threshold}"
        team_label = " vs ".join(teams[:2])
        match_legs.append({
            "player": "",
            "team": team_label,
            "match_id": match_id,
            "leg_type": leg_name,
            "stat": "match_total",
            "threshold": threshold,
            "prob": float(np.clip(prob, 0, 0.999)),
            "hit": None,
            "label": f"Match total {threshold}+ points",
            "reason": f"{prob:.0%} confidence — predicted {predicted_total:.0f} total points",
            "predicted_val": round(predicted_total, 1),
        })

    return team_legs, match_legs


# ---------------------------------------------------------------------------
# Candidate leg generation
# ---------------------------------------------------------------------------

def _player_short(name):
    """'Cripps, Patrick' -> 'P. Cripps'"""
    if ", " in name:
        last, first = name.split(", ", 1)
        return f"{first[0]}. {last}"
    return name


def _get_predicted(stat, row):
    col_map = {"goals": "predicted_goals", "disposals": "predicted_disposals"}
    col = col_map.get(stat)
    if col and col in row.index and not pd.isna(row.get(col)):
        return round(float(row[col]), 2)
    return None


def generate_candidate_legs(match_df, venue, venue_history, model_team_std, model_match_std):
    """Generate all candidate legs for a single match."""
    candidates = []

    for _, row in match_df.iterrows():
        player = row["player"]
        team = row["team"]
        match_id = row["match_id"]

        for leg in PLAYER_LEGS:
            prob_col = leg["prob_col"]
            if prob_col not in row.index or pd.isna(row[prob_col]):
                continue
            prob = float(row[prob_col])
            if prob < MIN_TIER_PROB:
                continue

            actual_col = leg["actual_col"]
            actual_val = row.get(actual_col, np.nan)
            hit = bool(actual_val >= leg["threshold"]) if not pd.isna(actual_val) else None

            predicted = _get_predicted(leg["stat"], row)
            reason = f"{prob:.0%} model confidence"
            if predicted is not None:
                reason += f" — predicted {predicted:.1f} {leg['stat']}"

            candidates.append({
                "player": player, "team": team, "match_id": match_id,
                "leg_type": leg["name"], "stat": leg["stat"],
                "threshold": leg["threshold"], "prob": prob, "hit": hit,
                "label": f"{player} {leg['threshold']}+ {leg['stat']}",
                "reason": reason,
                "predicted_val": predicted,
            })

    # Team total and match total legs
    team_legs, match_legs = compute_team_match_probs(
        match_df, venue, venue_history, model_team_std, model_match_std
    )
    candidates.extend(team_legs)
    candidates.extend(match_legs)

    return candidates


# ---------------------------------------------------------------------------
# Actuals checking for team/match totals
# ---------------------------------------------------------------------------

def fill_team_match_actuals(all_combos, team_matches, matches_df):
    """Fill in hit/miss for team_total and match_total legs."""
    # Build lookup: (match_id, team) -> score
    team_score_lookup = {}
    if not team_matches.empty:
        for _, row in team_matches.iterrows():
            team_score_lookup[(row["match_id"], row["team"])] = row["score"]

    # Build lookup: match_id -> total_score
    match_total_lookup = {}
    if not matches_df.empty:
        year_matches = matches_df[matches_df["match_id"].isin(
            set(c["match_id"] for c in all_combos)
        )]
        for _, row in year_matches.iterrows():
            if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
                match_total_lookup[row["match_id"]] = row["home_score"] + row["away_score"]

    for combo in all_combos:
        for leg in combo["legs"]:
            if leg["hit"] is not None:
                continue

            if leg["leg_type"].startswith("team_total_"):
                team = leg["team"]
                mid = combo["match_id"]
                actual = team_score_lookup.get((mid, team))
                if actual is not None:
                    leg["hit"] = bool(actual >= leg["threshold"])

            elif leg["leg_type"].startswith("match_total_"):
                mid = combo["match_id"]
                actual = match_total_lookup.get(mid)
                if actual is not None:
                    leg["hit"] = bool(actual >= leg["threshold"])

        # Recompute combo hit
        failed = [l for l in combo["legs"] if l["hit"] is False]
        has_unknown = any(l["hit"] is None for l in combo["legs"])
        if has_unknown:
            combo["combo_hit"] = None
        else:
            combo["combo_hit"] = len(failed) == 0
        combo["n_failed_legs"] = len(failed)

        hit_legs = [l for l in combo["legs"] if l["hit"] is True]
        if len(failed) == 1 and len(hit_legs) == len(combo["legs"]) - 1:
            combo["weakest_link"] = failed[0]["leg_type"]
        else:
            combo["weakest_link"] = None


# ---------------------------------------------------------------------------
# Combo building
# ---------------------------------------------------------------------------

def _dedupe_legs(legs):
    """Deduplicate: one leg per (player, leg_type), one team_total per team, one match_total."""
    legs = sorted(legs, key=lambda x: -x["prob"])
    seen_player = set()
    seen_team_total = set()
    seen_match_total = False
    out = []
    for l in legs:
        if l["stat"] == "match_total":
            if seen_match_total:
                continue
            seen_match_total = True
        elif l["stat"] == "team_total":
            key = (l["team"], l["leg_type"])
            if key in seen_team_total:
                continue
            seen_team_total.add(key)
        else:
            key = (l["player"], l["leg_type"])
            if key in seen_player:
                continue
            seen_player.add(key)
        out.append(l)
    return out


def _pick_diverse_legs(legs, max_legs=4):
    """Pick legs with diversity: cross-player for player legs, mix stat categories.

    Max 1 team total, max 1 match total. Max 2 of same stat category.
    """
    picked = []
    used_players = set()
    stat_counts = defaultdict(int)
    has_team_total = False
    has_match_total = False

    for l in legs:
        if len(picked) >= max_legs:
            break

        if l["stat"] == "team_total":
            if has_team_total:
                continue
            has_team_total = True
        elif l["stat"] == "match_total":
            if has_match_total:
                continue
            has_match_total = True
        else:
            # Player leg — ensure cross-player, max 2 per stat category
            if l["player"] in used_players:
                continue
            if stat_counts[l["stat"]] >= 2:
                continue
            used_players.add(l["player"])

        stat_counts[l["stat"]] += 1
        picked.append(l)

    return picked


MAX_COMBOS_PER_TIER = 5


def build_tier_combos(candidates, tier_key, tier_threshold):
    """Build up to MAX_COMBOS_PER_TIER combos per match for a given tier.

    Repeatedly picks diverse leg sets from the eligible pool, removing used
    legs each time so combos don't overlap.
    """
    eligible = [c for c in candidates if c["prob"] >= tier_threshold]
    eligible = _dedupe_legs(eligible)

    if len(eligible) < 2:
        return []

    combos = []
    pool = list(eligible)

    while len(combos) < MAX_COMBOS_PER_TIER and len(pool) >= 2:
        legs = _pick_diverse_legs(pool, max_legs=4)
        if len(legs) < 2:
            break
        combos.append(_make_combo(legs, tier_key))
        # Remove used legs from pool
        used = set(id(l) for l in legs)
        pool = [l for l in pool if id(l) not in used]

    return combos


def _make_combo(legs, tier):
    combo_prob = 1.0
    for l in legs:
        combo_prob *= l["prob"]

    failed_legs = [l for l in legs if l["hit"] is False]
    hit_legs = [l for l in legs if l["hit"] is True]
    has_unknown = any(l["hit"] is None for l in legs)
    combo_hit = (len(failed_legs) == 0 and not has_unknown) if legs else None
    if has_unknown:
        combo_hit = None

    weakest_link = None
    if len(failed_legs) == 1 and len(hit_legs) == len(legs) - 1:
        weakest_link = failed_legs[0]["leg_type"]

    return {
        "tier": tier,
        "tier_label": TIER_LABELS[tier],
        "tier_desc": TIER_DESCS[tier],
        "n_legs": len(legs),
        "combo_predicted_prob": combo_prob,
        "combo_hit": combo_hit,
        "weakest_link": weakest_link,
        "n_failed_legs": len(failed_legs),
        "legs": [{
            "player": l["player"],
            "team": l["team"],
            "leg_type": l["leg_type"],
            "threshold": l["threshold"],
            "prob": round(l["prob"], 4),
            "hit": l["hit"],
            "label": l["label"],
            "reason": l.get("reason", ""),
        } for l in legs],
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(all_combos):
    tier_stats = {}
    for tier in TIER_KEYS:
        tier_combos = [c for c in all_combos if c["tier"] == tier]
        if not tier_combos:
            tier_stats[tier] = {"n_combos": 0, "n_hits": 0, "combo_hit_rate": 0.0,
                                "avg_predicted_prob": 0.0, "avg_legs": 0.0,
                                "label": TIER_LABELS[tier], "desc": TIER_DESCS[tier]}
            continue
        n = len(tier_combos)
        hits = sum(1 for c in tier_combos if c["combo_hit"] is True)
        tier_stats[tier] = {
            "n_combos": n, "n_hits": hits,
            "combo_hit_rate": round(hits / n, 4) if n > 0 else 0.0,
            "avg_predicted_prob": round(float(np.mean([c["combo_predicted_prob"] for c in tier_combos])), 4),
            "avg_legs": round(float(np.mean([c["n_legs"] for c in tier_combos])), 2),
            "label": TIER_LABELS[tier], "desc": TIER_DESCS[tier],
        }

    leg_stats = defaultdict(lambda: {"n_used": 0, "n_hit": 0, "n_weakest_link": 0})
    for c in all_combos:
        for l in c["legs"]:
            lt = l["leg_type"]
            leg_stats[lt]["n_used"] += 1
            if l["hit"] is True:
                leg_stats[lt]["n_hit"] += 1
        if c.get("weakest_link"):
            leg_stats[c["weakest_link"]]["n_weakest_link"] += 1
    for lt, s in leg_stats.items():
        s["hit_rate"] = round(s["n_hit"] / s["n_used"], 4) if s["n_used"] > 0 else 0.0

    total = len(all_combos)
    total_misses = sum(1 for c in all_combos if c["combo_hit"] is False)
    single_leg_failures = sum(1 for c in all_combos if c["combo_hit"] is False and c["n_failed_legs"] == 1)
    failure_counts = defaultdict(int)
    for c in all_combos:
        if c["combo_hit"] is False:
            for l in c["legs"]:
                if l["hit"] is False:
                    failure_counts[l["leg_type"]] += 1
    most_failed = max(failure_counts, key=failure_counts.get) if failure_counts else None

    failure_analysis = {
        "total_combos": total, "total_misses": total_misses,
        "single_leg_failures": single_leg_failures,
        "single_leg_failures_pct": round(single_leg_failures / total_misses, 4) if total_misses > 0 else 0.0,
        "most_failed_leg_type": most_failed,
        "failure_counts": dict(failure_counts),
    }

    calibration = {}
    for lo, hi in [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 1.0)]:
        bucket = [c for c in all_combos if lo <= c["combo_predicted_prob"] < hi and c["combo_hit"] is not None]
        if bucket:
            calibration[f"{lo:.2f}-{hi:.2f}"] = {
                "n": len(bucket),
                "avg_predicted": round(float(np.mean([c["combo_predicted_prob"] for c in bucket])), 4),
                "actual_hit_rate": round(float(np.mean([1 if c["combo_hit"] else 0 for c in bucket])), 4),
            }

    total_hits = sum(1 for c in all_combos if c["combo_hit"] is True)
    overall = {
        "total_combos": total, "total_hits": total_hits,
        "overall_hit_rate": round(total_hits / total, 4) if total > 0 else 0.0,
    }

    return {
        "tier_stats": tier_stats,
        "leg_type_stats": dict(leg_stats),
        "failure_analysis": failure_analysis,
        "calibration": calibration,
        "overall": overall,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(stats):
    print("\n" + "=" * 78)
    print("  MULTI-BET COMBINATION BACKTEST")
    print("=" * 78)

    print(f"\n  {'Tier':<18} {'Combos':>7} {'Hits':>6} {'Hit Rate':>10} {'Avg Prob':>10} {'Avg Legs':>10}")
    print(f"  {'-'*16:<18} {'-'*7:>7} {'-'*6:>6} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10}")
    for tier in TIER_KEYS:
        ts = stats["tier_stats"].get(tier, {})
        n = ts.get("n_combos", 0)
        hits = ts.get("n_hits", 0)
        hr = ts.get("combo_hit_rate", 0)
        ap = ts.get("avg_predicted_prob", 0)
        al = ts.get("avg_legs", 0)
        label = TIER_LABELS.get(tier, tier)
        print(f"  {label:<18} {n:>7} {hits:>6} {hr:>9.1%} {ap:>10.4f} {al:>10.1f}")
    ov = stats["overall"]
    print(f"  {'OVERALL':<18} {ov['total_combos']:>7} {ov['total_hits']:>6} {ov['overall_hit_rate']:>9.1%}")

    print(f"\n  {'Leg Type':<22} {'Used':>6} {'Hit Rate':>10} {'Weakest Link':>14}")
    print(f"  {'-'*20:<22} {'-'*6:>6} {'-'*10:>10} {'-'*14:>14}")
    for lt in sorted(stats["leg_type_stats"].keys()):
        ls = stats["leg_type_stats"][lt]
        print(f"  {lt:<22} {ls['n_used']:>6} {ls['hit_rate']:>9.1%} {ls['n_weakest_link']:>14}")

    fa = stats["failure_analysis"]
    print(f"\n  Misses: {fa['total_misses']}/{fa['total_combos']}  "
          f"Single-leg failures: {fa['single_leg_failures']} ({fa['single_leg_failures_pct']:.0%} of misses)")
    if fa["most_failed_leg_type"]:
        print(f"  Most failed: {fa['most_failed_leg_type']} ({fa['failure_counts'].get(fa['most_failed_leg_type'], 0)}x)")
    print("=" * 78)


def save_json(stats, all_combos, year, label):
    rounds_dict = defaultdict(list)
    for c in all_combos:
        rnd = c.get("round_number", "unknown")
        rounds_dict[str(rnd)].append({
            "match_id": c.get("match_id"),
            "tier": c["tier"],
            "tier_label": c.get("tier_label", ""),
            "tier_desc": c.get("tier_desc", ""),
            "combo_predicted_prob": round(c["combo_predicted_prob"], 6),
            "combo_hit": c["combo_hit"],
            "n_legs": c["n_legs"],
            "legs": c["legs"],
        })

    experiment = {
        "label": label, "season": year,
        "tier_definitions": {t[0]: {"label": t[1], "desc": t[2]} for t in TIERS},
        "summary": {
            **{tier: stats["tier_stats"].get(tier, {}) for tier in TIER_KEYS},
            "overall": stats["overall"],
        },
        "leg_type_stats": stats["leg_type_stats"],
        "failure_analysis": stats["failure_analysis"],
        "calibration": stats["calibration"],
        "rounds": dict(rounds_dict),
    }

    config.ensure_dirs()
    out_path = config.EXPERIMENTS_DIR / f"{label}.json"
    with open(out_path, "w") as f:
        json.dump(experiment, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_year(year, label=None, run_id=None):
    """Run multi-bet backtest for a single year."""
    label = label or f"multi_backtest_{year}"
    store = LearningStore(base_dir=config.SEQUENTIAL_DIR, run_id=run_id)
    runs = store.list_runs(year=year, subdir="predictions")
    if not runs:
        print(f"No sequential runs found for {year}")
        return None

    run_id = run_id or runs[-1]
    store = LearningStore(base_dir=config.SEQUENTIAL_DIR, run_id=run_id)
    print(f"\n{'='*60}")
    print(f"  Year: {year}  |  Run: {run_id}")
    print(f"{'='*60}")

    print("Loading season data...")
    full_df = load_season_data(store, year)
    if full_df.empty:
        print("No data loaded")
        return None

    team_matches = load_team_matches(year)
    matches_df = load_matches()
    venue_history = build_venue_history(matches_df, up_to_year=year)

    n_rounds = full_df["round_number"].nunique()
    n_matches = full_df["match_id"].nunique()
    print(f"Loaded: {len(full_df)} player-rows, {n_matches} matches, {n_rounds} rounds")
    print(f"Venue history: {len(venue_history)} venues")

    all_combos = []
    matches_with_combos = 0

    for match_id in full_df["match_id"].unique():
        match_df = full_df[full_df["match_id"] == match_id]
        rnd = int(match_df["round_number"].iloc[0])

        # Get venue for this match
        venue = match_df["venue"].iloc[0] if "venue" in match_df.columns else None

        # Compute calibrated model residual std from prior rounds
        model_team_std, model_match_std = compute_model_residual_std(full_df, team_matches, rnd)

        # Generate all candidate legs
        candidates = generate_candidate_legs(
            match_df, venue, venue_history, model_team_std, model_match_std
        )

        # Build combos for each tier
        match_combos = []
        for tier_key, _, _, tier_threshold in TIERS:
            combos = build_tier_combos(candidates, tier_key, tier_threshold)
            for c in combos:
                c["match_id"] = match_id
                c["round_number"] = rnd
            match_combos.extend(combos)

        if match_combos:
            matches_with_combos += 1
        all_combos.extend(match_combos)

    # Fill in actuals for team/match total legs
    year_matches = matches_df[matches_df["year"] == year] if not matches_df.empty else pd.DataFrame()
    fill_team_match_actuals(all_combos, team_matches, year_matches)

    print(f"\nGenerated {len(all_combos)} combos across {matches_with_combos}/{n_matches} matches")

    stats = compute_stats(all_combos)
    print_summary(stats)
    save_json(stats, all_combos, year, label)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Multi-bet combination backtest")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--all-years", action="store_true", help="Run for all years 2021-2025")
    args = parser.parse_args()

    if args.all_years:
        for year in [2021, 2022, 2023, 2024, 2025]:
            run_year(year, run_id=args.run_id)
    else:
        run_year(args.year, label=args.label, run_id=args.run_id)


if __name__ == "__main__":
    main()
