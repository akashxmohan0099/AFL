"""Season-level aggregation services."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from api.data_loader import DataCache

logger = logging.getLogger(__name__)


def _fallback_game_prediction_from_player_predictions(
    predictions: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> dict:
    """Infer a lightweight pregame match snapshot from player-level predictions.

    This is only used when the dedicated game-winner prediction row is missing.
    The probability is a heuristic based on relative predicted team scores, not
    the calibrated game-winner model output.
    """
    if predictions.empty or not home_team or not away_team or "team" not in predictions.columns:
        return {}

    team_preds = predictions[predictions["team"].isin([home_team, away_team])].copy()
    if team_preds.empty:
        return {}

    def _team_totals(team: str) -> tuple[float | None, float | None, float | None]:
        grp = team_preds[team_preds["team"] == team]
        if grp.empty:
            return None, None, None
        goals = float(grp["predicted_goals"].sum()) if "predicted_goals" in grp.columns else None
        behinds = float(grp["predicted_behinds"].sum()) if "predicted_behinds" in grp.columns else None
        score = float(grp["predicted_score"].sum()) if "predicted_score" in grp.columns else None
        if score is None and goals is not None:
            score = goals * 6 + (behinds or 0.0)
        return goals, behinds, score

    _, _, home_score = _team_totals(home_team)
    _, _, away_score = _team_totals(away_team)
    if home_score is None or away_score is None:
        return {}

    margin = home_score - away_score
    total_score = home_score + away_score
    if total_score <= 0:
        home_win_prob = 0.5
    else:
        # Relative expected scoring edge -> pseudo win probability fallback.
        margin_ratio = margin / total_score
        home_win_prob = 1.0 / (1.0 + np.exp(-4.0 * margin_ratio))

    predicted_winner = home_team if margin > 0 else away_team if margin < 0 else "Draw"
    return {
        "home_win_prob": round(float(np.clip(home_win_prob, 0.05, 0.95)), 3),
        "predicted_margin": round(float(margin), 1),
        "predicted_winner": predicted_winner,
    }


def _stat_summary(series: pd.Series) -> dict:
    """Return min/max/median for a numeric series."""
    clean = series.dropna()
    if clean.empty:
        return {"min": None, "max": None, "median": None}
    return {
        "min": round(float(clean.min()), 1),
        "max": round(float(clean.max()), 1),
        "median": round(float(clean.median()), 1),
    }


def _get_store(cache: DataCache):
    """Return the best available store (sequential preferred, then learning)."""
    return cache.sequential_store or cache.store


def _get_fixture_round_files(year: int) -> list[Path]:
    import config as _cfg2

    return sorted(
        _cfg2.FIXTURES_DIR.glob(f"round_*_{year}.csv"),
        key=lambda f: int(f.stem.split("_")[1]),
    )


def _build_round_progress(year: int, matches: pd.DataFrame) -> dict[int, dict]:
    round_progress: dict[int, dict] = {}
    season_matches = matches[matches["year"] == year] if not matches.empty else pd.DataFrame()

    for fpath in _get_fixture_round_files(year):
        round_num = int(fpath.stem.split("_")[1])
        fixture_df = pd.read_csv(fpath)
        fixture_matches = int((fixture_df["is_home"] == 1).sum())

        played_matches = 0
        if not season_matches.empty:
            round_matches = season_matches[season_matches["round_number"] == round_num]
            if not round_matches.empty:
                played_matches = int(
                    (round_matches["home_score"].notna() & round_matches["away_score"].notna()).sum()
                )

        round_progress[round_num] = {
            "fixture_path": fpath,
            "fixture_matches": fixture_matches,
            "played_matches": played_matches,
        }

    if not season_matches.empty:
        for round_num, round_matches in season_matches.groupby("round_number", observed=True):
            if pd.isna(round_num):
                continue
            round_num = int(round_num)
            played_matches = int(
                (round_matches["home_score"].notna() & round_matches["away_score"].notna()).sum()
            )
            fixture_matches = len(round_matches)
            existing = round_progress.get(round_num)
            if existing is None:
                round_progress[round_num] = {
                    "fixture_path": None,
                    "fixture_matches": fixture_matches,
                    "played_matches": played_matches,
                }
            else:
                existing["fixture_matches"] = max(int(existing["fixture_matches"]), fixture_matches)
                existing["played_matches"] = max(int(existing["played_matches"]), played_matches)

    return round_progress


def _next_incomplete_round(round_progress: dict[int, dict]) -> int | None:
    for round_num in sorted(round_progress):
        info = round_progress[round_num]
        if info["played_matches"] < info["fixture_matches"]:
            return round_num
    return None


def _next_unstarted_round(round_progress: dict[int, dict]) -> int | None:
    for round_num in sorted(round_progress):
        if round_progress[round_num]["played_matches"] == 0:
            return round_num
    return None


def _round_status(round_num: int, round_progress: dict[int, dict], next_unstarted_round: int | None) -> str:
    info = round_progress[round_num]
    if info["fixture_matches"] > 0 and info["played_matches"] >= info["fixture_matches"]:
        return "completed"
    if info["played_matches"] > 0:
        return "in_progress"
    if round_num == next_unstarted_round:
        return "upcoming"
    return "future"


def _merge_round_predictions_outcomes(preds: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Merge per-round predictions/outcomes using the safest available key."""
    if preds.empty or outcomes.empty:
        return pd.DataFrame()

    base_cols = ["player", "team"]
    merge_cols = list(base_cols)
    used_match_id = False
    if "match_id" in preds.columns and "match_id" in outcomes.columns:
        merge_cols.append("match_id")
        used_match_id = True
    elif "round" in preds.columns and "round" in outcomes.columns:
        merge_cols.append("round")

    merged = preds.merge(outcomes, on=merge_cols, how="inner", suffixes=("", "_out"))

    # Upcoming-round predictions can be stored with synthetic negative match_ids;
    # once results land, outcomes use real match_ids. If the strict merge misses
    # entirely, fall back to per-round (player, team) only when that key is unique.
    if used_match_id and merged.empty:
        pred_unique = not preds.duplicated(base_cols).any()
        out_unique = not outcomes.duplicated(base_cols).any()
        if pred_unique and out_unique:
            merged = preds.merge(outcomes, on=base_cols, how="inner", suffixes=("", "_out"))

    if merged.empty:
        return merged

    rename_map = {}
    if "actual_goals" not in merged.columns and "GL" in merged.columns:
        rename_map["GL"] = "actual_goals"
    if "actual_disposals" not in merged.columns and "DI" in merged.columns:
        rename_map["DI"] = "actual_disposals"
    if "actual_marks" not in merged.columns and "MK" in merged.columns:
        rename_map["MK"] = "actual_marks"
    if rename_map:
        merged = merged.rename(columns=rename_map)

    return merged


def _build_schedule_round_matches(
    round_num: int,
    fixture_path: Path | None,
    season_matches: pd.DataFrame,
) -> list[dict]:
    """Build schedule match rows from fixtures when present, otherwise from played matches."""
    if fixture_path is not None and fixture_path.exists():
        fix_df = pd.read_csv(fixture_path)
        home_rows = fix_df[fix_df["is_home"] == 1]
        return [
            {
                "home_team": row["team"],
                "away_team": row["opponent"],
                "venue": row.get("venue", ""),
                "date": row.get("date", ""),
            }
            for _, row in home_rows.iterrows()
        ]

    if season_matches.empty:
        return []

    round_matches = season_matches[season_matches["round_number"] == round_num].copy()
    if round_matches.empty:
        return []

    sort_cols = [col for col in ["date", "home_team", "away_team"] if col in round_matches.columns]
    if sort_cols:
        round_matches = round_matches.sort_values(sort_cols)

    match_rows = []
    for _, row in round_matches.iterrows():
        date = row.get("date", "")
        if pd.notna(date):
            date = str(date)
        else:
            date = ""
        match_rows.append(
            {
                "home_team": row.get("home_team", ""),
                "away_team": row.get("away_team", ""),
                "venue": row.get("venue", ""),
                "date": date,
            }
        )

    return match_rows


def get_season_summary(year: int) -> dict:
    """Season overview: games played, model accuracy, current round."""
    cache = DataCache.get()
    matches = cache.matches
    round_progress = _build_round_progress(year, matches)

    season_matches = matches[matches["year"] == year]
    total_matches = len(season_matches)

    completed_rounds = sorted(
        round_num
        for round_num, info in round_progress.items()
        if info["fixture_matches"] > 0 and info["played_matches"] >= info["fixture_matches"]
    )
    played_rounds = sorted(
        round_num
        for round_num, info in round_progress.items()
        if info["played_matches"] > 0
    )
    current_round = int(max(played_rounds)) if played_rounds else 0
    total_rounds = len(round_progress)

    # Build actual winner lookup: (home_team, away_team, round) -> actual_winner
    actual_winners: dict[tuple, str] = {}
    if not season_matches.empty:
        for _, mrow in season_matches.iterrows():
            h = mrow.get("home_team", "")
            a = mrow.get("away_team", "")
            rn = int(mrow["round_number"]) if pd.notna(mrow.get("round_number")) else None
            hs = mrow.get("home_score")
            as_ = mrow.get("away_score")
            if rn is not None and pd.notna(hs) and pd.notna(as_):
                if int(hs) > int(as_):
                    actual_winners[(h, a, rn)] = h
                elif int(as_) > int(hs):
                    actual_winners[(h, a, rn)] = a
                else:
                    actual_winners[(h, a, rn)] = "Draw"

    # Model accuracy from sequential store — load per-round to avoid match_id collisions
    accuracy = {}
    store = _get_store(cache)
    if store is not None:
        all_merged = []
        gw_correct = 0
        gw_total = 0

        for rnum in played_rounds:
            rnum = int(rnum)
            # Player-level accuracy: merge predictions with outcomes by (player, team)
            try:
                preds = store.load_predictions(year=year, round_num=rnum)
                outcomes = store.load_outcomes(year=year, round_num=rnum)
                if not preds.empty and not outcomes.empty:
                    merged = _merge_round_predictions_outcomes(preds, outcomes)
                    if merged.empty:
                        continue
                    all_merged.append(merged)
            except Exception:
                logger.debug("Failed to load data", exc_info=True)

            # Game winner accuracy: match by team names
            try:
                gp = store.load_game_predictions(year=year, round_num=rnum)
                if gp is not None and not gp.empty:
                    for _, gp_row in gp.iterrows():
                        h = str(gp_row.get("home_team", ""))
                        a = str(gp_row.get("away_team", ""))
                        pw = gp_row.get("predicted_winner", "")
                        aw = actual_winners.get((h, a, rnum))
                        if aw is not None and pw:
                            gw_total += 1
                            if pw == aw:
                                gw_correct += 1
            except Exception:
                logger.debug("Failed to load data", exc_info=True)

        if all_merged:
            merged = pd.concat(all_merged, ignore_index=True)
            if "predicted_goals" in merged.columns and "actual_goals" in merged.columns:
                accuracy["goals_mae"] = round(float((merged["predicted_goals"] - merged["actual_goals"]).abs().mean()), 3)
                pred_scorer = merged["predicted_goals"] >= 0.5
                actual_scorer = merged["actual_goals"] >= 1
                accuracy["scorer_accuracy"] = round(float((pred_scorer == actual_scorer).mean() * 100), 1)
            if "predicted_disposals" in merged.columns and "actual_disposals" in merged.columns:
                accuracy["disposals_mae"] = round(float((merged["predicted_disposals"] - merged["actual_disposals"]).abs().mean()), 3)
            if "predicted_marks" in merged.columns and "actual_marks" in merged.columns:
                accuracy["marks_mae"] = round(float((merged["predicted_marks"] - merged["actual_marks"]).abs().mean()), 3)

        if gw_total > 0:
            accuracy["game_winner_accuracy"] = round(float(gw_correct / gw_total * 100), 1)
            accuracy["game_winner_correct"] = gw_correct
            accuracy["game_winner_total"] = gw_total

    return {
        "year": year,
        "total_matches": total_matches,
        "total_rounds": total_rounds,
        "completed_rounds": len(completed_rounds),
        "current_round": current_round,
        "rounds_list": [int(r) for r in completed_rounds],
        "accuracy": accuracy,
    }


def get_season_matches(year: int) -> list[dict]:
    """All matches for a season with actual scores, predicted outcomes, and team stat totals."""
    cache = DataCache.get()
    matches = cache.matches
    pg = cache.player_games

    season = matches[matches["year"] == year].sort_values(["round_number", "date"])

    # Load predictions per-round from sequential store (avoids match_id collisions
    # since synthetic negative IDs are reused across rounds)
    store = _get_store(cache)
    # Lookups keyed by (home_team, away_team, round_number) → prediction data
    game_pred_lookup: dict[tuple, dict] = {}
    pred_totals_lookup: dict[tuple, dict] = {}

    if store is not None:
        round_nums = set()
        if not season.empty:
            round_nums = set(int(r) for r in season["round_number"].dropna().unique())
        # Also check fixture files for rounds not yet in matches.parquet
        import config as _cfg2
        for fpath in _cfg2.FIXTURES_DIR.glob(f"round_*_{year}.csv"):
            try:
                round_nums.add(int(fpath.stem.split("_")[1]))
            except (ValueError, IndexError):
                pass

        for rnum in sorted(round_nums):
            # Game predictions (win prob, margin)
            try:
                gp = store.load_game_predictions(year=year, round_num=rnum)
                if gp is not None and not gp.empty:
                    for _, gp_row in gp.iterrows():
                        home_t = str(gp_row.get("home_team", ""))
                        away_t = str(gp_row.get("away_team", ""))
                        if home_t and away_t:
                            entry = {}
                            if "home_win_prob" in gp_row.index and pd.notna(gp_row["home_win_prob"]):
                                entry["home_win_prob"] = round(float(gp_row["home_win_prob"]), 3)
                                entry["predicted_winner"] = home_t if gp_row["home_win_prob"] > 0.5 else away_t
                            if "predicted_margin" in gp_row.index and pd.notna(gp_row["predicted_margin"]):
                                entry["predicted_margin"] = round(float(gp_row["predicted_margin"]), 1)
                            if entry:
                                game_pred_lookup[(home_t, away_t, rnum)] = entry
            except Exception:
                logger.debug("Failed to load data", exc_info=True)

            # Player predictions → team totals
            try:
                pp = store.load_predictions(year=year, round_num=rnum)
                if pp is not None and not pp.empty:
                    for _mid, grp in pp.groupby("match_id", observed=True):
                        teams_in_match = [str(t) for t in grp["team"].unique()]
                        team_totals = {}
                        for team, tgrp in grp.groupby("team", observed=True):
                            team_totals[str(team)] = {
                                "pred_gl": round(float(tgrp["predicted_goals"].sum()), 1) if "predicted_goals" in tgrp.columns else None,
                                "pred_di": round(float(tgrp["predicted_disposals"].sum()), 0) if "predicted_disposals" in tgrp.columns else None,
                                "pred_mk": round(float(tgrp["predicted_marks"].sum()), 0) if "predicted_marks" in tgrp.columns else None,
                            }
                        # Need to figure out home/away from the game_pred_lookup or opponent col
                        if len(teams_in_match) == 2:
                            t0, t1 = teams_in_match[0], teams_in_match[1]
                            # Try both orderings — the correct one will match
                            for h, a in [(t0, t1), (t1, t0)]:
                                pred_totals_lookup[(h, a, rnum)] = team_totals
            except Exception:
                logger.debug("Failed to load data", exc_info=True)

    # Pre-aggregate actual team totals from player_games
    season_pg = pg[pg["year"] == year]
    actual_team_totals = {}
    if not season_pg.empty:
        for mid, grp in season_pg.groupby("match_id", observed=True):
            mid_int = int(mid)
            actual_team_totals[mid_int] = {}
            for team, tgrp in grp.groupby("team", observed=True):
                actual_team_totals[mid_int][str(team)] = {
                    "actual_gl": int(tgrp["GL"].sum()),
                    "actual_di": int(tgrp["DI"].sum()),
                    "actual_mk": int(tgrp["MK"].sum()),
                }

    results = []
    for _, m in season.iterrows():
        mid = int(m["match_id"])
        home = m.get("home_team", "")
        away = m.get("away_team", "")
        rnum = int(m["round_number"]) if pd.notna(m.get("round_number")) else None

        entry = {
            "match_id": mid,
            "round_number": rnum,
            "date": str(m["date"])[:10] if pd.notna(m.get("date")) else None,
            "venue": m.get("venue", ""),
            "home_team": home,
            "away_team": away,
            "home_score": int(m["home_score"]) if pd.notna(m.get("home_score")) else None,
            "away_score": int(m["away_score"]) if pd.notna(m.get("away_score")) else None,
        }

        # Add actual winner
        if entry["home_score"] is not None and entry["away_score"] is not None:
            if entry["home_score"] > entry["away_score"]:
                entry["actual_winner"] = home
            elif entry["away_score"] > entry["home_score"]:
                entry["actual_winner"] = away
            else:
                entry["actual_winner"] = "Draw"

        # Add game predictions — match by team names + round (not match_id)
        if rnum is not None:
            gp_entry = game_pred_lookup.get((home, away, rnum))
            if gp_entry:
                entry.update(gp_entry)
                if "home_win_prob" in gp_entry:
                    entry["away_win_prob"] = round(1 - gp_entry["home_win_prob"], 3)
                if "predicted_winner" in entry and "actual_winner" in entry:
                    entry["correct"] = entry["predicted_winner"] == entry["actual_winner"]

        # Add team stat totals (predicted vs actual)
        if rnum is not None:
            pt = pred_totals_lookup.get((home, away, rnum))
            if pt:
                entry["home_pred"] = pt.get(home)
                entry["away_pred"] = pt.get(away)
        if mid in actual_team_totals:
            at = actual_team_totals[mid]
            entry["home_actual"] = at.get(home)
            entry["away_actual"] = at.get(away)

        results.append(entry)

    return results


def _enrich_player_from_pred(player_entry: dict, pr) -> None:
    """Add prediction fields from a prediction row to a player entry dict."""
    if "predicted_goals" in pr.index and pd.notna(pr["predicted_goals"]):
        player_entry["predicted_gl"] = round(float(pr["predicted_goals"]), 2)
    if "predicted_disposals" in pr.index and pd.notna(pr["predicted_disposals"]):
        player_entry["predicted_di"] = round(float(pr["predicted_disposals"]), 1)
    if "predicted_marks" in pr.index and pd.notna(pr["predicted_marks"]):
        player_entry["predicted_mk"] = round(float(pr["predicted_marks"]), 1)
    if "predicted_behinds" in pr.index and pd.notna(pr["predicted_behinds"]):
        player_entry["predicted_bh"] = round(float(pr["predicted_behinds"]), 2)

    # Confidence intervals
    for stat in ["gl", "di", "mk"]:
        lo = f"conf_lower_{stat}"
        hi = f"conf_upper_{stat}"
        if lo in pr.index and pd.notna(pr[lo]) and hi in pr.index and pd.notna(pr[hi]):
            player_entry[f"conf_{stat}"] = [round(float(pr[lo]), 1), round(float(pr[hi]), 1)]

    # Goal probabilities
    if "p_scorer" in pr.index and pd.notna(pr["p_scorer"]):
        player_entry["p_scorer"] = round(float(pr["p_scorer"]), 3)
    for col in ["p_2plus_goals", "p_3plus_goals"]:
        if col in pr.index and pd.notna(pr[col]):
            player_entry[col] = round(float(pr[col]), 3)

    # Disposal thresholds
    for col in ["p_15plus_disp", "p_20plus_disp", "p_25plus_disp", "p_30plus_disp"]:
        if col in pr.index and pd.notna(pr[col]):
            player_entry[col] = round(float(pr[col]), 3)

    # Marks thresholds
    for col in ["p_3plus_mk", "p_5plus_mk"]:
        if col in pr.index and pd.notna(pr[col]):
            player_entry[col] = round(float(pr[col]), 3)

    # Role and career
    if "player_role" in pr.index and pd.notna(pr["player_role"]):
        player_entry["player_role"] = pr["player_role"]
    if "career_goal_avg" in pr.index and pd.notna(pr["career_goal_avg"]):
        player_entry["career_goal_avg"] = round(float(pr["career_goal_avg"]), 2)


def _enrich_players_advanced(cache: DataCache, players: list, venue: str,
                             year: int, round_num: int) -> None:
    """Add advanced context per player: venue/opponent/form/season/career stats.

    Pre-groups player data to avoid per-player DataFrame filtering (N+1 fix).
    """
    import config as _cfg

    pg = cache.player_games
    if pg.empty or not year:
        return

    venue_canonical = _cfg.VENUE_NAME_MAP.get(venue, venue) if venue else ""
    venue_aliases = {v for v, c in _cfg.VENUE_NAME_MAP.items() if c == venue_canonical}
    venue_aliases.add(venue_canonical)
    if venue:
        venue_aliases.add(venue)

    # Build opponent lookup: team -> opponent team
    teams_in_match = {p.get("team", "") for p in players if p.get("team", "")}
    opponent_map = {}
    for t in teams_in_match:
        for t2 in teams_in_match:
            if t2 != t:
                opponent_map[t] = t2
                break

    # Collect all (player, team) pairs we need
    player_keys = {(p.get("player", ""), p.get("team", "")) for p in players
                   if p.get("player", "") and p.get("team", "")}

    # Pre-filter to recent data (last 5 years) for speed
    recent = pg[(pg["year"] >= year - 5) & (pg.get("did_not_play", 0) != 1)]

    # Single group-by to get all player data at once
    # Filter to only players in our match
    player_names = {k[0] for k in player_keys}
    player_teams = {k[1] for k in player_keys}
    filtered = recent[recent["player"].isin(player_names) & recent["team"].isin(player_teams)]

    # Pre-group by (player, team)
    grouped = {}
    for (pname, team), grp in filtered.groupby(["player", "team"], observed=True):
        grouped[(pname, team)] = grp

    # Stat columns to include in advanced enrichment
    _ADV_STATS = ["GL", "BH", "DI", "MK", "KI", "HB", "TK", "HO", "CP", "UP",
                  "IF", "CL", "CG", "FF", "FA"]
    # Columns available in the data
    _avail = [c for c in _ADV_STATS if c in filtered.columns]

    def _stat_block(data: pd.DataFrame, full: bool = False) -> dict:
        """Build a stat summary dict for a subset of games."""
        block: dict = {"games": len(data)}
        for col in _avail:
            vals = data[col].dropna()
            if vals.empty:
                continue
            key = col.lower()
            block[f"avg_{key}"] = round(float(vals.mean()), 2 if col in ("GL", "BH") else 1)
            if full and len(vals) >= 3:
                block[f"med_{key}"] = round(float(vals.median()), 1)
                block[f"max_{key}"] = int(vals.max())
                block[f"min_{key}"] = int(vals.min())
        return block

    # Build a lookup from player name+team to advanced stats
    player_lookup = {}
    for key, player_data in grouped.items():
        if key not in player_keys:
            continue

        pname, team = key
        opponent = opponent_map.get(team, "")

        # Exclude the current match from stats
        if round_num:
            before_match = player_data[
                (player_data["year"] < year) |
                ((player_data["year"] == year) & (player_data["round_number"] < round_num))
            ]
        else:
            before_match = player_data

        if before_match.empty:
            continue

        adv = {}

        # --- Season averages ---
        season = before_match[before_match["year"] == year]
        if not season.empty and len(season) >= 2:
            adv["season"] = _stat_block(season, full=True)

        # --- Career averages (all data) ---
        adv["career"] = _stat_block(before_match, full=True)

        # --- Last 5 form ---
        last5 = before_match.sort_values("date").tail(5)
        if len(last5) >= 3:
            adv["form_5"] = _stat_block(last5, full=True)

        # --- Venue history ---
        if venue_aliases:
            at_venue = before_match[before_match["venue"].isin(venue_aliases)]
            if not at_venue.empty:
                adv["venue"] = _stat_block(at_venue, full=len(at_venue) >= 3)

        # --- Opponent history ---
        if opponent:
            vs_opp = before_match[before_match["opponent"] == opponent]
            if not vs_opp.empty:
                adv["opponent"] = _stat_block(vs_opp, full=len(vs_opp) >= 3)
                # Individual game-by-game opponent history
                opp_games_list = []
                for _, g in vs_opp.sort_values("date").iterrows():
                    opp_entry: dict = {
                        "opponent": g.get("opponent", ""),
                        "venue": g.get("venue", ""),
                        "round": int(g["round_number"]) if pd.notna(g.get("round_number")) else None,
                        "year": int(g["year"]),
                    }
                    for col in _avail:
                        if pd.notna(g[col]):
                            opp_entry[col.lower()] = int(g[col])
                    opp_games_list.append(opp_entry)
                adv["opponent_games"] = opp_games_list

        # --- Last 10 games detail (for recent form + streaks) ---
        last10_detail = before_match.sort_values("date").tail(10)
        if not last10_detail.empty:
            games_list = []
            for _, g in last10_detail.iterrows():
                game_entry: dict = {
                    "opponent": g.get("opponent", ""),
                    "venue": g.get("venue", ""),
                    "round": int(g["round_number"]) if pd.notna(g.get("round_number")) else None,
                    "year": int(g["year"]),
                }
                for col in _avail:
                    if pd.notna(g[col]):
                        game_entry[col.lower()] = int(g[col])
                games_list.append(game_entry)
            adv["recent_games"] = games_list

            # Streaks: per-game values for last 10 (chronological order)
            adv["streak_gl"] = [int(g["GL"]) if pd.notna(g.get("GL")) else 0 for _, g in last10_detail.iterrows()]
            adv["streak_di"] = [int(g["DI"]) if pd.notna(g.get("DI")) else 0 for _, g in last10_detail.iterrows()]
            adv["streak_mk"] = [int(g["MK"]) if pd.notna(g.get("MK")) else 0 for _, g in last10_detail.iterrows()]
            adv["streak_tk"] = [int(g["TK"]) if pd.notna(g.get("TK")) else 0 for _, g in last10_detail.iterrows()]

        if adv:
            player_lookup[key] = adv

    # Apply to player dicts
    for p in players:
        key = (p.get("player", ""), p.get("team", ""))
        if key in player_lookup:
            p["advanced"] = player_lookup[key]


def _compute_match_context(cache: DataCache, home_team: str, away_team: str,
                           venue: str, match_date: str, year: int) -> dict:
    """Compute match context: game time, home/away records, venue history."""
    import config as _cfg

    context: dict = {}
    tm = cache.team_matches
    matches = cache.matches

    # --- Game time / day-night ---
    if match_date:
        try:
            dt = pd.Timestamp(match_date)
            if not pd.isna(dt):
                context["day_of_week"] = dt.strftime("%A")
                context["date_formatted"] = dt.strftime("%d/%m/%y")
                if dt.hour > 0:
                    context["time"] = dt.strftime("%H:%M")
                    context["day_night"] = "Night" if dt.hour >= 17 else "Day"
        except Exception:
            logger.debug("Unexpected error", exc_info=True)

    # --- Normalize venue for lookups ---
    venue_canonical = _cfg.VENUE_NAME_MAP.get(venue, venue) if venue else ""
    # Use display name: try raw venue, then canonical
    display = _cfg.VENUE_DISPLAY_MAP.get(venue, None) or _cfg.VENUE_DISPLAY_MAP.get(venue_canonical, None) or venue or ""
    context["venue_display"] = display

    if tm.empty or not home_team or not away_team:
        return context

    # Also build a set of all venue aliases that map to the same canonical
    venue_aliases = {v for v, c in _cfg.VENUE_NAME_MAP.items() if c == venue_canonical}
    venue_aliases.add(venue_canonical)
    if venue:
        venue_aliases.add(venue)

    # --- Team records (current season to date) ---
    def _team_record(team: str, season_year: int) -> dict:
        mask = (tm["team"] == team) & (tm["year"] == season_year)
        games = tm[mask]
        if games.empty:
            return {"played": 0, "wins": 0, "losses": 0}
        wins = int((games["result"] == "W").sum())
        losses = int((games["result"] == "L").sum())
        draws = len(games) - wins - losses
        home_g = games[games["is_home"] == True]
        away_g = games[games["is_home"] == False]
        home_w = int((home_g["result"] == "W").sum()) if not home_g.empty else 0
        away_w = int((away_g["result"] == "W").sum()) if not away_g.empty else 0
        avg_score = round(float(games["score"].mean()), 1)
        avg_conceded = round(float(games["opp_score"].mean()), 1)
        return {
            "played": len(games),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "home_record": f"{home_w}-{len(home_g) - home_w}",
            "home_played": len(home_g),
            "home_wins": home_w,
            "away_record": f"{away_w}-{len(away_g) - away_w}",
            "away_played": len(away_g),
            "away_wins": away_w,
            "avg_score": avg_score,
            "avg_conceded": avg_conceded,
            **{f"{k}_score": v for k, v in _stat_summary(games["score"]).items()},
            **{f"{k}_conceded": v for k, v in _stat_summary(games["opp_score"]).items()},
        }

    if year:
        context["home_team_season"] = _team_record(home_team, year)
        context["away_team_season"] = _team_record(away_team, year)
        # Always include last season for context
        if year > 2015:
            context["home_team_last_season"] = _team_record(home_team, year - 1)
            context["away_team_last_season"] = _team_record(away_team, year - 1)

    # --- Venue history (last 5 years) ---
    def _venue_record(team: str) -> dict:
        mask = (tm["team"] == team) & (tm["venue"].isin(venue_aliases))
        if year:
            mask = mask & (tm["year"] >= year - 5)
        games = tm[mask]
        if games.empty:
            return {"played": 0, "wins": 0, "losses": 0, "avg_score": None}
        wins = int((games["result"] == "W").sum())
        avg_score = round(float(games["score"].mean()), 1)
        avg_opp = round(float(games["opp_score"].mean()), 1)
        avg_margin = round(float(games["margin"].mean()), 1) if "margin" in games.columns else None
        # Current season at this venue
        season_games = games[games["year"] == year] if year else pd.DataFrame()
        season_avg = round(float(season_games["score"].mean()), 1) if not season_games.empty else None
        season_played = len(season_games) if not season_games.empty else 0
        return {
            "played": len(games),
            "wins": wins,
            "losses": len(games) - wins,
            "avg_score": avg_score,
            "avg_conceded": avg_opp,
            "avg_margin": avg_margin,
            **{f"{k}_score": v for k, v in _stat_summary(games["score"]).items()},
            **{f"{k}_conceded": v for k, v in _stat_summary(games["opp_score"]).items()},
            **({f"{k}_margin": v for k, v in _stat_summary(games["margin"]).items()} if "margin" in games.columns else {}),
            "season_played": season_played,
            "season_avg_score": season_avg,
        }

    if venue_canonical:
        context["home_team_venue"] = _venue_record(home_team)
        context["away_team_venue"] = _venue_record(away_team)

    # --- Ground-wide stats (all teams at this venue) ---
    if venue_canonical:
        venue_mask = tm["venue"].isin(venue_aliases)
        if year:
            venue_5y = tm[venue_mask & (tm["year"] >= year - 5)]
            venue_season = tm[venue_mask & (tm["year"] == year)]
        else:
            venue_5y = tm[venue_mask]
            venue_season = pd.DataFrame()

        ground = {}
        if not venue_5y.empty:
            # Each team_matches row is one team's view; total score per match = sum of both teams
            # avg_score is the average points per team per game at this venue
            totals = venue_5y["score"] + venue_5y["opp_score"]
            ground["avg_score_5y"] = round(float(venue_5y["score"].mean()), 1)
            ground["avg_total_5y"] = round(float(totals.mean()), 1)
            ground["median_total_5y"] = round(float(totals.median()), 1)
            ground["total_games_5y"] = len(venue_5y) // 2  # 2 rows per match
            ground["highest_total_5y"] = int(totals.max())
            ground["lowest_total_5y"] = int(totals.min())
            # Last 5 games: group by match_id (2 rows per match → sum gives home+away score)
            if "match_id" in venue_5y.columns:
                try:
                    match_totals = (
                        venue_5y.groupby("match_id", observed=True)
                        .agg(date=("date", "max"), total=("score", "sum"))
                        .sort_values("date")
                        .tail(5)
                    )
                    if not match_totals.empty:
                        ground["last_5_avg_total"] = round(float(match_totals["total"].mean()), 1)
                        ground["last_5_median_total"] = round(float(match_totals["total"].median()), 1)
                        ground["last_5_highest"] = int(match_totals["total"].max())
                        ground["last_5_lowest"] = int(match_totals["total"].min())
                except Exception:
                    logger.debug("Unexpected error", exc_info=True)

            # Total score distribution — one row per match using deduplication
            try:
                unique_matches = venue_5y.drop_duplicates(subset=["match_id"])
                totals = unique_matches["score"] + unique_matches["opp_score"]
                n = len(totals)
                if n >= 5:
                    brackets = []
                    for t in range(80, 241, 10):
                        p_over = round(float((totals >= t).sum() / n * 100), 1)
                        if p_over == 0.0 and t > totals.max():
                            break
                        p_under = round(100.0 - p_over, 1)
                        brackets.append({"threshold": t, "p_over": p_over, "p_under": p_under})
                    ground["total_score_distribution"] = {"brackets": brackets, "sample_size": n}
            except Exception:
                logger.debug("Unexpected error in score distribution", exc_info=True)

        if not venue_season.empty:
            season_totals = venue_season["score"] + venue_season["opp_score"]
            ground["avg_score_season"] = round(float(venue_season["score"].mean()), 1)
            ground["avg_total_season"] = round(float(season_totals.mean()), 1)
            ground["median_total_season"] = round(float(season_totals.median()), 1)
            ground["total_games_season"] = len(venue_season) // 2
        if ground:
            context["ground_stats"] = ground

    # --- Rest days ---
    if not tm.empty and home_team and match_date:
        try:
            match_dt = pd.Timestamp(match_date)
            for side, team in [("home", home_team), ("away", away_team)]:
                team_rows = tm[(tm["team"] == team) & (tm["date"] < match_dt)].sort_values("date")
                if not team_rows.empty:
                    last_game_date = team_rows.iloc[-1]["date"]
                    if pd.notna(last_game_date):
                        rest = (match_dt - pd.Timestamp(last_game_date)).days
                        if rest >= 0:
                            context[f"{side}_rest_days"] = int(rest)
        except Exception:
            logger.debug("Failed to compute match context", exc_info=True)

    # --- Rest days historical impact ---
    if not tm.empty:
        def _rest_buckets(team: str) -> dict | None:
            team_rows = tm[(tm["team"] == team) & (tm["year"] >= (year or 2025) - 3)]
            if team_rows.empty or "rest_days" not in team_rows.columns:
                return None
            rd = team_rows.dropna(subset=["rest_days"])
            if rd.empty:
                return None
            buckets = {}
            for label, lo, hi in [("short", 0, 5), ("normal", 6, 7), ("extended", 8, 999)]:
                b = rd[(rd["rest_days"] >= lo) & (rd["rest_days"] <= hi)]
                if b.empty:
                    continue
                wins = int((b["result"] == "W").sum())
                losses = int((b["result"] == "L").sum())
                buckets[label] = {
                    "label": label,
                    "played": len(b),
                    "wins": wins,
                    "losses": losses,
                    "avg_score": round(float(b["score"].mean()), 1),
                    "avg_conceded": round(float(b["opp_score"].mean()), 1),
                    "avg_margin": round(float(b["margin"].mean()), 1) if "margin" in b.columns else None,
                }
            return buckets if buckets else None

        home_rest_impact = _rest_buckets(home_team)
        away_rest_impact = _rest_buckets(away_team)
        if home_rest_impact:
            context["home_rest_impact"] = home_rest_impact
        if away_rest_impact:
            context["away_rest_impact"] = away_rest_impact

        # League-wide rest impact
        league = tm[tm["year"] >= (year or 2025) - 3].dropna(subset=["rest_days"]) if "rest_days" in tm.columns else pd.DataFrame()
        if not league.empty:
            league_buckets = {}
            for label, lo, hi in [("short", 0, 5), ("normal", 6, 7), ("extended", 8, 999)]:
                b = league[(league["rest_days"] >= lo) & (league["rest_days"] <= hi)]
                if b.empty:
                    continue
                wins = int((b["result"] == "W").sum())
                league_buckets[label] = {
                    "label": label,
                    "played": len(b),
                    "win_pct": round(float(wins / len(b) * 100), 1),
                    "avg_score": round(float(b["score"].mean()), 1),
                }
            if league_buckets:
                context["league_rest_impact"] = league_buckets

    # --- Recent form (last 5 games) ---
    if not tm.empty and home_team and match_date:
        try:
            match_dt = pd.Timestamp(match_date)
            for side, team in [("home", home_team), ("away", away_team)]:
                recent = tm[(tm["team"] == team) & (tm["date"] < match_dt)].sort_values("date").tail(5)
                if not recent.empty:
                    form = []
                    for _, r in recent.iterrows():
                        form.append({
                            "result": r.get("result", ""),
                            "opponent": r.get("opponent", ""),
                            "score": int(r["score"]) if pd.notna(r.get("score")) else None,
                            "opp_score": int(r["opp_score"]) if pd.notna(r.get("opp_score")) else None,
                            "margin": int(r["margin"]) if pd.notna(r.get("margin")) else None,
                            "venue": r.get("venue", ""),
                            "is_home": bool(r.get("is_home", False)),
                        })
                    context[f"{side}_recent_form"] = form
        except Exception:
            logger.debug("Unexpected error", exc_info=True)

    # --- Stat matchup (season averages) ---
    if not tm.empty and home_team and year:
        stat_cols = ["DI", "MK", "TK", "CL", "CP", "IF", "RB", "GL", "BH"]
        try:
            for side, team in [("home", home_team), ("away", away_team)]:
                season_mask = (tm["team"] == team) & (tm["year"] == year)
                if match_date:
                    try:
                        season_mask = season_mask & (tm["date"] < pd.Timestamp(match_date))
                    except Exception:
                        logger.debug("Unexpected error", exc_info=True)
                season = tm[season_mask]
                if season.empty:
                    # Fall back to last season
                    season = tm[(tm["team"] == team) & (tm["year"] == year - 1)]
                if not season.empty:
                    stats = {"games": len(season), "avg_score": round(float(season["score"].mean()), 1), "avg_conceded": round(float(season["opp_score"].mean()), 1)}
                    for col in stat_cols:
                        if col in season.columns:
                            vals = season[col].dropna()
                            if not vals.empty:
                                stats[f"avg_{col.lower()}"] = round(float(vals.mean()), 1)
                                ss = _stat_summary(vals)
                                stats[f"min_{col.lower()}"] = ss["min"]
                                stats[f"max_{col.lower()}"] = ss["max"]
                                stats[f"median_{col.lower()}"] = ss["median"]
                    context[f"{side}_stats"] = stats
        except Exception:
            logger.debug("Unexpected error", exc_info=True)

    # --- Scoring by quarter ---
    pg = cache.player_games
    if not pg.empty and home_team and year:
        try:
            for side, team in [("home", home_team), ("away", away_team)]:
                season_pg = pg[(pg["team"] == team) & (pg["year"] == year)]
                if match_date:
                    try:
                        season_pg = season_pg[season_pg["date"] < pd.Timestamp(match_date)]
                    except Exception:
                        logger.debug("Unexpected error", exc_info=True)
                # Fall back to last season if no data or quarter columns are all zeros
                has_quarter_data = (
                    not season_pg.empty
                    and "q1_goals" in season_pg.columns
                    and season_pg[["q1_goals", "q2_goals", "q3_goals", "q4_goals"]].sum().sum() > 0
                )
                if not has_quarter_data:
                    season_pg = pg[(pg["team"] == team) & (pg["year"] == year - 1)]
                if not season_pg.empty and "q1_goals" in season_pg.columns:
                    # Aggregate by match_id to get team totals per quarter
                    qdata = season_pg.groupby("match_id", observed=True).agg(
                        q1_gl=("q1_goals", "sum"), q2_gl=("q2_goals", "sum"),
                        q3_gl=("q3_goals", "sum"), q4_gl=("q4_goals", "sum"),
                        q1_bh=("q1_behinds", "sum"), q2_bh=("q2_behinds", "sum"),
                        q3_bh=("q3_behinds", "sum"), q4_bh=("q4_behinds", "sum"),
                    )
                    if qdata[["q1_gl", "q2_gl", "q3_gl", "q4_gl"]].sum().sum() > 0:
                        quarters = {}
                        for q in [1, 2, 3, 4]:
                            gl = qdata[f"q{q}_gl"]
                            bh = qdata[f"q{q}_bh"]
                            pts = gl * 6 + bh
                            quarters[f"q{q}"] = {
                                "avg_goals": round(float(gl.mean()), 1),
                                "avg_behinds": round(float(bh.mean()), 1),
                                "avg_points": round(float(pts.mean()), 1),
                                **{f"{k}_points": v for k, v in _stat_summary(pts).items()},
                            }
                        context[f"{side}_quarters"] = quarters
        except Exception:
            logger.debug("Unexpected error", exc_info=True)

    # --- Per-team scoring distribution (last 5 years, all venues) ---
    def _team_score_dist(team: str) -> dict | None:
        if not team:
            return None
        mask = tm["team"] == team
        if year:
            mask = mask & (tm["year"] >= year - 5)
        rows = tm[mask]
        if len(rows) < 5:
            return None
        scores = rows["score"]
        n = len(scores)
        brackets = []
        for t in range(40, 141, 10):
            p_over = round(float((scores >= t).sum() / n * 100), 1)
            if p_over == 0.0:
                break
            brackets.append({"threshold": t, "p_over": p_over, "p_under": round(100.0 - p_over, 1)})
        if not brackets:
            return None
        return {"brackets": brackets, "sample_size": n,
                "avg_score": round(float(scores.mean()), 1),
                "median_score": round(float(scores.median()), 1),
                "highest": int(scores.max()), "lowest": int(scores.min())}

    if home_team and away_team:
        home_dist = _team_score_dist(home_team)
        away_dist = _team_score_dist(away_team)
        if home_dist or away_dist:
            context["team_score_distribution"] = {
                "home_team": home_team,
                "away_team": away_team,
                "home": home_dist,
                "away": away_dist,
            }

    # --- Head to head (last 5 years) ---
    def _h2h(team_a: str, team_b: str) -> dict:
        mask = (tm["team"] == team_a) & (tm["opponent"] == team_b)
        if year:
            mask = mask & (tm["year"] >= year - 5)
        games = tm[mask]
        if games.empty:
            return {"played": 0, "wins": 0, "losses": 0}
        wins = int((games["result"] == "W").sum())
        avg_score = round(float(games["score"].mean()), 1)
        avg_conceded = round(float(games["opp_score"].mean()), 1)
        avg_margin = round(float(games["margin"].mean()), 1) if "margin" in games.columns else None
        # At this venue specifically
        venue_games = games[games["venue"].isin(venue_aliases)] if venue_canonical else pd.DataFrame()
        venue_avg = round(float(venue_games["score"].mean()), 1) if not venue_games.empty else None
        venue_played = len(venue_games) if not venue_games.empty else 0
        return {
            "played": len(games), "wins": wins, "losses": len(games) - wins,
            "avg_score": avg_score, "avg_conceded": avg_conceded, "avg_margin": avg_margin,
            **{f"{k}_score": v for k, v in _stat_summary(games["score"]).items()},
            **{f"{k}_conceded": v for k, v in _stat_summary(games["opp_score"]).items()},
            **({f"{k}_margin": v for k, v in _stat_summary(games["margin"]).items()} if "margin" in games.columns else {}),
            "at_venue_played": venue_played, "at_venue_avg_score": venue_avg,
        }

    context["h2h_home"] = _h2h(home_team, away_team)
    context["h2h_away"] = _h2h(away_team, home_team)

    # --- Scoring averages summary (per team) ---
    def _scoring_averages(team: str, opponent: str) -> dict | None:
        if tm.empty or not team:
            return None
        match_dt = pd.Timestamp(match_date) if match_date else None
        before = tm[(tm["team"] == team) & (tm["date"] < match_dt)] if match_dt else tm[tm["team"] == team]
        if before.empty:
            return None
        before = before.sort_values("date")

        def _avg(df):
            if df.empty:
                return None
            return {"scored": round(float(df["score"].mean()), 1),
                    "conceded": round(float(df["opp_score"].mean()), 1),
                    "games": len(df)}

        # Season average
        season = before[before["year"] == year] if year else before
        # Last 5 / Last 10
        last5 = before.tail(5)
        last10 = before.tail(10)
        # Vs opponent (last 5 years)
        vs_opp = before[before["opponent"] == opponent]
        if year:
            vs_opp = vs_opp[vs_opp["year"] >= year - 5]
        # At venue
        at_venue = before[before["venue"].isin(venue_aliases)] if venue_canonical else pd.DataFrame()
        if year and not at_venue.empty:
            at_venue = at_venue[at_venue["year"] >= year - 5]

        return {
            "season": _avg(season),
            "last_5": _avg(last5),
            "last_10": _avg(last10),
            "vs_opponent": _avg(vs_opp),
            "at_venue": _avg(at_venue),
        }

    context["home_scoring_averages"] = _scoring_averages(home_team, away_team)
    context["away_scoring_averages"] = _scoring_averages(away_team, home_team)

    return context


def get_match_comparison(match_id: int, year: int = None, round_number: int = None,
                         home_team: str = None, away_team: str = None) -> Optional[dict]:
    """Detailed player-by-player predicted vs actual for a specific match."""
    cache = DataCache.get()
    pg = cache.player_games

    match_data = pg[pg["match_id"] == match_id]
    has_actuals = not match_data.empty

    # Try to determine year from player_games, or use provided params
    round_num = round_number
    if has_actuals:
        first = match_data.iloc[0]
        year = int(first["year"])
        round_num = int(first["round_number"]) if pd.notna(first.get("round_number")) else round_num

    # Determine teams for this match from matches.parquet (needed for prediction lookup)
    matches = cache.matches
    match_info = matches[matches["match_id"] == match_id]
    _home_team = home_team or ""
    _away_team = away_team or ""
    if not _home_team and not match_info.empty:
        mi = match_info.iloc[0]
        _home_team = mi.get("home_team", "")
        _away_team = mi.get("away_team", "")
    elif not _home_team and has_actuals:
        # Derive from player_games
        teams = match_data["team"].unique().tolist()
        home_rows = match_data[match_data["is_home"] == True]
        if not home_rows.empty:
            _home_team = str(home_rows.iloc[0]["team"])
            away_candidates = [t for t in teams if str(t) != _home_team]
            _away_team = str(away_candidates[0]) if away_candidates else ""
        elif len(teams) == 2:
            _home_team, _away_team = str(teams[0]), str(teams[1])

    # Load predictions from sequential store — match by team names, not match_id
    predictions = pd.DataFrame()
    store = _get_store(cache)
    if store is not None:
        match_teams = {_home_team, _away_team} - {""}
        years_to_try = [year] if year else [2026, 2025, 2024]
        for y in years_to_try:
            try:
                if round_num is not None:
                    rnd_preds = store.load_predictions(year=y, round_num=round_num)
                    if rnd_preds is not None and not rnd_preds.empty:
                        if match_teams:
                            match_preds = rnd_preds[rnd_preds["team"].isin(match_teams)]
                        else:
                            import logging
                            logging.getLogger(__name__).warning(
                                "get_match_comparison: no team names resolved for match_id=%s; "
                                "cannot filter predictions", match_id)
                            match_preds = pd.DataFrame()
                        if not match_preds.empty:
                            predictions = match_preds
                            if year is None:
                                year = y
                            break
                else:
                    # No round known — scan all rounds
                    rounds = store.list_rounds("predictions", year=y)
                    for _, rnd_val in rounds:
                        rnd_preds = store.load_predictions(year=y, round_num=rnd_val)
                        if rnd_preds is not None and not rnd_preds.empty:
                            if match_teams:
                                match_preds = rnd_preds[rnd_preds["team"].isin(match_teams)]
                            else:
                                import logging
                                logging.getLogger(__name__).warning(
                                    "get_match_comparison: no team names resolved for match_id=%s; "
                                    "cannot filter predictions", match_id)
                                match_preds = pd.DataFrame()
                            if not match_preds.empty:
                                predictions = match_preds
                                if year is None:
                                    year = y
                                round_num = rnd_val
                                break
                if not predictions.empty:
                    break
            except Exception:
                logger.debug("Failed to load data", exc_info=True)

    # If no actuals AND no predictions, return minimal context if we have team names
    if not has_actuals and predictions.empty:
        if _home_team and _away_team:
            # Try to get venue/date from fixture
            _venue = ""
            _date = ""
            _date_full = ""
            if round_num is not None and year is not None:
                import config as _cfg_min
                import csv as _csv_min
                fixture_path = _cfg_min.FIXTURES_DIR / f"round_{round_num}_{year}.csv"
                if fixture_path.exists():
                    try:
                        with open(fixture_path) as fh:
                            reader = _csv_min.DictReader(fh)
                            for frow in reader:
                                if frow.get("team") == _home_team and frow.get("is_home") == "1":
                                    _venue = frow.get("venue", "")
                                    _date = frow.get("date", "")
                                    _date_full = _date
                                    break
                    except Exception:
                        logger.debug("Unexpected error", exc_info=True)
            return {
                "match_id": match_id,
                "year": year,
                "round_number": round_num,
                "date": _date,
                "venue": _venue,
                "is_roofed": None,
                "attendance": None,
                "home_team": _home_team,
                "away_team": _away_team,
                "home_score": None,
                "away_score": None,
                "weather": None,
                "weather_summary": None,
                "weather_impact": [],
                "game_prediction": {},
                "players": [],
                "match_context": _compute_match_context(
                    cache, _home_team, _away_team, _venue, _date_full, year
                ),
            }
        return None

    # Use match_info already loaded above
    home_team = _home_team
    away_team = _away_team
    home_score = None
    away_score = None
    venue = ""
    date = ""
    date_full = ""  # Full datetime for context (includes time)

    if has_actuals:
        first = match_data.iloc[0]
        venue = first.get("venue", "")
        if pd.notna(first.get("date")):
            date = str(first["date"])[:10]
            date_full = str(first["date"])

    if not match_info.empty:
        mi = match_info.iloc[0]
        home_score = int(mi["home_score"]) if pd.notna(mi.get("home_score")) else None
        away_score = int(mi["away_score"]) if pd.notna(mi.get("away_score")) else None
        venue = mi.get("venue", venue) or venue
        if not date_full and pd.notna(mi.get("date")):
            date_full = str(mi["date"])
            date = date or str(mi["date"])[:10]

    # If no match_info, derive from predictions
    if not home_team and not predictions.empty:
        # Find home/away from predictions
        teams_in_pred = predictions["team"].unique().tolist()
        opponents_in_pred = predictions["opponent"].unique().tolist() if "opponent" in predictions.columns else []
        if len(teams_in_pred) == 2:
            home_team = teams_in_pred[0]
            away_team = teams_in_pred[1]
        elif len(teams_in_pred) == 1 and len(opponents_in_pred) >= 1:
            home_team = teams_in_pred[0]
            away_team = opponents_in_pred[0]
        if "venue" in predictions.columns:
            venue = venue or str(predictions.iloc[0].get("venue", ""))

    # Try fixture CSV for venue/date if still missing
    if (not venue or not date) and home_team and round_num is not None and year is not None:
        import config as _cfg
        import csv
        fixture_path = _cfg.FIXTURES_DIR / f"round_{round_num}_{year}.csv"
        if fixture_path.exists():
            try:
                with open(fixture_path) as fh:
                    reader = csv.DictReader(fh)
                    for frow in reader:
                        if frow.get("team") == home_team and frow.get("is_home") == "1":
                            if not venue:
                                venue = frow.get("venue", "")
                            if not date:
                                date = frow.get("date", "")
                                date_full = date
                            break
            except Exception:
                logger.debug("Failed to read fixture CSV", exc_info=True)

    # Build player comparison
    players = []
    stat_cols = ["GL", "BH", "DI", "MK", "KI", "HB", "TK", "HO", "CP", "UP",
                 "IF", "CL", "CG", "FF", "FA"]

    if has_actuals:
        # We have actual game data — iterate over player_games
        for _, row in match_data.iterrows():
            player_entry = {
                "player": row["player"],
                "team": row["team"],
                "is_home": bool(row.get("is_home", False)),
            }

            # Actuals
            for col in stat_cols:
                if col in row.index:
                    player_entry[f"actual_{col.lower()}"] = int(row[col])

            # Predictions
            if not predictions.empty:
                pred_row = predictions[
                    (predictions["player"] == row["player"]) &
                    (predictions["team"] == row["team"])
                ]
                if not pred_row.empty:
                    _enrich_player_from_pred(player_entry, pred_row.iloc[0])

            players.append(player_entry)
    else:
        # Prediction-only mode — build from predictions
        # Determine is_home from game_predictions or team order
        home_set = set()
        if home_team:
            home_set.add(home_team)
        elif store is not None:
            try:
                gp = store.load_game_predictions(year=year, round_num=round_num)
                if gp is not None and not gp.empty:
                    # Match by team names, not match_id
                    pred_teams = set(str(t) for t in predictions["team"].unique())
                    for _, gp_row in gp.iterrows():
                        h = str(gp_row.get("home_team", ""))
                        a = str(gp_row.get("away_team", ""))
                        if h in pred_teams or a in pred_teams:
                            home_set.add(h)
                            break
            except Exception:
                logger.debug("Failed to load data", exc_info=True)

        for _, pr in predictions.iterrows():
            player_entry = {
                "player": pr["player"],
                "team": str(pr.get("team", "")),
                "is_home": str(pr.get("team", "")) in home_set if home_set else True,
            }
            _enrich_player_from_pred(player_entry, pr)
            players.append(player_entry)

        # Fix home_team / away_team from game predictions
        if home_set:
            home_team = list(home_set)[0]
            away_teams = [t for t in set(p["team"] for p in players) if t not in home_set]
            away_team = away_teams[0] if away_teams else away_team
            for p in players:
                p["is_home"] = p["team"] == home_team

    # Sort: home team first, then by predicted/actual goals desc
    players.sort(key=lambda p: (not p.get("is_home", False), -(p.get("actual_gl", 0) or p.get("predicted_gl", 0) or 0)))

    # Enrich with advanced stats (venue/opponent/form/season/career)
    _enrich_players_advanced(cache, players, venue, year, round_num)

    # Game prediction — match by team names, not match_id
    game_pred = {}
    if year is not None and store is not None and home_team and away_team:
        try:
            gp = store.load_game_predictions(year=year, round_num=round_num)
            if gp is not None and not gp.empty:
                gp_match = gp[
                    (gp["home_team"] == home_team) & (gp["away_team"] == away_team)
                ]
                if not gp_match.empty:
                    gpr = gp_match.iloc[0]
                    if "home_win_prob" in gpr.index and pd.notna(gpr["home_win_prob"]):
                        game_pred["home_win_prob"] = round(float(gpr["home_win_prob"]), 3)
                    if "predicted_margin" in gpr.index and pd.notna(gpr["predicted_margin"]):
                        game_pred["predicted_margin"] = round(float(gpr["predicted_margin"]), 1)
                    if "predicted_winner" in gpr.index and pd.notna(gpr["predicted_winner"]):
                        game_pred["predicted_winner"] = str(gpr["predicted_winner"])
        except Exception:
            logger.debug("Failed to load data", exc_info=True)

    if not game_pred and not predictions.empty and home_team and away_team:
        game_pred = _fallback_game_prediction_from_player_predictions(
            predictions,
            home_team,
            away_team,
        )

    if (
        game_pred
        and home_score is not None
        and away_score is not None
        and "predicted_winner" in game_pred
    ):
        actual_winner = home_team if home_score > away_score else away_team if away_score > home_score else "Draw"
        game_pred["correct"] = game_pred["predicted_winner"] == actual_winner

    # Weather conditions for this match
    weather_data = {}
    weather = cache.weather
    if not weather.empty:
        wx = weather[weather["match_id"] == match_id]
        if not wx.empty:
            wxr = wx.iloc[0]
            for col in ["temperature_avg", "precipitation_total", "wind_speed_avg",
                         "humidity_avg", "weather_difficulty_score"]:
                if col in wxr.index and pd.notna(wxr[col]):
                    weather_data[col] = round(float(wxr[col]), 2)
            for col in ["is_wet", "is_roofed"]:
                if col in wxr.index and pd.notna(wxr[col]):
                    weather_data[col] = bool(wxr[col])

    # Fall back to forecast data for upcoming matches
    if not weather_data and year is not None and home_team and away_team:
        import config as _cfg
        forecast_path = _cfg.DATA_DIR / "forecasts" / f"weather_forecast_{year}.parquet"
        if forecast_path.exists():
            try:
                fc_df = pd.read_parquet(forecast_path)
                fc_match = fc_df[
                    (fc_df["home_team"] == home_team) & (fc_df["away_team"] == away_team)
                ]
                if fc_match.empty:
                    fc_match = fc_df[
                        (fc_df["home_team"] == away_team) & (fc_df["away_team"] == home_team)
                    ]
                if not fc_match.empty:
                    frow = fc_match.iloc[0]
                    for col in ["temperature_avg", "precipitation_total", "wind_speed_avg",
                                 "humidity_avg", "weather_difficulty_score"]:
                        if col in frow.index and pd.notna(frow[col]):
                            weather_data[col] = round(float(frow[col]), 2)
                    for col in ["is_wet", "is_roofed"]:
                        if col in frow.index and pd.notna(frow[col]):
                            weather_data[col] = bool(frow[col])
            except Exception:
                logger.debug("Failed to load data", exc_info=True)

    # Attendance from matches.parquet
    attendance = None
    is_roofed = weather_data.get("is_roofed")
    if not match_info.empty:
        mi = match_info.iloc[0]
        if "attendance" in mi.index and pd.notna(mi["attendance"]):
            attendance = int(mi["attendance"])

    # --- Weather impact per player ---
    # Compare each player's historic stats in similar conditions vs overall
    weather_impact = []
    if weather_data:
        is_wet = weather_data.get("is_wet", False)
        temp = weather_data.get("temperature_avg")
        wind = weather_data.get("wind_speed_avg")

        # Categorize match conditions
        if temp is not None:
            temp_cat = "hot" if temp >= 28 else "cold" if temp <= 14 else "mild"
        else:
            temp_cat = "mild"
        windy = wind is not None and wind >= 20

        # Pre-filter player_games to relevant players before merging with weather
        match_players = set((p["player"], p["team"]) for p in players)
        relevant_pg = pg[pg.apply(lambda r: (r["player"], r["team"]) in match_players, axis=1)]
        pw = relevant_pg.merge(
            weather[["match_id", "is_wet", "temperature_avg", "wind_speed_avg"]],
            on="match_id", how="inner",
        )
        for player_name, team_name in match_players:
            pp = pw[(pw["player"] == player_name) & (pw["team"] == team_name)]
            if len(pp) < 5:
                continue

            overall_gl = float(pp["GL"].mean())
            overall_di = float(pp["DI"].mean())
            overall_mk = float(pp["MK"].mean())
            total_games = len(pp)

            # Filter to similar conditions
            similar = pp.copy()
            conditions = []
            if is_wet:
                similar = similar[similar["is_wet"] == 1]
                conditions.append("wet")
            else:
                similar = similar[similar["is_wet"] == 0]
                conditions.append("dry")

            if temp_cat == "hot":
                similar = similar[similar["temperature_avg"] >= 28]
                conditions.append("hot")
            elif temp_cat == "cold":
                similar = similar[similar["temperature_avg"] <= 14]
                conditions.append("cold")

            if len(similar) < 3:
                # Fall back to just wet/dry
                similar = pp[pp["is_wet"] == (1 if is_wet else 0)]

            if len(similar) < 2:
                continue

            cond_gl = float(similar["GL"].mean())
            cond_di = float(similar["DI"].mean())
            cond_mk = float(similar["MK"].mean())
            cond_games = len(similar)

            weather_impact.append({
                "player": player_name,
                "team": team_name,
                "total_games": total_games,
                "condition_games": cond_games,
                "conditions": conditions,
                "overall_gl": round(overall_gl, 2),
                "overall_di": round(overall_di, 1),
                "overall_mk": round(overall_mk, 1),
                "condition_gl": round(cond_gl, 2),
                "condition_di": round(cond_di, 1),
                "condition_mk": round(cond_mk, 1),
                "gl_diff": round(cond_gl - overall_gl, 2),
                "di_diff": round(cond_di - overall_di, 1),
                "mk_diff": round(cond_mk - overall_mk, 1),
            })

        weather_impact.sort(key=lambda x: -x["di_diff"])

    # Summarize: how many players benefit/suffer
    weather_summary = None
    if weather_data and weather_impact:
        favored = sum(1 for p in weather_impact if p["di_diff"] > 0)
        total_wp = len(weather_impact)
        weather_summary = {
            "conditions": weather_impact[0]["conditions"] if weather_impact else [],
            "players_assessed": total_wp,
            "players_favored": favored,
            "players_hindered": total_wp - favored,
            "avg_di_diff": round(np.mean([p["di_diff"] for p in weather_impact]), 1),
            "avg_gl_diff": round(np.mean([p["gl_diff"] for p in weather_impact]), 2),
        }

    # --- Umpires ---
    umpire_info = []
    umpires = cache.umpires
    if not umpires.empty:
        # Find umpires for this match
        match_umps = umpires[umpires["match_id"] == match_id]
        if match_umps.empty and round_num is not None and year is not None:
            # Try matching by round/year
            match_umps = umpires[(umpires["year"] == year)]
            # Can't match without match_id for upcoming games
            match_umps = pd.DataFrame()

        for _, u in match_umps.iterrows():
            name = u.get("umpire_name", "")
            career = int(u["umpire_career_games"]) if pd.notna(u.get("umpire_career_games")) else None
            umpire_info.append({"name": name, "career_games": career})

    # --- Coaches ---
    coach_info = {}
    coaches = cache.coaches
    if not coaches.empty and home_team and away_team:
        for side, team in [("home", home_team), ("away", away_team)]:
            # Find most recent coach for this team
            team_coaches = coaches[coaches["team"] == team].sort_values("year")
            if match_id > 0:
                exact = coaches[(coaches["match_id"] == match_id) & (coaches["team"] == team)]
                if not exact.empty:
                    team_coaches = exact
                else:
                    team_coaches = team_coaches.tail(1)
            else:
                team_coaches = team_coaches.tail(1)
            if not team_coaches.empty:
                c = team_coaches.iloc[-1]
                coach_entry = {"name": c.get("coach", "")}
                if pd.notna(c.get("coach_win_pct")):
                    coach_entry["win_pct"] = round(float(c["coach_win_pct"]), 1)
                if pd.notna(c.get("coach_career_games")):
                    coach_entry["career_games"] = int(c["coach_career_games"])
                if pd.notna(c.get("coach_wins")):
                    coach_entry["wins"] = int(c["coach_wins"])
                if pd.notna(c.get("coach_losses")):
                    coach_entry["losses"] = int(c["coach_losses"])
                coach_info[side] = coach_entry

    # --- Odds comparison ---
    odds_info = {}
    odds = cache.odds
    if not odds.empty:
        match_odds = odds[odds.index == match_id] if odds.index.name == "match_id" else odds[odds.get("match_id", pd.Series()) == match_id] if "match_id" in odds.columns else pd.DataFrame()
        # Try index-based lookup
        if match_odds.empty and match_id in odds.index:
            match_odds = odds.loc[[match_id]]
        if not match_odds.empty:
            o = match_odds.iloc[0]
            for col in ["market_home_implied_prob", "market_away_implied_prob", "market_handicap",
                         "market_total_score", "market_confidence", "betfair_home_implied_prob"]:
                if col in o.index and pd.notna(o[col]):
                    odds_info[col] = round(float(o[col]), 3)

    # --- Advanced stats from FootyWire ---
    adv_stats = {}
    fw = cache.footywire
    if not fw.empty and home_team and away_team:
        try:
            # Merge with player_games to get team info
            if "match_id" in fw.columns and not pg.empty:
                fw_with_team = fw.merge(pg[["match_id", "player", "team", "year"]].drop_duplicates(), on=["match_id", "player"], how="inner")
                fw_cols = ["ED", "DE_pct", "CCL", "SCL", "TO", "MG", "SI", "ITC", "T5", "TOG_pct"]
                for side, team in [("home", home_team), ("away", away_team)]:
                    team_fw = fw_with_team[(fw_with_team["team"] == team) & (fw_with_team["year"] >= (year or 2025) - 1)]
                    if not team_fw.empty:
                        team_avgs = {}
                        for col in fw_cols:
                            if col in team_fw.columns:
                                vals = team_fw[col].dropna()
                                if not vals.empty:
                                    team_avgs[col.lower()] = round(float(vals.mean()), 1)
                        if team_avgs:
                            adv_stats[side] = team_avgs
        except Exception:
            logger.debug("Unexpected error", exc_info=True)

    # --- Attendance stats ---
    attendance_stats = {}
    import config as _cfg_adv
    venue_canonical = _cfg_adv.VENUE_NAME_MAP.get(venue, venue) if venue else ""
    tm = cache.team_matches
    if not tm.empty and venue_canonical:
        venue_aliases_att = {v for v, c in _cfg_adv.VENUE_NAME_MAP.items() if c == venue_canonical}
        venue_aliases_att.add(venue_canonical)
        if venue:
            venue_aliases_att.add(venue)
        venue_att = tm[tm["venue"].isin(venue_aliases_att) & (tm["year"] >= (year or 2025) - 3)]
        if not venue_att.empty and "attendance" in venue_att.columns:
            atts = venue_att["attendance"].dropna()
            if not atts.empty:
                # Deduplicate by match_id (2 rows per match)
                att_by_match = venue_att.drop_duplicates(subset=["match_id"])["attendance"].dropna()
                if not att_by_match.empty:
                    attendance_stats["avg_attendance"] = int(att_by_match.mean())
                    attendance_stats["median_attendance"] = int(att_by_match.median())
                    attendance_stats["max_attendance"] = int(att_by_match.max())
                    attendance_stats["min_attendance"] = int(att_by_match.min())
                    last5_att = att_by_match.tail(5)
                    attendance_stats["last_5_avg"] = int(last5_att.mean())
                    attendance_stats["total_games"] = len(att_by_match)

    return {
        "match_id": match_id,
        "year": year,
        "round_number": round_num,
        "date": date,
        "venue": venue,
        "is_roofed": is_roofed,
        "attendance": attendance,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "weather": weather_data if weather_data else None,
        "weather_summary": weather_summary,
        "weather_impact": weather_impact,
        "game_prediction": game_pred,
        "players": players,
        "match_context": _compute_match_context(
            cache, home_team, away_team, venue, date_full, year
        ),
        "umpires": umpire_info,
        "coaches": coach_info,
        "odds": odds_info,
        "advanced_stats": adv_stats,
        "attendance_stats": attendance_stats,
    }


def get_upcoming_matches(year: int) -> dict:
    """Get the next unplayed round with predictions."""
    import config

    cache = DataCache.get()
    matches = cache.matches
    round_progress = _build_round_progress(year, matches)
    next_round = _next_incomplete_round(round_progress)
    if next_round is None:
        return {"year": year, "round_number": None, "matches": [], "predictions": []}

    next_fixture = round_progress[next_round]["fixture_path"]

    score_lookup = {}
    season_matches = matches[matches["year"] == year] if not matches.empty else pd.DataFrame()
    if not season_matches.empty:
        for _, mrow in season_matches.iterrows():
            round_num = int(mrow["round_number"]) if pd.notna(mrow.get("round_number")) else None
            if round_num is None:
                continue
            score_lookup[(str(mrow.get("home_team", "")), str(mrow.get("away_team", "")), round_num)] = (
                mrow.get("home_score"),
                mrow.get("away_score"),
            )

    # Parse fixture
    fixture_df = pd.read_csv(next_fixture)
    home_rows = fixture_df[fixture_df["is_home"] == 1]
    home_rows = home_rows.loc[
        home_rows.apply(
            lambda row: not (
                (row["team"], row["opponent"], next_round) in score_lookup
                and pd.notna(score_lookup[(row["team"], row["opponent"], next_round)][0])
                and pd.notna(score_lookup[(row["team"], row["opponent"], next_round)][1])
            ),
            axis=1,
        )
    ]

    matches = []
    for _, row in home_rows.iterrows():
        matches.append({
            "home_team": row["team"],
            "away_team": row["opponent"],
            "venue": row.get("venue", ""),
            "date": row.get("date", ""),
        })

    # Load predictions for this round
    predictions = []
    store = _get_store(cache)
    if store is not None:
        try:
            preds = store.load_predictions(year=year, round_num=next_round)
            if preds is not None and not preds.empty:
                relevant_teams = {m["home_team"] for m in matches} | {m["away_team"] for m in matches}
                if relevant_teams and "team" in preds.columns:
                    preds = preds[preds["team"].isin(relevant_teams)]
                for _, p in preds.iterrows():
                    entry = {
                        "player": p["player"],
                        "team": p.get("team", ""),
                        "opponent": p.get("opponent", ""),
                    }
                    for col in ["predicted_goals", "predicted_disposals", "predicted_marks",
                                "p_scorer", "p_2plus_goals", "p_3plus_goals",
                                "p_20plus_disp", "p_25plus_disp", "player_role"]:
                        if col in p.index and pd.notna(p[col]):
                            entry[col] = round(float(p[col]), 3) if isinstance(p[col], float) else p[col]
                    predictions.append(entry)
        except Exception:
            logger.debug("Failed to load data", exc_info=True)

    # Also try loading from prediction CSVs (from cmd_predict)
    if not predictions:
        pred_csv = config.DATA_DIR / "predictions" / str(year) / f"round_{next_round}_predictions.csv"
        if pred_csv.exists():
            pred_df = pd.read_csv(pred_csv)
            for _, p in pred_df.iterrows():
                entry = {
                    "player": p.get("player", ""),
                    "team": p.get("team", ""),
                    "opponent": p.get("opponent", ""),
                }
                for col in ["predicted_goals", "predicted_disposals", "predicted_marks",
                            "p_scorer", "p_2plus_goals", "p_3plus_goals",
                            "p_20plus_disp", "p_25plus_disp", "player_role"]:
                    if col in p.index and pd.notna(p[col]):
                        entry[col] = round(float(p[col]), 3) if isinstance(p[col], float) else p[col]
                predictions.append(entry)

    return {
        "year": year,
        "round_number": next_round,
        "matches": matches,
        "predictions": predictions,
    }


def get_season_schedule(year: int) -> dict:
    """Full season schedule with round status, scores, and weather forecasts.

    Returns structure with all rounds, their status (completed/upcoming/future),
    match details including scores for played matches and weather forecasts for
    upcoming ones.
    """
    import config

    cache = DataCache.get()
    matches = cache.matches

    fixture_files = _get_fixture_round_files(year)
    fixture_paths = {int(f.stem.split("_")[1]): f for f in fixture_files}
    round_progress = _build_round_progress(year, matches)
    all_round_nums = sorted(round_progress)
    next_unstarted_round = _next_unstarted_round(round_progress)

    # Load weather forecasts if available
    forecast_dir = config.DATA_DIR / "forecasts"
    forecast_path = forecast_dir / f"weather_forecast_{year}.parquet"
    forecast_df = pd.DataFrame()
    if forecast_path.exists():
        try:
            forecast_df = pd.read_parquet(forecast_path)
        except Exception:
            logger.debug("Failed to load data", exc_info=True)

    # Build forecast lookup: (home_team, away_team, date) -> forecast row
    forecast_lookup = {}
    if not forecast_df.empty:
        for _, frow in forecast_df.iterrows():
            key = (str(frow.get("home_team", "")),
                   str(frow.get("away_team", "")),
                   str(frow.get("date", "")))
            forecast_lookup[key] = frow

    # Get match scores from matches.parquet
    season_matches = matches[matches["year"] == year] if not matches.empty else pd.DataFrame()
    # Build score lookup: (home_team, away_team, round_number) -> (home_score, away_score)
    # Also build match_id lookup for completed matches (frontend needs real match_id for links)
    score_lookup = {}
    match_id_lookup = {}
    if not season_matches.empty:
        for _, mrow in season_matches.iterrows():
            home = mrow.get("home_team", "")
            away = mrow.get("away_team", "")
            rnd = int(mrow["round_number"]) if pd.notna(mrow.get("round_number")) else None
            if rnd is not None:
                h_score = int(mrow["home_score"]) if pd.notna(mrow.get("home_score")) else None
                a_score = int(mrow["away_score"]) if pd.notna(mrow.get("away_score")) else None
                score_lookup[(home, away, rnd)] = (h_score, a_score)
                match_id_lookup[(home, away, rnd)] = int(mrow["match_id"])

    # Load prediction data from sequential store (per-round to avoid match_id collisions)
    store = _get_store(cache)
    pred_last_updated: dict[int, str] = {}  # round_number -> ISO timestamp
    # Per-round lookups keyed by (home_team, away_team) — no match_id dependency
    gp_by_round: dict[int, dict] = {}      # round -> {(home, away) -> game_pred_entry}
    pt_by_round: dict[int, dict] = {}      # round -> {(home, away) -> {team -> pred_totals}}
    if store is not None:
        # Get last-updated timestamps from prediction file mtimes
        try:
            from datetime import datetime, timezone
            rid = store._resolve_read_run_id("predictions", year, run_id=None, latest=True)
            if rid is not None:
                run_dir = store._run_dir("predictions", year, rid)
                if run_dir.exists():
                    for pf in run_dir.glob("R*.parquet"):
                        rnum = int(pf.stem[1:])
                        mtime = datetime.fromtimestamp(pf.stat().st_mtime, tz=timezone.utc)
                        pred_last_updated[rnum] = mtime.isoformat()
        except Exception:
            logger.debug("Failed to load data", exc_info=True)

        # Load per-round to avoid match_id collisions across rounds
    for rnum in all_round_nums:
            try:
                gp = store.load_game_predictions(year=year, round_num=rnum)
                if gp is not None and not gp.empty:
                    gp_map = {}
                    for _, gp_row in gp.iterrows():
                        h = str(gp_row.get("home_team", ""))
                        a = str(gp_row.get("away_team", ""))
                        entry = {}
                        if "home_win_prob" in gp_row.index and pd.notna(gp_row["home_win_prob"]):
                            entry["home_win_prob"] = round(float(gp_row["home_win_prob"]), 3)
                            entry["predicted_winner"] = gp_row.get("predicted_winner", "")
                        if "predicted_margin" in gp_row.index and pd.notna(gp_row["predicted_margin"]):
                            entry["predicted_margin"] = round(float(gp_row["predicted_margin"]), 1)
                        if entry and h and a:
                            gp_map[(h, a)] = entry
                    gp_by_round[rnum] = gp_map
            except Exception:
                logger.debug("Failed to load data", exc_info=True)

            try:
                pp = store.load_predictions(year=year, round_num=rnum)
                if pp is not None and not pp.empty:
                    pt_map = {}
                    for _mid, grp in pp.groupby("match_id", observed=True):
                        team_totals = {}
                        for team, tgrp in grp.groupby("team", observed=True):
                            team_totals[str(team)] = {
                                "pred_gl": round(float(tgrp["predicted_goals"].sum()), 1) if "predicted_goals" in tgrp.columns else None,
                                "pred_di": round(float(tgrp["predicted_disposals"].sum()), 0) if "predicted_disposals" in tgrp.columns else None,
                                "pred_mk": round(float(tgrp["predicted_marks"].sum()), 0) if "predicted_marks" in tgrp.columns else None,
                            }
                        # Key by both team orderings
                        teams = [str(t) for t in grp["team"].unique()]
                        if len(teams) == 2:
                            pt_map[(teams[0], teams[1])] = team_totals
                            pt_map[(teams[1], teams[0])] = team_totals
                    pt_by_round[rnum] = pt_map

                    if rnum not in gp_by_round:
                        gp_by_round[rnum] = {}
                    for team_key, team_totals in pt_map.items():
                        if team_key in gp_by_round[rnum]:
                            continue
                        home_team_key, away_team_key = team_key
                        if home_team_key == away_team_key:
                            continue
                        fallback_pred = _fallback_game_prediction_from_player_predictions(
                            pp,
                            home_team_key,
                            away_team_key,
                        )
                        if fallback_pred:
                            gp_by_round[rnum][team_key] = fallback_pred
            except Exception:
                logger.debug("Failed to load data", exc_info=True)

    # Build rounds
    rounds = []
    for round_num in all_round_nums:
        fpath = fixture_paths.get(round_num)

        status = _round_status(round_num, round_progress, next_unstarted_round)

        base_matches = _build_schedule_round_matches(round_num, fpath, season_matches)

        match_list = []
        for row in base_matches:
            home_team = row["home_team"]
            away_team = row["away_team"]
            venue = row.get("venue", "")
            date = row.get("date", "")

            match_entry: dict = {
                "home_team": home_team,
                "away_team": away_team,
                "venue": venue,
                "date": date,
                "home_score": None,
                "away_score": None,
                "forecast": None,
            }

            # Fill in scores and real match_id for completed matches
            score_key = (home_team, away_team, round_num)
            if score_key in score_lookup:
                h_s, a_s = score_lookup[score_key]
                match_entry["home_score"] = h_s
                match_entry["away_score"] = a_s
            real_mid = match_id_lookup.get(score_key)
            if real_mid is not None:
                match_entry["match_id"] = real_mid

            # Add prediction data if available (keyed by team names)
            team_key = (home_team, away_team)
            r_gp = gp_by_round.get(round_num, {})
            if team_key in r_gp:
                match_entry["prediction"] = r_gp[team_key]
            r_pt = pt_by_round.get(round_num, {})
            if team_key in r_pt:
                pt = r_pt[team_key]
                match_entry["home_pred"] = pt.get(home_team)
                match_entry["away_pred"] = pt.get(away_team)

            # Add weather forecast for matches without scores yet.
            if match_entry["home_score"] is None and match_entry["away_score"] is None:
                forecast_key = (home_team, away_team, date)
                if forecast_key in forecast_lookup:
                    frow = forecast_lookup[forecast_key]
                    forecast_entry = {}
                    for col in ["temperature_avg", "precipitation_total",
                                "wind_speed_avg", "humidity_avg"]:
                        if col in frow.index and pd.notna(frow[col]):
                            forecast_entry[col] = round(float(frow[col]), 1)
                    for col in ["is_wet", "is_roofed"]:
                        if col in frow.index and pd.notna(frow[col]):
                            forecast_entry[col] = bool(frow[col])
                    if "weather_difficulty_score" in frow.index and pd.notna(frow["weather_difficulty_score"]):
                        forecast_entry["weather_difficulty_score"] = round(
                            float(frow["weather_difficulty_score"]), 1
                        )
                    if forecast_entry:
                        match_entry["forecast"] = forecast_entry

            match_list.append(match_entry)

        round_entry: dict = {
            "round_number": round_num,
            "status": status,
            "matches": match_list,
        }
        if round_num in pred_last_updated:
            round_entry["prediction_updated"] = pred_last_updated[round_num]

        rounds.append(round_entry)

    return {
        "year": year,
        "rounds": rounds,
    }


def get_round_accuracy(year: int) -> list[dict]:
    """Per-round model accuracy metrics for charts."""
    cache = DataCache.get()
    store = _get_store(cache)

    if store is None:
        return []

    round_progress = _build_round_progress(year, cache.matches)
    completed_rounds = sorted(
        round_num
        for round_num, info in round_progress.items()
        if info["played_matches"] > 0
    )

    results = []
    for rnum in completed_rounds:
        rnum = int(rnum)
        try:
            preds = store.load_predictions(year=year, round_num=rnum)
            outcomes = store.load_outcomes(year=year, round_num=rnum)
        except Exception:
            continue

        if preds.empty or outcomes.empty:
            continue

        merged = _merge_round_predictions_outcomes(preds, outcomes)
        if merged.empty:
            continue

        entry = {"round_number": rnum, "n_players": len(merged)}

        if "predicted_goals" in merged.columns and "actual_goals" in merged.columns:
            entry["goals_mae"] = round(float((merged["predicted_goals"] - merged["actual_goals"]).abs().mean()), 3)
            pred_scorer = merged["predicted_goals"] >= 0.5
            actual_scorer = merged["actual_goals"] >= 1
            entry["scorer_accuracy"] = round(float((pred_scorer == actual_scorer).mean() * 100), 1)

        if "predicted_disposals" in merged.columns and "actual_disposals" in merged.columns:
            entry["disposals_mae"] = round(float((merged["predicted_disposals"] - merged["actual_disposals"]).abs().mean()), 3)

        if "predicted_marks" in merged.columns and "actual_marks" in merged.columns:
            entry["marks_mae"] = round(float((merged["predicted_marks"] - merged["actual_marks"]).abs().mean()), 3)

        results.append(entry)

    return sorted(results, key=lambda r: r["round_number"])


# ---------------------------------------------------------------------------
# Venue Stats
# ---------------------------------------------------------------------------

from api.services.venue_service import _VENUE_CITY, _VENUE_ROOFED  # noqa: F401


def get_all_venues() -> list[dict]:
    """Delegate to venue_service."""
    from api.services.venue_service import get_all_venues as _get_all_venues
    return _get_all_venues()


def get_venue_detail(venue_name: str) -> Optional[dict]:
    """Delegate to venue_service."""
    from api.services.venue_service import get_venue_detail as _get_venue_detail
    return _get_venue_detail(venue_name)


# ---------------------------------------------------------------------------
# Predictions History
# ---------------------------------------------------------------------------

def get_predictions_history(year: int) -> dict:
    """All predictions vs actuals for a given year with summary stats."""
    cache = DataCache.get()
    store = _get_store(cache)

    if store is None:
        return {"entries": [], "summary": {}}

    round_progress = _build_round_progress(year, cache.matches)
    completed_rounds = sorted(
        round_num
        for round_num, info in round_progress.items()
        if info["played_matches"] > 0
    )

    records = []
    for rnum in completed_rounds:
        rnum = int(rnum)
        try:
            preds = store.load_predictions(year=year, round_num=rnum)
            outcomes = store.load_outcomes(year=year, round_num=rnum)
        except Exception:
            continue

        if preds.empty or outcomes.empty:
            continue

        merged = _merge_round_predictions_outcomes(preds, outcomes)

        for _, row in merged.iterrows():
            entry = {
                "player": str(row["player"]),
                "team": str(row.get("team", "")),
                "opponent": str(row.get("opponent", "")) if pd.notna(row.get("opponent")) else "",
                "round": rnum,
            }
            if "match_id" in row.index and pd.notna(row.get("match_id")):
                entry["match_id"] = int(row["match_id"])
            if "venue" in row.index and pd.notna(row.get("venue")):
                entry["venue"] = str(row["venue"])

            # Predictions
            for col in ["predicted_goals", "predicted_disposals", "predicted_marks", "p_scorer"]:
                if col in row.index and pd.notna(row[col]):
                    entry[col] = round(float(row[col]), 3)

            # Actuals
            for col in ["actual_goals", "actual_disposals", "actual_marks"]:
                if col in row.index and pd.notna(row[col]):
                    entry[col] = int(row[col])

            # Derived: actually scored
            if "actual_goals" in entry:
                entry["actually_scored"] = entry["actual_goals"] >= 1

            records.append(entry)

    # Summary stats
    summary = {}
    merged_df = pd.DataFrame(records)
    if not merged_df.empty:
        for metric in ["goals", "disposals", "marks"]:
            pred_col = f"predicted_{metric}"
            actual_col = f"actual_{metric}"
            if pred_col in merged_df.columns and actual_col in merged_df.columns:
                valid = merged_df.dropna(subset=[pred_col, actual_col])
                if not valid.empty:
                    summary[f"{metric}_mae"] = round(
                        float((valid[pred_col] - valid[actual_col]).abs().mean()), 3
                    )

        # Scorer accuracy
        if "p_scorer" in merged_df.columns and "actually_scored" in merged_df.columns:
            valid = merged_df.dropna(subset=["p_scorer", "actually_scored"])
            if not valid.empty:
                pred_scorer = valid["p_scorer"] >= 0.5
                actual_scorer = valid["actually_scored"]
                summary["scorer_accuracy"] = round(
                    float((pred_scorer == actual_scorer).mean() * 100), 1
                )

        summary["total_predictions"] = len(records)

    return {"entries": records, "summary": summary}


# ---------------------------------------------------------------------------
# Accuracy Breakdown
# ---------------------------------------------------------------------------

def get_accuracy_breakdown(year: int) -> dict:
    """Compute MAE breakdowns by team, venue, home/away, and season stage."""
    history = get_predictions_history(year)
    entries = history.get("entries", [])
    if not entries:
        return {"by_team": [], "by_venue": [], "by_home_away": {}, "by_stage": []}

    df = pd.DataFrame(entries)
    cache = DataCache.get()

    # Need is_home and match_id — merge with matches
    matches = cache.matches
    if not matches.empty and "venue" in df.columns:
        season_matches = matches[matches["year"] == year]
        # Build round→match mapping for is_home lookup
        home_away_map = {}  # (team, round) -> is_home
        for _, mrow in season_matches.iterrows():
            rn = int(mrow["round_number"]) if pd.notna(mrow.get("round_number")) else None
            if rn is not None:
                home_away_map[(str(mrow.get("home_team", "")), rn)] = True
                home_away_map[(str(mrow.get("away_team", "")), rn)] = False
        df["is_home"] = df.apply(
            lambda r: home_away_map.get((r.get("team", ""), r.get("round", 0))), axis=1
        )

    def _compute_mae(subset):
        result = {"n": len(subset)}
        for metric in ["goals", "disposals", "marks"]:
            pred_col = f"predicted_{metric}"
            actual_col = f"actual_{metric}"
            if pred_col in subset.columns and actual_col in subset.columns:
                valid = subset.dropna(subset=[pred_col, actual_col])
                if not valid.empty:
                    result[f"{metric}_mae"] = round(
                        float((valid[pred_col] - valid[actual_col]).abs().mean()), 3
                    )
        return result

    # Overall MAE for comparison
    overall = _compute_mae(df)

    # By team
    by_team = []
    if "team" in df.columns:
        for team, grp in df.groupby("team", observed=True):
            entry = _compute_mae(grp)
            entry["team"] = str(team)
            by_team.append(entry)
        by_team.sort(key=lambda x: x.get("goals_mae", 999))

    # By venue
    by_venue = []
    if "venue" in df.columns:
        for venue, grp in df.groupby("venue", observed=True):
            if len(grp) < 20:
                continue
            entry = _compute_mae(grp)
            entry["venue"] = str(venue)
            by_venue.append(entry)
        by_venue.sort(key=lambda x: x.get("goals_mae", 999))

    # By home/away
    by_home_away = {}
    if "is_home" in df.columns:
        home_df = df[df["is_home"] == True]
        away_df = df[df["is_home"] == False]
        if not home_df.empty:
            by_home_away["home"] = _compute_mae(home_df)
        if not away_df.empty:
            by_home_away["away"] = _compute_mae(away_df)

    # By season stage
    by_stage = []
    if "round" in df.columns:
        stage_map = {
            "Early (R1-8)": df[(df["round"] >= 1) & (df["round"] <= 8)],
            "Mid (R9-16)": df[(df["round"] >= 9) & (df["round"] <= 16)],
            "Late (R17-24)": df[(df["round"] >= 17) & (df["round"] <= 24)],
            "Finals (R25+)": df[df["round"] >= 25],
        }
        for stage_label, grp in stage_map.items():
            if grp.empty:
                continue
            entry = _compute_mae(grp)
            entry["stage"] = stage_label
            by_stage.append(entry)

    return {
        "overall": overall,
        "by_team": by_team,
        "by_venue": by_venue,
        "by_home_away": by_home_away,
        "by_stage": by_stage,
    }
