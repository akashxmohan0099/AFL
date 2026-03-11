"""League-wide services: stat leaders, ladder."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from api.data_loader import DataCache

VALID_STATS = {"GL", "BH", "DI", "MK", "KI", "HB", "TK", "HO",
               "CP", "UP", "IF", "CL", "CG", "FF", "FA", "CM",
               "one_pct", "RB", "MI", "BO", "GA"}


def _r(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), d)


def get_ladder(year: int = 2025) -> dict:
    """AFL ladder / standings for a given season."""
    cache = DataCache.get()
    tm = cache.team_matches

    if tm.empty:
        return {"year": year, "ladder": []}

    season = tm[tm["year"] == year]
    if season.empty:
        return {"year": year, "ladder": []}

    # --- Aggregate stats per team ---
    grouped = season.groupby("team", observed=True)

    agg = grouped.agg(
        played=("result", "size"),
        wins=("result", lambda x: (x == "W").sum()),
        losses=("result", lambda x: (x == "L").sum()),
        draws=("result", lambda x: (x == "D").sum()),
        points_for=("score", "sum"),
        points_against=("opp_score", "sum"),
        avg_margin=("margin", "mean"),
    ).reset_index()

    agg["points"] = agg["wins"] * 4 + agg["draws"] * 2
    agg["percentage"] = np.where(
        agg["points_against"] > 0,
        (agg["points_for"] / agg["points_against"]) * 100,
        0.0,
    )

    # --- Form: last 5 results (most recent last) ---
    # Sort by date/round so tail(5) gives most recent
    sorted_season = season.sort_values(["team", "round_number", "date"])
    form_map: dict[str, List[str]] = {}
    for team, grp in sorted_season.groupby("team", observed=True):
        form_map[str(team)] = grp["result"].tail(5).tolist()

    # --- Sort and assign positions ---
    agg = agg.sort_values(["points", "percentage"], ascending=[False, False]).reset_index(drop=True)

    ladder: List[dict] = []
    for pos, (_, row) in enumerate(agg.iterrows(), start=1):
        team_name = str(row["team"])
        ladder.append({
            "position": pos,
            "team": team_name,
            "played": int(row["played"]),
            "wins": int(row["wins"]),
            "losses": int(row["losses"]),
            "draws": int(row["draws"]),
            "points": int(row["points"]),
            "points_for": int(row["points_for"]),
            "points_against": int(row["points_against"]),
            "percentage": _r(row["percentage"], 1),
            "form": form_map.get(team_name, []),
            "avg_margin": _r(row["avg_margin"], 1),
        })

    return {"year": year, "ladder": ladder}


def get_leaders(stat: str = "GL", year: int = 2025, limit: int = 50, min_games: int = 5) -> dict:
    """Top players by a given stat for a season."""
    cache = DataCache.get()
    pg = cache.player_games

    if pg.empty or stat not in VALID_STATS:
        return {"stat": stat, "year": year, "leaders": []}

    season = pg[pg["year"] == year]
    if season.empty:
        return {"stat": stat, "year": year, "leaders": []}

    if stat not in season.columns:
        return {"stat": stat, "year": year, "leaders": []}

    grouped = season.groupby(["player_id", "player", "team"], observed=True)
    agg = grouped.agg(
        games=(stat, "size"),
        total=(stat, "sum"),
        avg=(stat, "mean"),
        best=(stat, "max"),
    ).reset_index()

    qualified = agg[agg["games"] >= min_games]
    if qualified.empty:
        return {"stat": stat, "year": year, "leaders": []}

    # All stats in VALID_STATS are counting stats — sort by total
    sort_col = "total"
    top = qualified.nlargest(limit, sort_col)

    leaders = []
    for _, r in top.iterrows():
        leaders.append({
            "player_id": str(r["player_id"]),
            "name": str(r["player"]),
            "team": str(r["team"]),
            "games": int(r["games"]),
            "total": int(r["total"]),
            "avg": _r(r["avg"], 1),
            "best": int(r["best"]),
        })

    return {"stat": stat, "year": year, "leaders": leaders}


def get_team_profile(team: str, year: int = 2025) -> dict | None:
    """Full team profile for a season: record, form, top players, averages, splits."""
    cache = DataCache.get()
    tm = cache.team_matches
    pg = cache.player_games

    if tm.empty:
        return None

    team_tm = tm[tm["team"] == team]
    if team_tm.empty:
        return None

    season = team_tm[team_tm["year"] == year]
    if season.empty:
        return {"team": team, "year": year, "record": None, "recent_form": [],
                "top_goals": [], "top_disposals": [], "top_marks": [],
                "season_averages": None, "home_away": None}

    # ── 1. Season record (reuse ladder-style aggregation) ──
    wins = int((season["result"] == "W").sum())
    losses = int((season["result"] == "L").sum())
    draws = int(len(season) - wins - losses)
    points_for = int(season["score"].sum())
    points_against = int(season["opp_score"].sum())
    points = wins * 4 + draws * 2
    percentage = _r((points_for / points_against) * 100, 1) if points_against > 0 else 0.0

    # Derive ladder position from full ladder
    ladder_data = get_ladder(year=year)
    ladder_pos = None
    for entry in ladder_data.get("ladder", []):
        if entry["team"] == team:
            ladder_pos = entry["position"]
            break

    record = {
        "played": int(len(season)),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "points": points,
        "points_for": points_for,
        "points_against": points_against,
        "percentage": percentage,
        "ladder_position": ladder_pos,
    }

    # ── 2. Recent form (last 5 games) ──
    recent = season.sort_values("date").tail(5)
    recent_form: List[dict] = []
    for _, r in recent.iterrows():
        recent_form.append({
            "round_number": int(r["round_number"]) if pd.notna(r.get("round_number")) else None,
            "opponent": str(r.get("opponent", "")),
            "score": int(r["score"]) if pd.notna(r.get("score")) else None,
            "opp_score": int(r["opp_score"]) if pd.notna(r.get("opp_score")) else None,
            "result": str(r.get("result", "")),
            "margin": int(r["margin"]) if pd.notna(r.get("margin")) else None,
            "venue": str(r.get("venue", "")),
            "is_home": bool(r.get("is_home", False)),
        })

    # ── 3. Top players (top 5 by goals, disposals, marks) ──
    top_goals: List[dict] = []
    top_disposals: List[dict] = []
    top_marks: List[dict] = []

    if not pg.empty:
        team_pg = pg[(pg["team"] == team) & (pg["year"] == year)]
        if not team_pg.empty:
            player_agg = team_pg.groupby(["player_id", "player"], observed=True).agg(
                games=("GL", "size"),
                total_gl=("GL", "sum"),
                avg_gl=("GL", "mean"),
                total_di=("DI", "sum"),
                avg_di=("DI", "mean"),
                total_mk=("MK", "sum"),
                avg_mk=("MK", "mean"),
            ).reset_index()

            def _top5(df: pd.DataFrame, total_col: str, avg_col: str, label: str) -> List[dict]:
                top = df.nlargest(5, total_col)
                out = []
                for _, r in top.iterrows():
                    out.append({
                        "player_id": str(r["player_id"]),
                        "name": str(r["player"]),
                        "games": int(r["games"]),
                        "total": int(r[total_col]),
                        "avg": _r(r[avg_col], 1),
                    })
                return out

            top_goals = _top5(player_agg, "total_gl", "avg_gl", "goals")
            top_disposals = _top5(player_agg, "total_di", "avg_di", "disposals")
            top_marks = _top5(player_agg, "total_mk", "avg_mk", "marks")

    # ── 4. Season averages ──
    season_averages = {
        "avg_score": _r(season["score"].mean(), 1),
        "avg_conceded": _r(season["opp_score"].mean(), 1),
        "avg_margin": _r(season["margin"].mean(), 1),
        "avg_rest_days": _r(season["rest_days"].mean(), 1) if "rest_days" in season.columns else None,
    }

    # ── 5. Home/away split ──
    def _split_record(games: pd.DataFrame) -> dict:
        if games.empty:
            return {"played": 0, "wins": 0, "losses": 0, "draws": 0,
                    "avg_score": None, "avg_conceded": None, "avg_margin": None}
        w = int((games["result"] == "W").sum())
        l = int((games["result"] == "L").sum())
        d = int(len(games) - w - l)
        return {
            "played": int(len(games)),
            "wins": w,
            "losses": l,
            "draws": d,
            "avg_score": _r(games["score"].mean(), 1),
            "avg_conceded": _r(games["opp_score"].mean(), 1),
            "avg_margin": _r(games["margin"].mean(), 1),
        }

    home_games = season[season["is_home"] == True]
    away_games = season[season["is_home"] == False]
    home_away = {
        "home": _split_record(home_games),
        "away": _split_record(away_games),
    }

    return {
        "team": team,
        "year": year,
        "record": record,
        "recent_form": recent_form,
        "top_goals": top_goals,
        "top_disposals": top_disposals,
        "top_marks": top_marks,
        "season_averages": season_averages,
        "home_away": home_away,
    }
