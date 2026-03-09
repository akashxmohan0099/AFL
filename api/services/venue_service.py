"""Venue-level services: list, detail."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from api.data_loader import DataCache


_VENUE_CITY = {
    "M.C.G.": "Melbourne",
    "Docklands": "Melbourne",
    "Adelaide Oval": "Adelaide",
    "S.C.G.": "Sydney",
    "SCG": "Sydney",
    "Gabba": "Brisbane",
    "Kardinia Park": "Geelong",
    "Perth Stadium": "Perth",
    "Giants Stadium": "Sydney",
    "Sydney Showground": "Sydney",
    "Carrara": "Gold Coast",
    "York Park": "Launceston",
    "Manuka Oval": "Canberra",
    "Subiaco": "Perth",
    "Bellerive Oval": "Hobart",
    "Stadium Australia": "Sydney",
    "Cazaly's Stadium": "Cairns",
    "Traeger Park": "Alice Springs",
    "Eureka Stadium": "Ballarat",
    "Marrara Oval": "Darwin",
    "Riverway Stadium": "Townsville",
    "Norwood Oval": "Adelaide",
    "Jiangwan Stadium": "Shanghai",
}

_VENUE_ROOFED = {"Docklands"}


def get_all_venues() -> list[dict]:
    """List all venues with aggregate stats, weather, home teams, city, years."""
    cache = DataCache.get()
    matches = cache.matches
    weather = cache.weather
    tm = cache.team_matches

    if matches.empty:
        return []

    # Merge matches with weather
    if not weather.empty:
        mw = matches.merge(
            weather[["match_id", "temperature_avg", "precipitation_total", "is_wet"]],
            on="match_id",
            how="left",
        )
    else:
        mw = matches.copy()

    # Pre-compute home team counts per venue from team_matches
    home_teams_by_venue: dict[str, dict[str, int]] = {}
    if not tm.empty:
        home_games = tm[tm["is_home"] == True]
        for (venue, team), cnt in home_games.groupby(
            ["venue", "team"], observed=True
        ).size().items():
            home_teams_by_venue.setdefault(str(venue), {})[str(team)] = int(cnt)

    venue_stats = []
    for venue, grp in mw.groupby("venue", observed=True):
        venue_str = str(venue)
        entry: dict = {
            "venue": venue_str,
            "total_games": len(grp),
        }

        # Year range
        if "year" in grp.columns:
            years = grp["year"].dropna()
            if not years.empty:
                entry["year_from"] = int(years.min())
                entry["year_to"] = int(years.max())

        # City
        entry["city"] = _VENUE_CITY.get(venue_str, None)

        # Roofed
        entry["is_roofed"] = venue_str in _VENUE_ROOFED

        # Home teams — top teams by home games at this venue
        ht = home_teams_by_venue.get(venue_str, {})
        if ht:
            sorted_ht = sorted(ht.items(), key=lambda x: -x[1])
            max_home = sorted_ht[0][1] if sorted_ht else 0
            threshold = max(max_home * 0.3, 5)
            entry["home_teams"] = [
                {"team": t, "home_games": g}
                for t, g in sorted_ht if g >= threshold
            ]

        # Score stats
        if "total_score" in grp.columns:
            entry["avg_total_score"] = round(float(grp["total_score"].mean()), 1)
        elif "home_score" in grp.columns and "away_score" in grp.columns:
            total = grp["home_score"].fillna(0) + grp["away_score"].fillna(0)
            entry["avg_total_score"] = round(float(total.mean()), 1)

        if "margin" in grp.columns:
            entry["avg_margin"] = round(float(grp["margin"].abs().mean()), 1)

        # Weather stats (if available)
        if "temperature_avg" in grp.columns:
            valid_temp = grp["temperature_avg"].dropna()
            entry["avg_temperature"] = round(float(valid_temp.mean()), 1) if not valid_temp.empty else None
        if "precipitation_total" in grp.columns:
            valid_rain = grp["precipitation_total"].dropna()
            entry["avg_precipitation"] = round(float(valid_rain.mean()), 2) if not valid_rain.empty else None
        if "is_wet" in grp.columns:
            valid_wet = grp["is_wet"].dropna()
            entry["pct_wet_games"] = round(float(valid_wet.mean() * 100), 1) if not valid_wet.empty else None

        venue_stats.append(entry)

    venue_stats.sort(key=lambda v: -v["total_games"])
    return venue_stats


def get_venue_detail(venue_name: str) -> Optional[dict]:
    """Detailed stats for a specific venue: matches, weather, top performers."""
    cache = DataCache.get()
    matches = cache.matches
    pg = cache.player_games
    weather = cache.weather

    # Filter matches at this venue (last 5 years)
    venue_matches = matches[matches["venue"] == venue_name].copy()
    if venue_matches.empty:
        return None

    min_year = int(venue_matches["year"].max()) - 4
    recent = venue_matches[venue_matches["year"] >= min_year].sort_values(
        ["year", "round_number"], ascending=[False, False]
    )

    # Aggregate venue stats
    scored = recent.dropna(subset=["home_score", "away_score"])
    avg_total_score = float((scored["home_score"] + scored["away_score"]).mean()) if not scored.empty else 0
    avg_margin = float(scored["margin"].abs().mean()) if (not scored.empty and "margin" in scored.columns) else 0

    # Build match list
    match_list = []
    for _, m in recent.iterrows():
        entry = {
            "match_id": int(m["match_id"]),
            "year": int(m["year"]),
            "round_number": int(m["round_number"]) if pd.notna(m.get("round_number")) else None,
            "date": str(m["date"])[:10] if pd.notna(m.get("date")) else None,
            "home_team": m.get("home_team", ""),
            "away_team": m.get("away_team", ""),
            "home_score": int(m["home_score"]) if pd.notna(m.get("home_score")) else None,
            "away_score": int(m["away_score"]) if pd.notna(m.get("away_score")) else None,
        }
        if "attendance" in m.index and pd.notna(m["attendance"]):
            entry["attendance"] = int(m["attendance"])
        match_list.append(entry)

    # Aggregate weather into summary
    weather_summary = None
    is_roofed = False
    if not weather.empty:
        venue_match_ids = set(recent["match_id"].tolist())
        wx = weather[weather["match_id"].isin(venue_match_ids)]
        if not wx.empty:
            avg_temp = float(wx["temperature_avg"].mean()) if "temperature_avg" in wx.columns else 0
            avg_wind = float(wx["wind_speed_avg"].mean()) if "wind_speed_avg" in wx.columns else 0
            avg_hum = float(wx["humidity_avg"].mean()) if "humidity_avg" in wx.columns else 0
            pct_wet = float(wx["is_wet"].mean()) if "is_wet" in wx.columns else 0
            if "is_roofed" in wx.columns:
                is_roofed = bool(wx["is_roofed"].mode().iloc[0]) if not wx["is_roofed"].dropna().empty else False
            weather_summary = {
                "avg_temperature": round(avg_temp, 1),
                "avg_wind_speed": round(avg_wind, 1),
                "pct_wet": round(pct_wet, 3),
                "avg_humidity": round(avg_hum, 1),
            }

    # Top performers at this venue (min 5 games)
    venue_pg = pg[pg["venue"] == venue_name]
    top_goal_scorers = []
    top_disposal_getters = []
    if not venue_pg.empty:
        player_agg = venue_pg.groupby(["player", "team"], observed=True).agg(
            games=("GL", "size"),
            total_gl=("GL", "sum"),
            avg_gl=("GL", "mean"),
            total_di=("DI", "sum"),
            avg_di=("DI", "mean"),
        ).reset_index()

        qualified = player_agg[player_agg["games"] >= 5]

        for _, r in qualified.nlargest(10, "avg_gl").iterrows():
            top_goal_scorers.append({
                "player": str(r["player"]),
                "team": str(r["team"]),
                "games": int(r["games"]),
                "total_goals": int(r["total_gl"]),
                "avg_goals": round(float(r["avg_gl"]), 2),
            })

        for _, r in qualified.nlargest(10, "avg_di").iterrows():
            top_disposal_getters.append({
                "player": str(r["player"]),
                "team": str(r["team"]),
                "games": int(r["games"]),
                "total_disposals": int(r["total_di"]),
                "avg_disposals": round(float(r["avg_di"]), 1),
            })

    return {
        "venue": venue_name,
        "total_games": len(venue_matches),
        "avg_total_score": round(avg_total_score, 1),
        "avg_margin": round(avg_margin, 1),
        "weather": weather_summary,
        "is_roofed": is_roofed,
        "top_goal_scorers": top_goal_scorers,
        "top_disposal_getters": top_disposal_getters,
        "recent_matches": match_list,
    }
