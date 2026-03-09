"""News service — reads cached news parquets, JSON files, and intel signals."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import config

NEWS_DIR = config.DATA_DIR / "news"
TEAM_LISTS_DIR = NEWS_DIR / "team_lists"
INJURIES_DIR = NEWS_DIR / "injuries"
INTEL_DIR = NEWS_DIR / "intel"


def _severity_label(severity: int) -> str:
    return ["Test", "1 week", "2-3 weeks", "4-8 weeks", "Season"][min(severity, 4)]


def get_injuries() -> dict:
    """Return current injury list grouped by team."""
    injuries_path = config.BASE_STORE_DIR / "injuries.parquet"

    if not injuries_path.exists():
        # Try latest JSON cache
        json_files = sorted(INJURIES_DIR.glob("injuries_*.json")) if INJURIES_DIR.exists() else []
        if not json_files:
            return {"teams": {}, "total": 0, "updated": None}

        latest = json_files[-1]
        with open(latest) as f:
            injuries = json.load(f)

        # Extract date from filename
        updated = latest.stem.replace("injuries_", "")
    else:
        import pandas as pd
        df = pd.read_parquet(injuries_path)
        injuries = df.to_dict("records")
        # Get updated date from latest cache file
        json_files = sorted(INJURIES_DIR.glob("injuries_*.json")) if INJURIES_DIR.exists() else []
        updated = json_files[-1].stem.replace("injuries_", "") if json_files else None

    # Group by team
    teams: dict[str, list] = {}
    for inj in injuries:
        team = inj.get("team", "Unknown")
        severity = inj.get("severity", 2)
        teams.setdefault(team, []).append({
            "player": inj.get("player", ""),
            "injury": inj.get("injury", ""),
            "estimated_return": inj.get("estimated_return", ""),
            "severity": severity,
            "severity_label": _severity_label(severity),
        })

    # Sort players within each team by severity (worst first)
    for team in teams:
        teams[team].sort(key=lambda x: -x["severity"])

    return {
        "teams": dict(sorted(teams.items())),
        "total": len(injuries),
        "updated": updated,
    }


def get_team_changes(year: int) -> dict:
    """Return team changes (ins/outs) for a season."""
    changes_path = config.BASE_STORE_DIR / "team_changes.parquet"

    if not changes_path.exists():
        return {"rounds": [], "year": year}

    import pandas as pd
    df = pd.read_parquet(changes_path)
    df = df[df["year"] == year]

    if df.empty:
        return {"rounds": [], "year": year}

    rounds = []
    for rnd in sorted(df["round_number"].unique()):
        rnd_df = df[df["round_number"] == rnd]
        teams = []
        for _, row in rnd_df.iterrows():
            ins = json.loads(row.get("ins_list", "[]")) if isinstance(row.get("ins_list"), str) else []
            outs = json.loads(row.get("outs_list", "[]")) if isinstance(row.get("outs_list"), str) else []
            debutants = json.loads(row.get("debutant_list", "[]")) if isinstance(row.get("debutant_list"), str) else []
            teams.append({
                "team": row["team"],
                "n_ins": int(row.get("n_ins", 0)),
                "n_outs": int(row.get("n_outs", 0)),
                "n_debutants": int(row.get("n_debutants", 0)),
                "stability": float(row.get("team_stability", 1.0)),
                "ins": ins,
                "outs": outs,
                "debutants": debutants,
            })
        rounds.append({
            "round_number": int(rnd),
            "teams": sorted(teams, key=lambda t: t["team"]),
        })

    return {"rounds": rounds, "year": year}


def get_match_news(home_team: str, away_team: str) -> dict:
    """Combined news for a specific matchup."""
    injuries = get_injuries()

    home_injuries = injuries["teams"].get(home_team, [])
    away_injuries = injuries["teams"].get(away_team, [])

    # Headline injuries (severity >= 2 and well-known — just use severity for now)
    home_headlines = [i for i in home_injuries if i["severity"] >= 2]
    away_headlines = [i for i in away_injuries if i["severity"] >= 2]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "injuries_updated": injuries.get("updated"),
        "home": {
            "injuries": home_injuries,
            "injury_count": len(home_injuries),
            "headline_injuries": home_headlines[:5],
        },
        "away": {
            "injuries": away_injuries,
            "injury_count": len(away_injuries),
            "headline_injuries": away_headlines[:5],
        },
    }


def get_round_news(year: int, round_number: int) -> dict:
    """All news for a round — injuries + team changes for all teams."""
    injuries = get_injuries()
    changes = get_team_changes(year)

    # Find the right round in changes
    round_changes = None
    for r in changes.get("rounds", []):
        if r["round_number"] == round_number:
            round_changes = r
            break

    # Build per-team news
    teams_news: dict[str, dict] = {}

    # Add injury data for all teams
    for team, team_injuries in injuries["teams"].items():
        teams_news[team] = {
            "team": team,
            "injuries": team_injuries,
            "injury_count": len(team_injuries),
            "injury_severity_total": sum(i["severity"] for i in team_injuries),
            "ins": [],
            "outs": [],
            "debutants": [],
            "stability": 1.0,
        }

    # Overlay team changes
    if round_changes:
        for tc in round_changes["teams"]:
            team = tc["team"]
            if team not in teams_news:
                teams_news[team] = {
                    "team": team,
                    "injuries": [],
                    "injury_count": 0,
                    "injury_severity_total": 0,
                    "ins": [],
                    "outs": [],
                    "debutants": [],
                    "stability": 1.0,
                }
            teams_news[team]["ins"] = tc.get("ins", [])
            teams_news[team]["outs"] = tc.get("outs", [])
            teams_news[team]["debutants"] = tc.get("debutants", [])
            teams_news[team]["stability"] = tc.get("stability", 1.0)

    return {
        "year": year,
        "round_number": round_number,
        "injuries_updated": injuries.get("updated"),
        "teams": dict(sorted(teams_news.items())),
    }


# ---------------------------------------------------------------------------
# Intel Feed — article-derived intelligence signals
# ---------------------------------------------------------------------------

def _load_latest_intel() -> dict:
    """Load latest.json intel feed."""
    latest_path = INTEL_DIR / "latest.json"
    if not latest_path.exists():
        return {"signals": [], "total": 0, "by_type": {}, "by_team": {}, "updated": None}
    with open(latest_path) as f:
        return json.load(f)


def get_intel_feed(
    signal_type: Optional[str] = None,
    team: Optional[str] = None,
    min_relevance: float = 0.0,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Return filtered intel signals as a paginated feed.

    Supports filtering by signal_type, team, min_relevance, with pagination.
    """
    data = _load_latest_intel()
    signals = data.get("signals", [])

    # Apply filters
    if signal_type:
        signals = [s for s in signals if s.get("signal_type") == signal_type]
    if team:
        team_lower = team.lower()
        signals = [s for s in signals if any(t.lower() == team_lower for t in s.get("teams", []))]
    if min_relevance > 0:
        signals = [s for s in signals if s.get("relevance_score", 0) >= min_relevance]

    total = len(signals)
    page = signals[offset:offset + limit]

    # Compute breaking signals (relevance >= 0.7, published in last 24h)
    from datetime import datetime, timedelta
    cutoff_24h = (datetime.now() - timedelta(hours=24)).isoformat()
    breaking = [
        s for s in data.get("signals", [])
        if s.get("relevance_score", 0) >= 0.7
        and (s.get("published_at") or "") >= cutoff_24h
    ]

    return {
        "signals": page,
        "total": total,
        "offset": offset,
        "limit": limit,
        "breaking_count": len(breaking),
        "by_type": data.get("by_type", {}),
        "by_team": data.get("by_team", {}),
        "updated": data.get("updated"),
    }


def get_intel_summary() -> dict:
    """High-level intel summary — counts by type, top signals, breaking news."""
    data = _load_latest_intel()
    signals = data.get("signals", [])

    from datetime import datetime, timedelta
    cutoff_24h = (datetime.now() - timedelta(hours=24)).isoformat()

    breaking = [
        s for s in signals
        if s.get("relevance_score", 0) >= 0.7
        and (s.get("published_at") or "") >= cutoff_24h
    ]

    # Top signals by relevance
    top = sorted(signals, key=lambda s: -s.get("relevance_score", 0))[:10]

    # Sentiment distribution
    sentiment_counts = {}
    for s in signals:
        sent = s.get("sentiment", "neutral")
        sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1

    # Direction summary — teams with most bearish/bullish signals
    team_direction: dict[str, dict[str, int]] = {}
    for s in signals:
        for entity, direction in s.get("direction", {}).items():
            if entity not in team_direction:
                team_direction[entity] = {"bullish": 0, "bearish": 0, "neutral": 0}
            team_direction[entity][direction] = team_direction[entity].get(direction, 0) + 1

    return {
        "total": len(signals),
        "breaking": breaking,
        "breaking_count": len(breaking),
        "top_signals": top,
        "by_type": data.get("by_type", {}),
        "by_team": data.get("by_team", {}),
        "sentiment": sentiment_counts,
        "team_direction": team_direction,
        "updated": data.get("updated"),
    }


def get_team_intel(team: str) -> dict:
    """All intel for a specific team — signals, direction, injury impact."""
    data = _load_latest_intel()
    signals = data.get("signals", [])

    team_lower = team.lower()
    team_signals = [
        s for s in signals
        if any(t.lower() == team_lower for t in s.get("teams", []))
    ]

    # Aggregate direction
    bullish = sum(1 for s in team_signals if s.get("direction", {}).get(team) == "bullish")
    bearish = sum(1 for s in team_signals if s.get("direction", {}).get(team) == "bearish")

    # Break down by type
    by_type: dict[str, list] = {}
    for s in team_signals:
        st = s.get("signal_type", "general")
        by_type.setdefault(st, []).append(s)

    return {
        "team": team,
        "total_signals": len(team_signals),
        "signals": team_signals,
        "bullish": bullish,
        "bearish": bearish,
        "net_sentiment": bullish - bearish,
        "by_type": {k: len(v) for k, v in by_type.items()},
        "injuries": by_type.get("injury", []),
        "suspensions": by_type.get("suspension", []),
        "form": by_type.get("form", []),
        "tactical": by_type.get("tactical", []),
        "updated": data.get("updated"),
    }
