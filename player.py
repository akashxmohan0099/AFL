"""
AFL Player Lookup — Comprehensive Player Profile
=================================================

Fuzzy name matching, data loading, profile building, and rendering.
Used by pipeline.py --player command. Can be reused by app/ later.
"""

import difflib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config


# ── Fuzzy name matching ──────────────────────────────────────────────────────

def find_players(search_term, player_games_df):
    """Multi-strategy fuzzy search for players.

    Returns list of dicts: {player_id, player, team, score, games}
    sorted by score desc then games desc.
    """
    search = search_term.strip().lower()
    if not search:
        return []

    # Build unique player list with game counts
    player_info = (
        player_games_df.groupby(["player_id", "player", "team"], observed=True)
        .agg(games=("match_id", "nunique"),
             last_year=("year", "max"))
        .reset_index()
    )

    matches = []
    for _, row in player_info.iterrows():
        pid = str(row["player_id"]).lower()
        name = str(row["player"]).lower()
        team = str(row["team"])
        games = int(row["games"])
        last_year = int(row["last_year"])

        score = _match_score(search, pid, name)
        if score >= 0.4:
            matches.append({
                "player_id": row["player_id"],
                "player": row["player"],
                "team": team,
                "score": score,
                "games": games,
                "last_year": last_year,
            })

    matches.sort(key=lambda m: (-m["score"], -m["games"]))

    # If we have strong matches, filter out the weak difflib noise
    if matches and matches[0]["score"] >= 0.8:
        cutoff = max(0.6, matches[0]["score"] - 0.2)
        matches = [m for m in matches if m["score"] >= cutoff]

    return matches


def _match_score(search, player_id, name):
    """Score a search term against a player name/id. Returns 0.0-1.0."""
    # 1. Exact player_id or full name
    if search == player_id or search == name:
        return 1.0

    # 2. Exact last name match ("cripps" → "cripps, patrick")
    if ", " in name:
        last_name = name.split(", ")[0]
        if search == last_name:
            return 0.95

    # 3. "first last" ↔ "last, first" token match
    search_tokens = set(search.replace(",", " ").split())
    name_tokens = set(name.replace(",", " ").split())
    if len(search_tokens) >= 2 and search_tokens == name_tokens:
        return 0.95

    # 4. Substring match
    if search in name or search in player_id:
        return 0.80

    # 5. difflib fallback
    ratio = difflib.SequenceMatcher(None, search, name).ratio()
    return ratio


def select_player(matches):
    """Interactive selection when multiple matches found.

    Returns selected match dict or None.
    """
    if not matches:
        print("No matching players found.")
        return None

    if len(matches) == 1:
        m = matches[0]
        print(f"Found: {m['player']} ({m['team']}, {m['games']} games)")
        return m

    # Multiple matches — show numbered list
    print(f"\nFound {len(matches)} matching players:\n")
    for i, m in enumerate(matches, 1):
        print(f"  {i:2d}. {m['player']:30s} {m['team']:20s} "
              f"{m['games']:4d} games  (last: {m['last_year']})")

    print()
    try:
        choice = input("Select player number (or Enter for #1): ").strip()
        if not choice:
            idx = 0
        else:
            idx = int(choice) - 1
        if 0 <= idx < len(matches):
            return matches[idx]
        print("Invalid selection.")
        return None
    except (ValueError, EOFError, KeyboardInterrupt):
        print()
        return None


# ── Data loading ─────────────────────────────────────────────────────────────

def load_player_data(player_id, year=None):
    """Load all available data for a player.

    Returns dict with dataframes + metadata. Gracefully skips missing files.
    """
    data = {"player_id": player_id, "year": year}

    # Base player games (always required)
    pg_path = config.BASE_STORE_DIR / "player_games.parquet"
    pg = pd.read_parquet(pg_path)
    pg = pg[pg["player_id"] == player_id].sort_values("date").copy()
    data["player_games"] = pg

    # Feature matrix (optional)
    fm_path = config.FEATURES_DIR / "feature_matrix.parquet"
    if fm_path.exists():
        fm = pd.read_parquet(fm_path)
        fm = fm[fm["player_id"] == player_id].sort_values("date").copy()
        data["features"] = fm if len(fm) > 0 else None
    else:
        data["features"] = None

    # Learning store predictions (optional)
    data["learning_predictions"] = _load_learning_parquets(
        config.LEARNING_DIR / "predictions", player_id
    )
    data["learning_outcomes"] = _load_learning_parquets(
        config.LEARNING_DIR / "outcomes", player_id
    )
    data["learning_diagnostics"] = _load_learning_parquets(
        config.LEARNING_DIR / "diagnostics", player_id
    )

    # Sequential learning predictions
    seq_pred_dir = config.SEQUENTIAL_DIR / "predictions"
    data["sequential_predictions"] = _load_learning_parquets(
        seq_pred_dir, player_id
    )

    # Latest prediction CSVs
    data["latest_predictions"] = _load_latest_predictions(player_id)

    return data


def _load_learning_parquets(directory, player_id):
    """Load and concatenate per-round parquets from a learning subdirectory."""
    if not directory.exists():
        return None

    def _sort_key(path):
        stem = path.stem
        legacy = re.match(r"^(\d{4})_R(\d+)$", stem)
        if legacy:
            year = int(legacy.group(1))
            rnd = int(legacy.group(2))
            return (year, rnd, "", path.name)

        run_file = re.match(r"^R(\d+)$", stem)
        if run_file and path.parent.name.startswith("run_") and path.parent.parent.name.isdigit():
            year = int(path.parent.parent.name)
            rnd = int(run_file.group(1))
            run_id = path.parent.name.replace("run_", "", 1)
            return (year, rnd, run_id, path.name)

        return (10**9, 10**9, "", str(path))

    files = sorted([f for f in directory.rglob("*.parquet") if f.is_file()], key=_sort_key)
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if "player_id" in df.columns:
                df = df[df["player_id"] == player_id]
            elif "player" in df.columns:
                # Fallback: match on player name from player_id
                player_name = player_id.rsplit("_", 1)[0] if "_" in str(player_id) else str(player_id)
                df = df[df["player"].str.lower() == player_name.lower()]
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            continue
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None


def _load_latest_predictions(player_id):
    """Load the most recent prediction CSV for this player."""
    pred_dir = config.PREDICTIONS_DIR
    if not pred_dir.exists():
        return None
    candidates = []
    for path in pred_dir.glob("round_*_predictions.csv"):
        m = re.match(r"^round_(\d+)_predictions\.csv$", path.name)
        round_num = int(m.group(1)) if m else -1
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((round_num, mtime, path))

    if not candidates:
        return None

    # Primary sort by numeric round, tie-break by modification time.
    candidates.sort(key=lambda x: (x[0], x[1]))
    latest = candidates[-1][2]
    try:
        df = pd.read_csv(latest)
        player_name = player_id.rsplit("_", 1)[0] if "_" in str(player_id) else str(player_id)
        mask = df["player"].str.lower() == player_name.lower()
        if "player_id" in df.columns:
            mask = mask | (df["player_id"] == player_id)
        filtered = df[mask]
        if len(filtered) > 0:
            filtered = filtered.copy()
            filtered["_source_file"] = latest.name
            return filtered
    except Exception:
        pass
    return None


# ── Profile building ─────────────────────────────────────────────────────────

STAT_COLS = ["GL", "BH", "DI", "MK", "TK", "KI", "HB", "HO", "CL",
             "CP", "IF", "RB", "FF", "FA", "one_pct", "GA"]


def build_identity(data):
    """Build identity section: name, team, age, career summary."""
    pg = data["player_games"]
    fm = data.get("features")
    if pg.empty:
        return None

    latest = pg.iloc[-1]
    info = {
        "player": latest.get("player", "Unknown"),
        "player_id": data["player_id"],
        "team": latest.get("team", "Unknown"),
        "career_games": len(pg[~pg.get("did_not_play", pd.Series(False, index=pg.index)).astype(bool)]),
        "first_game": pg["date"].iloc[0],
        "last_game": pg["date"].iloc[-1],
        "years_active": f"{int(pg['year'].min())}-{int(pg['year'].max())}",
    }

    if "jumper" in pg.columns and pd.notna(latest.get("jumper")):
        info["jumper"] = int(latest["jumper"])
    if "age_years" in pg.columns and pd.notna(latest.get("age_years")):
        info["age"] = round(float(latest["age_years"]), 1)

    # Role and archetype from features
    if fm is not None and len(fm) > 0:
        fl = fm.iloc[-1]
        if "player_role" in fm.columns and pd.notna(fl.get("player_role")):
            info["role"] = str(fl["player_role"])
        if "archetype" in fm.columns and pd.notna(fl.get("archetype")):
            info["archetype"] = int(fl["archetype"])
            info["archetype_label"] = _describe_archetype(fm)

    # Status: active if played in last year of data
    max_year = int(pg["year"].max())
    last_year_games = pg[pg["year"] == max_year]
    info["status"] = "Active" if len(last_year_games) > 0 else "Inactive"

    return info


def _describe_archetype(fm):
    """Describe archetype from player's dominant stats."""
    if fm is None or fm.empty:
        return "Unknown"
    recent = fm.tail(10)
    stats = {}
    for col, label in [("DI", "Disp"), ("GL", "Goal"), ("MK", "Mark"),
                        ("TK", "Tack"), ("CL", "Clear"), ("HO", "Ruck"),
                        ("RB", "Reb"), ("IF", "I50")]:
        if col in recent.columns:
            stats[label] = recent[col].mean()

    # Pick top 2 stats relative to typical values
    benchmarks = {"Disp": 15, "Goal": 1.0, "Mark": 4, "Tack": 4,
                  "Clear": 2, "Ruck": 5, "Reb": 2, "I50": 2}
    ratios = {}
    for label, val in stats.items():
        bench = benchmarks.get(label, 1)
        if bench > 0:
            ratios[label] = val / bench

    if not ratios:
        return "Unknown"
    top = sorted(ratios, key=ratios.get, reverse=True)[:2]
    arch_id = fm.iloc[-1].get("archetype", "?")
    return f"{'/'.join(top)} (arch {arch_id})"


def build_career_stats(data):
    """Career totals and averages."""
    pg = data["player_games"]
    if pg.empty:
        return None

    # Exclude DNP rows
    active = pg[~pg.get("did_not_play", pd.Series(False, index=pg.index)).astype(bool)]
    if active.empty:
        return None

    available = [c for c in STAT_COLS if c in active.columns]
    totals = active[available].sum()
    averages = active[available].mean()
    games = len(active)

    return {"games": games, "totals": totals.to_dict(), "averages": averages.to_dict()}


def build_recent_form(data, n_games=10):
    """Last N games detail table."""
    pg = data["player_games"]
    active = pg[~pg.get("did_not_play", pd.Series(False, index=pg.index)).astype(bool)]
    if active.empty:
        return None

    recent = active.tail(n_games).copy()
    cols = ["date", "round_number", "opponent", "venue", "GL", "BH", "DI",
            "MK", "TK", "KI", "HB"]
    if "pct_played" in recent.columns:
        cols.append("pct_played")
    available = [c for c in cols if c in recent.columns]
    return recent[available].reset_index(drop=True)


def build_rolling_averages(data):
    """Rolling averages from feature matrix."""
    fm = data.get("features")
    if fm is None or fm.empty:
        return None

    latest = fm.iloc[-1]
    # Feature columns use lowercase stat names (e.g. player_gl_avg_3)
    stats = [("GL", "gl"), ("DI", "di"), ("MK", "mk"), ("TK", "tk"), ("KI", "ki")]
    windows = [3, 5, 10]
    rows = []
    for label, col_key in stats:
        row = {"stat": label}
        for w in windows:
            col = f"player_{col_key}_avg_{w}"
            row[f"avg_{w}"] = latest.get(col, np.nan)
        ewm_col = f"player_{col_key}_ewm_5"
        row["ewm_5"] = latest.get(ewm_col, np.nan)
        rows.append(row)
    return rows


def build_streaks(data):
    """Streak and form indicators from features."""
    fm = data.get("features")
    if fm is None or fm.empty:
        return None

    latest = fm.iloc[-1]
    info = {}
    field_map = {
        "goal_streak": "player_gl_streak",
        "goal_streak_weighted": "player_gl_streak_weighted",
        "cold_streak": "player_gl_cold_streak",
        "form_ratio": "player_form_ratio",
        "is_hot": "player_is_hot",
        "is_cold": "player_is_cold",
        "gl_volatility": "player_gl_volatility_5",
        "gl_trend": "player_gl_trend_5",
        "di_trend": "player_di_trend_5",
        "di_volatility": "player_di_volatility_5",
    }
    for key, col in field_map.items():
        if col in fm.columns:
            val = latest.get(col)
            if pd.notna(val):
                info[key] = float(val)
    return info if info else None


def build_venue_splits(data, top_n=5):
    """Per-venue performance splits."""
    pg = data["player_games"]
    active = pg[~pg.get("did_not_play", pd.Series(False, index=pg.index)).astype(bool)]
    if len(active) < 3 or "venue" not in active.columns:
        return None

    overall_gl = active["GL"].mean() if "GL" in active.columns else 0
    overall_di = active["DI"].mean() if "DI" in active.columns else 0

    venue_stats = (
        active.groupby("venue", observed=True)
        .agg(
            games=("match_id", "nunique"),
            avg_GL=("GL", "mean") if "GL" in active.columns else ("match_id", "count"),
            avg_DI=("DI", "mean") if "DI" in active.columns else ("match_id", "count"),
        )
        .reset_index()
    )
    venue_stats["gl_diff"] = venue_stats["avg_GL"] - overall_gl
    venue_stats["di_diff"] = venue_stats["avg_DI"] - overall_di
    venue_stats = venue_stats.sort_values("games", ascending=False)

    return {
        "overall_gl": overall_gl,
        "overall_di": overall_di,
        "venues": venue_stats.head(top_n) if top_n else venue_stats,
    }


def build_opponent_matchups(data, top_n=5):
    """Best and worst opponents by avg goals."""
    pg = data["player_games"]
    active = pg[~pg.get("did_not_play", pd.Series(False, index=pg.index)).astype(bool)]
    if len(active) < 3 or "opponent" not in active.columns or "GL" not in active.columns:
        return None

    overall_gl = active["GL"].mean()

    opp_stats = (
        active.groupby("opponent", observed=True)
        .agg(games=("match_id", "nunique"), avg_GL=("GL", "mean"))
        .reset_index()
    )
    opp_stats["gl_diff"] = opp_stats["avg_GL"] - overall_gl
    opp_stats = opp_stats[opp_stats["games"] >= 2]

    if opp_stats.empty:
        return None

    best = opp_stats.sort_values("avg_GL", ascending=False).head(top_n)
    worst = opp_stats.sort_values("avg_GL", ascending=True).head(top_n)

    return {"overall_gl": overall_gl, "best": best, "worst": worst}


def build_scoring_patterns(data):
    """Quarter distribution, multi-goal rates, home/away, finals."""
    pg = data["player_games"]
    active = pg[~pg.get("did_not_play", pd.Series(False, index=pg.index)).astype(bool)]
    if active.empty or "GL" not in active.columns:
        return None

    info = {}
    total_goals = active["GL"].sum()

    # Quarter distribution
    q_cols = ["q1_goals", "q2_goals", "q3_goals", "q4_goals"]
    if all(c in active.columns for c in q_cols):
        q_totals = {q: active[q].sum() for q in q_cols}
        if total_goals > 0:
            info["quarter_pct"] = {
                q.replace("_goals", ""): round(v / total_goals * 100, 1)
                for q, v in q_totals.items()
            }

    # Goal rates
    games_played = len(active)
    if games_played > 0:
        info["multi_goal_rate"] = round(
            len(active[active["GL"] >= 2]) / games_played * 100, 1
        )
        info["one_plus_rate"] = round(
            len(active[active["GL"] >= 1]) / games_played * 100, 1
        )
        info["two_plus_rate"] = round(
            len(active[active["GL"] >= 2]) / games_played * 100, 1
        )
        info["three_plus_rate"] = round(
            len(active[active["GL"] >= 3]) / games_played * 100, 1
        )

    # Home / away split
    if "is_home" in active.columns:
        home = active[active["is_home"] == True]
        away = active[active["is_home"] == False]
        if len(home) > 0 and len(away) > 0:
            info["home_avg_gl"] = round(home["GL"].mean(), 2)
            info["away_avg_gl"] = round(away["GL"].mean(), 2)
            info["home_games"] = len(home)
            info["away_games"] = len(away)

    # Finals record
    if "is_finals" in active.columns:
        finals = active[active["is_finals"] == True]
        if len(finals) > 0:
            info["finals_games"] = len(finals)
            info["finals_avg_gl"] = round(finals["GL"].mean(), 2)
            info["finals_total_gl"] = int(finals["GL"].sum())

    return info


def build_season_summary(data, year=None):
    """Season-level summary for current or specified year."""
    pg = data["player_games"]
    if year is None:
        year = int(pg["year"].max())

    season = pg[pg["year"] == year].copy()
    active = season[~season.get("did_not_play", pd.Series(False, index=season.index)).astype(bool)]
    if active.empty:
        return None

    available = [c for c in STAT_COLS if c in active.columns]
    totals = active[available].sum()
    averages = active[available].mean()

    info = {
        "year": year,
        "games": len(active),
        "totals": totals.to_dict(),
        "averages": averages.to_dict(),
    }

    # Trend: compare first half to second half of season
    if len(active) >= 6:
        mid = len(active) // 2
        first_half = active.iloc[:mid]
        second_half = active.iloc[mid:]
        if "GL" in active.columns:
            info["gl_trend"] = round(
                second_half["GL"].mean() - first_half["GL"].mean(), 2
            )
        if "DI" in active.columns:
            info["di_trend"] = round(
                second_half["DI"].mean() - first_half["DI"].mean(), 2
            )

    return info


def build_predictions(data):
    """Latest upcoming predictions for this player."""
    preds = data.get("latest_predictions")
    if preds is None or preds.empty:
        return None

    row = preds.iloc[0]
    info = {"source": row.get("_source_file", "predictions")}
    for col in ["predicted_goals", "predicted_behinds", "predicted_score",
                "predicted_disposals", "p_scorer", "p_2plus_goals",
                "p_3plus_goals", "p_15plus_disp", "p_20plus_disp",
                "p_25plus_disp", "p_30plus_disp", "team", "opponent", "venue"]:
        if col in row.index and pd.notna(row[col]):
            info[col] = row[col]
    return info


def build_prediction_accuracy(data):
    """Prediction accuracy from learning store data."""
    preds = data.get("learning_predictions")
    if preds is None:
        preds = data.get("sequential_predictions")
    outcomes = data.get("learning_outcomes")

    if preds is None or outcomes is None:
        return None

    # Try to merge predictions with outcomes
    merge_cols = []
    for c in ["match_id", "round_number", "year"]:
        if c in preds.columns and c in outcomes.columns:
            merge_cols.append(c)
    if not merge_cols:
        return None

    try:
        merged = preds.merge(outcomes, on=merge_cols, suffixes=("_pred", "_out"))
    except Exception:
        return None

    if merged.empty:
        return None

    info = {}
    # Goals accuracy
    pred_col = next((c for c in ["predicted_goals", "predicted_goals_pred"]
                     if c in merged.columns), None)
    actual_col = next((c for c in ["GL", "GL_out", "actual_goals"]
                       if c in merged.columns), None)
    if pred_col and actual_col:
        errors = (merged[pred_col] - merged[actual_col]).abs()
        info["goals_mae"] = round(errors.mean(), 3)
        info["goals_n"] = len(errors)

    return info if info else None


# ── Rendering ────────────────────────────────────────────────────────────────

WIDTH = 72


def _header(title):
    return f"\n{'=' * WIDTH}\n{title.upper():^{WIDTH}}\n{'=' * WIDTH}"


def _subheader(title):
    return f"\n  {title}\n  {'-' * (WIDTH - 4)}"


def _row(label, value, indent=4):
    return f"{' ' * indent}{label + ':':<28s} {value}"


def render_profile(data, detail=False, year=None):
    """Print full player profile to stdout."""
    identity = build_identity(data)
    if identity is None:
        print("No data found for this player.")
        return

    # ── Identity ──
    print(_header(f"Player Profile: {identity['player']}"))
    print(_row("Team", identity["team"]))
    if "jumper" in identity:
        print(_row("Jumper", f"#{identity['jumper']}"))
    if "age" in identity:
        print(_row("Age", f"{identity['age']} years"))
    print(_row("Career games", identity["career_games"]))
    print(_row("Years active", identity["years_active"]))
    print(_row("Status", identity["status"]))
    if "role" in identity:
        print(_row("Role", identity["role"]))
    if "archetype_label" in identity:
        print(_row("Archetype", identity["archetype_label"]))

    # ── Career Stats ──
    career = build_career_stats(data)
    if career:
        print(_subheader(f"Career Stats ({career['games']} games)"))
        _render_stat_table(career["totals"], career["averages"])

    # ── Season Summary ──
    season = build_season_summary(data, year=year)
    if season:
        print(_subheader(f"Season {season['year']} ({season['games']} games)"))
        _render_stat_table(season["totals"], season["averages"])
        if "gl_trend" in season:
            direction = "up" if season["gl_trend"] > 0 else "down" if season["gl_trend"] < 0 else "flat"
            print(f"    Goals trend (2nd half vs 1st): {season['gl_trend']:+.2f} ({direction})")
        if "di_trend" in season:
            direction = "up" if season["di_trend"] > 0 else "down" if season["di_trend"] < 0 else "flat"
            print(f"    Disposals trend:               {season['di_trend']:+.2f} ({direction})")

    # ── Recent Form ──
    n_games = 20 if detail else 10
    recent = build_recent_form(data, n_games=n_games)
    if recent is not None and not recent.empty:
        print(_subheader(f"Recent Form (last {len(recent)} games)"))
        _render_recent_table(recent)

    # ── Rolling Averages ──
    rolling = build_rolling_averages(data)
    if rolling:
        print(_subheader("Rolling Averages"))
        print(f"    {'Stat':<6s} {'Avg3':>6s} {'Avg5':>6s} {'Avg10':>6s} {'EWM5':>6s}")
        for r in rolling:
            vals = []
            for k in ["avg_3", "avg_5", "avg_10", "ewm_5"]:
                v = r.get(k)
                vals.append(f"{v:6.2f}" if pd.notna(v) else "     -")
            print(f"    {r['stat']:<6s} {''.join(vals)}")

    # ── Streaks & Form ──
    streaks = build_streaks(data)
    if streaks:
        print(_subheader("Streaks & Form"))
        if "goal_streak" in streaks:
            print(_row("Goal streak", f"{streaks['goal_streak']:.0f} games"))
        if "cold_streak" in streaks:
            print(_row("Cold streak", f"{streaks['cold_streak']:.0f} games"))
        if "form_ratio" in streaks:
            ratio = streaks["form_ratio"]
            label = "HOT" if ratio >= 1.5 else "COLD" if ratio <= 0.5 else "normal"
            print(_row("Form ratio", f"{ratio:.2f} ({label})"))
        if "gl_trend" in streaks:
            print(_row("GL trend (5-game)", f"{streaks['gl_trend']:+.3f}"))
        if "gl_volatility" in streaks:
            print(_row("GL volatility (5-game)", f"{streaks['gl_volatility']:.3f}"))
        if "di_trend" in streaks:
            print(_row("DI trend (5-game)", f"{streaks['di_trend']:+.3f}"))

    # ── Scoring Patterns ──
    patterns = build_scoring_patterns(data)
    if patterns:
        print(_subheader("Scoring Patterns"))
        if "one_plus_rate" in patterns:
            print(_row("Scores 1+ goals", f"{patterns['one_plus_rate']:.1f}% of games"))
        if "two_plus_rate" in patterns:
            print(_row("Scores 2+ goals", f"{patterns['two_plus_rate']:.1f}% of games"))
        if "three_plus_rate" in patterns:
            print(_row("Scores 3+ goals", f"{patterns['three_plus_rate']:.1f}% of games"))
        if "quarter_pct" in patterns:
            qp = patterns["quarter_pct"]
            parts = [f"Q{i+1}: {qp.get(f'q{i+1}', 0):.0f}%" for i in range(4)]
            print(_row("Quarter distribution", "  ".join(parts)))
        if "home_avg_gl" in patterns:
            print(_row("Home avg goals",
                        f"{patterns['home_avg_gl']:.2f} ({patterns['home_games']} games)"))
            print(_row("Away avg goals",
                        f"{patterns['away_avg_gl']:.2f} ({patterns['away_games']} games)"))
        if "finals_games" in patterns:
            print(_row("Finals record",
                        f"{patterns['finals_total_gl']} goals in "
                        f"{patterns['finals_games']} games "
                        f"(avg {patterns['finals_avg_gl']:.2f})"))

    # ── Venue Splits ──
    top_n = None if detail else 5
    venues = build_venue_splits(data, top_n=top_n)
    if venues:
        venue_df = venues["venues"]
        label = "All Venues" if detail else f"Top {len(venue_df)} Venues"
        print(_subheader(f"Venue Splits ({label})"))
        print(f"    {'Venue':<25s} {'Games':>5s} {'AvgGL':>6s} {'Diff':>6s} "
              f"{'AvgDI':>6s} {'Diff':>6s}")
        for _, v in venue_df.iterrows():
            print(f"    {str(v['venue'])[:25]:<25s} {v['games']:5.0f} "
                  f"{v['avg_GL']:6.2f} {v['gl_diff']:+6.2f} "
                  f"{v['avg_DI']:6.2f} {v['di_diff']:+6.2f}")

    # ── Opponent Matchups ──
    top_n = None if detail else 5
    matchups = build_opponent_matchups(data, top_n=top_n if top_n else 5)
    if matchups:
        print(_subheader("Best Opponents (by avg goals)"))
        print(f"    {'Opponent':<25s} {'Games':>5s} {'AvgGL':>6s} {'Diff':>6s}")
        for _, r in matchups["best"].iterrows():
            print(f"    {str(r['opponent'])[:25]:<25s} {r['games']:5.0f} "
                  f"{r['avg_GL']:6.2f} {r['gl_diff']:+6.2f}")

        print(_subheader("Worst Opponents (by avg goals)"))
        print(f"    {'Opponent':<25s} {'Games':>5s} {'AvgGL':>6s} {'Diff':>6s}")
        for _, r in matchups["worst"].iterrows():
            print(f"    {str(r['opponent'])[:25]:<25s} {r['games']:5.0f} "
                  f"{r['avg_GL']:6.2f} {r['gl_diff']:+6.2f}")

    # ── Latest Predictions ──
    predictions = build_predictions(data)
    if predictions:
        print(_subheader("Latest Predictions"))
        print(_row("Source", predictions.get("source", "?")))
        if "opponent" in predictions:
            matchup = f"vs {predictions['opponent']}"
            if "venue" in predictions:
                matchup += f" at {predictions['venue']}"
            print(_row("Matchup", matchup))
        if "predicted_goals" in predictions:
            print(_row("Predicted goals", f"{predictions['predicted_goals']:.2f}"))
        if "predicted_behinds" in predictions:
            print(_row("Predicted behinds", f"{predictions['predicted_behinds']:.2f}"))
        if "predicted_disposals" in predictions:
            print(_row("Predicted disposals", f"{predictions['predicted_disposals']:.1f}"))
        if "p_scorer" in predictions:
            print(_row("P(1+ goals)", f"{predictions['p_scorer']:.1%}"))
        if "p_2plus_goals" in predictions:
            print(_row("P(2+ goals)", f"{predictions['p_2plus_goals']:.1%}"))
        if "p_3plus_goals" in predictions:
            print(_row("P(3+ goals)", f"{predictions['p_3plus_goals']:.1%}"))
        if "p_20plus_disp" in predictions:
            print(_row("P(20+ disposals)", f"{predictions['p_20plus_disp']:.1%}"))
        if "p_25plus_disp" in predictions:
            print(_row("P(25+ disposals)", f"{predictions['p_25plus_disp']:.1%}"))
    else:
        print(_subheader("Latest Predictions"))
        print("    No predictions available.")

    # ── Prediction Accuracy ──
    accuracy = build_prediction_accuracy(data)
    if accuracy:
        print(_subheader("Prediction Accuracy (historical)"))
        if "goals_mae" in accuracy:
            print(_row("Goals MAE", f"{accuracy['goals_mae']:.3f} "
                        f"(n={accuracy['goals_n']})"))

    print(f"\n{'=' * WIDTH}")


def _render_stat_table(totals, averages):
    """Render a compact stat totals + averages table."""
    stats_row1 = ["GL", "BH", "DI", "MK", "TK", "KI", "HB", "HO"]
    stats_row2 = ["CL", "CP", "IF", "RB", "FF", "FA", "one_pct", "GA"]

    for stats in [stats_row1, stats_row2]:
        available = [s for s in stats if s in totals]
        if not available:
            continue
        # Header
        hdr = "    " + "".join(f"{s:>8s}" for s in available)
        # Totals
        tot = "Tot " + "".join(f"{totals.get(s, 0):8.0f}" for s in available)
        # Averages
        avg = "Avg " + "".join(f"{averages.get(s, 0):8.2f}" for s in available)
        print(hdr)
        print(tot)
        print(avg)


def _render_recent_table(recent):
    """Render recent games table."""
    print(f"    {'Date':<12s} {'Rnd':>3s} {'Opponent':<18s} "
          f"{'GL':>3s} {'BH':>3s} {'DI':>3s} {'MK':>3s} {'TK':>3s} "
          f"{'KI':>3s} {'HB':>3s} {'%Pl':>4s}")

    for _, row in recent.iterrows():
        date_str = str(row.get("date", ""))[:10]
        rnd = str(int(row["round_number"])) if "round_number" in row.index else "?"
        opp = str(row.get("opponent", "?"))[:18]
        pct = f"{row['pct_played']:4.0f}" if "pct_played" in row.index and pd.notna(row.get("pct_played")) else "   -"

        vals = []
        for col in ["GL", "BH", "DI", "MK", "TK", "KI", "HB"]:
            v = row.get(col)
            vals.append(f"{int(v):3d}" if pd.notna(v) else "  -")

        print(f"    {date_str:<12s} {rnd:>3s} {opp:<18s} "
              f"{''.join(vals)} {pct}")


# ── CLI entry point ──────────────────────────────────────────────────────────

def cmd_player(args):
    """CLI handler for --player command. Called from pipeline.py."""
    search_term = args.player
    detail = getattr(args, "player_detail", False)
    year = getattr(args, "year", None)

    # Load base data
    pg_path = config.BASE_STORE_DIR / "player_games.parquet"
    if not pg_path.exists():
        print("Error: No player_games.parquet found. Run --clean first.")
        sys.exit(1)

    print(f"Searching for '{search_term}'...")
    pg = pd.read_parquet(pg_path)

    matches = find_players(search_term, pg)
    selected = select_player(matches)
    if selected is None:
        if matches:
            return
        # Suggest closest names
        all_names = pg["player"].dropna().unique().tolist()
        close = difflib.get_close_matches(search_term, [str(n) for n in all_names], n=5, cutoff=0.4)
        if close:
            print(f"\nDid you mean: {', '.join(close)}?")
        return

    player_id = selected["player_id"]
    data = load_player_data(player_id, year=year)
    render_profile(data, detail=detail, year=year)
