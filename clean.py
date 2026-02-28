"""
AFL Prediction Pipeline — Data Cleaning & Normalization
========================================================
Loads raw CSVs from the scraper, normalizes names/dates/types,
parses career strings, and joins into a single analysis-ready
DataFrame saved as parquet.

Output: data/cleaned/player_matches.parquet
  - One row per (player, team, match_id)
  - All stat columns numeric
  - Parsed career/age fields
  - Scoring aggregates per quarter joined in
"""

import re
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

import config

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_age(age_str):
    """Parse '24y 187d' → 24.51 (float years)."""
    if pd.isna(age_str) or not isinstance(age_str, str):
        return np.nan
    m = re.match(r"(\d+)y\s+(\d+)d", age_str.strip())
    if m:
        return int(m.group(1)) + int(m.group(2)) / 365.25
    return np.nan


def parse_career_goals(goals_str):
    """Parse '228 (1.28)' → (228, 1.28) i.e. (total, average).
    Returns (np.nan, np.nan) on failure."""
    if pd.isna(goals_str) or not isinstance(goals_str, str):
        return np.nan, np.nan
    goals_str = goals_str.strip()
    m = re.match(r"(\d+)\s*\((\d+\.?\d*)\)", goals_str)
    if m:
        return int(m.group(1)), float(m.group(2))
    # Sometimes just a number with no average (debutants)
    m2 = re.match(r"^(\d+)$", goals_str)
    if m2:
        return int(m2.group(1)), np.nan
    return np.nan, np.nan


def parse_career_games(games_str):
    """Parse '178 (112-1-65 63.20%)' → (178, 112, 1, 65, 63.20).
    Returns tuple of NaNs on failure."""
    if pd.isna(games_str) or not isinstance(games_str, str):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    games_str = games_str.strip()
    m = re.match(
        r"(\d+)\s*\((\d+)-(\d+)-(\d+)\s+([\d.]+)%\)", games_str
    )
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), float(m.group(5)))
    # Just a bare number
    m2 = re.match(r"^(\d+)$", games_str)
    if m2:
        return int(m2.group(1)), np.nan, np.nan, np.nan, np.nan
    return np.nan, np.nan, np.nan, np.nan, np.nan


def ensure_date_iso(df):
    """Ensure a 'date_iso' column exists by parsing the 'date' column
    when date_iso is missing (e.g. older scraped data)."""
    if "date_iso" not in df.columns:
        df["date_iso"] = df["date"].apply(_parse_afl_date)
    # Convert to pandas datetime
    df["date_iso"] = pd.to_datetime(df["date_iso"], errors="coerce")
    return df


def _parse_afl_date(date_str):
    """Mirror of scraper.parse_afl_date — parse AFL Tables date string."""
    if pd.isna(date_str) or not isinstance(date_str, str):
        return pd.NaT
    clean = re.sub(r"\s*\([^)]*\)\s*$", "", date_str.strip())
    try:
        from datetime import datetime
        dt = datetime.strptime(clean, "%a, %d-%b-%Y %I:%M %p")
        return dt.isoformat()
    except ValueError:
        return pd.NaT


def normalize_player_name(name):
    """Normalize any player name to 'Last, First' format.
    Handles:
      - 'Last, First' → pass-through
      - 'First Last' → 'Last, First'
      - 'Rushed' → 'Rushed'
    """
    if pd.isna(name) or not isinstance(name, str):
        return name
    name = name.strip()
    if name in ("Rushed", ""):
        return name
    if "," in name:
        return name  # already canonical
    parts = name.split()
    if len(parts) >= 2:
        return f"{' '.join(parts[1:])}, {parts[0]}"
    return name


def parse_round_number(round_str):
    """Convert round string to a sortable integer.
    '1' → 1, 'QF' → 25, 'EF' → 25, 'SF' → 26, 'PF' → 27, 'GF' → 28."""
    if pd.isna(round_str):
        return np.nan
    s = str(round_str).strip()
    try:
        return int(s)
    except ValueError:
        finals_map = {"QF": 25, "EF": 25, "SF": 26, "PF": 27, "GF": 28}
        return finals_map.get(s.upper(), np.nan)


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------

def load_player_stats(data_dir=None):
    """Load all player_stats CSVs (per-year or master).
    Returns a cleaned DataFrame."""
    data_dir = Path(data_dir or config.DATA_DIR)

    # Try master file first
    master = data_dir / "all_player_stats.csv"
    if master.exists():
        df = pd.read_csv(master, low_memory=False)
    else:
        # Fall back to per-year files
        ps_dir = data_dir / "player_stats"
        files = sorted(ps_dir.glob("player_stats_*.csv"))
        if not files:
            # Try test files
            test_files = sorted(data_dir.glob("test_*player_stats*.csv"))
            files = test_files
        if not files:
            raise FileNotFoundError(f"No player_stats files in {data_dir}")
        df = pd.concat([pd.read_csv(f, low_memory=False) for f in files],
                       ignore_index=True)

    # Normalize
    df = ensure_date_iso(df)
    df["player"] = df["player"].apply(normalize_player_name)
    df["round_number"] = df["round"].apply(parse_round_number)

    # Coerce stat columns to numeric
    stat_cols = [
        "KI", "MK", "HB", "DI", "GL", "BH", "HO", "TK", "RB", "IF",
        "CL", "CG", "FF", "FA", "BR", "CP", "UP", "CM", "MI",
        "one_pct", "BO", "GA", "pct_played",
    ]
    for col in stat_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Sort chronologically
    df = df.sort_values(["date_iso", "match_id", "team", "player"]).reset_index(drop=True)
    return df


def load_player_details(data_dir=None):
    """Load all player_details CSVs. Parse career strings."""
    data_dir = Path(data_dir or config.DATA_DIR)

    master = data_dir / "all_player_details.csv"
    if master.exists():
        df = pd.read_csv(master, low_memory=False)
    else:
        pd_dir = data_dir / "player_details"
        files = sorted(pd_dir.glob("player_details_*.csv"))
        if not files:
            test_files = sorted(data_dir.glob("test_*player_details*.csv"))
            files = test_files
        if not files:
            raise FileNotFoundError(f"No player_details files in {data_dir}")
        df = pd.concat([pd.read_csv(f, low_memory=False) for f in files],
                       ignore_index=True)

    # Filter out coach rows (jumper == "C")
    df = df[df["jumper"].astype(str).str.strip() != "C"].copy()

    df["player"] = df["player"].apply(normalize_player_name)

    # Parse age
    df["age_years"] = df["Age"].apply(parse_age)

    # Parse career games
    parsed_games = df["Career Games (W-D-L W%)"].apply(parse_career_games)
    df["career_games"] = parsed_games.apply(lambda x: x[0])
    df["career_wins"] = parsed_games.apply(lambda x: x[1])
    df["career_draws"] = parsed_games.apply(lambda x: x[2])
    df["career_losses"] = parsed_games.apply(lambda x: x[3])
    df["career_win_pct"] = parsed_games.apply(lambda x: x[4])

    # Parse career goals
    parsed_goals = df["Career Goals (Ave.)"].apply(parse_career_goals)
    df["career_goals_total"] = parsed_goals.apply(lambda x: x[0])
    df["career_goal_avg"] = parsed_goals.apply(lambda x: x[1])

    # Parse team-level games/goals similarly
    parsed_tg = df["team_games"].apply(parse_career_games)
    df["team_games_total"] = parsed_tg.apply(lambda x: x[0])
    df["team_win_pct"] = parsed_tg.apply(lambda x: x[4])

    parsed_tgl = df["team_goals"].apply(parse_career_goals)
    df["team_goals_total"] = parsed_tgl.apply(lambda x: x[0])
    df["team_goal_avg"] = parsed_tgl.apply(lambda x: x[1])

    return df


def load_scoring(data_dir=None):
    """Load all scoring CSVs. Normalize player names, parse quarters."""
    data_dir = Path(data_dir or config.DATA_DIR)

    master = data_dir / "all_scoring.csv"
    if master.exists():
        df = pd.read_csv(master, low_memory=False)
    else:
        sc_dir = data_dir / "scoring"
        files = sorted(sc_dir.glob("scoring_*.csv"))
        if not files:
            test_files = sorted(data_dir.glob("test_*scoring*.csv"))
            files = test_files
        if not files:
            raise FileNotFoundError(f"No scoring files in {data_dir}")
        df = pd.concat([pd.read_csv(f, low_memory=False) for f in files],
                       ignore_index=True)

    df["player"] = df["player"].apply(normalize_player_name)

    # Normalize quarter labels: '1st'→1, '2nd'→2, '3rd'→3, '4th'→4, 'Final'→4
    quarter_map = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "Final": 4}
    df["quarter_num"] = df["quarter"].map(quarter_map).fillna(4).astype(int)

    return df


# ---------------------------------------------------------------------------
# Aggregation: scoring events → per-player-match-quarter counts
# ---------------------------------------------------------------------------

def aggregate_scoring_per_player_match(scoring_df):
    """Pivot scoring events into per-(player, team, match_id) quarter counts.

    Returns DataFrame with columns:
      match_id, player, team, q1_goals, q1_behinds, ..., q4_goals, q4_behinds
    """
    # Filter out rushed behinds — they're team events, not player events
    player_scoring = scoring_df[scoring_df["player"] != "Rushed"].copy()

    if player_scoring.empty:
        return pd.DataFrame()

    # Pivot: count goals and behinds per quarter per player per match
    agg = (
        player_scoring
        .groupby(["match_id", "player", "team", "quarter_num", "score_type"])
        .size()
        .reset_index(name="count")
    )

    # Pivot wider: one column per (quarter, score_type)
    pivoted = agg.pivot_table(
        index=["match_id", "player", "team"],
        columns=["quarter_num", "score_type"],
        values="count",
        fill_value=0,
    )

    # Flatten column names: (1, 'goal') → 'q1_goals'
    pivoted.columns = [f"q{q}_{st}s" for q, st in pivoted.columns]
    pivoted = pivoted.reset_index()

    # Ensure all q columns exist
    for q in range(1, 5):
        for st in ["goals", "behinds"]:
            col = f"q{q}_{st}"
            if col not in pivoted.columns:
                pivoted[col] = 0

    return pivoted


# ---------------------------------------------------------------------------
# Main join: produce the clean player_matches DataFrame
# ---------------------------------------------------------------------------

def build_player_matches(data_dir=None, save=True):
    """Load all raw data, normalize, join, and produce the clean dataset.

    Returns a DataFrame with one row per (player, team, match_id) containing:
      - All player_stats columns (normalized)
      - Parsed career/age details from player_details
      - Per-quarter scoring aggregates from scoring progression
    """
    data_dir = Path(data_dir or config.DATA_DIR)

    print("Loading player stats...")
    stats = load_player_stats(data_dir)
    print(f"  {len(stats)} player-match rows")

    print("Loading player details...")
    details = load_player_details(data_dir)
    print(f"  {len(details)} detail rows")

    print("Loading scoring events...")
    scoring = load_scoring(data_dir)
    print(f"  {len(scoring)} scoring events")

    # --- Join details onto stats ---
    detail_cols = [
        "match_id", "player", "team",
        "age_years", "career_games", "career_wins", "career_draws",
        "career_losses", "career_win_pct", "career_goals_total",
        "career_goal_avg", "team_games_total", "team_win_pct",
        "team_goals_total", "team_goal_avg",
    ]
    # Keep only the columns we need, deduplicate
    det = details[
        [c for c in detail_cols if c in details.columns]
    ].drop_duplicates(subset=["match_id", "player", "team"])

    df = stats.merge(det, on=["match_id", "player", "team"], how="left")
    print(f"  After details join: {len(df)} rows")

    # --- Aggregate scoring and join ---
    scoring_agg = aggregate_scoring_per_player_match(scoring)
    if not scoring_agg.empty:
        df = df.merge(scoring_agg, on=["match_id", "player", "team"], how="left")
        # Fill missing quarter scoring with 0
        for q in range(1, 5):
            for st in ["goals", "behinds"]:
                col = f"q{q}_{st}"
                if col in df.columns:
                    df[col] = df[col].fillna(0).astype(int)
    else:
        for q in range(1, 5):
            for st in ["goals", "behinds"]:
                df[f"q{q}_{st}"] = 0

    # --- Derived columns ---
    df["is_home"] = (df["home_away"] == "home").astype(int)
    df["is_finals"] = df["round_number"].apply(
        lambda x: 1 if pd.notna(x) and x >= 25 else 0
    )

    # Save
    if save:
        config.ensure_dirs()
        out_path = config.CLEANED_DIR / "player_matches.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Saved cleaned data to {out_path} ({len(df)} rows)")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = build_player_matches()
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nYear range: {df['year'].min()} - {df['year'].max()}")
    print(f"Unique players: {df[['player', 'team']].drop_duplicates().shape[0]}")
    print(f"\nSample row:\n{df.iloc[0].to_dict()}")
