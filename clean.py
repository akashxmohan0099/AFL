"""
AFL Prediction Pipeline — Data Cleaning & Base Store
=====================================================
Loads raw CSVs from the scraper, normalizes names/dates/types,
parses career strings, computes rate-normalised stats, and produces
two optimised parquet stores:

  data/base/player_games.parquet  — one row per (player, team, match_id)
  data/base/matches.parquet       — one row per match_id

All columns are explicitly typed for minimal memory footprint.
"""

import re
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

import config

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# ---------------------------------------------------------------------------
# Stat columns — the 22 raw stats from AFLTables + pct_played
# ---------------------------------------------------------------------------

STAT_COLS = [
    "KI", "MK", "HB", "DI", "GL", "BH", "HO", "TK", "RB", "IF",
    "CL", "CG", "FF", "FA", "BR", "CP", "UP", "CM", "MI",
    "one_pct", "BO", "GA",
]

# Rate-normalised versions (per 90% game time)
RATE_COLS = [f"{c}_rate" for c in STAT_COLS]

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
    m2 = re.match(r"^(\d+)$", goals_str)
    if m2:
        return int(m2.group(1)), np.nan
    return np.nan, np.nan


def parse_career_games(games_str):
    """Parse '178 (112-1-65 63.20%)' → total games (int).
    Returns np.nan on failure."""
    if pd.isna(games_str) or not isinstance(games_str, str):
        return np.nan
    games_str = games_str.strip()
    m = re.match(r"(\d+)\s*\(", games_str)
    if m:
        return int(m.group(1))
    m2 = re.match(r"^(\d+)$", games_str)
    if m2:
        return int(m2.group(1))
    return np.nan


def ensure_date_iso(df):
    """Ensure a 'date_iso' column exists by parsing the 'date' column."""
    if "date_iso" not in df.columns:
        df["date_iso"] = df["date"].apply(_parse_afl_date)
    df["date_iso"] = pd.to_datetime(df["date_iso"], errors="coerce", format="mixed")
    return df


def _parse_afl_date(date_str):
    """Parse AFL Tables date string."""
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
    """Normalize any player name to 'Last, First' format."""
    if pd.isna(name) or not isinstance(name, str):
        return name
    name = name.strip()
    if name in ("Rushed", ""):
        return name
    if "," in name:
        return name
    parts = name.split()
    if len(parts) >= 2:
        return f"{' '.join(parts[1:])}, {parts[0]}"
    return name


FINALS_MAP = {
    # Abbreviations
    "QF": 25, "EF": 25, "SF": 26, "PF": 27, "GF": 28,
    # Full names (as they appear in raw CSVs)
    "QUALIFYING": 25, "ELIMINATION": 25,
    "SEMI": 26,
    "PRELIMINARY": 27,
    "GRAND": 28,
    # Hyphenated variants
    "QUALIFYING FINAL": 25, "ELIMINATION FINAL": 25,
    "SEMI FINAL": 26, "SEMI-FINAL": 26,
    "PRELIMINARY FINAL": 27,
    "GRAND FINAL": 28,
}

# Opening Round (introduced 2026): maps to round_number 0
OPENING_ROUND_STRINGS = {"OPENING ROUND", "OPENING"}

# Canonical round label for finals (abbreviations)
FINALS_LABEL_MAP = {
    25: "EF",   # Elimination/Qualifying both map to 25
    26: "SF",
    27: "PF",
    28: "GF",
}


def parse_round_number(round_str):
    """Convert round string to a sortable integer.
    'Opening Round' → 0,
    '1' → 1, 'Qualifying' → 25, 'Elimination' → 25, 'Semi' → 26,
    'Preliminary' → 27, 'Grand' → 28.
    Also handles abbreviations: QF/EF → 25, SF → 26, PF → 27, GF → 28."""
    if pd.isna(round_str):
        return np.nan
    s = str(round_str).strip()
    try:
        return int(s)
    except ValueError:
        upper = s.upper()
        if upper in OPENING_ROUND_STRINGS:
            return 0
        return FINALS_MAP.get(upper, np.nan)


def is_finals_round(round_str):
    """Check whether a round string represents a finals match.
    Works on the raw round string before numeric conversion."""
    if pd.isna(round_str):
        return False
    s = str(round_str).strip().upper()
    return s in FINALS_MAP


def normalize_round_label(round_str, round_number):
    """Produce a canonical round label.
    Opening Round: '0'.
    Regular rounds: '1', '2', etc.
    Finals: 'EF', 'SF', 'PF', 'GF' (abbreviations)."""
    if pd.isna(round_str):
        return str(int(round_number)) if pd.notna(round_number) else "0"
    s = str(round_str).strip().upper()
    if s in OPENING_ROUND_STRINGS:
        return "0"
    if s in FINALS_MAP:
        rn = FINALS_MAP[s]
        return FINALS_LABEL_MAP.get(rn, s)
    # Regular round — just use the numeric string
    return str(round_str).strip()


def _parse_game_time(time_str):
    """Parse '118m 47s' → 118.78 minutes."""
    if pd.isna(time_str) or not isinstance(time_str, str):
        return np.nan
    m = re.match(r"(\d+)m\s+(\d+)s", time_str.strip())
    if m:
        return int(m.group(1)) + int(m.group(2)) / 60.0
    return np.nan


# ---------------------------------------------------------------------------
# Loading functions — always from year-specific CSVs
# ---------------------------------------------------------------------------

def load_player_stats(data_dir=None):
    """Load all player_stats CSVs (per-year).
    Returns a cleaned DataFrame."""
    data_dir = Path(data_dir or config.DATA_DIR)

    # Load year-specific files
    ps_dir = data_dir / "player_stats"
    files = sorted(ps_dir.glob("player_stats_*.csv"))
    if not files:
        # Fallback to master file if year-specific don't exist
        master = data_dir / "all_player_stats.csv"
        if master.exists():
            files = [master]
        else:
            raise FileNotFoundError(f"No player_stats files in {data_dir}")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files],
                   ignore_index=True)

    # Normalize
    df = ensure_date_iso(df)
    df["player"] = df["player"].apply(normalize_player_name)
    df["round_number"] = df["round"].apply(parse_round_number)

    # Coerce stat columns to numeric
    for col in STAT_COLS + ["pct_played"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df.sort_values(["date_iso", "match_id", "team", "player"]).reset_index(drop=True)
    return df


def load_player_details(data_dir=None):
    """Load all player_details CSVs. Parse career strings."""
    data_dir = Path(data_dir or config.DATA_DIR)

    pd_dir = data_dir / "player_details"
    files = sorted(pd_dir.glob("player_details_*.csv"))
    if not files:
        master = data_dir / "all_player_details.csv"
        if master.exists():
            files = [master]
        else:
            raise FileNotFoundError(f"No player_details files in {data_dir}")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files],
                   ignore_index=True)

    # Filter out coach rows
    df = df[df["jumper"].astype(str).str.strip() != "C"].copy()
    df["player"] = df["player"].apply(normalize_player_name)

    # Parse age
    df["age_years"] = df["Age"].apply(parse_age)

    # Parse career games (just total count)
    df["career_games"] = df["Career Games (W-D-L W%)"].apply(parse_career_games)

    # Parse career goals
    parsed_goals = df["Career Goals (Ave.)"].apply(parse_career_goals)
    df["career_goals_total"] = parsed_goals.apply(lambda x: x[0])
    df["career_goal_avg"] = parsed_goals.apply(lambda x: x[1])

    return df


def load_scoring(data_dir=None):
    """Load all scoring CSVs. Normalize player names, parse quarters."""
    data_dir = Path(data_dir or config.DATA_DIR)

    sc_dir = data_dir / "scoring"
    files = sorted(sc_dir.glob("scoring_*.csv"))
    if not files:
        master = data_dir / "all_scoring.csv"
        if master.exists():
            files = [master]
        else:
            raise FileNotFoundError(f"No scoring files in {data_dir}")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files],
                   ignore_index=True)

    df["player"] = df["player"].apply(normalize_player_name)
    quarter_map = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "Final": 4}
    df["quarter_num"] = df["quarter"].map(quarter_map).fillna(4).astype(int)

    return df


# ---------------------------------------------------------------------------
# Load umpire assignments
# ---------------------------------------------------------------------------

def load_umpires(data_dir=None):
    """Load umpire assignment CSVs (per-year).
    Returns a cleaned DataFrame with one row per (match_id, umpire_name)."""
    data_dir = Path(data_dir or config.DATA_DIR)

    ump_dir = data_dir / "umpires"
    files = sorted(ump_dir.glob("umpires_*.csv"))
    if not files:
        master = data_dir / "all_umpires.csv"
        if master.exists():
            files = [master]
        else:
            return pd.DataFrame()

    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files],
                   ignore_index=True)

    if df.empty:
        return df

    required = {"match_id", "umpire_name", "umpire_career_games"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    # Normalize IDs / names
    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce")
    df = df.dropna(subset=["match_id"]).copy()
    df["match_id"] = df["match_id"].astype(np.int64)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(np.int16)
    else:
        df["year"] = np.int16(0)

    df["umpire_name"] = (
        df["umpire_name"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    invalid_name = (
        df["umpire_name"].isna()
        | (df["umpire_name"] == "")
        | (df["umpire_name"].str.lower() == "nan")
    )
    df = df[~invalid_name].copy()

    # Parse career games and repair sparse zeros/missing values without lookahead:
    # if an umpire has known non-zero values, carry their latest known value forward.
    df["umpire_career_games"] = pd.to_numeric(
        df["umpire_career_games"], errors="coerce"
    )
    df.loc[df["umpire_career_games"] < 0, "umpire_career_games"] = np.nan
    df = df.sort_values(["umpire_name", "year", "match_id"])

    def _repair_games(series):
        s = pd.to_numeric(series, errors="coerce")
        if not (s > 0).any():
            return s.fillna(0)
        repaired = s.mask(s <= 0, np.nan).ffill().fillna(0)
        return repaired.cummax()

    df["umpire_career_games"] = (
        df.groupby("umpire_name", observed=True)["umpire_career_games"]
        .transform(_repair_games)
        .fillna(0)
        .clip(lower=0)
        .round()
        .astype(np.int16)
    )

    # Drop duplicates (same umpire listed twice for a match)
    df = df.drop_duplicates(subset=["match_id", "umpire_name"], keep="last")
    return df


# ---------------------------------------------------------------------------
# Load coach data (from player_details rows with jumper=="C")
# ---------------------------------------------------------------------------

def load_coaches(data_dir=None):
    """Extract coach rows from player_details CSVs.
    Coaches have jumper='C' in the details table.
    Parse career record string into wins/draws/losses/win_pct.
    Returns DataFrame with one row per (match_id, team, coach)."""
    data_dir = Path(data_dir or config.DATA_DIR)

    pd_dir = data_dir / "player_details"
    files = sorted(pd_dir.glob("player_details_*.csv"))
    if not files:
        master = data_dir / "all_player_details.csv"
        if master.exists():
            files = [master]
        else:
            return pd.DataFrame()

    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files],
                   ignore_index=True)

    # Filter to coach rows only (jumper == "C")
    df["jumper"] = df["jumper"].astype(str).str.strip()
    df = df[df["jumper"] == "C"].copy()
    if df.empty:
        return df

    df["coach"] = df["player"].apply(normalize_player_name)

    # Parse career record: "308 (190-3-115 62.18%)"
    def _parse_coach_record(record_str):
        if pd.isna(record_str) or not isinstance(record_str, str):
            return np.nan, np.nan, np.nan, np.nan, np.nan
        record_str = record_str.strip()
        m = re.match(
            r'(\d+)\s*\((\d+)-(\d+)-(\d+)\s+([\d.]+)%\)',
            record_str,
        )
        if m:
            total = int(m.group(1))
            wins = int(m.group(2))
            draws = int(m.group(3))
            losses = int(m.group(4))
            win_pct = float(m.group(5))
            return total, wins, draws, losses, win_pct
        # Try just total
        m2 = re.match(r'^(\d+)$', record_str)
        if m2:
            return int(m2.group(1)), np.nan, np.nan, np.nan, np.nan
        return np.nan, np.nan, np.nan, np.nan, np.nan

    career_col = "Career Games (W-D-L W%)"
    if career_col in df.columns:
        parsed = df[career_col].apply(_parse_coach_record)
        df["coach_career_games"] = parsed.apply(lambda x: x[0])
        df["coach_wins"] = parsed.apply(lambda x: x[1])
        df["coach_draws"] = parsed.apply(lambda x: x[2])
        df["coach_losses"] = parsed.apply(lambda x: x[3])
        df["coach_win_pct"] = parsed.apply(lambda x: x[4])
    else:
        for c in ["coach_career_games", "coach_wins", "coach_draws",
                   "coach_losses", "coach_win_pct"]:
            df[c] = np.nan

    # Select output columns
    keep_cols = ["match_id", "year", "team", "coach",
                 "coach_career_games", "coach_wins", "coach_draws",
                 "coach_losses", "coach_win_pct"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    df = df.drop_duplicates(subset=["match_id", "team"])
    return df


# ---------------------------------------------------------------------------
# Load player profiles (height/weight/DOB from scraped profile pages)
# ---------------------------------------------------------------------------

def load_player_profiles(data_dir=None):
    """Load player physical attribute profiles.
    Returns DataFrame with one row per player."""
    data_dir = Path(data_dir or config.DATA_DIR)

    profiles_path = data_dir / "player_profiles" / "profiles.csv"
    if not profiles_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(profiles_path)
    if df.empty:
        return df

    df["player"] = df["player"].apply(normalize_player_name)

    # Parse DOB
    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], format="%d-%b-%Y", errors="coerce")

    # Cast physical attributes
    for col in ["height_cm", "weight_kg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    df = df.drop_duplicates(subset=["player"], keep="last")
    return df


# ---------------------------------------------------------------------------
# Load career split tables
# ---------------------------------------------------------------------------

def load_career_splits(data_dir=None):
    """Load player career splits by opponent and venue.
    Returns (vs_opponent_df, vs_venue_df)."""
    data_dir = Path(data_dir or config.DATA_DIR)
    profiles_dir = data_dir / "player_profiles"

    opp_path = profiles_dir / "player_vs_opponent.csv"
    venue_path = profiles_dir / "player_vs_venue.csv"

    opp_df = pd.DataFrame()
    venue_df = pd.DataFrame()

    if opp_path.exists():
        opp_df = pd.read_csv(opp_path)
        if not opp_df.empty:
            opp_df["player"] = opp_df["player"].apply(normalize_player_name)
            # Normalize opponent names via config map
            if "opponent" in opp_df.columns:
                opp_df["opponent"] = opp_df["opponent"].map(
                    config.TEAM_NAME_MAP
                ).fillna(opp_df["opponent"])
            # Cast stat columns to numeric
            for c in ["P", "KI", "MK", "HB", "DI", "GL", "BH", "HO", "TK"]:
                if c in opp_df.columns:
                    opp_df[c] = pd.to_numeric(opp_df[c], errors="coerce")

    if venue_path.exists():
        venue_df = pd.read_csv(venue_path)
        if not venue_df.empty:
            venue_df["player"] = venue_df["player"].apply(normalize_player_name)
            # Normalize venue names via config map
            if "venue" in venue_df.columns:
                venue_df["venue"] = venue_df["venue"].map(
                    config.VENUE_NAME_MAP
                ).fillna(venue_df["venue"])
            for c in ["P", "KI", "MK", "HB", "DI", "GL", "BH", "HO", "TK"]:
                if c in venue_df.columns:
                    venue_df[c] = pd.to_numeric(venue_df[c], errors="coerce")

    return opp_df, venue_df


# ---------------------------------------------------------------------------
# Aggregation: scoring events → per-player-match-quarter counts
# ---------------------------------------------------------------------------

def aggregate_scoring_per_player_match(scoring_df):
    """Pivot scoring events into per-(player, team, match_id) quarter counts."""
    player_scoring = scoring_df[scoring_df["player"] != "Rushed"].copy()
    if player_scoring.empty:
        return pd.DataFrame()

    agg = (
        player_scoring
        .groupby(["match_id", "player", "team", "quarter_num", "score_type"])
        .size()
        .reset_index(name="count")
    )

    pivoted = agg.pivot_table(
        index=["match_id", "player", "team"],
        columns=["quarter_num", "score_type"],
        values="count",
        fill_value=0,
    )

    pivoted.columns = [f"q{q}_{st}s" for q, st in pivoted.columns]
    pivoted = pivoted.reset_index()

    for q in range(1, 5):
        for st in ["goals", "behinds"]:
            col = f"q{q}_{st}"
            if col not in pivoted.columns:
                pivoted[col] = 0

    return pivoted


# ---------------------------------------------------------------------------
# Rate normalisation
# ---------------------------------------------------------------------------

def add_rate_columns(df):
    """Add %P-normalised rate columns for all stats.
    Rate = stat / (pct_played / 90.0), normalised to 90% game time as reference."""
    pct = df["pct_played"].values
    # Avoid division by zero for players with 0% played
    divisor = np.where(pct > 0, pct / 90.0, np.nan)

    for col in STAT_COLS:
        if col in df.columns:
            rate_col = f"{col}_rate"
            df[rate_col] = (df[col].values / divisor).astype(np.float32)
            # Fill NaN rates (0% played) with 0
            df[rate_col] = df[rate_col].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Dtype optimisation
# ---------------------------------------------------------------------------

def optimize_dtypes(df):
    """Downcast all columns to their minimal types."""

    # Integer downcasts
    int8_cols = ["jumper", "q1_goals", "q1_behinds", "q2_goals", "q2_behinds",
                 "q3_goals", "q3_behinds", "q4_goals", "q4_behinds"]
    int16_cols = ["year", "career_games", "career_goals_total",
                   "career_games_pre", "career_goals_pre"]
    int32_cols = []  # match_id stays int64 (values exceed int32 range)

    for col in int8_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(np.int8)
    for col in int16_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(np.int16)
    for col in int32_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.int32)

    # round_number: int8 (1-28, with NaN → 0)
    if "round_number" in df.columns:
        df["round_number"] = df["round_number"].fillna(0).astype(np.int8)

    # Boolean columns
    for col in ["is_home", "is_finals", "did_not_play"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Float32 for all stat and rate columns
    float32_cols = STAT_COLS + RATE_COLS + ["pct_played", "age_years",
                   "career_goal_avg", "career_goal_avg_pre"]
    for col in float32_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    # Category columns
    cat_cols = ["player", "team", "opponent", "venue", "round_label", "player_id"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


# ---------------------------------------------------------------------------
# Match store builder
# ---------------------------------------------------------------------------

def build_match_store(player_games_df, data_dir=None):
    """Build match-level aggregates from player data + match metadata.
    Returns a DataFrame with one row per match_id."""
    data_dir = Path(data_dir or config.DATA_DIR)

    # Aggregate player stats per (match_id, team)
    agg_cols = ["GL", "BH", "DI", "IF", "CL", "CP", "TK", "RB", "MK"]
    team_agg = (
        player_games_df.groupby(["match_id", "team", "is_home"], observed=True)
        [agg_cols]
        .sum()
        .reset_index()
    )

    # Split into home and away
    home = team_agg[team_agg["is_home"]].drop(columns=["is_home"]).copy()
    away = team_agg[~team_agg["is_home"]].drop(columns=["is_home"]).copy()

    # Rename columns
    home = home.rename(columns={c: f"home_total_{c}" for c in agg_cols})
    home = home.rename(columns={"team": "home_team"})
    away = away.rename(columns={c: f"away_total_{c}" for c in agg_cols})
    away = away.rename(columns={"team": "away_team"})

    # Merge on match_id
    matches = home.merge(away, on="match_id", how="outer")

    # Add match-level info from player data (take first row per match)
    match_info = (
        player_games_df.groupby("match_id")
        .agg(
            year=("year", "first"),
            round_number=("round_number", "first"),
            date=("date", "first"),
            venue=("venue", "first"),
            is_finals=("is_finals", "first"),
            round_label=("round_label", "first"),
        )
        .reset_index()
    )
    matches = matches.merge(match_info, on="match_id", how="left")

    # Compute derived columns
    matches["home_score"] = (matches["home_total_GL"] * 6 + matches["home_total_BH"]).astype(np.int16)
    matches["away_score"] = (matches["away_total_GL"] * 6 + matches["away_total_BH"]).astype(np.int16)
    matches["margin"] = (matches["home_score"] - matches["away_score"]).astype(np.int16)
    matches["total_score"] = (matches["home_score"] + matches["away_score"]).astype(np.int16)
    matches["total_DI"] = (matches["home_total_DI"] + matches["away_total_DI"]).astype(np.int16)
    matches["total_IF"] = (matches["home_total_IF"] + matches["away_total_IF"]).astype(np.int16)
    matches["total_CP"] = (matches["home_total_CP"] + matches["away_total_CP"]).astype(np.int16)

    # Try to merge attendance and game_time from matches CSV
    matches_csv = data_dir / "all_matches.csv"
    if matches_csv.exists():
        meta = pd.read_csv(matches_csv, low_memory=False)
        meta["match_id"] = pd.to_numeric(meta["match_id"], errors="coerce")
        meta["attendance"] = pd.to_numeric(meta["attendance"], errors="coerce").fillna(0).astype(np.int32)
        if "game_time" in meta.columns:
            meta["game_time_minutes"] = meta["game_time"].apply(_parse_game_time).astype(np.float32)
        else:
            meta["game_time_minutes"] = np.float32(0)
        # Rushed behinds (defensive pressure indicator)
        for rb_col in ["home_rushed_behinds", "away_rushed_behinds"]:
            if rb_col in meta.columns:
                meta[rb_col] = pd.to_numeric(meta[rb_col], errors="coerce").fillna(0).astype(np.int8)
        keep_cols = ["match_id", "attendance", "game_time_minutes"]
        for rb_col in ["home_rushed_behinds", "away_rushed_behinds"]:
            if rb_col in meta.columns:
                keep_cols.append(rb_col)
        meta_cols = meta[keep_cols].drop_duplicates("match_id")
        matches = matches.merge(meta_cols, on="match_id", how="left")
    else:
        # Try year-specific match files
        m_dir = data_dir / "matches"
        m_files = sorted(m_dir.glob("matches_*.csv"))
        if m_files:
            meta = pd.concat([pd.read_csv(f, low_memory=False) for f in m_files], ignore_index=True)
            meta["match_id"] = pd.to_numeric(meta["match_id"], errors="coerce")
            meta["attendance"] = pd.to_numeric(meta["attendance"], errors="coerce").fillna(0).astype(np.int32)
            if "game_time" in meta.columns:
                meta["game_time_minutes"] = meta["game_time"].apply(_parse_game_time).astype(np.float32)
            else:
                meta["game_time_minutes"] = np.float32(0)
            for rb_col in ["home_rushed_behinds", "away_rushed_behinds"]:
                if rb_col in meta.columns:
                    meta[rb_col] = pd.to_numeric(meta[rb_col], errors="coerce").fillna(0).astype(np.int8)
            keep_cols = ["match_id", "attendance", "game_time_minutes"]
            for rb_col in ["home_rushed_behinds", "away_rushed_behinds"]:
                if rb_col in meta.columns:
                    keep_cols.append(rb_col)
            meta_cols = meta[keep_cols].drop_duplicates("match_id")
            matches = matches.merge(meta_cols, on="match_id", how="left")

    # Optimize dtypes (match_id stays int64 — values exceed int32 range)
    matches["year"] = matches["year"].astype(np.int16)
    matches["round_number"] = matches["round_number"].fillna(0).astype(np.int8)
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    if "is_finals" in matches.columns:
        matches["is_finals"] = matches["is_finals"].astype(bool)
    if "round_label" in matches.columns:
        matches["round_label"] = matches["round_label"].astype("category")
    for col in ["venue", "home_team", "away_team"]:
        if col in matches.columns:
            matches[col] = matches[col].astype("category")
    for col in agg_cols:
        for prefix in ["home_total_", "away_total_"]:
            c = f"{prefix}{col}"
            if c in matches.columns:
                matches[c] = matches[c].fillna(0).astype(np.int16)

    matches = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
    return matches


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_player_games(data_dir=None, save=True):
    """Load all raw data, normalize, join, and produce the clean base stores.

    Outputs:
      data/base/player_games.parquet  — player-level, with rate columns
      data/base/matches.parquet       — match-level aggregates

    Also writes legacy data/cleaned/player_matches.parquet for backward compat.
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
        "age_years", "career_games", "career_goals_total", "career_goal_avg",
    ]
    det = details[
        [c for c in detail_cols if c in details.columns]
    ].drop_duplicates(subset=["match_id", "player", "team"])

    df = stats.merge(det, on=["match_id", "player", "team"], how="left")
    print(f"  After details join: {len(df)} rows")

    # --- Aggregate scoring and join ---
    scoring_agg = aggregate_scoring_per_player_match(scoring)
    if not scoring_agg.empty:
        df = df.merge(scoring_agg, on=["match_id", "player", "team"], how="left")
        for q in range(1, 5):
            for st in ["goals", "behinds"]:
                col = f"q{q}_{st}"
                if col in df.columns:
                    df[col] = df[col].fillna(0).astype(int)
    else:
        for q in range(1, 5):
            for st in ["goals", "behinds"]:
                df[f"q{q}_{st}"] = 0

    # --- Fix career stat leakage: AFLTables career stats include current game ---
    # Compute pre-game versions (what was known BEFORE this match started)
    df["career_games_pre"] = (df["career_games"] - 1).clip(lower=0)
    df["career_goals_pre"] = (df["career_goals_total"] - df["GL"]).clip(lower=0)
    df["career_goal_avg_pre"] = np.where(
        df["career_games_pre"] > 0,
        df["career_goals_pre"] / df["career_games_pre"],
        np.nan,
    )

    # --- Derived columns ---
    df["is_home"] = (df["home_away"] == "home").astype(int)
    # Derive is_finals from the raw round string (not from round_number,
    # which loses the distinction between R25 H&A and finals)
    df["is_finals"] = df["round"].apply(is_finals_round).astype(int)
    # Canonical round labels: "1"-"25" for H&A, "EF"/"SF"/"PF"/"GF" for finals
    df["round_label"] = df.apply(
        lambda row: normalize_round_label(row["round"], row["round_number"]),
        axis=1,
    )

    # Player ID: disambiguate same-name players across teams
    df["player_id"] = df["player"].astype(str) + "_" + df["team"].astype(str)

    # Flag rows with 0% time on ground (medical subs, emergencies)
    df["did_not_play"] = (df["pct_played"] == 0)

    # --- Season era / rule regime ---
    df["season_era"] = df["year"].map(config.ERA_MAP).fillna(
        config.CURRENT_PREDICTION_ERA
    ).astype(np.int8)
    df["is_covid_season"] = (df["year"] == config.COVID_SEASON_YEAR).astype(np.int8)
    df["quarter_length_ratio"] = np.where(
        df["year"] == config.COVID_SEASON_YEAR,
        config.COVID_QUARTER_LENGTH_RATIO, 1.0
    ).astype(np.float32)

    # --- Add rate-normalised columns ---
    print("  Computing rate-normalised stats...")
    df = add_rate_columns(df)

    # --- Drop columns not needed in the base store ---
    drop_cols = [
        "date",  # raw string date; we keep date_iso (datetime)
        "round",  # raw string; we keep round_number (int) + round_label (category)
        "home_away",  # replaced by is_home bool
        "sub_status",  # rarely populated
        # Raw career strings
        "Age", "Career Games (W-D-L W%)", "Career Goals (Ave.)",
        "team_games", "team_goals",
        # Leaky career stats (team-level)
        "team_games_total", "team_win_pct", "team_goals_total", "team_goal_avg",
        # Leaky career stats (player-level — include current game's data)
        "career_games", "career_goals_total", "career_goal_avg",
        # Derivable career breakdown
        "career_wins", "career_draws", "career_losses", "career_win_pct",
    ]
    existing_drops = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drops, errors="ignore")

    # Rename date_iso → date (the only date column remaining)
    if "date_iso" in df.columns:
        df = df.rename(columns={"date_iso": "date"})

    # Validate before dtype optimization
    from validate import validate_cleaned
    validate_cleaned(df)

    # --- Optimize dtypes ---
    print("  Optimizing dtypes...")
    df = optimize_dtypes(df)

    # --- Save ---
    if save:
        config.ensure_dirs()

        # Base store
        base_path = config.BASE_STORE_DIR / "player_games.parquet"
        df.to_parquet(base_path, index=False)
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        print(f"  Saved {base_path} ({len(df)} rows, {len(df.columns)} cols, {mem_mb:.1f} MB memory)")

        # Match store
        print("  Building match store...")
        matches_df = build_match_store(df, data_dir)
        matches_path = config.BASE_STORE_DIR / "matches.parquet"
        matches_df.to_parquet(matches_path, index=False)
        print(f"  Saved {matches_path} ({len(matches_df)} rows)")

        # Team-match table (for game winner model)
        print("  Building team-match table...")
        team_match_df = build_team_match_table(matches_df)
        team_match_path = config.BASE_STORE_DIR / "team_matches.parquet"
        team_match_df.to_parquet(team_match_path, index=False)
        print(f"  Saved {team_match_path} ({len(team_match_df)} rows)")

        # Umpire store
        print("  Loading umpire data...")
        umpires_df = load_umpires(data_dir)
        if not umpires_df.empty:
            from validate import validate_umpires
            validate_umpires(umpires_df)
            umpires_path = config.BASE_STORE_DIR / "umpires.parquet"
            umpires_df.to_parquet(umpires_path, index=False)
            print(f"  Saved {umpires_path} ({len(umpires_df)} rows)")
        else:
            print("  No umpire data found — skipping")

        # Coach store
        print("  Loading coach data...")
        coaches_df = load_coaches(data_dir)
        if not coaches_df.empty:
            from validate import validate_coaches
            validate_coaches(coaches_df)
            coaches_path = config.BASE_STORE_DIR / "coaches.parquet"
            coaches_df.to_parquet(coaches_path, index=False)
            print(f"  Saved {coaches_path} ({len(coaches_df)} rows)")
        else:
            print("  No coach data found — skipping")

        # Player profiles store
        print("  Loading player profiles...")
        profiles_df = load_player_profiles(data_dir)
        if not profiles_df.empty:
            from validate import validate_player_profiles
            validate_player_profiles(profiles_df)
            profiles_path = config.BASE_STORE_DIR / "player_profiles.parquet"
            profiles_df.to_parquet(profiles_path, index=False)
            print(f"  Saved {profiles_path} ({len(profiles_df)} rows)")
        else:
            print("  No player profile data found — skipping")

        # Career splits store
        print("  Loading career splits...")
        opp_splits_df, venue_splits_df = load_career_splits(data_dir)
        if not opp_splits_df.empty:
            opp_path = config.BASE_STORE_DIR / "career_splits_opponent.parquet"
            opp_splits_df.to_parquet(opp_path, index=False)
            print(f"  Saved {opp_path} ({len(opp_splits_df)} rows)")
        if not venue_splits_df.empty:
            venue_path = config.BASE_STORE_DIR / "career_splits_venue.parquet"
            venue_splits_df.to_parquet(venue_path, index=False)
            print(f"  Saved {venue_path} ({len(venue_splits_df)} rows)")
        if opp_splits_df.empty and venue_splits_df.empty:
            print("  No career split data found — skipping")

    return df


# ---------------------------------------------------------------------------
# Team-match table builder (for game winner model)
# ---------------------------------------------------------------------------

def build_team_match_table(matches_df):
    """Build a team-level per-match table from match-level data.

    One row per (team, match) with: result (W/L/D), score, margin,
    rest_days, is_home, venue, stat totals, opponent info.

    Returns DataFrame with columns:
      match_id, team, opponent, date, year, round_number, venue, is_home,
      score, opp_score, margin, result (W/L/D), rest_days,
      stat totals (GL, BH, DI, IF, CL, CP, TK, RB, MK)
    """
    rows = []
    agg_stats = ["GL", "BH", "DI", "IF", "CL", "CP", "TK", "RB", "MK"]

    for _, m in matches_df.iterrows():
        shared = {
            "match_id": m["match_id"],
            "date": m["date"],
            "year": m["year"],
            "round_number": m["round_number"],
            "venue": m["venue"],
            "attendance": m.get("attendance", 0),
            "is_finals": m.get("is_finals", False),
        }

        # Home team row
        home_row = {
            **shared,
            "team": m["home_team"],
            "opponent": m["away_team"],
            "is_home": True,
            "score": m["home_score"],
            "opp_score": m["away_score"],
            "margin": m["margin"],
        }
        for stat in agg_stats:
            home_col = f"home_total_{stat}"
            if home_col in m.index:
                home_row[stat] = m[home_col]

        # Result
        if m["margin"] > 0:
            home_row["result"] = "W"
        elif m["margin"] < 0:
            home_row["result"] = "L"
        else:
            home_row["result"] = "D"
        rows.append(home_row)

        # Away team row
        away_row = {
            **shared,
            "team": m["away_team"],
            "opponent": m["home_team"],
            "is_home": False,
            "score": m["away_score"],
            "opp_score": m["home_score"],
            "margin": -m["margin"],
        }
        for stat in agg_stats:
            away_col = f"away_total_{stat}"
            if away_col in m.index:
                away_row[stat] = m[away_col]

        if m["margin"] < 0:
            away_row["result"] = "W"
        elif m["margin"] > 0:
            away_row["result"] = "L"
        else:
            away_row["result"] = "D"
        rows.append(away_row)

    team_df = pd.DataFrame(rows)
    team_df = team_df.sort_values(["date", "match_id", "team"]).reset_index(drop=True)

    # Compute rest days per team
    team_df["date"] = pd.to_datetime(team_df["date"])
    team_df["rest_days"] = (
        team_df.groupby("team")["date"]
        .transform(lambda s: s.diff().dt.days)
    )

    # Optimize dtypes
    team_df["is_home"] = team_df["is_home"].astype(bool)
    team_df["is_finals"] = team_df["is_finals"].astype(bool)
    team_df["score"] = team_df["score"].astype(np.int16)
    team_df["opp_score"] = team_df["opp_score"].astype(np.int16)
    team_df["margin"] = team_df["margin"].astype(np.int16)
    team_df["year"] = team_df["year"].astype(np.int16)
    team_df["round_number"] = team_df["round_number"].astype(np.int8)
    team_df["attendance"] = team_df["attendance"].fillna(0).astype(np.int32)
    for col in ["team", "opponent", "venue", "result"]:
        team_df[col] = team_df[col].astype("category")
    for stat in agg_stats:
        if stat in team_df.columns:
            team_df[stat] = team_df[stat].fillna(0).astype(np.int16)

    return team_df


# Keep backward-compatible alias
build_player_matches = build_player_games


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = build_player_games()
    print(f"\nDataset shape: {df.shape}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"Unique players: {df['player'].nunique()}")

    print(f"\nColumn dtypes:")
    for col in sorted(df.columns):
        print(f"  {col:30s}  {str(df[col].dtype):15s}")

    print(f"\nMemory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Verify rate columns
    rate_present = [c for c in df.columns if c.endswith("_rate")]
    print(f"\nRate columns: {len(rate_present)}")
    if rate_present:
        print(f"  Sample GL_rate: {df['GL_rate'].describe().to_dict()}")


# ---------------------------------------------------------------------------
# FootyWire Advanced Stats — Build Parquet
# ---------------------------------------------------------------------------

def _normalize_footywire_name(name):
    """Convert FootyWire 'I Heeney' format to pipeline 'Heeney, I' for matching."""
    if pd.isna(name) or not isinstance(name, str):
        return name
    name = name.strip()
    # Remove sub indicators
    name = re.sub(r"[↗↙]", "", name).strip()
    if not name:
        return name
    parts = name.split()
    if len(parts) >= 2:
        # "I Heeney" → "Heeney, I"
        first_initial = parts[0]
        last_name = " ".join(parts[1:])
        return f"{last_name}, {first_initial}"
    return name


def build_footywire_parquet():
    """Load FootyWire CSVs, match to pipeline players, save as parquet.

    Matches FootyWire rows to pipeline by (date, team, player_name).
    FootyWire uses 'First Last' names; pipeline uses 'Last, First'.
    We match on last name + first initial for robustness.

    Output: data/base/footywire_advanced.parquet keyed by (match_id, player)
    """
    fw_dir = config.FOOTYWIRE_DIR
    if not fw_dir.exists():
        print("  FootyWire directory not found — skipping")
        return None

    csv_files = sorted(fw_dir.glob("advanced_stats_*.csv"))
    if not csv_files:
        print("  No FootyWire CSV files found — run --scrape-footywire first")
        return None

    # Load all FootyWire CSVs
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)
    fw = pd.concat(dfs, ignore_index=True)

    print(f"  Loaded {len(fw)} FootyWire rows from {len(csv_files)} files")

    # Normalize player names for matching
    fw["player_match_key"] = fw["player"].apply(_normalize_footywire_name)

    # Clean team names using config map
    team_map = config.TEAM_NAME_MAP.copy()
    # FootyWire-specific aliases
    fw_team_map = {
        "GWS": "Greater Western Sydney",
        "GWS Giants": "Greater Western Sydney",
        "Footscray": "Western Bulldogs",
        "Brisbane": "Brisbane Lions",
        "Kangaroos": "North Melbourne",
    }
    team_map.update(fw_team_map)
    fw["team_clean"] = fw["team"].map(team_map).fillna(fw["team"])

    # Parse dates from FootyWire format
    # Format: "Thursday, 7th March 2024"
    def _parse_fw_date(d):
        if pd.isna(d):
            return pd.NaT
        d = str(d).strip()
        # Remove day name and ordinal suffixes
        d = re.sub(r"^\w+day,?\s*", "", d)
        d = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", d)
        try:
            return pd.to_datetime(d, format="%d %B %Y")
        except Exception:
            return pd.NaT

    fw["date_parsed"] = fw["date"].apply(_parse_fw_date)

    # Load pipeline player_games for matching
    pg_path = config.BASE_STORE_DIR / "player_games.parquet"
    if not pg_path.exists():
        print("  player_games.parquet not found — run --clean first")
        return None

    pg = pd.read_parquet(pg_path, columns=["match_id", "player", "team", "date", "year"])
    pg["date"] = pd.to_datetime(pg["date"])
    pg["date_only"] = pg["date"].dt.date

    # Create matching key: last name + first initial from pipeline names
    def _pipeline_match_key(name):
        if pd.isna(name):
            return name
        # "Heeney, Isaac" → "Heeney, I"
        if "," in str(name):
            parts = str(name).split(",")
            last = parts[0].strip()
            first = parts[1].strip()
            return f"{last}, {first[0]}" if first else name
        return name

    pg["player_match_key"] = pg["player"].apply(_pipeline_match_key)

    # Build join keys
    fw["date_only"] = fw["date_parsed"].dt.date

    # Merge on (date_only, team, player_match_key)
    merged = fw.merge(
        pg[["match_id", "player", "team", "date_only", "player_match_key"]],
        left_on=["date_only", "team_clean", "player_match_key"],
        right_on=["date_only", "team", "player_match_key"],
        how="inner",
        suffixes=("_fw", "_pg"),
    )

    n_matched = len(merged)
    n_total = len(fw)
    match_rate = n_matched / n_total * 100 if n_total > 0 else 0
    print(f"  Matched {n_matched}/{n_total} FootyWire rows ({match_rate:.1f}%)")

    if merged.empty:
        print("  WARNING: No matches found — check team name mapping")
        return None

    # Select output columns — use player_pg (pipeline canonical name)
    # After merge with suffixes, player column becomes player_fw and player_pg
    player_col = "player_pg" if "player_pg" in merged.columns else "player"
    keep_stats = ["ED", "DE_pct", "CCL", "SCL", "TO", "MG", "SI", "ITC", "T5", "TOG_pct"]
    result = merged[["match_id", player_col] + keep_stats].copy()
    result = result.rename(columns={player_col: "player"})

    # Convert stats to numeric
    for col in keep_stats:
        result[col] = pd.to_numeric(result[col], errors="coerce").astype("float32")

    # Deduplicate — keep first occurrence per (match_id, player)
    result = result.drop_duplicates(subset=["match_id", "player"], keep="first")

    # Save
    out_path = config.BASE_STORE_DIR / "footywire_advanced.parquet"
    result.to_parquet(out_path, index=False)
    print(f"  Saved {len(result)} rows to {out_path}")
    print(f"  Columns: {keep_stats}")

    return result
