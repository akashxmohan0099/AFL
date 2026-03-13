"""
AFL Prediction Pipeline — News / Team Changes Fetcher
======================================================
Scrapes team lineups (ins/outs/emergencies) from FootyWire and injury lists
from AFL.com.au. Derives structured features for the prediction model.

Stages:
  1. Scrape team selections from FootyWire (per round)
  2. Scrape injury list from AFL.com.au (current snapshot)
  3. Build structured parquets (team_changes.parquet, injuries.parquet)
  4. Derive news features for the feature matrix

Usage:
    python news.py scrape --year 2026                  # Scrape latest team selections
    python news.py scrape --year 2026 --round 5        # Scrape specific round
    python news.py injuries                            # Scrape current injury list
    python news.py build --year 2026                   # Build parquets from cached data
    python news.py features                            # Show derived feature summary
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NEWS_DIR = config.DATA_DIR / "news"
TEAM_LISTS_DIR = NEWS_DIR / "team_lists"
INJURIES_DIR = NEWS_DIR / "injuries"
ARTICLES_DIR = NEWS_DIR / "articles"
INTEL_DIR = NEWS_DIR / "intel"

REQUEST_DELAY = 2.0
HEADERS = {
    "User-Agent": "AFL-Stats-Research-Bot/1.0 (personal research project)"
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# FootyWire team name -> pipeline canonical name (reuse from scrape_footywire_live)
# Includes both readable names and FootyWire slug->Title variants
TEAM_MAP = {
    "Adelaide": "Adelaide",
    "Adelaide Crows": "Adelaide",
    "Brisbane Lions": "Brisbane Lions",
    "Brisbane": "Brisbane Lions",
    "Carlton": "Carlton",
    "Carlton Blues": "Carlton",
    "Collingwood": "Collingwood",
    "Collingwood Magpies": "Collingwood",
    "Essendon": "Essendon",
    "Essendon Bombers": "Essendon",
    "Fremantle": "Fremantle",
    "Fremantle Dockers": "Fremantle",
    "Geelong": "Geelong",
    "Geelong Cats": "Geelong",
    "Gold Coast": "Gold Coast",
    "Gold Coast Suns": "Gold Coast",
    "GWS": "Greater Western Sydney",
    "GWS Giants": "Greater Western Sydney",
    "Greater Western Sydney": "Greater Western Sydney",
    "Greater Western Sydney Giants": "Greater Western Sydney",
    "Hawthorn": "Hawthorn",
    "Hawthorn Hawks": "Hawthorn",
    "Melbourne": "Melbourne",
    "Melbourne Demons": "Melbourne",
    "North Melbourne": "North Melbourne",
    "North Melbourne Kangaroos": "North Melbourne",
    "Port Adelaide": "Port Adelaide",
    "Port Adelaide Power": "Port Adelaide",
    "Richmond": "Richmond",
    "Richmond Tigers": "Richmond",
    "St Kilda": "St Kilda",
    "St Kilda Saints": "St Kilda",
    "Sydney": "Sydney",
    "Sydney Swans": "Sydney",
    "West Coast": "West Coast",
    "West Coast Eagles": "West Coast",
    "Western Bulldogs": "Western Bulldogs",
}

# AFL.com.au team name -> pipeline canonical name
# Includes exact strings as they appear on the injury list page
AFL_TEAM_MAP = {
    "Adelaide Crows": "Adelaide",
    "Brisbane Lions": "Brisbane Lions",
    "Brisbane": "Brisbane Lions",
    "Carlton": "Carlton",
    "Collingwood": "Collingwood",
    "Essendon": "Essendon",
    "Fremantle": "Fremantle",
    "Geelong Cats": "Geelong",
    "Geelong": "Geelong",
    "Gold Coast Suns": "Gold Coast",
    "Gold Coast SUNS": "Gold Coast",
    "GWS Giants": "Greater Western Sydney",
    "GWS GIANTS": "Greater Western Sydney",
    "Hawthorn": "Hawthorn",
    "Melbourne": "Melbourne",
    "North Melbourne": "North Melbourne",
    "Port Adelaide": "Port Adelaide",
    "Richmond": "Richmond",
    "St Kilda": "St Kilda",
    "Sydney Swans": "Sydney",
    "Sydney": "Sydney",
    "West Coast Eagles": "West Coast",
    "West Coast": "West Coast",
    "Western Bulldogs": "Western Bulldogs",
}

# Ordered list of teams as they appear on the AFL.com.au injury page (alphabetical)
AFL_INJURY_TEAM_ORDER = [
    "Adelaide", "Brisbane Lions", "Carlton", "Collingwood",
    "Essendon", "Fremantle", "Geelong", "Gold Coast",
    "Greater Western Sydney", "Hawthorn", "Melbourne", "North Melbourne",
    "Port Adelaide", "Richmond", "St Kilda", "Sydney",
    "West Coast", "Western Bulldogs",
]


def _fetch(url, retries=3):
    """Fetch URL with retries and polite delays."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.text
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(REQUEST_DELAY * (attempt + 2))
            else:
                log.error(f"Failed: {url}: {e}")
                return None


def _normalize_player_name(name):
    """Normalize player name to 'Last, First' format."""
    name = name.strip()
    # Already in "Last, First" format
    if "," in name:
        return name
    # "First Last" -> "Last, First"
    parts = name.rsplit(" ", 1)
    if len(parts) == 2:
        return f"{parts[1]}, {parts[0]}"
    return name


def _ensure_dirs():
    """Create news data directories."""
    for d in [NEWS_DIR, TEAM_LISTS_DIR, INJURIES_DIR, ARTICLES_DIR, INTEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Stage 1: Scrape Team Selections from FootyWire
# ---------------------------------------------------------------------------

TEAM_SELECTIONS_URL = "https://www.footywire.com/afl/footy/afl_team_selections"


def scrape_team_selections(year=None):
    """Scrape current team selections from FootyWire.

    The page shows the latest round's team selections with:
    - Selected 18 + interchange + emergencies per team
    - Player positions (FB, HB, C, HF, FF, Fol)

    Returns:
        dict with keys:
            - round_number: int
            - year: int
            - teams: list of dicts per team with:
                - team: str (canonical name)
                - selected: list of player names (starting 18)
                - interchange: list of player names
                - emergencies: list of player names
    """
    _ensure_dirs()

    html = _fetch(TEAM_SELECTIONS_URL)
    if not html:
        log.error("Failed to fetch team selections page")
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Extract round info from page title/heading, then fallback to body text
    title = soup.find("title")
    title_text = title.get_text(strip=True) if title else ""
    round_match = re.search(r"Round\s+(\d+)", title_text)
    if not round_match:
        # Fallback: search headings and body text for round number
        page_text = soup.get_text()
        round_match = re.search(r"Round\s+(\d+)", page_text)
    round_num = int(round_match.group(1)) if round_match else 0

    year_match = re.search(r"(\d{4})", title_text)
    detected_year = int(year_match.group(1)) if year_match else (year or config.CURRENT_SEASON_YEAR)
    if year is None:
        year = detected_year

    log.info(f"Parsing team selections for {year} Round {round_num}")

    # Parse match blocks - FootyWire uses tables for team selections.
    # The page has two table types per match:
    #   1. Interchange/Emergencies/Ins/Outs table (links with <b> section headers)
    #   2. Position grid table (FB/HB/C/HF/FF/Fol positions = Selected 18)
    # We walk all tables, tracking section via <b> labels to correctly assign players.
    teams_data = []

    POSITION_LABELS = {"FB", "HB", "C", "HF", "FF", "Fol", "R", "RR", "IC"}
    SECTION_LABELS = {"Interchange", "Emergencies", "Ins", "Outs"}

    # Collect players per team per section by walking tables in document order
    team_sections = {}  # team_slug -> {selected: [], interchange: [], emergencies: []}

    for table in soup.find_all("table"):
        section = "pre"
        for elem in table.descendants:
            if not hasattr(elem, "name"):
                continue
            if elem.name == "b":
                text = elem.get_text(strip=True)
                if text in SECTION_LABELS:
                    section = text
                elif text in POSITION_LABELS:
                    section = "selected"
            elif elem.name == "a":
                href = elem.get("href", "")
                if not href.startswith("pp-") and "/pp-" not in href:
                    continue
                href_clean = href.split("/")[-1] if "/" in href else href
                if not href_clean.startswith("pp-"):
                    continue
                parts = href_clean[3:].split("--")
                if len(parts) != 2:
                    continue
                team_slug = parts[0]
                player_name = elem.get_text(strip=True)
                if not player_name:
                    continue

                if team_slug not in team_sections:
                    team_sections[team_slug] = {
                        "selected": [], "interchange": [], "emergencies": [],
                    }

                if section == "selected":
                    team_sections[team_slug]["selected"].append(player_name)
                elif section == "Interchange":
                    team_sections[team_slug]["interchange"].append(player_name)
                elif section == "Emergencies":
                    team_sections[team_slug]["emergencies"].append(player_name)
                # Skip "Ins" and "Outs" — they're change tracking, not squad membership

    # Map team slugs to canonical names and build output
    for slug, sections in team_sections.items():
        readable = slug.replace("-", " ").title()
        canonical = TEAM_MAP.get(readable)
        if not canonical:
            for key, val in TEAM_MAP.items():
                if key.lower().replace(" ", "-") == slug:
                    canonical = val
                    break
        if not canonical:
            canonical = readable
            log.warning(f"Unknown team slug: {slug} -> {readable}")

        # Deduplicate (player links may appear as both <a> and <b> in position grid)
        selected = list(dict.fromkeys(sections["selected"]))
        interchange = list(dict.fromkeys(sections["interchange"]))
        emergencies = list(dict.fromkeys(sections["emergencies"]))

        # Normalize all player names
        selected = [_normalize_player_name(p) for p in selected]
        interchange = [_normalize_player_name(p) for p in interchange]
        emergencies = [_normalize_player_name(p) for p in emergencies]

        n_total = len(selected) + len(interchange) + len(emergencies)

        teams_data.append({
            "team": canonical,
            "selected": selected,
            "interchange": interchange,
            "emergencies": emergencies,
            "total_named": n_total,
        })

    if not teams_data:
        log.warning("No team data parsed from selections page")
        return None

    result = {
        "round_number": round_num,
        "year": year,
        "scraped_at": datetime.now().isoformat(),
        "teams": teams_data,
    }

    # Cache to JSON
    cache_path = TEAM_LISTS_DIR / f"round_{round_num}_{year}.json"
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved team selections to {cache_path} ({len(teams_data)} teams)")

    return result


# ---------------------------------------------------------------------------
# Stage 2: Scrape Injury List from AFL.com.au
# ---------------------------------------------------------------------------

AFL_INJURY_URL = "https://www.afl.com.au/matches/injury-list"


def _parse_return_severity(return_text):
    """Convert estimated return text to numeric severity (0-4).

    0 = test/available (likely to play)
    1 = 1 week
    2 = 2-3 weeks
    3 = 4-8 weeks
    4 = season / indefinite
    """
    text = return_text.lower().strip()
    if not text or text in ("tbc", "tbd", "to be confirmed"):
        return 2  # unknown defaults to moderate
    if text in ("test", "available", "managed"):
        return 0
    if text in ("season", "indefinite"):
        return 4
    # Parse week ranges: "1 week", "2-3 weeks", "4-6 weeks"
    week_match = re.search(r"(\d+)(?:\s*-\s*(\d+))?\s*week", text)
    if week_match:
        weeks_lo = int(week_match.group(1))
        weeks_hi = int(week_match.group(2)) if week_match.group(2) else weeks_lo
        avg_weeks = (weeks_lo + weeks_hi) / 2
        if avg_weeks <= 1:
            return 1
        elif avg_weeks <= 3:
            return 2
        elif avg_weeks <= 8:
            return 3
        else:
            return 4
    return 2  # default moderate


def scrape_injury_list():
    """Scrape current injury list from AFL.com.au.

    The page has 18 tables (one per team) in alphabetical team order.
    Each table has rows: PLAYER | INJURY | ESTIMATED RETURN.

    Returns list of dicts:
        {
            "team": str (canonical name),
            "player": str (normalized name),
            "injury": str (injury description),
            "estimated_return": str (raw return text),
            "severity": int (0-4),
        }
    """
    _ensure_dirs()

    html = _fetch(AFL_INJURY_URL)
    if not html:
        log.error("Failed to fetch AFL injury list")
        return []

    soup = BeautifulSoup(html, "html.parser")
    injuries = []

    tables = soup.find_all("table")

    if not tables:
        log.warning("No tables found on AFL injury page")
        return []

    # Strategy 1: Try to identify team from text near each table
    # Walk backwards from each table to find a team name
    team_for_table = []
    used_teams = set()
    page_text = soup.get_text()

    for table in tables:
        assigned_team = None
        # Search the text content between this table and the previous one
        # by walking previous siblings and parent's previous siblings
        search_elements = []
        for sib in table.previous_siblings:
            if hasattr(sib, "get_text"):
                search_elements.append(sib)
            if len(search_elements) > 20:
                break

        for elem in search_elements:
            text = elem.get_text(strip=True) if hasattr(elem, "get_text") else str(elem).strip()
            if not text:
                continue
            # Check against AFL team names
            for afl_name, canonical in AFL_TEAM_MAP.items():
                if text == afl_name and canonical not in used_teams:
                    assigned_team = canonical
                    used_teams.add(canonical)
                    break
            if assigned_team:
                break

        team_for_table.append(assigned_team)

    # Strategy 2: If we have exactly 18 tables and couldn't identify all teams,
    # fall back to alphabetical order
    if len(tables) == 18:
        for i, team in enumerate(AFL_INJURY_TEAM_ORDER):
            if team_for_table[i] is None:
                team_for_table[i] = team

    # Parse each table's rows
    for table_idx, table in enumerate(tables):
        team = team_for_table[table_idx] if table_idx < len(team_for_table) else None
        if not team:
            continue

        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) >= 3:
                player_text = cells[0].get_text(strip=True)
                injury_text = cells[1].get_text(strip=True)
                return_text = cells[2].get_text(strip=True)

                # Skip header rows
                if player_text.upper() in ("PLAYER", "NAME", ""):
                    continue

                injuries.append({
                    "team": team,
                    "player": _normalize_player_name(player_text),
                    "injury": injury_text,
                    "estimated_return": return_text,
                    "severity": _parse_return_severity(return_text),
                })

    # Cache with date stamp
    date_str = datetime.now().strftime("%Y-%m-%d")
    cache_path = INJURIES_DIR / f"injuries_{date_str}.json"
    with open(cache_path, "w") as f:
        json.dump(injuries, f, indent=2)
    log.info(f"Saved {len(injuries)} injury records to {cache_path}")

    return injuries


# ---------------------------------------------------------------------------
# Stage 3: Build Structured Parquets
# ---------------------------------------------------------------------------

def _detect_debutants(team_players, player_games_df):
    """Identify debutants by cross-referencing with historical player data.

    A player is a debutant if they appear in the team list but have
    zero rows in player_games_df.
    """
    if player_games_df is None or player_games_df.empty:
        return []

    all_known = set(player_games_df["player"].unique())
    debutants = [p for p in team_players if p not in all_known]
    return debutants


def _compute_ins_outs(year, round_num):
    """Compute ins/outs by comparing current round selections to previous round.

    Returns dict of {team: {"ins": [...], "outs": [...]}}
    """
    current_path = TEAM_LISTS_DIR / f"round_{round_num}_{year}.json"
    prev_path = TEAM_LISTS_DIR / f"round_{round_num - 1}_{year}.json"

    if not current_path.exists():
        return {}

    with open(current_path) as f:
        current = json.load(f)

    # If no previous round, can't compute ins/outs
    if not prev_path.exists() or round_num <= 1:
        return {t["team"]: {"ins": [], "outs": []} for t in current.get("teams", [])}

    with open(prev_path) as f:
        prev = json.load(f)

    # Build previous round lookup: team -> set of all named players
    prev_squads = {}
    for t in prev.get("teams", []):
        all_players = set(t.get("selected", []) + t.get("interchange", []))
        prev_squads[t["team"]] = all_players

    result = {}
    for t in current.get("teams", []):
        team = t["team"]
        current_squad = set(t.get("selected", []) + t.get("interchange", []))
        prev_squad = prev_squads.get(team, set())

        if prev_squad:
            ins = list(current_squad - prev_squad)
            outs = list(prev_squad - current_squad)
        else:
            ins = []
            outs = []

        result[team] = {"ins": ins, "outs": outs}

    return result


def build_team_changes_parquet(year):
    """Build team_changes.parquet from cached team list JSONs.

    One row per (team, round_number, year) with columns:
        team, round_number, year, n_ins, n_outs, n_debutants,
        ins_list, outs_list, debutant_list, team_stability,
        cumulative_changes_3r
    """
    _ensure_dirs()

    # Load player history for debutant detection
    player_games_path = config.BASE_STORE_DIR / "player_games.parquet"
    player_games = None
    if player_games_path.exists():
        player_games = pd.read_parquet(player_games_path, columns=["player"])

    # Load all cached team list JSONs for this year
    json_files = sorted(TEAM_LISTS_DIR.glob(f"round_*_{year}.json"))
    if not json_files:
        log.info(f"No team list JSONs found for {year}")
        return None

    rows = []
    for fpath in json_files:
        with open(fpath) as f:
            data = json.load(f)

        round_num = data.get("round_number", 0)

        # Compute ins/outs vs previous round
        ins_outs = _compute_ins_outs(year, round_num)

        for team_data in data.get("teams", []):
            team = team_data["team"]
            all_named = (
                team_data.get("selected", [])
                + team_data.get("interchange", [])
            )

            # Detect debutants
            debutants = _detect_debutants(all_named, player_games)

            team_io = ins_outs.get(team, {"ins": [], "outs": []})
            n_ins = len(team_io["ins"])
            n_outs = len(team_io["outs"])

            # Team stability: fraction of unchanged lineup (22 - changes) / 22
            n_changes = max(n_ins, n_outs)
            squad_size = max(len(all_named), 22)
            stability = (squad_size - n_changes) / squad_size

            rows.append({
                "team": team,
                "round_number": round_num,
                "year": year,
                "n_ins": n_ins,
                "n_outs": n_outs,
                "n_debutants": len(debutants),
                "ins_list": json.dumps(team_io["ins"]),
                "outs_list": json.dumps(team_io["outs"]),
                "debutant_list": json.dumps(debutants),
                "team_stability": round(stability, 3),
                "total_named": team_data.get("total_named", len(all_named)),
            })

    if not rows:
        log.info("No team change data to save")
        return None

    df = pd.DataFrame(rows)

    # Compute cumulative changes over last 3 rounds
    df = df.sort_values(["team", "round_number"])
    df["n_changes"] = df["n_ins"] + df["n_outs"]
    df["cumulative_changes_3r"] = (
        df.groupby("team", observed=True)["n_changes"]
        .transform(lambda s: s.rolling(3, min_periods=1).sum())
    )
    df = df.drop(columns=["n_changes"])

    # Optimize dtypes
    df["round_number"] = df["round_number"].astype(np.int8)
    df["year"] = df["year"].astype(np.int16)
    df["n_ins"] = df["n_ins"].astype(np.int8)
    df["n_outs"] = df["n_outs"].astype(np.int8)
    df["n_debutants"] = df["n_debutants"].astype(np.int8)
    df["team_stability"] = df["team_stability"].astype(np.float32)
    df["cumulative_changes_3r"] = df["cumulative_changes_3r"].astype(np.float32)

    out_path = config.BASE_STORE_DIR / "team_changes.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"Saved {len(df)} team change records to {out_path}")
    return out_path


def build_injuries_parquet():
    """Build injuries.parquet from the latest injury snapshot.

    One row per (team, player) with columns:
        team, player, injury, severity, estimated_return
    Also computes team-level aggregates saved alongside.
    """
    _ensure_dirs()

    # Find latest injury cache file
    injury_files = sorted(INJURIES_DIR.glob("injuries_*.json"))
    if not injury_files:
        log.info("No injury cache files found")
        return None

    latest = injury_files[-1]
    with open(latest) as f:
        injuries = json.load(f)

    if not injuries:
        log.info("No injuries in latest snapshot")
        return None

    df = pd.DataFrame(injuries)
    df["severity"] = df["severity"].astype(np.int8)

    out_path = config.BASE_STORE_DIR / "injuries.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"Saved {len(df)} injury records to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Stage 4: Derive News Features for Feature Matrix
# ---------------------------------------------------------------------------

def add_news_features(df):
    """Join news-derived features to the player-match DataFrame.

    Features added (all match-level, per team):
        team_n_ins             (int) players brought in this round
        team_n_outs            (int) players dropped this round
        team_n_debutants       (int) debutants in the team
        team_stability         (float 0-1) fraction of unchanged lineup
        team_churn_3r          (float) cumulative changes over last 3 rounds
        team_injured_count     (int) players on the injury list
        team_injury_severity   (float) sum of injury severity scores
        opp_n_ins              (int) opponent's ins
        opp_n_outs             (int) opponent's outs
        opp_n_debutants        (int) opponent debutants
        opp_stability          (float) opponent lineup stability
        opp_injured_count      (int) opponent injured player count
        is_debutant            (bool) whether this specific player is debuting
    """
    # --- Team changes features ---
    changes_path = config.BASE_STORE_DIR / "team_changes.parquet"
    has_changes = changes_path.exists()

    if has_changes:
        changes = pd.read_parquet(changes_path)
        change_cols = ["team", "round_number", "year",
                       "n_ins", "n_outs", "n_debutants",
                       "team_stability", "cumulative_changes_3r"]
        changes = changes[[c for c in change_cols if c in changes.columns]]

        # Rename for team merge
        team_rename = {
            "n_ins": "team_n_ins",
            "n_outs": "team_n_outs",
            "n_debutants": "team_n_debutants",
            "team_stability": "team_stability",
            "cumulative_changes_3r": "team_churn_3r",
        }
        changes_team = changes.rename(columns=team_rename)

        # Merge on team + round + year
        df = df.merge(
            changes_team,
            on=["team", "round_number", "year"],
            how="left",
        )

        # Opponent merge
        opp_rename = {
            "team": "opponent",
            "n_ins": "opp_n_ins",
            "n_outs": "opp_n_outs",
            "n_debutants": "opp_n_debutants",
            "team_stability": "opp_stability",
            "cumulative_changes_3r": "opp_churn_3r",
        }
        changes_opp = changes.rename(columns=opp_rename)
        # Drop duplicated columns that already exist
        opp_cols_to_merge = ["opponent", "round_number", "year",
                             "opp_n_ins", "opp_n_outs", "opp_n_debutants",
                             "opp_stability", "opp_churn_3r"]
        changes_opp = changes_opp[[c for c in opp_cols_to_merge if c in changes_opp.columns]]

        df = df.merge(
            changes_opp,
            on=["opponent", "round_number", "year"],
            how="left",
        )

        # Detect if this specific player is a debutant
        # Load debutant lists from team changes
        full_changes = pd.read_parquet(changes_path)
        if "debutant_list" in full_changes.columns:
            debutant_lookup = set()
            for _, row in full_changes.iterrows():
                try:
                    debs = json.loads(row["debutant_list"])
                    for d in debs:
                        debutant_lookup.add((row["team"], row["round_number"], row["year"], d))
                except (json.JSONDecodeError, TypeError):
                    pass
            df["is_debutant"] = df.apply(
                lambda r: 1 if (r["team"], r["round_number"], r["year"], r["player"])
                in debutant_lookup else 0,
                axis=1
            ).astype(np.int8)
        else:
            df["is_debutant"] = np.int8(0)

        n_matched = df["team_n_ins"].notna().sum()
        print(f"    Joined team changes: {n_matched}/{len(df)} rows matched")
    else:
        print("    WARNING: team_changes.parquet not found — using defaults")

    # --- Injury features ---
    injuries_path = config.BASE_STORE_DIR / "injuries.parquet"
    has_injuries = injuries_path.exists()

    if has_injuries:
        injuries = pd.read_parquet(injuries_path)

        # Team-level injury aggregates
        team_inj = (
            injuries.groupby("team", observed=True)
            .agg(
                team_injured_count=("player", "count"),
                team_injury_severity=("severity", "sum"),
            )
            .reset_index()
        )
        team_inj["team_injured_count"] = team_inj["team_injured_count"].astype(np.int8)
        team_inj["team_injury_severity"] = team_inj["team_injury_severity"].astype(np.float32)

        df = df.merge(team_inj, on="team", how="left")

        # Opponent injury aggregates
        opp_inj = team_inj.rename(columns={
            "team": "opponent",
            "team_injured_count": "opp_injured_count",
            "team_injury_severity": "opp_injury_severity",
        })
        df = df.merge(opp_inj, on="opponent", how="left")

        n_inj_matched = df["team_injured_count"].notna().sum()
        print(f"    Joined injury data: {n_inj_matched}/{len(df)} rows matched")
    else:
        print("    WARNING: injuries.parquet not found — using defaults")

    # --- Fill defaults for all news features ---
    news_feature_defaults = {
        "team_n_ins": 0,
        "team_n_outs": 0,
        "team_n_debutants": 0,
        "team_stability": 1.0,
        "team_churn_3r": 0.0,
        "team_injured_count": 0,
        "team_injury_severity": 0.0,
        "opp_n_ins": 0,
        "opp_n_outs": 0,
        "opp_n_debutants": 0,
        "opp_stability": 1.0,
        "opp_churn_3r": 0.0,
        "opp_injured_count": 0,
        "opp_injury_severity": 0.0,
        "is_debutant": 0,
    }
    for col, default in news_feature_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)
        # Cast to appropriate dtype
        if isinstance(default, float):
            df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype(np.int8)

    n_features = len(news_feature_defaults)
    print(f"    Added {n_features} news features")
    return df


# ---------------------------------------------------------------------------
# Stage 5: Scrape News Articles from AFL.com.au
# ---------------------------------------------------------------------------

AFL_RSS_URL = "https://www.afl.com.au/rss"
MAX_ARTICLES_PER_RUN = 30
MAX_FULL_TEXT_FETCHES = 15

# AFLW / historical / off-field articles are low-value for predictions
_SKIP_PATTERNS = re.compile(
    r"aflw|under-\d+|u\d+ champ|draft combine|"
    r"hall of fame|podcast|coming soon|"
    r"1966 premiership|immortals reunite",
    re.IGNORECASE,
)


def scrape_afl_news():
    """Fetch latest AFL news via RSS feed + full text extraction.

    Uses the AFL.com.au RSS feed for reliable article discovery (no JS rendering),
    then fetches full text from individual article pages for high-value items.

    Returns list of article dicts cached to data/news/articles/.
    """
    _ensure_dirs()

    xml = _fetch(AFL_RSS_URL)
    if not xml:
        log.error("Failed to fetch AFL RSS feed")
        return []

    soup = BeautifulSoup(xml, "xml")
    items = soup.find_all("item")
    if not items:
        # Fallback to html.parser if lxml-xml not available
        soup = BeautifulSoup(xml, "html.parser")
        items = soup.find_all("item")

    if not items:
        log.warning("No items found in AFL RSS feed")
        return []

    articles = []
    for item in items[:MAX_ARTICLES_PER_RUN]:
        title_el = item.find("title")
        link_el = item.find("link")
        desc_el = item.find("description")
        pub_el = item.find("pubDate")

        headline = title_el.get_text(strip=True) if title_el else ""
        url = link_el.get_text(strip=True) if link_el else ""
        if not url or not headline:
            continue

        # Skip low-value articles
        if _SKIP_PATTERNS.search(headline):
            continue

        summary = desc_el.get_text(strip=True) if desc_el else ""
        published_at = pub_el.get_text(strip=True) if pub_el else None

        articles.append({
            "url": url,
            "headline": headline,
            "summary": summary,
            "full_text": None,
            "teams": [],
            "players": [],
            "published_at": published_at,
        })

    if not articles:
        log.warning("No relevant articles found in RSS feed")
        return []

    log.info(f"Found {len(articles)} articles from RSS, fetching full text for top {MAX_FULL_TEXT_FETCHES}...")

    # Fetch full text for the most valuable articles
    for i, article in enumerate(articles[:MAX_FULL_TEXT_FETCHES]):
        html = _fetch(article["url"])
        if not html:
            continue

        article_soup = BeautifulSoup(html, "html.parser")

        # Extract published date from meta tags (more precise than RSS pubDate)
        date_meta = article_soup.find("meta", attrs={"property": "article:published_time"})
        if date_meta and date_meta.get("content"):
            article["published_at"] = date_meta["content"]

        # Extract article body text
        body_text = []
        for tag in ["article", "main"]:
            el = article_soup.find(tag)
            if el:
                for p in el.find_all("p"):
                    text = p.get_text(strip=True)
                    if text and len(text) > 20:
                        body_text.append(text)
                break

        if not body_text:
            for p in article_soup.find_all("p"):
                text = p.get_text(strip=True)
                if text and len(text) > 50:
                    body_text.append(text)

        if body_text:
            article["full_text"] = "\n".join(body_text[:30])
            article["summary"] = article["summary"] or body_text[0][:300]

        # Tag teams mentioned in the article
        full_content = article.get("full_text", "") or article["headline"]
        for team_name in set(TEAM_MAP.values()):
            if re.search(r"\b" + re.escape(team_name) + r"\b", full_content, re.IGNORECASE):
                if team_name not in article["teams"]:
                    article["teams"].append(team_name)

        if (i + 1) % 5 == 0:
            log.info(f"  Fetched {i + 1}/{min(len(articles), MAX_FULL_TEXT_FETCHES)} articles")

    # Add scraped timestamp
    now = datetime.now().isoformat()
    for a in articles:
        a["scraped_at"] = now

    # Cache to JSON
    date_str = datetime.now().strftime("%Y-%m-%d")
    cache_path = ARTICLES_DIR / f"articles_{date_str}.json"

    existing = []
    if cache_path.exists():
        with open(cache_path) as f:
            existing = json.load(f)

    existing_urls = {a["url"] for a in existing}
    new_articles = [a for a in articles if a["url"] not in existing_urls]
    combined = existing + new_articles

    with open(cache_path, "w") as f:
        json.dump(combined, f, indent=2)

    log.info(f"Saved {len(combined)} articles ({len(new_articles)} new) to {cache_path}")
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python news.py [scrape|injuries|articles|build|features]")
        print("  scrape              Scrape team selections from FootyWire")
        print("  scrape --year Y     Scrape for specific year")
        print("  injuries            Scrape current injury list from AFL.com.au")
        print("  articles            Scrape news articles from AFL.com.au")
        print("  build --year Y      Build parquets from cached data")
        print("  features            Show feature summary from built parquets")
        sys.exit(1)

    cmd = sys.argv[1]

    # Parse optional args
    year = config.CURRENT_SEASON_YEAR
    round_num = None
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--year" and i + 1 < len(sys.argv):
            year = int(sys.argv[i + 1])
        if arg == "--round" and i + 1 < len(sys.argv):
            round_num = int(sys.argv[i + 1])

    if cmd == "scrape":
        result = scrape_team_selections(year)
        if result:
            print(f"\nTeam selections for {result['year']} Round {result['round_number']}:")
            for t in result["teams"]:
                print(f"  {t['team']}: {len(t['selected'])} selected, "
                      f"{len(t['interchange'])} interchange, "
                      f"{len(t['emergencies'])} emergency")

    elif cmd == "injuries":
        injuries = scrape_injury_list()
        if injuries:
            print(f"\nInjury list ({len(injuries)} players):")
            by_team = {}
            for inj in injuries:
                by_team.setdefault(inj["team"], []).append(inj)
            for team in sorted(by_team):
                players = by_team[team]
                print(f"  {team} ({len(players)}):")
                for p in players:
                    sev_label = ["test", "1wk", "2-3wk", "4-8wk", "season"][p["severity"]]
                    print(f"    {p['player']}: {p['injury']} ({sev_label})")

    elif cmd == "articles":
        articles = scrape_afl_news()
        if articles:
            print(f"\nScraped {len(articles)} articles:")
            for a in articles[:10]:
                teams = ", ".join(a.get("teams", [])) or "—"
                has_text = "+" if a.get("full_text") else " "
                print(f"  [{has_text}] {a['headline'][:80]}")
                print(f"       Teams: {teams}")

    elif cmd == "build":
        print(f"Building parquets for {year}...")
        build_team_changes_parquet(year)
        build_injuries_parquet()
        print("Done.")

    elif cmd == "features":
        changes_path = config.BASE_STORE_DIR / "team_changes.parquet"
        injuries_path = config.BASE_STORE_DIR / "injuries.parquet"

        if changes_path.exists():
            changes = pd.read_parquet(changes_path)
            print(f"\nTeam changes: {len(changes)} records")
            print(changes.describe())
        else:
            print("No team_changes.parquet found")

        if injuries_path.exists():
            injuries = pd.read_parquet(injuries_path)
            print(f"\nInjuries: {len(injuries)} records")
            print(injuries.groupby("team")["severity"].agg(["count", "sum", "mean"]))
        else:
            print("No injuries.parquet found")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
