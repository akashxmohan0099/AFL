"""
FootyWire Live Match Stats Scraper
===================================
Scrapes basic per-player match stats from footywire.com.
Designed to fill the gap when AFLTables hasn't updated yet (e.g. current season).

Outputs CSV files in the same schema as scraper.py (AFLTables) so they can be
ingested by clean.py without changes.

Usage:
    python scrape_footywire_live.py --year 2026                    # scrape full season
    python scrape_footywire_live.py --year 2026 --incremental      # only new matches
    python scrape_footywire_live.py --year 2026 --daily             # for cron: incremental + clean + features

Output:
    data/player_stats/player_stats_{year}_footywire.csv
"""

import argparse
import logging
import re
import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.footywire.com/afl/footy"
MATCH_LIST_URL = f"{BASE_URL}/ft_match_list?year={{year}}"
MATCH_STATS_URL = f"{BASE_URL}/ft_match_statistics?mid={{mid}}"
REQUEST_DELAY = 1.5
HEADERS = {
    "User-Agent": "AFL-Stats-Research-Bot/1.0 (personal research project)"
}

# Map FootyWire columns -> pipeline schema
FW_TO_PIPELINE = {
    "K": "KI", "HB": "HB", "D": "DI", "M": "MK",
    "G": "GL", "B": "BH", "T": "TK", "HO": "HO",
    "GA": "GA", "I50": "IF", "CL": "CL", "CG": "CG",
    "R50": "RB", "FF": "FF", "FA": "FA",
}

# FootyWire team name -> pipeline canonical name
TEAM_MAP = {
    "Adelaide": "Adelaide",
    "Brisbane Lions": "Brisbane Lions",
    "Brisbane": "Brisbane Lions",
    "Carlton": "Carlton",
    "Collingwood": "Collingwood",
    "Essendon": "Essendon",
    "Fremantle": "Fremantle",
    "Geelong": "Geelong",
    "Gold Coast": "Gold Coast",
    "GWS": "Greater Western Sydney",
    "GWS Giants": "Greater Western Sydney",
    "Greater Western Sydney": "Greater Western Sydney",
    "Hawthorn": "Hawthorn",
    "Melbourne": "Melbourne",
    "North Melbourne": "North Melbourne",
    "Port Adelaide": "Port Adelaide",
    "Richmond": "Richmond",
    "St Kilda": "St Kilda",
    "Sydney": "Sydney",
    "West Coast": "West Coast",
    "Western Bulldogs": "Western Bulldogs",
}

DATA_DIR = Path(__file__).resolve().parent / "data"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _fetch(url, retries=3):
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
    """Convert 'First Last' -> 'Last, First' (pipeline format)."""
    name = name.strip()
    parts = name.rsplit(" ", 1)
    if len(parts) == 2:
        return f"{parts[1]}, {parts[0]}"
    return name


def scrape_match_list(year):
    """Get all match IDs and metadata for a season from FootyWire."""
    html = _fetch(MATCH_LIST_URL.format(year=year))
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    matches = []
    current_round = ""

    # FootyWire structures rounds as sections
    for elem in soup.find_all(["a", "tr"]):
        # Detect round headers
        if elem.name == "a" and elem.get("name", "").startswith("round"):
            current_round = elem.get("name", "").replace("_", " ").title()

        if elem.name == "tr":
            # Only process actual match rows (7 direct td children)
            direct_tds = elem.find_all("td", recursive=False)
            if len(direct_tds) < 5:
                continue
            links = elem.find_all("a", href=True)
            for link in links:
                href = link["href"]
                if "ft_match_statistics" in href and "mid=" in href:
                    mid_match = re.search(r"mid=(\d+)", href)
                    if not mid_match:
                        continue
                    mid = int(mid_match.group(1))

                    cells = [td.get_text(strip=True) for td in direct_tds]
                    date_str = cells[0] if cells else ""
                    teams_str = cells[1] if len(cells) > 1 else ""
                    venue = cells[2] if len(cells) > 2 else ""
                    score_str = cells[4] if len(cells) > 4 else ""

                    # Parse teams: "SydneyvCarlton" or "Sydney v Carlton"
                    teams_match = re.match(r"(.+?)v(.+)", teams_str.replace(" v ", "v"))
                    home_team = teams_match.group(1).strip() if teams_match else ""
                    away_team = teams_match.group(2).strip() if teams_match else ""

                    # Parse date
                    date_iso = _parse_fw_date(date_str, year)

                    # Parse score
                    scores = re.match(r"(\d+)-(\d+)", score_str) if score_str else None
                    has_result = scores is not None

                    matches.append({
                        "mid": mid,
                        "round": current_round,
                        "date_str": date_str,
                        "date_iso": date_iso,
                        "venue": venue,
                        "home_team": TEAM_MAP.get(home_team, home_team),
                        "away_team": TEAM_MAP.get(away_team, away_team),
                        "home_score": int(scores.group(1)) if scores else None,
                        "away_score": int(scores.group(2)) if scores else None,
                        "has_result": has_result,
                    })

    # Deduplicate
    seen = set()
    unique = []
    for m in matches:
        if m["mid"] not in seen:
            seen.add(m["mid"])
            unique.append(m)

    log.info(f"{year}: Found {len(unique)} matches ({sum(m['has_result'] for m in unique)} with results)")
    return unique


def _parse_fw_date(date_str, year):
    """Parse FootyWire date 'Thu 5 Mar 7:30pm' -> ISO date string."""
    date_str = date_str.strip()
    if not date_str:
        return None
    # Try: "Thu 5 Mar 7:30pm"
    match = re.match(r"\w+\s+(\d+)\s+(\w+)\s+", date_str)
    if match:
        day = match.group(1)
        month = match.group(2)
        try:
            dt = datetime.strptime(f"{day} {month} {year}", "%d %b %Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def scrape_match_stats(mid, match_meta, year):
    """Scrape basic player stats for a single match."""
    html = _fetch(MATCH_STATS_URL.format(mid=mid))
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")

    # Parse title for teams/venue/round
    title = soup.find("title")
    title_text = title.get_text(strip=True) if title else ""

    title_match = re.search(
        r":\s*(.+?)\s+(?:defeats?|defeated by|drew with|lost to|def\.?)\s+(.+?)\s+at\s+(.+?)\s+Round\s+(\d+)",
        title_text,
    )
    if title_match:
        team1 = TEAM_MAP.get(title_match.group(1).strip(), title_match.group(1).strip())
        team2 = TEAM_MAP.get(title_match.group(2).strip(), title_match.group(2).strip())
        venue = title_match.group(3).strip()
        round_num = int(title_match.group(4))
    else:
        team1 = match_meta.get("home_team", "")
        team2 = match_meta.get("away_team", "")
        venue = match_meta.get("venue", "")
        round_match = re.search(r"Round\s*(\d+)", match_meta.get("round", ""))
        round_num = int(round_match.group(1)) if round_match else 0

    # Find basic stats tables using pandas
    try:
        dfs = pd.read_html(StringIO(html))
    except Exception:
        log.warning(f"  Could not parse tables for mid={mid}")
        return []

    # Find the two player stats tables (18 columns, 20+ rows)
    stat_tables = []
    for df in dfs:
        if df.shape[1] >= 16 and df.shape[0] >= 18:
            # Check if first row looks like headers
            first_row = [str(v) for v in df.iloc[0].values]
            if "Player" in first_row and ("K" in first_row or "D" in first_row):
                stat_tables.append(df)

    if len(stat_tables) < 2:
        log.warning(f"  mid={mid}: Found {len(stat_tables)} stat tables (expected 2)")
        return []

    teams = [team1, team2]
    rows = []
    date_iso = match_meta.get("date_iso", "")

    for idx, df in enumerate(stat_tables[:2]):
        team = teams[idx]
        opponent = teams[1 - idx]

        # Use first row as headers
        df.columns = [str(v) for v in df.iloc[0].values]
        df = df.iloc[1:].reset_index(drop=True)

        for _, player_row in df.iterrows():
            player_name = str(player_row.get("Player", "")).strip()
            if not player_name or player_name.lower() in ("player", "totals", "total"):
                continue

            row = {
                "match_id": mid,
                "year": year,
                "round": str(round_num) if round_num is not None else match_meta.get("round", ""),
                "venue": venue,
                "date": match_meta.get("date_str", ""),
                "date_iso": date_iso,
                "team": team,
                "opponent": opponent,
                "home_away": "home" if idx == 0 else "away",
                "jumper": "",
                "player": _normalize_player_name(player_name),
            }

            # Map FootyWire stat columns to pipeline columns
            for fw_col, pipe_col in FW_TO_PIPELINE.items():
                val = player_row.get(fw_col)
                try:
                    row[pipe_col] = int(val) if pd.notna(val) else 0
                except (ValueError, TypeError):
                    row[pipe_col] = 0

            # Columns not available from FootyWire basic stats
            row["BR"] = 0
            row["CP"] = 0
            row["UP"] = 0
            row["CM"] = 0
            row["MI"] = 0
            row["one_pct"] = 0
            row["BO"] = 0
            row["pct_played"] = 100
            row["sub_status"] = ""

            rows.append(row)

    return rows


def scrape_season(year, incremental=False):
    """Scrape all completed matches for a season."""
    output_dir = DATA_DIR / "player_stats"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"player_stats_{year}_footywire.csv"

    # Load existing data for incremental mode
    existing_mids = set()
    if incremental and out_path.exists():
        existing = pd.read_csv(out_path, low_memory=False)
        existing_mids = set(existing["match_id"].unique())
        log.info(f"Incremental mode: {len(existing_mids)} matches already scraped")

    matches = scrape_match_list(year)
    completed = [m for m in matches if m["has_result"]]

    if incremental:
        completed = [m for m in completed if m["mid"] not in existing_mids]

    if not completed:
        log.info(f"No new completed matches to scrape for {year}")
        return out_path if out_path.exists() else None

    log.info(f"Scraping {len(completed)} matches for {year}...")

    all_rows = []
    for i, match in enumerate(completed):
        if (i + 1) % 10 == 0:
            log.info(f"  Progress: {i+1}/{len(completed)}")
        rows = scrape_match_stats(match["mid"], match, year)
        all_rows.extend(rows)

    if not all_rows:
        log.warning(f"No player stats collected for {year}")
        return None

    new_df = pd.DataFrame(all_rows)

    # Merge with existing data in incremental mode
    if incremental and out_path.exists():
        existing = pd.read_csv(out_path, low_memory=False)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(out_path, index=False)
        log.info(f"Appended {len(new_df)} rows (total: {len(combined)}) to {out_path}")
    else:
        new_df.to_csv(out_path, index=False)
        log.info(f"Saved {len(new_df)} rows to {out_path}")

    return out_path


def daily_update(year):
    """Full daily update: scrape new matches, rebuild clean data and features."""
    log.info(f"=== Daily update for {year} ===")

    # 1. Scrape new matches
    result = scrape_season(year, incremental=True)
    if result is None:
        log.info("No new data. Done.")
        return

    # 2. Rebuild clean data
    log.info("Rebuilding clean data...")
    try:
        from clean import build_player_games
        build_player_games()
        log.info("Clean data rebuilt.")
    except Exception as e:
        log.error(f"Clean step failed: {e}")
        return

    # 3. Rebuild features (delete cache to force)
    features_path = DATA_DIR / "features" / "feature_matrix.parquet"
    if features_path.exists():
        features_path.unlink()
        log.info("Deleted feature cache to force rebuild.")

    try:
        from features import build_features
        build_features()
        log.info("Features rebuilt.")
    except Exception as e:
        log.error(f"Features step failed: {e}")

    log.info("=== Daily update complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape FootyWire basic match stats")
    parser.add_argument("--year", type=int, default=2026, help="Season year")
    parser.add_argument("--incremental", action="store_true",
                        help="Only scrape new matches not already in output")
    parser.add_argument("--daily", action="store_true",
                        help="Full daily pipeline: scrape + clean + features")
    args = parser.parse_args()

    if args.daily:
        daily_update(args.year)
    else:
        scrape_season(args.year, incremental=args.incremental)
