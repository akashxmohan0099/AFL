"""
FootyWire Advanced Stats Scraper
=================================
Scrapes advanced player statistics from footywire.com for seasons 2015-2025.

These stats are NOT available from AFLTables:
  - TOG% (Time on Ground percentage)
  - ED (Effective Disposals)
  - DE% (Disposal Efficiency)
  - CCL (Centre Clearances)
  - SCL (Stoppage Clearances)
  - TO (Turnovers)
  - MG (Metres Gained)
  - SI (Score Involvements)
  - ITC (Intercepts)
  - T5 (Tackles Inside 50)

Usage:
    python scrape_footywire.py --start 2015 --end 2025
    python scrape_footywire.py --start 2024 --end 2024  # single season test

Output:
    data/footywire/advanced_stats_{year}.csv per season
"""

import os
import re
import sys
import time
import argparse
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://www.footywire.com/afl/footy"
MATCH_LIST_URL = f"{BASE_URL}/ft_match_list?year={{year}}"
MATCH_STATS_URL = f"{BASE_URL}/ft_match_statistics?mid={{mid}}&advv=Y"
REQUEST_DELAY = 1.5  # seconds between requests — be polite
HEADERS = {
    "User-Agent": "AFL-Stats-Research-Bot/1.0 (personal research project)"
}

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "footywire"

# Advanced stats columns in the order they appear on FootyWire
ADVANCED_COLS = [
    "CP", "UP", "ED", "DE_pct", "CM", "GA", "MI5", "one_pct",
    "BO", "CCL", "SCL", "SI", "MG", "TO", "ITC", "T5", "TOG_pct",
]

# Columns we actually want to keep (the ones not already in AFLTables)
KEEP_COLS = ["ED", "DE_pct", "CCL", "SCL", "TO", "MG", "SI", "ITC", "T5", "TOG_pct"]

OUTPUT_SCHEMA = [
    "mid", "year", "round", "date", "venue", "team", "opponent",
    "player", *KEEP_COLS,
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch(url, retries=3):
    """Fetch URL with retry and delay."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.text
        except requests.RequestException as e:
            if attempt < retries - 1:
                wait = REQUEST_DELAY * (attempt + 2)
                log.warning(f"Retry {attempt+1}/{retries} for {url}: {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                log.error(f"Failed after {retries} attempts: {url}: {e}")
                return None


def _parse_number(text):
    """Parse a number from text, returning None on failure."""
    if text is None:
        return None
    text = text.strip().replace(",", "")
    if text == "" or text == "-":
        return None
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Match List Scraper
# ---------------------------------------------------------------------------

def scrape_match_list(year):
    """Fetch match list page and extract all match IDs with metadata.

    Returns list of dicts: {mid, round, date_str, venue, home_team, away_team}
    """
    url = MATCH_LIST_URL.format(year=year)
    html = _fetch(url)
    if html is None:
        return []

    soup = BeautifulSoup(html, "html.parser")
    matches = []

    # Find all links containing mid= parameter
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "ft_match_statistics" not in href or "mid=" not in href:
            continue

        # Extract mid
        mid_match = re.search(r"mid=(\d+)", href)
        if not mid_match:
            continue
        mid = int(mid_match.group(1))

        # Navigate up to the table row to get match context
        row = link.find_parent("tr")
        if row is None:
            continue

        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        # Try to extract round from section header
        # Look for preceding round anchor or header
        round_label = ""
        prev = row.find_previous("a", {"name": True})
        if prev and prev.get("name", "").startswith("round"):
            round_label = prev.get("name", "").replace("round_", "R")

        matches.append({
            "mid": mid,
            "round": round_label,
            "year": year,
        })

    # Deduplicate by mid
    seen = set()
    unique = []
    for m in matches:
        if m["mid"] not in seen:
            seen.add(m["mid"])
            unique.append(m)

    log.info(f"  {year}: Found {len(unique)} matches")
    return unique


# ---------------------------------------------------------------------------
# Match Advanced Stats Scraper
# ---------------------------------------------------------------------------

def scrape_match_advanced(mid, year):
    """Fetch advanced stats page for a match and parse both team tables.

    Returns list of dicts with player-level advanced stats.
    """
    url = MATCH_STATS_URL.format(mid=mid)
    html = _fetch(url)
    if html is None:
        return []

    soup = BeautifulSoup(html, "html.parser")
    rows = []

    # Extract match metadata from the page title
    # Format: 'AFL Match Statistics : TeamA defeats TeamB at Venue Round N Day, Date'
    date_str = ""
    venue = ""
    teams = []
    round_label = ""

    title_elem = soup.find("title")
    title_text = title_elem.get_text(strip=True) if title_elem else ""

    # Parse teams, venue from title
    title_match = re.search(
        r":\s*(.+?)\s+(?:defeats?|defeated by|drew with|lost to)\s+(.+?)\s+at\s+(.+?)\s+Round",
        title_text,
    )
    if title_match:
        teams = [title_match.group(1).strip(), title_match.group(2).strip()]
        venue = title_match.group(3).strip()

    # Parse date from title
    date_match = re.search(r"(\w+day,\s+\d+\w*\s+\w+\s+\d{4})", title_text)
    if date_match:
        date_str = date_match.group(1)

    # Parse round from title
    round_match = re.search(r"Round\s+(\d+)", title_text)
    if round_match:
        round_label = f"R{round_match.group(1)}"

    # Find the stats tables — look for tables with proper column headers
    all_tables = soup.find_all("table")

    # Filter for player stats tables with proper "Player" column header
    stats_tables = []
    for table in all_tables:
        header_row = table.find("tr")
        if header_row is None:
            continue
        headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
        # Must have "Player" as first column AND advanced stat columns
        if (len(headers) >= 10
                and headers[0] == "Player"
                and ("DE%" in headers or "TOG%" in headers)):
            stats_tables.append((table, headers))

    # Use team names from matchscoretable, falling back to detection
    team_labels = list(teams) if len(teams) >= 2 else []
    if len(team_labels) < len(stats_tables):
        # Pad with fallback names
        for i in range(len(team_labels), len(stats_tables)):
            team_labels.append(f"Team{i+1}")

    # Parse each team's stats table
    for idx, (table, headers) in enumerate(stats_tables):
        team_name = team_labels[idx] if idx < len(team_labels) else f"Team{idx+1}"
        opp_name = team_labels[1 - idx] if len(team_labels) >= 2 and idx < 2 else ""

        # Map header positions
        col_map = {}
        for i, h in enumerate(headers):
            h_clean = h.replace("%", "_pct").replace("1%", "one_pct")
            if h == "DE%":
                col_map["DE_pct"] = i
            elif h == "TOG%":
                col_map["TOG_pct"] = i
            elif h == "1%":
                col_map["one_pct"] = i
            elif h == "MI5":
                col_map["MI5"] = i
            elif h in ("CP", "UP", "ED", "CM", "GA", "BO",
                        "CCL", "SCL", "SI", "MG", "TO", "ITC", "T5"):
                col_map[h] = i

        # Find player name column — usually first column or column with links
        player_col_idx = 0
        for i, h in enumerate(headers):
            if h.lower() in ("player", ""):
                player_col_idx = i
                break

        # Parse data rows (skip header)
        data_rows = table.find_all("tr")[1:]
        for tr in data_rows:
            cells = tr.find_all(["td", "th"])
            if len(cells) < 5:
                continue

            # Get player name
            player_cell = cells[player_col_idx] if player_col_idx < len(cells) else None
            if player_cell is None:
                continue

            # Player name from link or text
            player_link = player_cell.find("a")
            player_name = (player_link.get_text(strip=True) if player_link
                           else player_cell.get_text(strip=True))

            # Skip totals/summary rows
            if not player_name or player_name.lower() in ("totals", "total", "team"):
                continue

            row = {
                "mid": mid,
                "year": year,
                "date": date_str,
                "venue": venue,
                "team": team_name,
                "opponent": opp_name,
                "player": player_name,
            }

            # Extract each stat
            for stat_name, col_idx in col_map.items():
                if col_idx < len(cells):
                    row[stat_name] = _parse_number(cells[col_idx].get_text(strip=True))
                else:
                    row[stat_name] = None

            # Only keep if we got at least some stats
            has_data = any(row.get(c) is not None for c in KEEP_COLS)
            if has_data:
                rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Season Scraper
# ---------------------------------------------------------------------------

def scrape_season(year, output_dir=None):
    """Scrape all advanced stats for a season."""
    output_dir = output_dir or OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"advanced_stats_{year}.csv"

    log.info(f"Scraping FootyWire advanced stats for {year}...")
    matches = scrape_match_list(year)
    if not matches:
        log.warning(f"No matches found for {year}")
        return None

    all_rows = []
    for i, match in enumerate(matches):
        mid = match["mid"]
        if (i + 1) % 20 == 0:
            log.info(f"  Progress: {i+1}/{len(matches)} matches ({year})")

        match_rows = scrape_match_advanced(mid, year)
        if match_rows:
            # Add round info
            for r in match_rows:
                r["round"] = match.get("round", "")
            all_rows.extend(match_rows)

    if not all_rows:
        log.warning(f"No advanced stats collected for {year}")
        return None

    df = pd.DataFrame(all_rows)

    # Ensure all output columns exist
    for col in OUTPUT_SCHEMA:
        if col not in df.columns:
            df[col] = None

    # Reorder and save
    df = df[OUTPUT_SCHEMA]
    df.to_csv(out_path, index=False)
    log.info(f"  Saved {len(df)} rows to {out_path}")

    return df


def scrape_all_seasons(start_year, end_year, output_dir=None):
    """Scrape FootyWire advanced stats for all seasons in range."""
    output_dir = output_dir or OUTPUT_DIR
    all_dfs = []

    for year in range(start_year, end_year + 1):
        out_path = Path(output_dir) / f"advanced_stats_{year}.csv"
        if out_path.exists():
            log.info(f"  {year}: Already scraped ({out_path}), skipping")
            df = pd.read_csv(out_path)
            all_dfs.append(df)
            continue

        df = scrape_season(year, output_dir)
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        log.info(f"\nTotal: {len(combined)} player-match rows across "
                 f"{start_year}-{end_year}")
        return combined
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape FootyWire advanced stats")
    parser.add_argument("--start", type=int, default=2015, help="Start year")
    parser.add_argument("--end", type=int, default=2025, help="End year")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    scrape_all_seasons(args.start, args.end, args.output)
