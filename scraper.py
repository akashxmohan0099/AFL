"""
AFL Tables Scraper
==================
Scrapes match results and per-game player stats from afltables.com
for seasons 2005-2025.

Usage:
    python scraper.py --start 2005 --end 2025 --output ./data
    python scraper.py --start 2024 --end 2024 --output ./data  # single season test

Data collected:
    1. Match results: date, round, venue, teams, quarter scores, attendance
    2. Player match stats: per-game stats for every player in every match
"""

import os
import re
import sys
import time
import json
import logging
import argparse
import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://afltables.com/afl"
SEASON_URL = f"{BASE_URL}/seas/{{year}}.html"
STATS_GAME_URL = f"{BASE_URL}/stats/games/{{year}}/{{match_id}}.html"
REQUEST_DELAY = 1.5  # seconds between requests -- be polite
HEADERS = {
    "User-Agent": "AFL-Stats-Research-Bot/1.0 (personal research project)"
}

# ---------------------------------------------------------------------------
# Canonical output schemas — every CSV will have exactly these columns in this
# order, regardless of what the source page provides.  Missing values → None.
# ---------------------------------------------------------------------------
PLAYER_STATS_SCHEMA = [
    "match_id", "year", "round", "venue", "date", "date_iso", "team", "opponent",
    "home_away", "jumper", "player",
    "KI", "MK", "HB", "DI", "GL", "BH", "HO", "TK", "RB", "IF",
    "CL", "CG", "FF", "FA", "BR", "CP", "UP", "CM", "MI",
    "one_pct", "BO", "GA", "pct_played", "sub_status",
]

PLAYER_DETAILS_SCHEMA = [
    "match_id", "year", "team", "jumper", "player", "Age",
    "Career Games (W-D-L W%)", "Career Goals (Ave.)",
    "team_games", "team_goals",
]

SCORING_SCHEMA = [
    "match_id", "year", "quarter", "time", "team", "player",
    "score_type", "score",
]


def parse_afl_date(date_str):
    """Parse AFL Tables date string to ISO datetime.
    Input:  'Thu, 7-Mar-2024 7:30 PM (6:30 PM)' or 'Thu, 7-Mar-2024 7:30 PM'
    Output: '2024-03-07T19:30:00'
    """
    if not date_str:
        return None
    # Strip optional timezone part in parens
    clean = re.sub(r'\s*\([^)]*\)\s*$', '', date_str.strip())
    try:
        dt = datetime.strptime(clean, "%a, %d-%b-%Y %I:%M %p")
        return dt.isoformat()
    except ValueError:
        return None


def enforce_schema(rows, schema):
    """Ensure every row dict has exactly the canonical columns (in order).
    Extra columns are dropped; missing columns are filled with None."""
    out = []
    for row in rows:
        out.append({col: row.get(col) for col in schema})
    return out

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scraper.log")
    ]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def fetch_page(url, retries=3):
    """Fetch a page with retries and polite delay."""
    for attempt in range(retries):
        try:
            time.sleep(REQUEST_DELAY)
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt+1}/{retries} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
    logger.error(f"Failed to fetch {url} after {retries} attempts")
    return None


# ---------------------------------------------------------------------------
# 1. Scrape season scores page to get match list and results
# ---------------------------------------------------------------------------
def scrape_season_matches(year):
    """
    Scrape the season scores page to extract:
    - Round number
    - Date and time
    - Venue
    - Home/away teams
    - Quarter-by-quarter scores
    - Final scores
    - Attendance
    - Match page URL (for player stats)
    """
    url = SEASON_URL.format(year=year)
    logger.info(f"Scraping season {year} matches from {url}")
    
    html = fetch_page(url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, "lxml")
    matches = []
    
    # Find all match tables -- AFL Tables uses <table> for each match
    # The structure has game info in tables within the page
    # We need to find all links to individual match stat pages
    all_links = soup.find_all("a", href=True)
    match_links = []
    
    for link in all_links:
        href = link.get("href", "")
        # Match stat page links look like: ../stats/games/2024/011020240728.html
        if f"stats/games/{year}/" in href and href.endswith(".html"):
            full_url = f"{BASE_URL}/stats/games/{year}/{href.split('/')[-1]}"
            if full_url not in match_links:
                match_links.append(full_url)
    
    logger.info(f"Found {len(match_links)} match links for {year}")
    
    # Now parse the season page for match results
    # AFL Tables structures each round with match details in tables
    tables = soup.find_all("table")
    
    current_round = None
    
    for table in tables:
        text = table.get_text()
        
        # Check if this is a round header
        round_match = re.search(r"Round:\s*(\d+|QF|EF|SF|PF|GF)", text)
        if round_match:
            current_round = round_match.group(1)
        
        # Look for match rows with team names and scores
        rows = table.find_all("tr")
        if len(rows) >= 2:
            # Check for venue, date, attendance patterns
            venue_match = re.search(r"Venue:\s*([^\n]+?)(?:\s*Date:)", text)
            date_match = re.search(r"Date:\s*([^\n]+?)(?:\s*Attendance:|\s*$)", text)
            att_match = re.search(r"Attendance:\s*([\d,]+)", text)
            
            # Look for score patterns: Team Q1 Q2 Q3 Q4
            score_pattern = re.findall(
                r"([\w\s]+?)\s+(\d+\.\d+\.\d+)\s+(\d+\.\d+\.\d+)\s+(\d+\.\d+\.\d+)\s+(\d+\.\d+\.\d+)",
                text
            )
            
            if len(score_pattern) >= 2:
                home_team = score_pattern[0][0].strip()
                away_team = score_pattern[1][0].strip()
                
                match_data = {
                    "year": year,
                    "round": current_round,
                    "venue": venue_match.group(1).strip() if venue_match else None,
                    "date": date_match.group(1).strip() if date_match else None,
                    "attendance": int(att_match.group(1).replace(",", "")) if att_match else None,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_q1": score_pattern[0][1],
                    "home_q2": score_pattern[0][2],
                    "home_q3": score_pattern[0][3],
                    "home_q4": score_pattern[0][4],
                    "away_q1": score_pattern[1][1],
                    "away_q2": score_pattern[1][2],
                    "away_q3": score_pattern[1][3],
                    "away_q4": score_pattern[1][4],
                }
                
                # Extract final total from Q4 score (format: G.B.Total)
                home_total = score_pattern[0][4].split(".")[-1]
                away_total = score_pattern[1][4].split(".")[-1]
                match_data["home_score"] = int(home_total)
                match_data["away_score"] = int(away_total)
                match_data["margin"] = int(home_total) - int(away_total)
                
                matches.append(match_data)
    
    # Store match links for later use
    return matches, match_links


# ---------------------------------------------------------------------------
# 2. Scrape individual match page for player stats
# ---------------------------------------------------------------------------
def scrape_match_player_stats(match_url, year):
    """
    Scrape a single match page to get per-player stats for both teams.
    Returns a list of dicts with player stats.
    """
    logger.info(f"Scraping match stats: {match_url}")
    
    html = fetch_page(match_url)
    if not html:
        return {"player_stats": [], "player_details": [], "scoring": [], "match_info": {}}
    
    soup = BeautifulSoup(html, "lxml")
    
    # ---- Extract match context from the header ----
    match_info = {}
    page_text = soup.get_text()
    
    # Extract teams from title
    title = soup.find("title")
    if title:
        title_text = title.get_text()
        # Format: "AFL Tables - Adelaide v Hawthorn - Sun, 28-Jul-2024 ..."
        team_match = re.search(r"AFL Tables - (.+?) v (.+?) -", title_text)
        if team_match:
            match_info["home_team"] = team_match.group(1).strip()
            match_info["away_team"] = team_match.group(2).strip()
    
    # Extract round, venue, date, attendance from the info table
    round_match = re.search(r"Round:\s*(\S+)", page_text)
    venue_match = re.search(r"Venue:\s*(.+?)(?:Date:|$)", page_text)
    date_match = re.search(r"Date:\s*(\w+,\s*\d+-\w+-\d+\s+\d+:\d+\s+[AP]M(?:\s*\([^)]*\))?)", page_text)
    att_match = re.search(r"Attendance:\s*([\d,]+)", page_text)
    
    match_info["round"] = round_match.group(1).strip() if round_match else None
    match_info["venue"] = venue_match.group(1).strip() if venue_match else None
    match_info["date"] = date_match.group(1).strip() if date_match else None
    match_info["date_iso"] = parse_afl_date(match_info["date"])
    match_info["attendance"] = int(att_match.group(1).replace(",", "")) if att_match else None
    match_info["year"] = year
    match_info["match_url"] = match_url
    match_info["match_id"] = match_url.split("/")[-1].replace(".html", "")
    
    # ---- Extract quarter scores ----
    # Find the scoring table (has team rows with quarter scores)
    score_rows = re.findall(
        r"([\w\s]+?)\s+(\d+\.\d+\.\d+)\s+(\d+\.\d+\.\d+)\s+(\d+\.\d+\.\d+)\s+(\d+\.\d+\.\d+)",
        page_text
    )
    if len(score_rows) >= 2:
        match_info["home_q1"] = score_rows[0][1]
        match_info["home_q2"] = score_rows[0][2]
        match_info["home_q3"] = score_rows[0][3]
        match_info["home_q4"] = score_rows[0][4]
        match_info["away_q1"] = score_rows[1][1]
        match_info["away_q2"] = score_rows[1][2]
        match_info["away_q3"] = score_rows[1][3]
        match_info["away_q4"] = score_rows[1][4]
        
        match_info["home_score"] = int(score_rows[0][4].split(".")[-1])
        match_info["away_score"] = int(score_rows[1][4].split(".")[-1])
        match_info["margin"] = match_info["home_score"] - match_info["away_score"]
    
    # ---- Extract player stats tables ----
    # Use pandas to read HTML tables -- much cleaner for structured tables
    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        logger.error(f"Failed to parse tables from {match_url}: {e}")
        return {"player_stats": [], "player_details": [], "scoring": [], "match_info": match_info}
    
    player_stats = []
    player_details = []
    scoring_events = []
    team_index = 0  # 0 = home, 1 = away
    detail_team_index = 0
    teams = [match_info.get("home_team", "Unknown"), match_info.get("away_team", "Unknown")]

    for table in tables:
        # Flatten multi-level column headers (e.g. ('Team Match Statistics', 'KI') -> 'KI')
        if hasattr(table.columns, 'levels'):
            table.columns = table.columns.get_level_values(-1)

        # Player stat tables have specific columns
        cols = [str(c).strip() for c in table.columns.tolist()]

        # Check if this looks like a player stats table
        # It should have columns like KI, MK, HB, DI, GL etc
        has_player_cols = any(c in cols for c in ["KI", "DI", "GL", "TK"])
        has_player_col = any("Player" in str(c) or "player" in str(c) for c in cols)
        
        if has_player_cols and has_player_col:
            team_name = teams[team_index] if team_index < len(teams) else f"Team_{team_index}"
            
            for _, row in table.iterrows():
                player_data = {
                    "match_id": match_info.get("match_id"),
                    "year": year,
                    "round": match_info.get("round"),
                    "venue": match_info.get("venue"),
                    "date": match_info.get("date"),
                    "date_iso": match_info.get("date_iso"),
                    "team": team_name,
                    "opponent": teams[1 - team_index] if team_index < len(teams) else "Unknown",
                    "home_away": "home" if team_index == 0 else "away",
                }
                
                # Map each column
                for col in cols:
                    col_clean = str(col).strip()
                    val = row.get(col, None)
                    
                    # Clean the value
                    if pd.isna(val):
                        val = 0
                    
                    # Rename specific columns
                    col_map = {
                        "#": "jumper",
                        "Player": "player",
                        "1%": "one_pct",
                        "%P": "pct_played",
                        "SU": "sub_status",
                    }
                    
                    mapped_col = col_map.get(col_clean, col_clean)
                    player_data[mapped_col] = val
                
                # Clean jumper number and extract sub status from arrows
                raw_jumper = str(player_data.get("jumper", ""))
                if "↑" in raw_jumper:
                    player_data["sub_status"] = "on"
                    player_data["jumper"] = raw_jumper.replace("↑", "").strip()
                elif "↓" in raw_jumper:
                    player_data["sub_status"] = "off"
                    player_data["jumper"] = raw_jumper.replace("↓", "").strip()
                else:
                    player_data["sub_status"] = None

                # Skip header/total rows and fake player rows
                SKIP_PLAYER_NAMES = {"Player", "nan", "Totals", "", "Rushed", "Opposition"}
                player_name = str(player_data.get("player", ""))
                if player_name and player_name not in SKIP_PLAYER_NAMES:
                    player_stats.append(player_data)
            
            team_index += 1
            continue

        # ---- Detect Player Details tables (have "Age" and "Career" columns) ----
        has_age = "Age" in cols
        has_career = any("Career" in c for c in cols)

        if has_age and has_career and has_player_col:
            team_name = teams[detail_team_index] if detail_team_index < len(teams) else f"Team_{detail_team_index}"

            for _, row in table.iterrows():
                detail = {
                    "match_id": match_info.get("match_id"),
                    "year": year,
                    "team": team_name,
                }
                for col in cols:
                    col_clean = str(col).strip()
                    val = row.get(col, None)
                    if pd.isna(val):
                        val = None

                    # Normalize team-specific column names to generic names
                    if "Games" in col_clean and "Career" not in col_clean:
                        mapped = "team_games"
                    elif "Goals" in col_clean and "Career" not in col_clean:
                        mapped = "team_goals"
                    else:
                        mapped = {"#": "jumper", "Player": "player"}.get(col_clean, col_clean)
                    detail[mapped] = val

                # Replace ← with career values (means single-team player)
                if detail.get("team_games") == "\u2190":
                    detail["team_games"] = detail.get("Career Games (W-D-L W%)")
                    detail["team_goals"] = detail.get("Career Goals (Ave.)")

                player_name = str(detail.get("player", ""))
                # Skip title/totals rows, but KEEP coach row (jumper == "C")
                if player_name and player_name not in {"Player", "nan", "Totals", ""}:
                    player_details.append(detail)

            detail_team_index += 1

    # ---- Extract Scoring Progression table via BeautifulSoup ----
    for bs_table in soup.find_all('table'):
        first_text = bs_table.get_text()[:30]
        if 'Scoring progression' not in first_text:
            continue

        current_quarter = None
        rows = bs_table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            cell_texts = [c.get_text(strip=True) for c in cells]

            if len(cells) > 5:
                # Quarter header row — extract quarter name
                qtr_match = re.search(r'(\d+(?:st|nd|rd|th)|Final)\s+quarter', cell_texts[0])
                if qtr_match:
                    current_quarter = qtr_match.group(1)
                continue

            if len(cells) != 5:
                continue

            # Skip column header row (e.g. ['Sydney', 'Time', 'Score', 'Time', 'Melbourne'])
            if cell_texts[1] == 'Time' and cell_texts[2] == 'Score':
                continue

            # Skip summary row
            if 'Biggest lead' in cell_texts[0] or 'Game time' in cell_texts[2]:
                gt = re.search(r'Game time:(\S+(?:\s\S+)?)', cell_texts[2])
                if gt:
                    match_info["game_time"] = gt.group(1)
                continue

            # Determine which team scored
            home_event, home_time, score, away_time, away_event = cell_texts
            if home_event:
                event_text = home_event
                event_time = home_time
                scoring_team = "home"
            elif away_event:
                event_text = away_event
                event_time = away_time
                scoring_team = "away"
            else:
                continue

            # Parse player and score type from event text
            score_type = "goal" if "goal" in event_text.lower() else "behind"
            player = event_text.replace(" goal", "").replace(" behind", "").strip()

            # Normalize "First Last" to "Last, First" to match player_stats format
            if player != "Rushed":
                parts = player.strip().split()
                if len(parts) >= 2:
                    player = f"{' '.join(parts[1:])}, {parts[0]}"

            scoring_events.append({
                "match_id": match_info.get("match_id"),
                "year": year,
                "quarter": current_quarter,
                "time": event_time,
                "team": teams[0] if scoring_team == "home" else teams[1],
                "player": player,
                "score_type": score_type,
                "score": score,
            })
        break  # Only process first matching table

    # Count rushed behinds per team from scoring events
    for side, team in [("home", teams[0]), ("away", teams[1])]:
        rushed = sum(1 for e in scoring_events
                     if e.get("player") == "Rushed" and e.get("team") == team)
        match_info[f"{side}_rushed_behinds"] = rushed

    logger.info(f"Extracted {len(player_stats)} player stat records, {len(player_details)} player details, {len(scoring_events)} scoring events")
    return {
        "player_stats": enforce_schema(player_stats, PLAYER_STATS_SCHEMA),
        "player_details": enforce_schema(player_details, PLAYER_DETAILS_SCHEMA),
        "scoring": enforce_schema(scoring_events, SCORING_SCHEMA),
        "match_info": match_info,
    }


# ---------------------------------------------------------------------------
# 3. Main scraping pipeline
# ---------------------------------------------------------------------------
def scrape_seasons(start_year, end_year, output_dir):
    """
    Main pipeline: scrape all matches and player stats for given year range.
    Saves data incrementally to avoid losing progress.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "matches"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "player_stats"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "player_details"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "scoring"), exist_ok=True)
    
    all_matches_summary = []
    
    for year in range(start_year, end_year + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"SCRAPING SEASON {year}")
        logger.info(f"{'='*60}")
        
        # Check if we already have this season's data
        match_file = os.path.join(output_dir, "matches", f"matches_{year}.csv")
        player_file = os.path.join(output_dir, "player_stats", f"player_stats_{year}.csv")
        
        if os.path.exists(match_file) and os.path.exists(player_file):
            logger.info(f"Season {year} already scraped, skipping. Delete files to re-scrape.")
            continue
        
        # Step 1: Get match list from season page
        result = scrape_season_matches(year)
        if not result:
            logger.error(f"Failed to scrape season {year}")
            continue
        
        matches, match_links = result
        logger.info(f"Found {len(match_links)} matches for {year}")
        
        # Step 2: Scrape each match for player stats
        season_player_stats = []
        season_player_details = []
        season_scoring = []
        season_match_details = []

        for i, match_url in enumerate(match_links):
            logger.info(f"Match {i+1}/{len(match_links)} for {year}")

            result = scrape_match_player_stats(match_url, year)
            player_stats = result["player_stats"]
            player_details = result["player_details"]
            scoring = result["scoring"]
            match_info = result["match_info"]

            if player_stats:
                season_player_stats.extend(player_stats)
            if player_details:
                season_player_details.extend(player_details)
            if scoring:
                season_scoring.extend(scoring)
            if match_info:
                season_match_details.append(match_info)
            
            # Save progress every 20 matches
            if (i + 1) % 20 == 0:
                logger.info(f"Progress checkpoint: {i+1}/{len(match_links)} matches scraped")
        
        # Step 3: Save season data
        if season_match_details:
            match_df = pd.DataFrame(season_match_details)
            match_df.to_csv(match_file, index=False)
            logger.info(f"Saved {len(match_df)} match records to {match_file}")
        
        if season_player_stats:
            player_df = pd.DataFrame(season_player_stats)
            player_df.to_csv(player_file, index=False)
            logger.info(f"Saved {len(player_df)} player stat records to {player_file}")

        if season_player_details:
            details_file = os.path.join(output_dir, "player_details", f"player_details_{year}.csv")
            details_df = pd.DataFrame(season_player_details)
            details_df.to_csv(details_file, index=False)
            logger.info(f"Saved {len(details_df)} player detail records to {details_file}")

        if season_scoring:
            scoring_file = os.path.join(output_dir, "scoring", f"scoring_{year}.csv")
            scoring_df = pd.DataFrame(season_scoring)
            scoring_df.to_csv(scoring_file, index=False)
            logger.info(f"Saved {len(scoring_df)} scoring events to {scoring_file}")

        all_matches_summary.append({
            "year": year,
            "matches_scraped": len(match_links),
            "player_records": len(season_player_stats)
        })
    
    # Step 4: Combine all seasons into master files
    logger.info(f"\n{'='*60}")
    logger.info("COMBINING ALL SEASONS")
    logger.info(f"{'='*60}")
    
    combine_season_files(output_dir)
    
    # Print summary
    summary_df = pd.DataFrame(all_matches_summary)
    logger.info(f"\nScraping Summary:\n{summary_df.to_string()}")
    
    return summary_df


def combine_season_files(output_dir):
    """Combine individual season CSVs into master files."""
    
    # Combine match files
    match_dir = os.path.join(output_dir, "matches")
    match_files = sorted([f for f in os.listdir(match_dir) if f.endswith(".csv")])
    
    if match_files:
        match_dfs = []
        for f in match_files:
            df = pd.read_csv(os.path.join(match_dir, f))
            match_dfs.append(df)
        
        all_matches = pd.concat(match_dfs, ignore_index=True)
        all_matches.to_csv(os.path.join(output_dir, "all_matches.csv"), index=False)
        logger.info(f"Combined {len(all_matches)} total match records")
    
    # Combine player stats files
    player_dir = os.path.join(output_dir, "player_stats")
    player_files = sorted([f for f in os.listdir(player_dir) if f.endswith(".csv")])

    if player_files:
        player_dfs = []
        for f in player_files:
            df = pd.read_csv(os.path.join(player_dir, f))
            player_dfs.append(df)

        all_players = pd.concat(player_dfs, ignore_index=True)
        all_players.to_csv(os.path.join(output_dir, "all_player_stats.csv"), index=False)
        logger.info(f"Combined {len(all_players)} total player stat records")

    # Combine player details files
    details_dir = os.path.join(output_dir, "player_details")
    if os.path.exists(details_dir):
        details_files = sorted([f for f in os.listdir(details_dir) if f.endswith(".csv")])
        if details_files:
            details_dfs = [pd.read_csv(os.path.join(details_dir, f)) for f in details_files]
            all_details = pd.concat(details_dfs, ignore_index=True)
            all_details.to_csv(os.path.join(output_dir, "all_player_details.csv"), index=False)
            logger.info(f"Combined {len(all_details)} total player detail records")

    # Combine scoring files
    scoring_dir = os.path.join(output_dir, "scoring")
    if os.path.exists(scoring_dir):
        scoring_files = sorted([f for f in os.listdir(scoring_dir) if f.endswith(".csv")])
        if scoring_files:
            scoring_dfs = [pd.read_csv(os.path.join(scoring_dir, f)) for f in scoring_files]
            all_scoring = pd.concat(scoring_dfs, ignore_index=True)
            all_scoring.to_csv(os.path.join(output_dir, "all_scoring.csv"), index=False)
            logger.info(f"Combined {len(all_scoring)} total scoring events")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AFL Tables Data Scraper")
    parser.add_argument("--start", type=int, default=2005, help="Start year (default: 2005)")
    parser.add_argument("--end", type=int, default=2025, help="End year (default: 2025)")
    parser.add_argument("--output", type=str, default="./data", help="Output directory (default: ./data)")
    parser.add_argument("--test", action="store_true", help="Test mode: scrape 1 match only")
    
    args = parser.parse_args()
    
    logger.info(f"AFL Tables Scraper")
    logger.info(f"Scraping seasons {args.start} to {args.end}")
    logger.info(f"Output directory: {args.output}")
    
    if args.test:
        logger.info("TEST MODE: Scraping 1 match from most recent season")
        result = scrape_season_matches(args.end)
        if result:
            matches, match_links = result
            if match_links:
                result = scrape_match_player_stats(match_links[0], args.end)
                player_stats = result["player_stats"]
                player_details = result["player_details"]
                scoring = result["scoring"]
                match_info = result["match_info"]

                logger.info(f"\nMatch Info:\n{json.dumps(match_info, indent=2, default=str)}")
                os.makedirs(args.output, exist_ok=True)

                if player_stats:
                    df = pd.DataFrame(player_stats)
                    logger.info(f"\nPlayer Stats ({len(df)} rows):\n{df.head(10).to_string()}")
                    df.to_csv(os.path.join(args.output, "test_player_stats.csv"), index=False)
                    logger.info(f"Saved to {args.output}/test_player_stats.csv")

                if player_details:
                    df = pd.DataFrame(player_details)
                    logger.info(f"\nPlayer Details ({len(df)} rows):\n{df.head(10).to_string()}")
                    df.to_csv(os.path.join(args.output, "test_player_details.csv"), index=False)
                    logger.info(f"Saved to {args.output}/test_player_details.csv")

                if scoring:
                    df = pd.DataFrame(scoring)
                    logger.info(f"\nScoring Progression ({len(df)} rows):\n{df.head(10).to_string()}")
                    df.to_csv(os.path.join(args.output, "test_scoring.csv"), index=False)
                    logger.info(f"Saved to {args.output}/test_scoring.csv")
    else:
        scrape_seasons(args.start, args.end, args.output)


if __name__ == "__main__":
    main()
