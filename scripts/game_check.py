"""
Lightweight AFL game completion checker for GitHub Actions.
============================================================

Compares FootyWire's completed games vs what's in Supabase to determine
if there are new results to process. Designed to run fast (<30s) with
minimal dependencies (just requests + beautifulsoup4).

Outputs GitHub Actions step outputs:
  - has_new_games: 'true' if new completed games found
  - round_complete: 'true' if the current active round is fully completed
  - current_round: the latest round number with any results

Usage:
    python scripts/game_check.py              # check current year
    YEAR=2026 python scripts/game_check.py    # explicit year
"""

import json
import os
import re
import sys
from datetime import date
from urllib.request import Request, urlopen

YEAR = int(os.environ.get("YEAR", date.today().year))
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

FW_MATCH_LIST_URL = f"https://www.footywire.com/afl/footy/ft_match_list?year={YEAR}"
HEADERS = {"User-Agent": "AFL-Stats-Research-Bot/1.0 (research project)"}


def get_footywire_completed():
    """Fetch FootyWire match list and count completed games per round.

    Returns dict of {round_number: completed_game_count}.
    """
    req = Request(FW_MATCH_LIST_URL, headers=HEADERS)
    html = urlopen(req, timeout=20).read().decode("utf-8", errors="replace")

    from html.parser import HTMLParser

    # Simple state-machine HTML parser (no bs4 dependency needed)
    completed = {}
    current_round = 0

    # Parse round anchors: <a name="round_1">
    for m in re.finditer(r'<a\s+name="round[_\s]*(\d+)"', html, re.IGNORECASE):
        pass  # just confirming format

    # Split by round anchors
    parts = re.split(r'<a\s+name="round[_\s]*(\d+)"', html, flags=re.IGNORECASE)
    # parts = [before_round_1, "1", html_for_round_1, "2", html_for_round_2, ...]
    i = 1
    while i < len(parts) - 1:
        round_num = int(parts[i])
        round_html = parts[i + 1]

        # Count score patterns like "123-99" in this round's HTML
        # These appear in the score column of the match list table
        scores = re.findall(r'>\s*(\d{2,3})\s*-\s*(\d{2,3})\s*<', round_html)
        completed[round_num] = len(scores)

        i += 2

    return completed


def get_supabase_match_counts():
    """Query Supabase for completed match counts per round this year.

    Returns dict of {round_number: match_count}.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Warning: Supabase credentials not set, skipping DB check")
        return {}

    url = (
        f"{SUPABASE_URL}/rest/v1/matches"
        f"?year=eq.{YEAR}&select=round_number"
    )
    req = Request(url, headers={
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    })
    try:
        data = json.loads(urlopen(req, timeout=10).read())
        counts = {}
        for row in data:
            rnd = row.get("round_number")
            if rnd is not None:
                counts[rnd] = counts.get(rnd, 0) + 1
        return counts
    except Exception as e:
        print(f"Warning: Supabase query failed: {e}")
        return {}


def get_fixture_counts():
    """Query Supabase for expected games per round from fixtures.

    Returns dict of {round_number: expected_game_count}.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {}

    url = (
        f"{SUPABASE_URL}/rest/v1/fixtures"
        f"?year=eq.{YEAR}&is_home=eq.true&select=round_number"
    )
    req = Request(url, headers={
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    })
    try:
        data = json.loads(urlopen(req, timeout=10).read())
        counts = {}
        for row in data:
            rnd = row.get("round_number")
            if rnd is not None:
                counts[rnd] = counts.get(rnd, 0) + 1
        return counts
    except Exception as e:
        print(f"Warning: Supabase fixtures query failed: {e}")
        return {}


def write_output(key, value):
    """Write a GitHub Actions step output."""
    gh_output = os.environ.get("GITHUB_OUTPUT", "")
    if gh_output:
        with open(gh_output, "a") as f:
            f.write(f"{key}={value}\n")
    print(f"  {key}={value}")


def main():
    print(f"Checking for new AFL games ({YEAR})...")

    # 1. Get FootyWire completed games per round
    try:
        fw = get_footywire_completed()
    except Exception as e:
        print(f"Error fetching FootyWire: {e}")
        write_output("has_new_games", "false")
        write_output("round_complete", "false")
        write_output("current_round", "0")
        return

    if not fw:
        print("No completed games found on FootyWire")
        write_output("has_new_games", "false")
        write_output("round_complete", "false")
        write_output("current_round", "0")
        return

    print(f"  FootyWire: {sum(fw.values())} total completed games across {len(fw)} rounds")
    for rnd in sorted(fw.keys()):
        if fw[rnd] > 0:
            print(f"    Round {rnd}: {fw[rnd]} games")

    # 2. Get Supabase match counts
    sb = get_supabase_match_counts()
    print(f"  Supabase: {sum(sb.values())} matches in DB for {YEAR}")

    # 3. Compare: are there new games?
    has_new = False
    new_game_rounds = []
    for rnd, fw_count in sorted(fw.items()):
        sb_count = sb.get(rnd, 0)
        if fw_count > sb_count:
            has_new = True
            new_game_rounds.append(rnd)
            print(f"  NEW: Round {rnd} has {fw_count} on FootyWire vs {sb_count} in DB")

    # 4. Determine current round (highest round WITH completed games)
    rounds_with_games = [r for r, c in fw.items() if c > 0]
    current_round = max(rounds_with_games) if rounds_with_games else 0
    fixture_counts = get_fixture_counts()

    round_complete = False
    if current_round:
        expected = fixture_counts.get(current_round, 9)  # default 9 games per round
        actual = fw.get(current_round, 0)
        if actual >= expected and expected > 0:
            round_complete = True
            print(f"  Round {current_round} COMPLETE ({actual}/{expected} games)")
        else:
            print(f"  Round {current_round} in progress ({actual}/{expected} games)")

    # 5. Write outputs
    write_output("has_new_games", "true" if has_new else "false")
    write_output("round_complete", "true" if round_complete else "false")
    write_output("current_round", str(current_round))

    if not has_new:
        print("\nNo new games to process.")
    else:
        print(f"\nNew games detected in round(s): {new_game_rounds}")


if __name__ == "__main__":
    main()
