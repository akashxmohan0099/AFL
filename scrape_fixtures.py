"""
Scrape AFL fixtures from FootyWire and write per-round CSVs.

Usage:
    python scrape_fixtures.py              # current year
    python scrape_fixtures.py --year 2026  # specific year

Output:
    data/fixtures/round_{N}_{YEAR}.csv
    Each CSV has: team, opponent, venue, date, is_home
"""
from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

import config

# FootyWire team name → canonical name mapping
FW_TEAM_MAP = {
    "Adelaide": "Adelaide",
    "Brisbane": "Brisbane Lions",
    "Carlton": "Carlton",
    "Collingwood": "Collingwood",
    "Essendon": "Essendon",
    "Fremantle": "Fremantle",
    "Geelong": "Geelong",
    "Gold Coast": "Gold Coast",
    "GWS": "Greater Western Sydney",
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

VENUE_MAP = getattr(config, "VENUE_NAME_MAP", {})
HEADERS = {"User-Agent": "AFL-Stats-Research-Bot/1.0 (research project)"}


def _normalise_team(name: str) -> str:
    """Map FootyWire display name to canonical team name."""
    return FW_TEAM_MAP.get(name.strip(), name.strip())


def _normalise_venue(venue: str) -> str:
    """Keep FootyWire venue names as-is (pipeline expects FootyWire names)."""
    return venue.strip()


def _parse_date(date_str: str, year: int) -> str:
    """Parse 'Thu 26 Mar 7:30pm' into 'YYYY-MM-DD'."""
    date_str = date_str.strip().lstrip("\xa0").strip()
    # Try multiple formats
    for fmt in ["%a %d %b %I:%M%p", "%a %d %b %I:%M %p", "%a %d %b"]:
        try:
            dt = datetime.strptime(date_str, fmt).replace(year=year)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


def scrape_fixtures(year: int) -> dict[int, list[dict]]:
    """Scrape all fixtures for a season from FootyWire.

    Returns dict of {round_number: [match_dicts]}.
    Each match_dict has: home, away, venue, date.
    """
    url = f"https://www.footywire.com/afl/footy/ft_match_list?year={year}"
    req = Request(url, headers=HEADERS)
    html = urlopen(req, timeout=30).read().decode("utf-8", errors="replace")

    # Split by round anchors
    parts = re.split(r'<a\s+name="round[_\s]*(\d+)"', html, flags=re.IGNORECASE)

    fixtures: dict[int, list[dict]] = {}

    for i in range(1, len(parts) - 1, 2):
        rnd = int(parts[i])
        rnd_html = parts[i + 1]

        # Extract matches: each row has date, "Home v Away", venue
        rows = re.findall(
            r'<td[^>]*class="data"[^>]*>\s*(?:&nbsp;)?\s*([^<]*)</td>\s*'
            r'<td[^>]*class="data"[^>]*>\s*'
            r'<a[^>]*>([^<]+)</a>\s*v\s*<a[^>]*>([^<]+)</a>\s*'
            r'</td>\s*'
            r'<td[^>]*class="data"[^>]*>([^<]*)</td>',
            rnd_html,
            re.DOTALL,
        )

        matches = []
        for date_raw, home_raw, away_raw, venue_raw in rows:
            home = _normalise_team(home_raw)
            away = _normalise_team(away_raw)
            venue = _normalise_venue(venue_raw)
            date = _parse_date(date_raw, year)
            matches.append({
                "home": home,
                "away": away,
                "venue": venue,
                "date": date,
            })

        if matches:
            fixtures[rnd] = matches

    return fixtures


def write_fixture_csvs(
    fixtures: dict[int, list[dict]], year: int, out_dir: Path | None = None
) -> list[Path]:
    """Write per-round CSV files in the format the pipeline expects.

    Format: team, opponent, venue, date, is_home
    Two rows per match (one for each team).
    """
    out_dir = out_dir or config.FIXTURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for rnd, matches in sorted(fixtures.items()):
        path = out_dir / f"round_{rnd}_{year}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["team", "opponent", "venue", "date", "is_home"])
            writer.writeheader()
            for m in matches:
                writer.writerow({
                    "team": m["home"],
                    "opponent": m["away"],
                    "venue": m["venue"],
                    "date": m["date"],
                    "is_home": 1,
                })
                writer.writerow({
                    "team": m["away"],
                    "opponent": m["home"],
                    "venue": m["venue"],
                    "date": m["date"],
                    "is_home": 0,
                })
        written.append(path)

    return written


def main():
    parser = argparse.ArgumentParser(description="Scrape AFL fixtures from FootyWire")
    parser.add_argument("--year", type=int, default=config.CURRENT_SEASON_YEAR)
    args = parser.parse_args()

    print(f"Scraping {args.year} fixtures from FootyWire...")
    fixtures = scrape_fixtures(args.year)

    if not fixtures:
        print("No fixtures found!")
        return

    total_matches = sum(len(m) for m in fixtures.values())
    print(f"  Found {total_matches} matches across {len(fixtures)} rounds")

    paths = write_fixture_csvs(fixtures, args.year)
    for p in paths:
        print(f"  Wrote {p.name}")

    print(f"\nDone: {len(paths)} fixture files written to {config.FIXTURES_DIR}")


if __name__ == "__main__":
    main()
