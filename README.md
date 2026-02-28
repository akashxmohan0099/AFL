# AFL Tables Scraper

A Python tool to scrape historical AFL match data and per-game player statistics from afltables.com.

## What it collects

### Match Data
- Round, date, venue, attendance
- Home and away teams
- Quarter-by-quarter scores (G.B.Total format)
- Final scores and margin

### Player Stats (per game, per player)
- KI (Kicks), MK (Marks), HB (Handballs), DI (Disposals)
- GL (Goals), BH (Behinds)
- HO (Hit Outs), TK (Tackles)
- RB (Rebound 50s), IF (Inside 50s)
- CL (Clearances), CG (Clangers)
- FF (Free Kicks For), FA (Free Kicks Against)
- BR (Brownlow Votes)
- CP (Contested Possessions), UP (Uncontested Possessions)
- CM (Contested Marks), MI (Marks Inside 50)
- 1% (One Percenters), BO (Bounces), GA (Goal Assists)
- %P (Percentage of Game Played)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Test with a single match first
```bash
python scraper.py --test --end 2024
```

### Scrape a single season
```bash
python scraper.py --start 2024 --end 2024 --output ./data
```

### Scrape full range (2005-2025)
```bash
python scraper.py --start 2005 --end 2025 --output ./data
```

## Output Structure

```
data/
  matches/
    matches_2005.csv
    matches_2006.csv
    ...
  player_stats/
    player_stats_2005.csv
    player_stats_2006.csv
    ...
  all_matches.csv          # combined master file
  all_player_stats.csv     # combined master file
```

## Resumable

The scraper checks for existing season files before scraping. If a season's files already exist, it skips that season. Delete the files to re-scrape.

## Rate Limiting

The scraper waits 1.5 seconds between requests to be respectful to the afltables.com server. For ~4,200 matches across 21 seasons, expect roughly 2 hours total scrape time.

## Estimated Data Volume

- ~4,200 matches (2005-2025)
- ~185,000 player-match records
- ~50MB total CSV data

## Next Steps After Scraping

1. Load CSVs into PostgreSQL or SQLite
2. Build feature engineering pipeline (rolling averages, form indicators)
3. Train prediction models
