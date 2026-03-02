"""
AFL Prediction Pipeline — Weather Data Fetcher
================================================
Fetches historical weather data from Open-Meteo's free Historical Weather API
for all AFL match venues and dates. No API key required.

Stages:
  1. Venue coordinate mapping (25 venues → lat/lon)
  2. Fetch historical weather per match (cached)
  3. Store as data/base/weather.parquet
  4. Derive weather features for the feature matrix
"""

import time
import json
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
from pathlib import Path

import config

# ---------------------------------------------------------------------------
# Stage 1: Venue Coordinates
# ---------------------------------------------------------------------------

# All 25 venues that appear in matches.parquet, mapped to (lat, lon, is_roofed).
# Coordinates are for the stadium centre. is_roofed=True only for Docklands
# (Marvel Stadium) which has a retractable roof that is typically closed in
# poor weather, making outdoor weather less relevant.

VENUE_COORDINATES = {
    "M.C.G.":              (-37.8200, 144.9834, False),
    "Docklands":           (-37.8165, 144.9475, True),   # Marvel Stadium — retractable roof
    "Adelaide Oval":       (-34.9156, 138.5961, False),
    "Perth Stadium":       (-31.9505, 115.8890, False),  # Optus Stadium
    "Gabba":               (-27.4858, 153.0381, False),
    "Carrara":             (-28.0067, 153.3661, False),   # Heritage Bank Stadium
    "S.C.G.":              (-33.8917, 151.2247, False),
    "Kardinia Park":       (-38.1581, 144.3544, False),   # GMHBA Stadium
    "Sydney Showground":   (-33.8441, 151.0694, False),
    "Subiaco":             (-31.9440, 115.8292, False),
    "York Park":           (-41.4282, 147.1384, False),   # UTAS Stadium, Launceston
    "Bellerive Oval":      (-42.8776, 147.3748, False),   # Hobart
    "Manuka Oval":         (-35.3181, 149.1347, False),   # Canberra
    "Marrara Oval":        (-12.3984, 130.8738, False),   # TIO Stadium, Darwin
    "Eureka Stadium":      (-37.5450, 143.8371, False),   # Mars Stadium, Ballarat
    "Cazaly's Stadium":    (-16.9274, 145.7482, False),   # Cairns
    "Traeger Park":        (-23.7066, 133.8751, False),   # Alice Springs
    "Norwood Oval":        (-34.9214, 138.6351, False),   # Adelaide
    "Stadium Australia":   (-33.8473, 151.0634, False),   # Accor Stadium, Sydney Olympic Park
    "Jiangwan Stadium":    (31.2800,  121.5200, False),   # Shanghai, China
    "Summit Sports Park":  (-37.7439, 145.0978, False),   # Chirnside Park, Melbourne
    "Barossa Oval":        (-34.6200, 138.9500, False),   # Tanunda, SA
    "Hands Oval":          (-37.5614, 143.8542, False),   # Buninyong near Ballarat
    "Riverway Stadium":    (-19.2850, 146.7293, False),   # Townsville
    "Wellington":          (-41.2729, 174.7866, False),   # Wellington, New Zealand
}


# ---------------------------------------------------------------------------
# Stage 2: Fetch Weather from Open-Meteo
# ---------------------------------------------------------------------------

CACHE_DIR = Path("data/base/weather_cache")
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

RATE_LIMIT_SECONDS = 0.5

# Default game duration if not available (minutes)
DEFAULT_GAME_DURATION = 120


def _cache_path(match_id):
    """Return path to cached JSON for a given match_id."""
    return CACHE_DIR / f"{match_id}.json"


def _aggregate_game_window(hourly, game_indices):
    """Aggregate hourly weather data for the game window into match-level metrics.

    Returns dict of aggregated values, or None if no valid data.
    """
    def _safe_vals(key):
        vals = hourly.get(key, [])
        return [vals[i] for i in game_indices if i < len(vals) and vals[i] is not None]

    temps = _safe_vals("temperature_2m")
    apparent = _safe_vals("apparent_temperature")
    precip = _safe_vals("precipitation")
    rain = _safe_vals("rain")
    wind = _safe_vals("wind_speed_10m")
    gusts = _safe_vals("wind_gusts_10m")
    humidity = _safe_vals("relative_humidity_2m")
    dewpoint = _safe_vals("dew_point_2m")
    cloud = _safe_vals("cloud_cover")
    pressure = _safe_vals("surface_pressure")

    if not temps:
        return None

    return {
        # Temperature
        "temperature_avg": round(np.mean(temps), 1),
        "temperature_min": round(min(temps), 1),
        "temperature_max": round(max(temps), 1),
        "apparent_temperature_avg": round(np.mean(apparent), 1) if apparent else None,
        # Precipitation
        "precipitation_total": round(sum(precip), 1) if precip else 0.0,
        "rain_total": round(sum(rain), 1) if rain else 0.0,
        # Wind
        "wind_speed_avg": round(np.mean(wind), 1) if wind else None,
        "wind_speed_max": round(max(wind), 1) if wind else None,
        "wind_gusts_max": round(max(gusts), 1) if gusts else None,
        # Humidity & dew point
        "humidity_avg": round(np.mean(humidity), 1) if humidity else None,
        "dew_point_avg": round(np.mean(dewpoint), 1) if dewpoint else None,
        # Cloud & pressure
        "cloud_cover_avg": round(np.mean(cloud), 1) if cloud else None,
        "pressure_avg": round(np.mean(pressure), 1) if pressure else None,
    }


def fetch_weather_for_match(lat, lon, date_str, match_id, start_hour=None, game_duration_min=None):
    """Fetch hourly weather from Open-Meteo for a single match date/location.

    Uses actual match start time + duration to select the precise weather window.
    For example, a 19:20 start with 120-minute duration → hours 19, 20, 21.

    Args:
        lat, lon: venue coordinates
        date_str: "YYYY-MM-DD"
        match_id: unique match identifier
        start_hour: hour the match started (e.g. 19 for 19:20). If None, uses 12-21 window.
        game_duration_min: game duration in minutes. If None, uses DEFAULT_GAME_DURATION.

    Returns dict with aggregated weather metrics, or None on failure.
    Uses file-based cache to avoid re-fetching.
    """
    cache_file = _cache_path(match_id)
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    # Fetch expanded set of hourly variables
    hourly_vars = (
        "temperature_2m,apparent_temperature,precipitation,rain,"
        "wind_speed_10m,wind_gusts_10m,relative_humidity_2m,"
        "dew_point_2m,cloud_cover,surface_pressure"
    )
    url = (
        f"{OPEN_METEO_URL}"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&hourly={hourly_vars}"
        f"&timezone=Australia%2FSydney"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AFL-Pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"    WARNING: Failed to fetch weather for match {match_id}: {e}")
        return None

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    # Determine game window hours
    if start_hour is not None:
        duration = game_duration_min if game_duration_min else DEFAULT_GAME_DURATION
        end_hour = start_hour + int(np.ceil(duration / 60))
        game_hours = set(range(start_hour, end_hour + 1))
    else:
        game_hours = set(range(12, 22))

    game_indices = []
    for i, t in enumerate(times):
        hour = int(t.split("T")[1].split(":")[0])
        if hour in game_hours:
            game_indices.append(i)

    if not game_indices:
        print(f"    WARNING: No game-window hours for match {match_id}")
        return None

    result = _aggregate_game_window(hourly, game_indices)
    if result is None:
        print(f"    WARNING: No valid temperature data for match {match_id}")
        return None

    result["match_id"] = int(match_id)

    # Cache the raw hourly data for the game window alongside aggregates,
    # so we can re-derive features without re-fetching
    raw_window = {}
    for key in hourly:
        if key == "time":
            raw_window[key] = [hourly[key][i] for i in game_indices]
        else:
            raw_window[key] = [hourly[key][i] for i in game_indices]
    result["_raw_hourly"] = raw_window

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(result, f)

    return result


def fetch_all_weather(matches_df, batch_size=None):
    """Fetch weather for all matches. Returns DataFrame.

    Args:
        matches_df: DataFrame with match_id, venue, date, game_time_minutes columns
        batch_size: If set, only fetch this many (for testing)
    """
    records = []
    total = len(matches_df) if batch_size is None else min(batch_size, len(matches_df))
    fetched_from_api = 0

    for i, (_, row) in enumerate(matches_df.iterrows()):
        if batch_size is not None and i >= batch_size:
            break

        match_id = row["match_id"]
        venue = row["venue"]
        match_dt = pd.Timestamp(row["date"])
        date_str = match_dt.strftime("%Y-%m-%d")

        if venue not in VENUE_COORDINATES:
            print(f"    WARNING: Unknown venue '{venue}' for match {match_id}")
            continue

        lat, lon, _ = VENUE_COORDINATES[venue]

        # Extract actual match start hour and game duration
        start_hour = match_dt.hour if match_dt.hour > 0 else None
        game_dur = row.get("game_time_minutes")
        if pd.notna(game_dur) and game_dur > 0:
            game_dur = float(game_dur)
        else:
            game_dur = None

        # Check cache first (don't count toward rate limiting)
        cache_file = _cache_path(match_id)
        is_cached = cache_file.exists()

        result = fetch_weather_for_match(lat, lon, date_str, match_id, start_hour, game_dur)
        if result:
            result["is_roofed"] = VENUE_COORDINATES[venue][2]
            records.append(result)

        if not is_cached and result is not None:
            fetched_from_api += 1
            if fetched_from_api % 100 == 0:
                print(f"  Fetched {i + 1}/{total}...")
            time.sleep(RATE_LIMIT_SECONDS)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total}...")

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# Stage 3: Save weather.parquet
# ---------------------------------------------------------------------------

WEATHER_FLOAT_COLS = [
    "temperature_avg", "temperature_min", "temperature_max",
    "apparent_temperature_avg", "precipitation_total", "rain_total",
    "wind_speed_avg", "wind_speed_max", "wind_gusts_max",
    "humidity_avg", "dew_point_avg", "cloud_cover_avg", "pressure_avg",
]


def save_weather_parquet(weather_df):
    """Save weather data to parquet with optimized dtypes."""
    out = weather_df.copy()

    # Drop raw hourly cache column (only for JSON cache files)
    out = out.drop(columns=["_raw_hourly"], errors="ignore")

    out["match_id"] = out["match_id"].astype(np.int64)
    for col in WEATHER_FLOAT_COLS:
        if col in out.columns:
            out[col] = out[col].astype(np.float32)
    out["is_roofed"] = out["is_roofed"].astype(bool)

    path = config.BASE_STORE_DIR / "weather.parquet"
    out.to_parquet(path, index=False)
    print(f"  Saved {len(out)} rows to {path}")
    return path


# ---------------------------------------------------------------------------
# Stage 4: Derive Weather Features
# ---------------------------------------------------------------------------

def derive_weather_features(weather_df):
    """Add derived weather feature columns to the weather DataFrame."""
    df = weather_df.copy()

    # --- Binary condition flags ---

    # is_wet: precipitation > 0.5mm during game window
    df["is_wet"] = (df["precipitation_total"] > 0.5).astype(np.int8)

    # is_heavy_rain: precipitation > 5mm
    df["is_heavy_rain"] = (df["precipitation_total"] > 5.0).astype(np.int8)

    # --- Categorical conditions ---

    # wind_severity: 0=calm (<15), 1=moderate (15-30), 2=strong (30+)
    df["wind_severity"] = pd.cut(
        df["wind_gusts_max"],
        bins=[-np.inf, 15, 30, np.inf],
        labels=[0, 1, 2],
    ).astype(np.int8)

    # temperature_category: 0=cool (<12C), 1=mild (12-22C), 2=hot (>22C)
    df["temperature_category"] = pd.cut(
        df["temperature_avg"],
        bins=[-np.inf, 12, 22, np.inf],
        labels=[0, 1, 2],
    ).astype(np.int8)

    # --- Continuous derived features ---

    # Wind chill / heat stress: difference between actual and apparent temperature
    # Negative = feels colder than actual (wind chill), Positive = feels hotter (humidity)
    if "apparent_temperature_avg" in df.columns:
        df["feels_like_delta"] = (
            df["apparent_temperature_avg"] - df["temperature_avg"]
        ).round(1).astype(np.float32)
    else:
        df["feels_like_delta"] = np.float32(0.0)

    # Humidity discomfort: high humidity impairs performance
    # Dew point > 16°C is uncomfortable, > 20°C is oppressive for sport
    if "dew_point_avg" in df.columns:
        df["humidity_discomfort"] = (
            df["dew_point_avg"].clip(lower=10) - 10
        ).clip(lower=0).round(1).astype(np.float32)
    else:
        df["humidity_discomfort"] = np.float32(0.0)

    # Temperature swing during the game: affects player adjustment
    if "temperature_max" in df.columns and "temperature_min" in df.columns:
        df["temperature_range"] = (
            df["temperature_max"] - df["temperature_min"]
        ).round(1).astype(np.float32)
    else:
        df["temperature_range"] = np.float32(0.0)

    # Overcast factor: high cloud cover affects visibility, mood, crowd energy
    if "cloud_cover_avg" in df.columns:
        df["is_overcast"] = (df["cloud_cover_avg"] > 80).astype(np.int8)
    else:
        df["is_overcast"] = np.int8(0)

    # --- Composite scores ---

    # weather_difficulty_score: composite 0-10 score
    # Higher = harder conditions (more rain, more wind, extreme temps)
    rain_score = df["precipitation_total"].clip(0, 20) / 20 * 4  # 0-4 points
    wind_score = df["wind_gusts_max"].clip(0, 60) / 60 * 3       # 0-3 points
    temp_dev = (df["temperature_avg"] - 17).abs()
    temp_score = temp_dev.clip(0, 15) / 15 * 3                   # 0-3 points
    df["weather_difficulty_score"] = (rain_score + wind_score + temp_score).round(2).astype(np.float32)

    # Slippery conditions score: combines rain + humidity + dew point
    # Wet ground affects contested ball, marking, kicking accuracy
    rain_factor = df["precipitation_total"].clip(0, 10) / 10     # 0-1
    humid_factor = df["humidity_avg"].clip(60, 100) / 100         # 0.6-1 → 0-1 range
    humid_factor = (humid_factor - 0.6) / 0.4                    # normalize to 0-1
    df["slippery_conditions"] = ((rain_factor * 0.7 + humid_factor * 0.3) * 10).round(2).astype(np.float32)

    # --- Roofed venue override ---
    # For roofed venues, zero out outdoor-dependent weather features
    roofed_mask = df["is_roofed"]
    roofed_zero_cols = [
        "is_wet", "is_heavy_rain", "wind_severity", "is_overcast",
        "weather_difficulty_score", "slippery_conditions", "feels_like_delta",
    ]
    for col in roofed_zero_cols:
        df.loc[roofed_mask, col] = 0

    return df


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python weather.py [test|fetch|features]")
        sys.exit(1)

    cmd = sys.argv[1]

    matches = pd.read_parquet(config.BASE_STORE_DIR / "matches.parquet")
    print(f"Loaded {len(matches)} matches")

    if cmd == "test":
        # Stage 2 test: fetch 10 matches
        sample = matches.sample(10, random_state=42).sort_values("date")
        print(f"\nFetching weather for {len(sample)} test matches...")
        weather_df = fetch_all_weather(sample, batch_size=10)
        print(f"\nResults ({len(weather_df)} rows):")
        print(weather_df.to_string())

    elif cmd == "fetch":
        # Stage 3: fetch all matches
        print(f"\nFetching weather for all {len(matches)} matches...")
        weather_df = fetch_all_weather(matches)
        weather_df = derive_weather_features(weather_df)
        save_weather_parquet(weather_df)
        print(f"\nSummary:")
        print(weather_df.describe())

    elif cmd == "features":
        # Stage 4: derive features from existing weather.parquet
        weather_path = config.BASE_STORE_DIR / "weather.parquet"
        if not weather_path.exists():
            print("ERROR: weather.parquet not found. Run 'fetch' first.")
            sys.exit(1)
        weather_df = pd.read_parquet(weather_path)
        weather_df = derive_weather_features(weather_df)
        save_weather_parquet(weather_df)
        print(f"\nFeature distributions:")
        print(f"  Wet games:       {weather_df['is_wet'].sum()} / {len(weather_df)}")
        print(f"  Heavy rain:      {weather_df['is_heavy_rain'].sum()} / {len(weather_df)}")
        print(f"  Roofed games:    {weather_df['is_roofed'].sum()} / {len(weather_df)}")
        print(f"  Wind severity:   {weather_df['wind_severity'].value_counts().to_dict()}")
        print(f"  Temp category:   {weather_df['temperature_category'].value_counts().to_dict()}")
