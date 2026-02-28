"""
AFL Prediction Pipeline — Configuration
========================================
All tunable parameters in one place. Import from here, never hardcode
magic numbers in feature engineering or model code.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"
FIXTURES_DIR = DATA_DIR / "fixtures"

# Scraper output subdirectories (matches scraper.py layout)
PLAYER_STATS_DIR = DATA_DIR / "player_stats"
PLAYER_DETAILS_DIR = DATA_DIR / "player_details"
SCORING_DIR = DATA_DIR / "scoring"
MATCHES_DIR = DATA_DIR / "matches"

# ---------------------------------------------------------------------------
# Era weights for recency — recent seasons matter more
# ---------------------------------------------------------------------------
ERA_WEIGHTS = {
    (2015, 2019): 0.4,
    (2020, 2022): 0.7,
    (2023, 2024): 0.9,
    (2025, 2025): 1.0,
}
RECENCY_DECAY_HALF_LIFE = 365  # days — used in exponential decay

# ---------------------------------------------------------------------------
# Feature engineering parameters
# ---------------------------------------------------------------------------
ROLLING_WINDOWS = [3, 5, 10]
VENUE_LOOKBACK_YEARS = 3
MATCHUP_MIN_MATCHES = 3
ENABLER_COUNT = 3  # top-N teammate enablers per player

# ---------------------------------------------------------------------------
# Role classification thresholds (from last 10 matches averages)
# ---------------------------------------------------------------------------
RUCK_HO_THRESHOLD = 15
KEY_FORWARD_GL_THRESHOLD = 1.0
KEY_FORWARD_MI_THRESHOLD = 1.5
SMALL_FORWARD_GL_THRESHOLD = 0.5
SMALL_FORWARD_IF_THRESHOLD = 3
KEY_DEFENDER_RB_THRESHOLD = 3
KEY_DEFENDER_ONE_PCT_THRESHOLD = 3
MIDFIELDER_DI_THRESHOLD = 20
MIDFIELDER_CL_THRESHOLD = 3

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
ENSEMBLE_WEIGHTS = {"poisson": 0.4, "gbt": 0.6}

GBT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_samples_leaf": 20,
    "subsample": 0.8,
}

POISSON_PARAMS = {
    "alpha": 1.0,       # regularization strength
    "max_iter": 1000,
}

# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
VALIDATION_YEAR = 2024
MIN_PLAYER_MATCHES = 5
HISTORICAL_START_YEAR = 2015
HISTORICAL_END_YEAR = 2025
CURRENT_SEASON_YEAR = 2025

# ---------------------------------------------------------------------------
# Ensure directories exist
# ---------------------------------------------------------------------------
def ensure_dirs():
    """Create all required data directories."""
    for d in [DATA_DIR, RAW_DIR, CLEANED_DIR, FEATURES_DIR, MODELS_DIR,
              PREDICTIONS_DIR, FIXTURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
