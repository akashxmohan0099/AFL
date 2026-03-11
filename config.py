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
BASE_STORE_DIR = DATA_DIR / "base"         # Layer 1-3: typed parquets
CLEANED_DIR = DATA_DIR / "cleaned"          # Legacy (kept for migration)
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"
FIXTURES_DIR = DATA_DIR / "fixtures"
LEARNING_DIR = DATA_DIR / "learning"        # Layer 4: persistent learning store

# Scraper output subdirectories (matches scraper.py layout)
PLAYER_STATS_DIR = DATA_DIR / "player_stats"
PLAYER_DETAILS_DIR = DATA_DIR / "player_details"
SCORING_DIR = DATA_DIR / "scoring"
MATCHES_DIR = DATA_DIR / "matches"
UMPIRES_DIR = DATA_DIR / "umpires"
COACHES_DIR = DATA_DIR / "coaches"
PLAYER_PROFILES_DIR = DATA_DIR / "player_profiles"
FOOTYWIRE_DIR = DATA_DIR / "footywire"
SUPERCOACH_DIR = DATA_DIR / "supercoach"
NEWS_DIR = DATA_DIR / "news"

# ---------------------------------------------------------------------------
# Era weights for recency — recent seasons matter more
# ---------------------------------------------------------------------------
ERA_WEIGHTS = {
    (2015, 2019): 0.4,
    (2020, 2022): 0.7,
    (2023, 2024): 0.9,
    (2025, 2026): 1.0,
}
RECENCY_DECAY_HALF_LIFE = 250  # days — faster decay (was 365)

# Dynamic current-season weighting (used in sequential mode)
CURRENT_SEASON_BOOST_BASE = 1.3       # base multiplier for current-season rows
CURRENT_SEASON_BOOST_PER_ROUND = 0.08 # additional boost per round of current-season data available
CURRENT_SEASON_BOOST_MAX = 3.0        # cap
WITHIN_SEASON_RECENCY_HALF_LIFE = 8   # rounds — within current season, recent rounds matter more

# ---------------------------------------------------------------------------
# Season era / rule regime mapping
# ---------------------------------------------------------------------------
ERA_MAP = {
    2015: 1,                    # 120 interchange cap, substitute rule
    2016: 2, 2017: 2, 2018: 2, # 90 interchange cap, no substitute
    2019: 3,                    # 6-6-6 starting positions introduced, major tactical shift
    2020: 4,                    # COVID — 16-min quarters, hubs, compressed schedule
    2021: 5, 2022: 5,          # 75 interchange cap, medical substitute, normal quarters
    2023: 6, 2024: 6, 2025: 6, # General substitute, stand rule matured
    2026: 7,                    # No centre bounces, 5-man interchange, top-10 wildcard finals
}
COVID_SEASON_YEAR = 2020
COVID_QUARTER_LENGTH_RATIO = 0.8  # 16/20 minutes
CURRENT_PREDICTION_ERA = 7  # era for upcoming/future predictions

# ---------------------------------------------------------------------------
# Feature engineering parameters
# ---------------------------------------------------------------------------
ROLLING_WINDOWS = [3, 5, 10]
VENUE_LOOKBACK_YEARS = 3
MATCHUP_MIN_MATCHES = 3
ENABLER_COUNT = 3  # top-N teammate enablers per player
N_ARCHETYPES = 6    # GMM archetype clusters (Forward, Midfielder, Ruck, Defender, Tagger, Utility)

# Umpire features
UMPIRE_LOOKBACK_MATCHES = 20
UMPIRE_MIN_GAMES = 10

# Coach features
COACH_MIN_GAMES = 10
COACH_H2H_MIN_MATCHES = 3

# Player physical attributes
PLAYER_PROFILE_HEIGHT_FALLBACK = 186.0   # median AFL height cm
PLAYER_PROFILE_WEIGHT_FALLBACK = 86.0    # median AFL weight kg
PLAYER_PROFILE_AGE_FALLBACK = 25.0

# Career splits
CAREER_SPLIT_MIN_GAMES = 3
CAREER_SPLIT_FEATURES_ENABLED = False  # snapshot career splits are leaky for historical rows unless explicitly approved

# Injury list
INJURY_LOOKBACK_MONTHS = 12

# UI-only/news intel features that should not be used for model training
# until we have properly versioned historical snapshots in a remote store.
MODEL_EXCLUDED_FEATURES = {
    "team_n_ins",
    "team_n_outs",
    "team_n_debutants",
    "team_stability",
    "team_churn_3r",
    "opp_n_ins",
    "opp_n_outs",
    "opp_n_debutants",
    "opp_stability",
    "opp_churn_3r",
    "is_debutant",
    "team_injured_count",
    "team_injury_severity",
    "opp_injured_count",
    "opp_injury_severity",
}

# Career disposal cold-start fallback (used when no prior match history exists)
CAREER_DISP_AVG_FALLBACK = 15.0

# Team venue records
TEAM_VENUE_LOOKBACK_YEARS = 5
TEAM_VENUE_MIN_GAMES = 3
TEAM_VENUE_SCORE_FALLBACK = 80.0

# ---------------------------------------------------------------------------
# Role classification thresholds (from last 10 matches averages)
# Lowered from original to classify more players (previously all got "general")
# ---------------------------------------------------------------------------
RUCK_HO_THRESHOLD = 10
KEY_FORWARD_GL_THRESHOLD = 0.7
KEY_FORWARD_MI_THRESHOLD = 1.0
SMALL_FORWARD_GL_THRESHOLD = 0.3
SMALL_FORWARD_IF_THRESHOLD = 2
KEY_DEFENDER_RB_THRESHOLD = 2.5
KEY_DEFENDER_ONE_PCT_THRESHOLD = 2.5
MIDFIELDER_DI_THRESHOLD = 18
MIDFIELDER_CL_THRESHOLD = 2

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
ENSEMBLE_WEIGHTS = {"poisson": 0.8, "gbt": 0.2}

# ---------------------------------------------------------------------------
# XGBoost parameters (Phase 1 deep learning upgrade)
# ---------------------------------------------------------------------------
XGB_ENABLED = True
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
    "verbosity": 0,
}
XGB_PARAMS_BACKTEST = {
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 15,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
    "verbosity": 0,
}
ENSEMBLE_WEIGHTS_3WAY = {"poisson": 0.5, "gbt": 0.2, "xgb": 0.3}

# LightGBM parameters (Phase 3 stacking)
LGBM_PARAMS_BACKTEST = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

# Stacking meta-ensemble (Phase 3)
STACKING_ENABLED = False
STACKING_N_FOLDS = 4
STACKING_META_ALPHA = 1.0
STACKING_META_ALPHA_BACKTEST = 10.0

# Entity embeddings (Phase 2)
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDING_ENABLED = False  # v5.0 test: no improvement over baseline
EMBEDDING_DIMS = {"player_id": 32, "team": 8, "opponent": 8, "venue": 8, "archetype": 4}
EMBEDDING_HIDDEN_DIM = 128
EMBEDDING_EPOCHS = 30
EMBEDDING_BATCH_SIZE = 2048
EMBEDDING_LR = 1e-3

# Sequence model (Phase 4)
SEQUENCE_DIR = DATA_DIR / "sequence"
SEQUENCE_ENABLED = False  # v5.0 test: no improvement over baseline
SEQUENCE_LOOKBACK = 20
SEQUENCE_HIDDEN_DIM = 64
SEQUENCE_OUTPUT_DIM = 32
SEQUENCE_N_LAYERS = 2
SEQUENCE_EPOCHS = 20
SEQUENCE_BATCH_SIZE = 512
SEQUENCE_LR = 5e-4

# Optuna tuning ranges for XGBoost
TUNE_XGB_RANGES = {
    "n_estimators": (100, 600),
    "max_depth": (3, 8),
    "learning_rate": (0.01, 0.2),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "min_child_weight": (5, 50),
    "reg_alpha": (0.0, 2.0),
    "reg_lambda": (0.5, 5.0),
}
DISPOSAL_DISTRIBUTION = "gaussian"  # 'poisson', 'gaussian', or 'negbin' — Gaussian wins at all thresholds (2025 backtest)

# Upper-tail correction for Gaussian disposal probabilities.
# DISABLED — raw Gaussian CDF is well-calibrated (std ratio ~0.98 vs actuals).
# These heuristics were compensating for broken isotonic calibration.
DISPOSAL_UPPER_TAIL_ENABLED = False
DISPOSAL_UPPER_TAIL_THRESHOLDS = [25, 30]
DISPOSAL_UPPER_TAIL_STD_MULTIPLIER = 1.2
DISPOSAL_UPPER_TAIL_SKEW_ALPHA = 2.0
DISPOSAL_30PLUS_PROB_SCALE = 1.0
DISPOSAL_30PLUS_PROB_CAP = 1.0

GBT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_samples_leaf": 20,
    "subsample": 0.8,
    "random_state": RANDOM_SEED,
}

# Lighter params for backtest (faster iteration, similar diagnostic value)
GBT_PARAMS_BACKTEST = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_samples_leaf": 25,
    "subsample": 0.8,
    "random_state": RANDOM_SEED,
}

# HistGradientBoosting params for backtest (scoring model)
HIST_GBT_PARAMS_BACKTEST = {
    "max_iter": 250,
    "max_depth": 5,
    "learning_rate": 0.15,
    "min_samples_leaf": 20,
    "random_state": RANDOM_SEED,
}

POISSON_PARAMS = {
    "alpha": 0.014,
    "max_iter": 1000,
}

# Disposal-specific params (separate from scoring)
DISPOSAL_GBT_PARAMS_BACKTEST = {
    "max_iter": 217,
    "max_depth": 5,
    "learning_rate": 0.052,
    "min_samples_leaf": 10,
    "random_state": RANDOM_SEED,
}
DISPOSAL_POISSON_PARAMS = {
    "alpha": 0.015,
    "max_iter": 1000,
}

# Marks-specific params (separate from scoring/disposals)
MARKS_DISTRIBUTION = "negbin"
MARKS_GBT_PARAMS_BACKTEST = {
    "max_iter": 250,
    "max_depth": 5,
    "learning_rate": 0.06,
    "min_samples_leaf": 15,
    "random_state": RANDOM_SEED,
}
MARKS_POISSON_PARAMS = {
    "alpha": 0.02,
    "max_iter": 1000,
}
MARKS_THRESHOLDS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
MARKS_TAKER_THRESHOLD = 4     # binary classifier: P(marks >= 4) — cleaner 50/50 split than threshold 3
MARKS_TAKER_BLEND = 0.55      # adjustment weight for mark-taker prob — stronger separation for threshold 4

# ---------------------------------------------------------------------------
# Market / odds features
# ---------------------------------------------------------------------------
MARKET_FEATURES_ENABLED = True
# Raw bookmaker / Betfair source directory (used by integrate_odds*.py).
# Override with AFL_ODDS_DIR for portability across machines.
ODDS_DIR = Path(os.getenv("AFL_ODDS_DIR", BASE_DIR / "AFL Betting odds"))
ODDS_PARQUET_PATH = BASE_STORE_DIR / "odds.parquet"
MARKET_POISSON_BLEND = 0.10   # blend market-implied baseline into Poisson component
# Market player-feature ablation mode:
#   "full"            -> full chain (environment + player implied goals/disposals)
#   "env_only"        -> keep only market_expected_match_goals + market_expected_team_goals
#   "player_goal_only"-> keep only market_implied_player_goals (intermediates hidden)
#   "v31_legacy"      -> revert to v3.1 market feature behavior
MARKET_PLAYER_FEATURE_CONFIG = "v31_legacy"
MARKET_POINTS_PER_GOAL_FALLBACK = 6.2
MARKET_POINTS_PER_DISPOSAL_FALLBACK = 0.38

# ---------------------------------------------------------------------------
# Elo rating system parameters
# ---------------------------------------------------------------------------
ELO_K_FACTOR = 12.3
ELO_HOME_ADVANTAGE = 16.2
ELO_SEASON_REGRESSION = 0.285

# ---------------------------------------------------------------------------
# Game winner model parameters
# ---------------------------------------------------------------------------
GAME_WINNER_PARAMS = {
    "max_iter": 64,
    "max_depth": 4,
    "learning_rate": 0.032,
    "min_samples_leaf": 13,
    "random_state": RANDOM_SEED,
}
GAME_WINNER_PARAMS_BACKTEST = {
    "max_iter": 64,
    "max_depth": 4,
    "learning_rate": 0.032,
    "min_samples_leaf": 13,
    "random_state": RANDOM_SEED,
}

# Hybrid winner mode: market prior + residual ML logit
WINNER_HYBRID_ENABLED = True
WINNER_HYBRID_ALPHA = 0.5
WINNER_HYBRID_BETA = 0.5
WINNER_HYBRID_BIAS = 0.0
WINNER_MARKET_EPS = 1e-6
WINNER_MIN_MARKET_COVERAGE = 0.30

# ---------------------------------------------------------------------------
# Streak feature parameters
# ---------------------------------------------------------------------------
STREAK_DECAY = 0.85          # Exponential decay factor for weighted streaks
HOT_THRESHOLD = 1.5          # form_ratio above this = hot
COLD_THRESHOLD = 0.5         # form_ratio below this = cold
STREAK_BROKE_MIN = 3         # Minimum streak length to trigger "just broke"

# ---------------------------------------------------------------------------
# Miss classification thresholds
# ---------------------------------------------------------------------------
MISS_ERROR_THRESHOLD = 1.5          # abs goal error >= this triggers classification
MISS_OPPORTUNITY_PCT_PLAYED = 60    # pct_played < this → opportunity miss
MISS_TEAM_ENV_DEVIATION = 0.25      # team scored 25%+ fewer/more goals than avg → team env
MISS_MATCHUP_CONCESSION_DEV = 0.3   # opponent conceded 30%+ differently from profile → matchup

# ---------------------------------------------------------------------------
# Probability calibration evaluation
# ---------------------------------------------------------------------------
GOAL_THRESHOLDS = {"1plus_goals": 1, "2plus_goals": 2, "3plus_goals": 3}
DISPOSAL_THRESHOLDS_EVAL = {"10plus_disp": 10, "15plus_disp": 15, "20plus_disp": 20, "25plus_disp": 25, "30plus_disp": 30}
MARKS_THRESHOLDS_EVAL = {"2plus_mk": 2, "3plus_mk": 3, "4plus_mk": 4, "5plus_mk": 5, "6plus_mk": 6, "7plus_mk": 7, "8plus_mk": 8, "9plus_mk": 9, "10plus_mk": 10}
CALIBRATION_METHOD = "isotonic"    # isotonic regression calibration (bucket params kept for analysis)
ISOTONIC_MIN_SAMPLES = 100         # min predictions before fitting isotonic calibrator
ISOTONIC_REFIT_INTERVAL = 5        # refit every N rounds in sequential mode
ISOTONIC_SKIP_TARGETS = {
    "1plus_goals", "2plus_goals", "3plus_goals",  # two-stage scorer already calibrates well
    "game_winner",  # hybrid blend already calibrated; isotonic creates feedback loop in sequential
    # Disposals + marks: isotonic ENABLED — raw Gaussian/NegBin CDFs compress predictions
    # into a narrow band (~0.28-0.32). Isotonic calibration spreads them to full [0,1] range
    # and dramatically improves BSS (3% → 40%+ for disposals).
}
CALIBRATION_N_BUCKETS = 10
CALIBRATION_MIN_BUCKET_SIZE = 5

# ---------------------------------------------------------------------------
# Model targets (future: add "disposals" entry)
# ---------------------------------------------------------------------------
MODEL_TARGETS = {
    "scoring": {
        "targets": ["GL", "BH"],
        "primary_target": "GL",
        "description": "Goals and behinds prediction",
    },
    "disposals": {
        "targets": ["DI"],
        "primary_target": "DI",
        "description": "Disposals prediction",
        "thresholds": [20, 25, 30],
    },
    "marks": {
        "targets": ["MK"],
        "primary_target": "MK",
        "description": "Marks prediction",
        "thresholds": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    },
}

# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
VALIDATION_YEAR = 2024
MIN_PLAYER_MATCHES = 5
HISTORICAL_START_YEAR = 2015
HISTORICAL_END_YEAR = 2026
CURRENT_SEASON_YEAR = 2026

# Backtesting
BACKTEST_TRAIN_MIN_YEARS = 3
BACKTEST_DIR = DATA_DIR / "backtest"
EXPERIMENTS_DIR = DATA_DIR / "experiments"

# Sequential learning
SEQUENTIAL_DIR = DATA_DIR / "sequential"
SEQUENTIAL_YEAR = 2025
CALIBRATION_MIN_SAMPLES = 30       # min predictions per bucket before applying adjustment
CALIBRATION_MAX_ADJUSTMENT = 0.3   # cap adjustment to prevent wild swings
DISPOSAL_THRESHOLDS = [10, 15, 20, 25, 30]
GOAL_DISTRIBUTION_MAX_K = 7       # compute P(0), P(1), ..., P(6), P(7+)

# ---------------------------------------------------------------------------
# Hyperparameter tuning (Optuna)
# ---------------------------------------------------------------------------
TUNING_DIR = DATA_DIR / "tuning"

TUNE_N_TRIALS = 50
TUNE_WALK_FORWARD_FOLDS = 3  # number of walk-forward validation folds

# Parameter search ranges (for Optuna suggest_*)
TUNE_GBT_RANGES = {
    "n_estimators": (100, 500),
    "max_depth": (2, 6),
    "learning_rate": (0.01, 0.15),
    "min_samples_leaf": (10, 50),
    "subsample": (0.6, 1.0),
}
TUNE_POISSON_ALPHA_RANGE = (0.01, 2.0)
TUNE_ENSEMBLE_POISSON_WEIGHT_RANGE = (0.2, 0.8)
TUNE_ELO_RANGES = {
    "k_factor": (10, 60),
    "home_advantage": (10, 50),
    "season_regression": (0.2, 0.8),
}

# ---------------------------------------------------------------------------
# Name matching maps (AFLTables profile pages → pipeline names)
# ---------------------------------------------------------------------------
VENUE_NAME_MAP = {
    "M.C.G.": "M.C.G.",
    "MCG": "M.C.G.",
    "Melbourne Cricket Ground": "M.C.G.",
    "Docklands": "Docklands",
    "Marvel Stadium": "Docklands",
    "Etihad Stadium": "Docklands",
    "S.C.G.": "S.C.G.",
    "SCG": "S.C.G.",
    "Sydney Cricket Ground": "S.C.G.",
    "Gabba": "Gabba",
    "The Gabba": "Gabba",
    "Brisbane Cricket Ground": "Gabba",
    "Kardinia Park": "Kardinia Park",
    "GMHBA Stadium": "Kardinia Park",
    "Simonds Stadium": "Kardinia Park",
    "Skilled Stadium": "Kardinia Park",
    "Subiaco": "Subiaco",
    "Football Park": "Football Park",
    "AAMI Stadium": "Football Park",
    "Adelaide Oval": "Adelaide Oval",
    "Optus Stadium": "Perth Stadium",
    "Perth Stadium": "Perth Stadium",
    "York Park": "York Park",
    "Aurora Stadium": "York Park",
    "University of Tasmania Stadium": "York Park",
    "Cazalys Stadium": "Cazalys Stadium",
    "TIO Stadium": "TIO Stadium",
    "Marrara Oval": "TIO Stadium",
    "Manuka Oval": "Manuka Oval",
    "Bellerive Oval": "Bellerive Oval",
    "Blundstone Arena": "Bellerive Oval",
    "Carrara": "Carrara",
    "Metricon Stadium": "Carrara",
    "Heritage Bank Stadium": "Carrara",
    "People First Stadium": "Carrara",
    "Giants Stadium": "Giants Stadium",
    "Sydney Showground": "Giants Stadium",
    "ENGIE Stadium": "Giants Stadium",
    "Showground Stadium": "Giants Stadium",
    "Jiangwan Stadium": "Jiangwan Stadium",
    "Adelaide Arena at Jiangwan Stadium": "Jiangwan Stadium",
    "Stadium Australia": "Stadium Australia",
    "ANZ Stadium": "Stadium Australia",
    "Accor Stadium": "Stadium Australia",
    "Traeger Park": "Traeger Park",
    "Eureka Stadium": "Eureka Stadium",
    "Mars Stadium": "Eureka Stadium",
    "Riverway Stadium": "Riverway Stadium",
    "Norwood Oval": "Norwood Oval",
}

# Canonical venue → display name (current sponsor + city)
VENUE_DISPLAY_MAP = {
    "M.C.G.":           "MCG, Melbourne",
    "Docklands":        "Marvel Stadium, Melbourne",
    "S.C.G.":           "SCG, Sydney",
    "Gabba":            "The Gabba, Brisbane",
    "Kardinia Park":    "GMHBA Stadium, Geelong",
    "Adelaide Oval":    "Adelaide Oval, Adelaide",
    "Perth Stadium":    "Optus Stadium, Perth",
    "Carrara":          "People First Stadium, Gold Coast",
    "Sydney Showground": "Engie Stadium, Sydney",
    "Giants Stadium":   "Engie Stadium, Sydney",
    "York Park":        "UTAS Stadium, Launceston",
    "Bellerive Oval":   "Blundstone Arena, Hobart",
    "Manuka Oval":      "Manuka Oval, Canberra",
    "Subiaco":          "Subiaco Oval, Perth",
    "Eureka Stadium":   "Mars Stadium, Ballarat",
    "TIO Stadium":      "TIO Stadium, Darwin",
    "Marrara Oval":     "TIO Stadium, Darwin",
    "Cazalys Stadium":  "Cazalys Stadium, Cairns",
    "Cazaly's Stadium": "Cazalys Stadium, Cairns",
    "Traeger Park":     "Traeger Park, Alice Springs",
    "Stadium Australia": "Accor Stadium, Sydney",
    "Jiangwan Stadium": "Jiangwan Stadium, Shanghai",
    "Riverway Stadium": "Riverway Stadium, Townsville",
    "Norwood Oval":     "Norwood Oval, Adelaide",
    "Football Park":    "Football Park, Adelaide",
    "Summit Sports Park": "Summit Sports Park, Gold Coast",
    "Barossa Oval":     "Barossa Oval, Adelaide",
    "Hands Oval":       "Hands Oval, Bunbury",
    "Wellington":       "Wellington, NZ",
    "SCG":              "SCG, Sydney",
}

TEAM_NAME_MAP = {
    # Profile page names → pipeline canonical names
    "Adelaide": "Adelaide",
    "Brisbane Lions": "Brisbane Lions",
    "Brisbane Bears": "Brisbane Bears",
    "Carlton": "Carlton",
    "Collingwood": "Collingwood",
    "Essendon": "Essendon",
    "Fremantle": "Fremantle",
    "Geelong": "Geelong",
    "Gold Coast": "Gold Coast",
    "GWS Giants": "Greater Western Sydney",
    "Greater Western Sydney": "Greater Western Sydney",
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
    "Footscray": "Western Bulldogs",
}

TEAM_HOME_GROUNDS = {
    "Adelaide": "Adelaide Oval",
    "Brisbane Lions": "Gabba",
    "Carlton": "M.C.G.",
    "Collingwood": "M.C.G.",
    "Essendon": "Docklands",
    "Fremantle": "Perth Stadium",
    "Geelong": "Kardinia Park",
    "Gold Coast": "Carrara",
    "Greater Western Sydney": "Sydney Showground",
    "Hawthorn": "M.C.G.",
    "Melbourne": "M.C.G.",
    "North Melbourne": "Docklands",
    "Port Adelaide": "Adelaide Oval",
    "Richmond": "M.C.G.",
    "St Kilda": "Docklands",
    "Sydney": "S.C.G.",
    "West Coast": "Perth Stadium",
    "Western Bulldogs": "Docklands",
}

# ---------------------------------------------------------------------------
# Multi-bet correlation engine
# ---------------------------------------------------------------------------
MULTI_N_SIMS = 10_000
MULTI_MAX_LEGS = 4
MULTI_MIN_EDGE = 0.03
MULTI_MIN_LEG_PROB = 0.30
MULTI_DIR = DATA_DIR / "multi"

# Market-type-aware overround margins (from bet365 analysis)
# These are the bookmaker's built-in margins that inflate implied probabilities.
# Our edge = model_prob - (book_implied_prob / (1 + overround))
# Note: _book_implied() in multi.py uses model_prob * (1 + overround) as a
# pessimistic placeholder when real odds are unavailable, ensuring no false edges.
# Only overlay_real_odds() with actual Betfair data produces meaningful edge signals.
# Lower overround = harder to find edge; higher = more margin baked in.
BOOK_OVERROUND = {
    "h2h":              0.047,   # Head-to-head / match winner (~4.7%)
    "line":             0.047,   # Line / handicap (~4.7%)
    "totals":           0.047,   # Over/under total score (~4.7%)
    "player_goals":     0.10,    # Anytime goalscorer (~10%)
    "player_disposals": 0.08,    # Player disposal milestones (~8%)
    "player_marks":     0.08,    # Player marks milestones (~8%)
    "first_goal":       0.15,    # First goalscorer (~15%)
    "margin_bands":     0.12,    # Margin band markets (~12%)
    "ht_ft":            0.11,    # Half-time/full-time (~11%)
    "total_5way":       0.14,    # Total 5-way (~14%)
    "sgm":              0.20,    # Same-game multi (~15-25%)
}

# ---------------------------------------------------------------------------
# Ensure directories exist
# ---------------------------------------------------------------------------
def ensure_dirs():
    """Create all required data directories."""
    for d in [DATA_DIR, RAW_DIR, BASE_STORE_DIR, CLEANED_DIR, FEATURES_DIR,
              MODELS_DIR, PREDICTIONS_DIR, FIXTURES_DIR, BACKTEST_DIR,
              LEARNING_DIR, SEQUENTIAL_DIR, TUNING_DIR, EXPERIMENTS_DIR,
              UMPIRES_DIR, COACHES_DIR, PLAYER_PROFILES_DIR, FOOTYWIRE_DIR,
              SUPERCOACH_DIR, NEWS_DIR, EMBEDDINGS_DIR, SEQUENCE_DIR,
              MULTI_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    # Sequential learning subdirectories
    for subdir in ["predictions", "outcomes", "diagnostics", "analysis",
                   "game_predictions", "calibration", "archetypes", "concessions"]:
        (SEQUENTIAL_DIR / subdir).mkdir(parents=True, exist_ok=True)
