-- AFL Predict Pro — Supabase Schema
-- Run this in the Supabase SQL Editor to create all tables.

-- ============================================================
-- 1. MATCHES — one row per match
-- ============================================================
CREATE TABLE matches (
  match_id         BIGINT PRIMARY KEY,
  home_team        TEXT NOT NULL,
  away_team        TEXT NOT NULL,
  year             SMALLINT NOT NULL,
  round_number     SMALLINT NOT NULL,
  date             TIMESTAMPTZ,
  venue            TEXT,
  is_finals        BOOLEAN DEFAULT FALSE,
  round_label      TEXT,
  home_score       SMALLINT,
  away_score       SMALLINT,
  margin           SMALLINT,
  total_score      SMALLINT,
  home_total_gl    SMALLINT,
  home_total_bh    SMALLINT,
  home_total_di    SMALLINT,
  home_total_mk    SMALLINT,
  home_total_tk    SMALLINT,
  home_total_if    SMALLINT,
  home_total_cl    SMALLINT,
  home_total_cp    SMALLINT,
  home_total_rb    SMALLINT,
  away_total_gl    SMALLINT,
  away_total_bh    SMALLINT,
  away_total_di    SMALLINT,
  away_total_mk    SMALLINT,
  away_total_tk    SMALLINT,
  away_total_if    SMALLINT,
  away_total_cl    SMALLINT,
  away_total_cp    SMALLINT,
  away_total_rb    SMALLINT,
  total_di         SMALLINT,
  total_if         SMALLINT,
  total_cp         SMALLINT,
  attendance       INTEGER,
  game_time_minutes REAL,
  home_rushed_behinds REAL,
  away_rushed_behinds REAL
);

CREATE INDEX idx_matches_year_round ON matches (year, round_number);
CREATE INDEX idx_matches_venue ON matches (venue);
CREATE INDEX idx_matches_home ON matches (home_team, year);
CREATE INDEX idx_matches_away ON matches (away_team, year);

-- ============================================================
-- 2. PLAYER_GAMES — one row per (player, match)
-- ============================================================
CREATE TABLE player_games (
  match_id           BIGINT NOT NULL REFERENCES matches(match_id),
  player_id          TEXT NOT NULL,
  player             TEXT NOT NULL,
  team               TEXT NOT NULL,
  opponent           TEXT NOT NULL,
  venue              TEXT,
  date               TIMESTAMPTZ,
  year               SMALLINT NOT NULL,
  round_number       SMALLINT NOT NULL,
  round_label        TEXT,
  jumper             SMALLINT,
  is_home            BOOLEAN,
  is_finals          BOOLEAN,
  did_not_play       BOOLEAN DEFAULT FALSE,
  age_years          REAL,
  pct_played         REAL,
  -- Core stats
  ki                 REAL, -- kicks
  mk                 REAL, -- marks
  hb                 REAL, -- handballs
  di                 REAL, -- disposals
  gl                 REAL, -- goals
  bh                 REAL, -- behinds
  ho                 REAL, -- hitouts
  tk                 REAL, -- tackles
  rb                 REAL, -- rebounds
  "if"               REAL, -- inside 50s
  cl                 REAL, -- clearances
  cg                 REAL, -- clangers
  ff                 REAL, -- frees for
  fa                 REAL, -- frees against
  br                 REAL, -- brownlow votes
  cp                 REAL, -- contested possessions
  up                 REAL, -- uncontested possessions
  cm                 REAL, -- contested marks
  mi                 REAL, -- marks inside 50
  one_pct            REAL, -- one percenters
  bo                 REAL, -- bounces
  ga                 REAL, -- goal assists
  -- Quarter scoring
  q1_goals SMALLINT, q1_behinds SMALLINT,
  q2_goals SMALLINT, q2_behinds SMALLINT,
  q3_goals SMALLINT, q3_behinds SMALLINT,
  q4_goals SMALLINT, q4_behinds SMALLINT,
  -- Career context
  career_games_pre   SMALLINT,
  career_goals_pre   SMALLINT,
  career_goal_avg_pre REAL,
  -- Era context
  season_era         SMALLINT,
  is_covid_season    SMALLINT,
  quarter_length_ratio REAL,

  PRIMARY KEY (match_id, player_id)
);

CREATE INDEX idx_pg_player ON player_games (player_id);
CREATE INDEX idx_pg_player_year ON player_games (player_id, year);
CREATE INDEX idx_pg_team_year ON player_games (team, year);
CREATE INDEX idx_pg_year_round ON player_games (year, round_number);
CREATE INDEX idx_pg_venue ON player_games (venue);
CREATE INDEX idx_pg_opponent ON player_games (opponent);

-- ============================================================
-- 3. TEAM_MATCHES — one row per (team, match)
-- ============================================================
CREATE TABLE team_matches (
  match_id      BIGINT NOT NULL REFERENCES matches(match_id),
  team          TEXT NOT NULL,
  opponent      TEXT NOT NULL,
  year          SMALLINT NOT NULL,
  round_number  SMALLINT NOT NULL,
  date          TIMESTAMPTZ,
  venue         TEXT,
  is_home       BOOLEAN,
  is_finals     BOOLEAN,
  attendance    INTEGER,
  score         SMALLINT,
  opp_score     SMALLINT,
  margin        SMALLINT,
  result        TEXT, -- W/L/D
  rest_days     REAL,
  gl SMALLINT, bh SMALLINT, di SMALLINT, mk SMALLINT,
  tk SMALLINT, cp SMALLINT, "if" SMALLINT, cl SMALLINT, rb SMALLINT,

  PRIMARY KEY (match_id, team)
);

CREATE INDEX idx_tm_team_year ON team_matches (team, year);
CREATE INDEX idx_tm_year_round ON team_matches (year, round_number);

-- ============================================================
-- 4. PREDICTIONS — player-level model predictions
-- ============================================================
CREATE TABLE predictions (
  year              SMALLINT NOT NULL,
  round_number      SMALLINT NOT NULL,
  match_id          BIGINT,
  player            TEXT NOT NULL,
  team              TEXT NOT NULL,
  opponent          TEXT,
  venue             TEXT,
  player_role       TEXT,
  career_goal_avg   REAL,
  -- Goals
  predicted_goals   REAL,
  predicted_behinds REAL,
  predicted_score   REAL,
  lambda_goals      REAL,
  lambda_behinds    REAL,
  p_scorer          REAL,
  p_scorer_raw      REAL,
  p_1plus_goals     REAL,
  p_2plus_goals     REAL,
  p_3plus_goals     REAL,
  p_1plus_goals_raw REAL,
  p_2plus_goals_raw REAL,
  p_3plus_goals_raw REAL,
  p_goals_0 REAL, p_goals_1 REAL, p_goals_2 REAL, p_goals_3 REAL,
  p_goals_4 REAL, p_goals_5 REAL, p_goals_6 REAL, p_goals_7plus REAL,
  p_behinds_0 REAL, p_behinds_1 REAL, p_behinds_2 REAL,
  p_behinds_3 REAL, p_behinds_4plus REAL,
  conf_lower_gl     INTEGER,
  conf_upper_gl     INTEGER,
  -- Disposals
  predicted_disposals REAL,
  lambda_disposals    REAL,
  p_10plus_disp REAL, p_15plus_disp REAL, p_20plus_disp REAL,
  p_25plus_disp REAL, p_30plus_disp REAL,
  conf_lower_di     INTEGER,
  conf_upper_di     INTEGER,
  -- Marks
  predicted_marks   REAL,
  lambda_marks      REAL,
  p_mark_taker      REAL,
  p_2plus_mk REAL, p_3plus_mk REAL, p_4plus_mk REAL,
  p_5plus_mk REAL, p_6plus_mk REAL, p_7plus_mk REAL,
  p_8plus_mk REAL, p_9plus_mk REAL, p_10plus_mk REAL,
  conf_lower_mk     INTEGER,
  conf_upper_mk     INTEGER,

  PRIMARY KEY (year, round_number, player, team)
);

CREATE INDEX idx_pred_year_round ON predictions (year, round_number);
CREATE INDEX idx_pred_match ON predictions (match_id);
CREATE INDEX idx_pred_player ON predictions (player, team);

-- ============================================================
-- 5. OUTCOMES — actual results matched to predictions
-- ============================================================
CREATE TABLE outcomes (
  year              SMALLINT NOT NULL,
  round_number      SMALLINT NOT NULL,
  match_id          BIGINT,
  player            TEXT NOT NULL,
  team              TEXT NOT NULL,
  actual_goals      REAL,
  actual_behinds    REAL,
  actual_disposals  REAL,
  actual_marks      REAL,

  PRIMARY KEY (year, round_number, player, team)
);

CREATE INDEX idx_out_year_round ON outcomes (year, round_number);
CREATE INDEX idx_out_match ON outcomes (match_id);

-- ============================================================
-- 6. GAME_PREDICTIONS — match winner predictions
-- ============================================================
CREATE TABLE game_predictions (
  year                     SMALLINT NOT NULL,
  round_number             SMALLINT NOT NULL,
  match_id                 BIGINT,
  home_team                TEXT NOT NULL,
  away_team                TEXT NOT NULL,
  venue                    TEXT,
  home_win_prob            REAL,
  away_win_prob            REAL,
  predicted_margin         REAL,
  predicted_winner         TEXT,
  home_elo                 REAL,
  away_elo                 REAL,
  hybrid_prob_home         REAL,
  residual_prob_home       REAL,
  market_prior_prob_home   REAL,
  market_prior_available   SMALLINT,

  PRIMARY KEY (year, round_number, home_team, away_team)
);

CREATE INDEX idx_gp_year_round ON game_predictions (year, round_number);
CREATE INDEX idx_gp_match ON game_predictions (match_id);

-- ============================================================
-- 7. FIXTURES — upcoming match schedule
-- ============================================================
CREATE TABLE fixtures (
  year          SMALLINT NOT NULL,
  round_number  SMALLINT NOT NULL,
  team          TEXT NOT NULL,
  opponent      TEXT NOT NULL,
  venue         TEXT,
  date          TEXT,
  is_home       BOOLEAN,

  PRIMARY KEY (year, round_number, team)
);

CREATE INDEX idx_fix_year_round ON fixtures (year, round_number);

-- ============================================================
-- 8. ODDS — match-level betting odds
-- ============================================================
CREATE TABLE odds (
  match_id                    BIGINT PRIMARY KEY REFERENCES matches(match_id),
  market_home_implied_prob    REAL,
  market_away_implied_prob    REAL,
  market_handicap             REAL,
  market_total_score          REAL,
  market_confidence           REAL,
  odds_movement_home          REAL,
  odds_movement_line          REAL,
  betfair_home_implied_prob   REAL
);

-- ============================================================
-- 9. PLAYER_ODDS — player-level betting markets
-- ============================================================
CREATE TABLE player_odds (
  match_id                     BIGINT NOT NULL,
  player                       TEXT NOT NULL,
  market_disposal_line         REAL,
  market_disposal_over_price   REAL,
  market_disposal_implied_over REAL,
  market_fgs_price             REAL,
  market_fgs_implied_prob      REAL,
  market_2goals_price          REAL,
  market_2goals_implied_prob   REAL,
  market_3goals_price          REAL,

  PRIMARY KEY (match_id, player)
);

CREATE INDEX idx_po_match ON player_odds (match_id);

-- ============================================================
-- 10. UMPIRES
-- ============================================================
CREATE TABLE umpires (
  match_id            BIGINT NOT NULL REFERENCES matches(match_id),
  year                SMALLINT,
  umpire_name         TEXT NOT NULL,
  umpire_career_games SMALLINT,
  umpire_url          TEXT,

  PRIMARY KEY (match_id, umpire_name)
);

-- ============================================================
-- 11. COACHES
-- ============================================================
CREATE TABLE coaches (
  match_id           BIGINT NOT NULL REFERENCES matches(match_id),
  year               SMALLINT,
  team               TEXT NOT NULL,
  coach              TEXT,
  coach_career_games INTEGER,
  coach_wins         INTEGER,
  coach_draws        INTEGER,
  coach_losses       INTEGER,
  coach_win_pct      REAL,

  PRIMARY KEY (match_id, team)
);

-- ============================================================
-- 12. WEATHER
-- ============================================================
CREATE TABLE weather (
  match_id                   BIGINT PRIMARY KEY REFERENCES matches(match_id),
  temperature_avg            REAL,
  temperature_min            REAL,
  temperature_max            REAL,
  apparent_temperature_avg   REAL,
  precipitation_total        REAL,
  rain_total                 REAL,
  wind_speed_avg             REAL,
  wind_speed_max             REAL,
  wind_gusts_max             REAL,
  wind_direction_avg         REAL,
  wind_direction_variability REAL,
  humidity_avg               REAL,
  dew_point_avg              REAL,
  cloud_cover_avg            REAL,
  pressure_avg               REAL,
  is_roofed                  BOOLEAN,
  is_wet                     SMALLINT,
  is_heavy_rain              SMALLINT,
  wind_severity              SMALLINT,
  temperature_category       SMALLINT,
  feels_like_delta           REAL,
  humidity_discomfort         REAL,
  temperature_range          REAL,
  is_overcast                SMALLINT,
  weather_difficulty_score   REAL,
  slippery_conditions        REAL
);

-- ============================================================
-- 13. EXPERIMENTS — backtest results (stored as JSONB)
-- ============================================================
CREATE TABLE experiments (
  id       SERIAL PRIMARY KEY,
  filename TEXT UNIQUE NOT NULL,
  label    TEXT,
  data     JSONB NOT NULL
);

-- ============================================================
-- 14. NEWS — injuries + intel (stored as JSONB, refreshed daily)
-- ============================================================
CREATE TABLE news_injuries (
  id         SERIAL PRIMARY KEY,
  team       TEXT NOT NULL,
  player     TEXT NOT NULL,
  injury     TEXT,
  severity   SMALLINT,
  severity_label TEXT,
  estimated_return TEXT,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_injuries_team ON news_injuries (team);

CREATE TABLE news_intel (
  id         SERIAL PRIMARY KEY,
  data       JSONB NOT NULL, -- full intel JSON blob
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE news_team_lists (
  year          SMALLINT NOT NULL,
  round_number  SMALLINT NOT NULL,
  data          JSONB NOT NULL, -- full team changes JSON
  updated_at    TIMESTAMPTZ DEFAULT NOW(),

  PRIMARY KEY (year, round_number)
);

-- ============================================================
-- 15. SIMULATIONS — pre-computed Monte Carlo results
-- ============================================================
CREATE TABLE simulations (
  match_id      BIGINT NOT NULL,
  year          SMALLINT NOT NULL,
  round_number  SMALLINT NOT NULL,
  home_team     TEXT NOT NULL,
  away_team     TEXT NOT NULL,
  n_sims        INTEGER NOT NULL,
  -- Match-level outcomes
  home_win_pct  REAL,
  away_win_pct  REAL,
  draw_pct      REAL,
  avg_home_score REAL,
  avg_away_score REAL,
  avg_total     REAL,
  avg_margin    REAL,
  -- Full result stored as JSONB for flexibility
  match_outcomes JSONB, -- score distributions, percentiles, brackets
  players       JSONB, -- per-player simulation distributions
  suggested_multis JSONB, -- correlated multi suggestions

  PRIMARY KEY (match_id)
);

CREATE INDEX idx_sim_year_round ON simulations (year, round_number);
