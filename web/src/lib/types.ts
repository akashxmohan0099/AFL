export interface PlayerSearchResult {
  player_id: string;
  name: string;
  team: string;
  total_games: number;
  score: number;
}

export interface PlayerDirectoryEntry {
  player_id: string;
  name: string;
  team: string;
  games: number;
  avg_goals: number;
  avg_disposals: number;
  avg_marks: number;
  avg_tackles: number;
  avg_kicks: number;
  avg_handballs: number;
  avg_hitouts: number;
}

export interface SeasonAvg {
  year: number;
  games: number;
  GL: number;
  BH: number;
  DI: number;
  MK: number;
  KI: number;
  HB: number;
  TK: number;
  HO: number;
}

export interface FormEntry {
  date: string;
  opponent: string;
  GL: number;
  BH: number;
  DI: number;
  MK: number;
  venue: string;
}

export interface PlayerProfile {
  player_id: string;
  name: string;
  team: string;
  total_games: number;
  career_goals: number;
  career_goal_avg: number;
  seasons: SeasonAvg[];
  recent_form: FormEntry[];
}

export interface VenueSplit {
  venue: string;
  games: number;
  avg_gl: number;
  avg_di: number;
  avg_mk: number;
  avg_ki: number;
  avg_hb: number;
  avg_tk: number;
}

export interface WeatherSplit {
  condition: string;
  category: string;
  games: number;
  avg_gl: number;
  avg_di: number;
  avg_mk: number;
}

export interface PredictionVsActual {
  year: number;
  round: number;
  opponent: string;
  venue: string;
  predicted_goals: number;
  actual_goals: number;
  predicted_disposals: number;
  actual_disposals: number;
  predicted_marks: number;
  actual_marks: number;
}

export interface PlayerProfileEnhanced extends PlayerProfile {
  venue_splits: VenueSplit[];
  weather_splits: WeatherSplit[];
  predictions_vs_actuals: PredictionVsActual[];
  prediction_mae: { goals: number; disposals: number; marks: number };
}

export interface GameLogEntry {
  match_id: number;
  date: string;
  year: number;
  round_number: number;
  team: string;
  opponent: string;
  venue: string;
  GL: number;
  BH: number;
  DI: number;
  MK: number;
  KI: number;
  HB: number;
  TK: number;
  HO: number;
}

export interface RoundInfo {
  year: number;
  round_number: number;
}

export interface PlayerPrediction {
  player: string;
  team: string;
  opponent: string;
  predicted_goals?: number;
  predicted_disposals?: number;
  predicted_marks?: number;
  predicted_behinds?: number;
  p_scorer?: number;
  p_2plus_goals?: number;
  p_3plus_goals?: number;
  p_10plus_disp?: number;
  p_15plus_disp?: number;
  p_20plus_disp?: number;
  p_25plus_disp?: number;
  p_30plus_disp?: number;
  player_role?: string;
  match_id?: number;
  venue?: string;
  round?: number;
  round_number?: number;
  lambda_goals?: number;
  lambda_disposals?: number;
  career_goal_avg?: number;
}

export interface MatchResult {
  match_id: number;
  home_team: string;
  away_team: string;
  venue?: string;
  date?: string;
  home_score?: number;
  away_score?: number;
  home_win_prob?: number;
  away_win_prob?: number;
  predicted_winner?: string;
  actual_winner?: string;
  correct?: boolean;
}

export interface MatchDetail extends MatchResult {
  year: number;
  round_number: number;
  home_players: PlayerPrediction[];
  away_players: PlayerPrediction[];
}

export interface OddsComparison {
  match_id: number;
  home_team: string;
  away_team: string;
  model_home_prob?: number;
  market_home_prob?: number;
  model_away_prob?: number;
  market_away_prob?: number;
  edge_home?: number;
  edge_away?: number;
}

export interface PlayerOddsComparison {
  player: string;
  team: string;
  market_type: string;
  market_line?: number;
  market_price?: number;
  market_implied_prob?: number;
  model_prob?: number;
  edge?: number;
}

export interface ExperimentSummary {
  filename: string;
  label: string;
  mae_goals?: number;
  mae_disposals?: number;
  mae_marks?: number;
  brier_1plus?: number;
  bss_1plus?: number;
  brier_2plus?: number;
  bss_20plus_disp?: number;
  bss_25plus_disp?: number;
  bss_5plus_mk?: number;
  game_winner_accuracy?: number;
}

export interface HealthStatus {
  status: string;
  player_games: number;
  matches: number;
  experiments: number;
  latest_data?: string;
  latest_data_full?: string;
}

// --- Season/match endpoints ---

export interface SeasonSummary {
  year: number;
  total_matches: number;
  total_rounds: number;
  completed_rounds: number;
  current_round: number;
  rounds_list: number[];
  accuracy: {
    goals_mae?: number;
    scorer_accuracy?: number;
    disposals_mae?: number;
    marks_mae?: number;
    game_winner_accuracy?: number;
  };
}

export interface TeamStatTotals {
  pred_gl?: number;
  pred_di?: number;
  pred_mk?: number;
  actual_gl?: number;
  actual_di?: number;
  actual_mk?: number;
}

export interface SeasonMatch {
  match_id: number;
  round_number: number;
  date?: string;
  venue?: string;
  home_team: string;
  away_team: string;
  home_score?: number;
  away_score?: number;
  actual_winner?: string;
  predicted_winner?: string;
  home_win_prob?: number;
  predicted_margin?: number;
  correct?: boolean;
  home_pred?: { pred_gl?: number; pred_di?: number; pred_mk?: number } | null;
  away_pred?: { pred_gl?: number; pred_di?: number; pred_mk?: number } | null;
  home_actual?: { actual_gl?: number; actual_di?: number; actual_mk?: number } | null;
  away_actual?: { actual_gl?: number; actual_di?: number; actual_mk?: number } | null;
}

export interface UpcomingRound {
  year: number;
  round_number: number;
  matches: {
    home_team: string;
    away_team: string;
    venue?: string;
    date?: string;
  }[];
  predictions: PlayerPrediction[];
}

export interface RoundAccuracy {
  round_number: number;
  n_players: number;
  goals_mae?: number;
  scorer_accuracy?: number;
  disposals_mae?: number;
  marks_mae?: number;
}

export interface MatchWeather {
  temperature_avg: number;
  precipitation_total: number;
  wind_speed_avg: number;
  humidity_avg: number;
  is_wet: boolean;
  is_roofed: boolean;
  weather_difficulty_score: number;
}

export interface MatchComparisonPlayer {
  player: string;
  team: string;
  is_home: boolean;
  actual_gl?: number;
  actual_bh?: number;
  actual_di?: number;
  actual_mk?: number;
  actual_ki?: number;
  actual_hb?: number;
  actual_tk?: number;
  actual_ho?: number;
  actual_cp?: number;
  actual_up?: number;
  actual_if?: number;
  actual_cl?: number;
  actual_cg?: number;
  actual_ff?: number;
  actual_fa?: number;
  predicted_gl?: number;
  predicted_di?: number;
  predicted_mk?: number;
  predicted_bh?: number;
  p_scorer?: number;
  player_role?: string;
  career_goal_avg?: number;
  conf_gl?: [number, number];
  conf_di?: [number, number];
  conf_mk?: [number, number];
  p_2plus_goals?: number;
  p_3plus_goals?: number;
  p_15plus_disp?: number;
  p_20plus_disp?: number;
  p_25plus_disp?: number;
  p_30plus_disp?: number;
  p_3plus_mk?: number;
  p_5plus_mk?: number;
  advanced?: PlayerAdvanced;
}

export interface PlayerAdvancedStats {
  games: number;
  // Averages (always present)
  avg_gl?: number;
  avg_bh?: number;
  avg_di?: number;
  avg_mk?: number;
  avg_ki?: number;
  avg_hb?: number;
  avg_tk?: number;
  avg_ho?: number;
  avg_cp?: number;
  avg_up?: number;
  avg_if?: number;
  avg_cl?: number;
  avg_cg?: number;
  avg_ff?: number;
  avg_fa?: number;
  // Medians (when full=true, >= 3 games)
  med_gl?: number;
  med_di?: number;
  med_mk?: number;
  med_tk?: number;
  med_ki?: number;
  med_hb?: number;
  med_ho?: number;
  // Max/Min (when full=true, >= 3 games)
  max_gl?: number;
  max_di?: number;
  max_mk?: number;
  max_tk?: number;
  max_ki?: number;
  max_hb?: number;
  max_ho?: number;
  min_gl?: number;
  min_di?: number;
  min_mk?: number;
  min_tk?: number;
  min_ki?: number;
  min_hb?: number;
  min_ho?: number;
  // Extra stats
  max_cp?: number; min_cp?: number;
  max_if?: number; min_if?: number;
  max_ff?: number; min_ff?: number;
  [key: string]: number | undefined;
}

export interface PlayerRecentGame {
  opponent: string;
  venue: string;
  gl?: number;
  bh?: number;
  di?: number;
  mk?: number;
  ki?: number;
  hb?: number;
  tk?: number;
  ho?: number;
  cp?: number;
  up?: number;
  ff?: number;
  fa?: number;
  round?: number;
  year: number;
  [key: string]: string | number | undefined;
}

export interface PlayerAdvanced {
  season?: PlayerAdvancedStats;
  career?: PlayerAdvancedStats;
  form_5?: PlayerAdvancedStats;
  venue?: PlayerAdvancedStats;
  opponent?: PlayerAdvancedStats;
  opponent_games?: PlayerRecentGame[];
  recent_games?: PlayerRecentGame[];
  streak_gl?: number[];
  streak_di?: number[];
  streak_mk?: number[];
  streak_tk?: number[];
}

export interface WeatherImpactPlayer {
  player: string;
  team: string;
  total_games: number;
  condition_games: number;
  conditions: string[];
  overall_gl: number;
  overall_di: number;
  overall_mk: number;
  condition_gl: number;
  condition_di: number;
  condition_mk: number;
  gl_diff: number;
  di_diff: number;
  mk_diff: number;
}

export interface WeatherSummaryInfo {
  conditions: string[];
  players_assessed: number;
  players_favored: number;
  players_hindered: number;
  avg_di_diff: number;
  avg_gl_diff: number;
}

export interface TeamRecord {
  played: number;
  wins: number;
  losses: number;
  draws?: number;
  home_record?: string;
  home_played?: number;
  home_wins?: number;
  away_record?: string;
  away_played?: number;
  away_wins?: number;
  avg_score?: number;
  avg_conceded?: number;
  min_score?: number;
  max_score?: number;
  median_score?: number;
  min_conceded?: number;
  max_conceded?: number;
  median_conceded?: number;
}

export interface VenueRecord {
  played: number;
  wins: number;
  losses: number;
  avg_score?: number;
  avg_conceded?: number;
  avg_margin?: number;
  min_score?: number;
  max_score?: number;
  median_score?: number;
  min_conceded?: number;
  max_conceded?: number;
  median_conceded?: number;
  min_margin?: number;
  max_margin?: number;
  median_margin?: number;
  season_played?: number;
  season_avg_score?: number;
}

export interface H2HRecord {
  played: number;
  wins: number;
  losses: number;
  avg_score?: number;
  avg_conceded?: number;
  avg_margin?: number;
  min_score?: number;
  max_score?: number;
  median_score?: number;
  min_conceded?: number;
  max_conceded?: number;
  median_conceded?: number;
  min_margin?: number;
  max_margin?: number;
  median_margin?: number;
  at_venue_played?: number;
  at_venue_avg_score?: number;
}

export interface TotalScoreBracket {
  threshold: number;
  p_over: number;
  p_under: number;
}

export interface TeamScoreDist {
  brackets: TotalScoreBracket[];
  sample_size: number;
  avg_score?: number;
  median_score?: number;
  highest?: number;
  lowest?: number;
}

export interface TeamScoreDistribution {
  home_team: string;
  away_team: string;
  home?: TeamScoreDist;
  away?: TeamScoreDist;
}

export interface GroundStats {
  avg_score_5y?: number;
  avg_total_5y?: number;
  median_total_5y?: number;
  total_games_5y?: number;
  highest_total_5y?: number;
  lowest_total_5y?: number;
  last_5_avg_total?: number;
  last_5_median_total?: number;
  last_5_highest?: number;
  last_5_lowest?: number;
  avg_score_season?: number;
  avg_total_season?: number;
  median_total_season?: number;
  total_games_season?: number;
  total_score_distribution?: {
    brackets: TotalScoreBracket[];
    sample_size: number;
  };
}

export interface RecentFormGame {
  result: string;
  opponent: string;
  score?: number;
  opp_score?: number;
  margin?: number;
  venue?: string;
  is_home?: boolean;
}

export interface TeamStatMatchup {
  games: number;
  avg_score: number;
  avg_conceded: number;
  avg_di?: number;
  avg_mk?: number;
  avg_tk?: number;
  avg_cl?: number;
  avg_cp?: number;
  avg_if?: number;
  avg_rb?: number;
  avg_gl?: number;
  avg_bh?: number;
  min_di?: number; max_di?: number; median_di?: number;
  min_mk?: number; max_mk?: number; median_mk?: number;
  min_tk?: number; max_tk?: number; median_tk?: number;
  min_cl?: number; max_cl?: number; median_cl?: number;
  min_cp?: number; max_cp?: number; median_cp?: number;
  min_if?: number; max_if?: number; median_if?: number;
  min_rb?: number; max_rb?: number; median_rb?: number;
  min_gl?: number; max_gl?: number; median_gl?: number;
  min_bh?: number; max_bh?: number; median_bh?: number;
}

export interface QuarterData {
  avg_goals: number;
  avg_behinds: number;
  avg_points: number;
  min_points?: number;
  max_points?: number;
  median_points?: number;
}

export interface QuarterScoring {
  q1?: QuarterData;
  q2?: QuarterData;
  q3?: QuarterData;
  q4?: QuarterData;
}

export interface CoachInfo {
  name: string;
  win_pct?: number;
  career_games?: number;
  wins?: number;
  losses?: number;
}

export interface UmpireInfo {
  name: string;
  career_games?: number;
}

export interface AttendanceStats {
  avg_attendance?: number;
  median_attendance?: number;
  max_attendance?: number;
  min_attendance?: number;
  last_5_avg?: number;
  total_games?: number;
}

export interface AdvancedTeamStats {
  ed?: number;
  de_pct?: number;
  ccl?: number;
  scl?: number;
  to?: number;
  mg?: number;
  si?: number;
  itc?: number;
  t5?: number;
  tog_pct?: number;
}

export interface RestDayBucket {
  label: string;
  played: number;
  wins: number;
  losses: number;
  avg_score: number;
  avg_conceded: number;
  avg_margin?: number;
}

export interface RestDayImpact {
  short?: RestDayBucket;
  normal?: RestDayBucket;
  extended?: RestDayBucket;
}

export interface LeagueRestBucket {
  label: string;
  played: number;
  win_pct: number;
  avg_score: number;
}

export interface LeagueRestImpact {
  short?: LeagueRestBucket;
  normal?: LeagueRestBucket;
  extended?: LeagueRestBucket;
}

export interface MatchContext {
  day_of_week?: string;
  date_formatted?: string;
  time?: string;
  day_night?: string;
  venue_display?: string;
  home_team_season?: TeamRecord;
  away_team_season?: TeamRecord;
  home_team_last_season?: TeamRecord;
  away_team_last_season?: TeamRecord;
  home_team_venue?: VenueRecord;
  away_team_venue?: VenueRecord;
  h2h_home?: H2HRecord;
  h2h_away?: H2HRecord;
  ground_stats?: GroundStats;
  team_score_distribution?: TeamScoreDistribution;
  home_rest_days?: number;
  away_rest_days?: number;
  home_rest_impact?: RestDayImpact;
  away_rest_impact?: RestDayImpact;
  league_rest_impact?: LeagueRestImpact;
  home_recent_form?: RecentFormGame[];
  away_recent_form?: RecentFormGame[];
  home_stats?: TeamStatMatchup;
  away_stats?: TeamStatMatchup;
  home_quarters?: QuarterScoring;
  away_quarters?: QuarterScoring;
  home_scoring_averages?: ScoringAverages;
  away_scoring_averages?: ScoringAverages;
}

export interface ScoringAverageEntry {
  scored: number;
  conceded: number;
  games: number;
}

export interface ScoringAverages {
  season?: ScoringAverageEntry | null;
  last_5?: ScoringAverageEntry | null;
  last_10?: ScoringAverageEntry | null;
  vs_opponent?: ScoringAverageEntry | null;
  at_venue?: ScoringAverageEntry | null;
}

export interface MatchComparison {
  match_id: number;
  year: number;
  round_number: number;
  date?: string;
  venue?: string;
  home_team: string;
  away_team: string;
  home_score?: number;
  away_score?: number;
  game_prediction?: {
    predicted_winner?: string;
    home_win_prob?: number;
    predicted_margin?: number;
    correct?: boolean;
  };
  players: MatchComparisonPlayer[];
  weather?: MatchWeather;
  weather_summary?: WeatherSummaryInfo;
  weather_impact?: WeatherImpactPlayer[];
  attendance?: number;
  match_context?: MatchContext;
  umpires?: UmpireInfo[];
  coaches?: { home?: CoachInfo; away?: CoachInfo };
  odds?: Record<string, number>;
  advanced_stats?: { home?: AdvancedTeamStats; away?: AdvancedTeamStats };
  attendance_stats?: AttendanceStats;
}

// --- Prediction History ---

export interface PredictionHistoryEntry {
  player: string;
  team: string;
  opponent: string;
  round: number;
  venue: string;
  match_id: number;
  predicted_goals: number;
  actual_goals: number;
  predicted_disposals: number;
  actual_disposals: number;
  predicted_marks: number;
  actual_marks: number;
  p_scorer: number;
  actually_scored: boolean;
}

export interface PredictionHistorySummary {
  entries: PredictionHistoryEntry[];
  summary: {
    goals_mae: number;
    disposals_mae: number;
    marks_mae: number;
    scorer_accuracy: number;
    total_predictions: number;
  };
}

// --- Calibration ---

export interface CalibrationBin {
  bin_lower: number;
  bin_upper: number;
  predicted_mean: number;
  observed_mean: number;
  count: number;
}

export interface CalibrationThreshold {
  curve: CalibrationBin[];
  ece: number | null;
  bss: number | null;
  n: number | null;
  category: string;
}

export interface CalibrationData {
  [threshold: string]: CalibrationThreshold;
}

// --- Accuracy Breakdown ---

export interface BreakdownEntry {
  n: number;
  goals_mae?: number;
  disposals_mae?: number;
  marks_mae?: number;
  team?: string;
  venue?: string;
  stage?: string;
}

export interface AccuracyBreakdown {
  overall: BreakdownEntry;
  by_team: BreakdownEntry[];
  by_venue: BreakdownEntry[];
  by_home_away: {
    home?: BreakdownEntry;
    away?: BreakdownEntry;
  };
  by_stage: BreakdownEntry[];
}

// --- Monte Carlo Simulation ---

export interface SimPercentiles {
  p10: number;
  p25: number;
  p50: number;
  p75: number;
  p90: number;
}

export interface SimTotalBracket {
  threshold: number;
  p_over: number;
}

export interface SimMatchOutcomes {
  home_win_pct: number;
  away_win_pct: number;
  draw_pct: number;
  avg_home_score: number;
  avg_away_score: number;
  avg_total: number;
  avg_margin: number;
  score_distribution: {
    home: SimPercentiles;
    away: SimPercentiles;
    total: SimPercentiles;
    margin: SimPercentiles;
  };
  total_brackets: SimTotalBracket[];
}

export interface SimPlayerGoals {
  avg: number;
  p_1plus: number;
  p_2plus: number;
  p_3plus: number;
  distribution: number[];
}

export interface SimPlayerDisposals {
  avg: number;
  p_10plus: number;
  p_15plus: number;
  p_20plus: number;
  p_25plus: number;
  p_30plus: number;
  percentiles: SimPercentiles;
}

export interface SimPlayerMarks {
  avg: number;
  p_3plus: number;
  p_5plus: number;
  p_7plus: number;
  p_10plus: number;
  percentiles: SimPercentiles;
}

export interface SimPlayer {
  player: string;
  team: string;
  is_home: boolean;
  goals: SimPlayerGoals;
  disposals: SimPlayerDisposals;
  marks: SimPlayerMarks;
}

export interface SimMultiLeg {
  player: string;
  team: string;
  type: string;
  threshold: number;
  label: string;
  solo_prob: number;
  book_implied_prob: number;
}

export interface SimMultiSuggestion {
  legs: SimMultiLeg[];
  n_legs: number;
  joint_prob: number;
  indep_prob: number;
  correlation_lift: number;
}

export interface MatchSimulation {
  match_id: number;
  home_team: string;
  away_team: string;
  n_sims: number;
  match_outcomes: SimMatchOutcomes;
  players: SimPlayer[];
  suggested_multis: SimMultiSuggestion[];
}

export interface RoundSimSummary {
  home_team: string;
  away_team: string;
  n_sims: number;
  home_win_pct: number;
  away_win_pct: number;
  draw_pct: number;
  avg_total: number;
  avg_margin: number;
  avg_home_score?: number;
  avg_away_score?: number;
  median_total?: number;
  score_range: {
    home: SimPercentiles;
    away: SimPercentiles;
  };
  top_scorers: { player: string; team: string; p_1plus: number; avg_goals?: number }[];
  has_mc: boolean;
}

// --- Multi-Bet Backtest ---

export interface MultiLeg {
  player: string;
  team: string;
  leg_type: string;
  threshold: number | null;
  prob: number;
  hit: boolean | null;
  label: string;
  reason?: string;
}

export interface MultiCombo {
  match_id: number;
  tier: string;
  tier_label?: string;
  tier_desc?: string;
  combo_predicted_prob: number;
  combo_hit: boolean | null;
  n_legs: number;
  legs: MultiLeg[];
}

export interface MultiTierStats {
  n_combos: number;
  n_hits?: number;
  combo_hit_rate: number;
  avg_predicted_prob: number;
  avg_legs: number;
  label?: string;
  desc?: string;
}

export interface MultiLegStats {
  n_used: number;
  n_hit: number;
  hit_rate: number;
  n_weakest_link: number;
}

export interface MultiCalibrationBucket {
  n: number;
  avg_predicted: number;
  actual_hit_rate: number;
}

export interface MultiBacktestData {
  label: string;
  season: number;
  tier_definitions?: Record<string, { label: string; desc: string }>;
  summary: {
    [key: string]: MultiTierStats | { total_combos: number; total_hits: number; overall_hit_rate: number };
    overall: { total_combos: number; total_hits: number; overall_hit_rate: number };
  };
  leg_type_stats: Record<string, MultiLegStats>;
  failure_analysis: {
    total_combos: number;
    total_misses: number;
    single_leg_failures: number;
    single_leg_failures_pct: number;
    most_failed_leg_type: string | null;
    failure_counts: Record<string, number>;
  };
  calibration: Record<string, MultiCalibrationBucket>;
  rounds: Record<string, MultiCombo[]>;
  error?: string;
}

// --- Venues ---

export interface VenueInfo {
  venue: string;
  total_games: number;
  avg_total_score: number;
  avg_margin: number;
  avg_temperature?: number;
  avg_precipitation?: number;
  pct_wet_games?: number;
  is_roofed: boolean;
  city?: string;
  year_from?: number;
  year_to?: number;
  home_teams?: { team: string; home_games: number }[];
}

// --- Schedule ---

export interface ScheduleForecast {
  temperature_avg?: number;
  precipitation_total?: number;
  wind_speed_avg?: number;
  humidity_avg?: number;
  is_wet?: boolean;
  is_roofed?: boolean;
  weather_difficulty_score?: number;
}

export interface ScheduleMatchPrediction {
  home_win_prob?: number;
  predicted_winner?: string;
  predicted_margin?: number;
}

export interface ScheduleTeamPred {
  pred_gl?: number;
  pred_di?: number;
  pred_mk?: number;
}

export interface ScheduleMatch {
  home_team: string;
  away_team: string;
  venue?: string;
  date?: string;
  home_score?: number | null;
  away_score?: number | null;
  match_id?: number | null;
  forecast?: ScheduleForecast | null;
  prediction?: ScheduleMatchPrediction | null;
  home_pred?: ScheduleTeamPred | null;
  away_pred?: ScheduleTeamPred | null;
}

export interface ScheduleRound {
  round_number: number;
  status: "completed" | "in_progress" | "upcoming" | "future";
  matches: ScheduleMatch[];
  prediction_updated?: string;
}

export interface SeasonSchedule {
  year: number;
  rounds: ScheduleRound[];
}

export interface VenueDetail {
  venue: string;
  total_games: number;
  avg_total_score: number;
  avg_margin: number;
  weather: {
    avg_temperature: number;
    avg_wind_speed: number;
    pct_wet: number;
    avg_humidity: number;
  };
  is_roofed: boolean;
  top_goal_scorers: {
    player: string;
    team: string;
    games: number;
    total_goals: number;
    avg_goals: number;
  }[];
  top_disposal_getters: {
    player: string;
    team: string;
    games: number;
    total_disposals: number;
    avg_disposals: number;
  }[];
  recent_matches: SeasonMatch[];
}

// ---------------------------------------------------------------------------
// Ladder
// ---------------------------------------------------------------------------

export interface LadderEntry {
  position: number;
  team: string;
  played: number;
  wins: number;
  losses: number;
  draws: number;
  points: number;
  points_for: number;
  points_against: number;
  percentage: number;
  form: string[];
  avg_margin: number;
}

export interface LadderResponse {
  year: number;
  ladder: LadderEntry[];
}

// ---------------------------------------------------------------------------
// Team Profile
// ---------------------------------------------------------------------------

export interface TeamProfileRecord {
  played: number;
  wins: number;
  losses: number;
  draws: number;
  points: number;
  points_for: number;
  points_against: number;
  percentage: number;
  ladder_position?: number | null;
}

export interface TeamProfileFormGame {
  opponent: string;
  result: string;
  score: number;
  opp_score: number;
  margin: number;
  venue: string;
  round_number: number;
  is_home?: boolean;
}

export interface TeamProfileSeasonAverages {
  avg_score: number;
  avg_conceded: number;
  avg_margin: number;
  avg_rest_days?: number | null;
}

export interface TeamProfileHomeAwaySplit {
  played: number;
  wins: number;
  losses: number;
  draws: number;
  avg_score: number | null;
  avg_conceded: number | null;
  avg_margin: number | null;
}

export interface TeamProfileTopPlayer {
  name: string;
  player_id: string;
  games: number;
  total: number;
  avg: number;
}

export interface TeamProfile {
  team: string;
  year: number;
  record: TeamProfileRecord | null;
  recent_form: TeamProfileFormGame[];
  season_averages: TeamProfileSeasonAverages | null;
  home_away: { home: TeamProfileHomeAwaySplit; away: TeamProfileHomeAwaySplit } | null;
  top_goals: TeamProfileTopPlayer[];
  top_disposals: TeamProfileTopPlayer[];
  top_marks: TeamProfileTopPlayer[];
}

// ---------------------------------------------------------------------------
// News / Injuries / Team Changes
// ---------------------------------------------------------------------------

export interface InjuryRecord {
  player: string;
  injury: string;
  estimated_return: string;
  severity: number;
  severity_label: string;
}

export interface InjuryList {
  teams: Record<string, InjuryRecord[]>;
  total: number;
  updated: string | null;
}

export interface TeamChangeRecord {
  team: string;
  n_ins: number;
  n_outs: number;
  n_debutants: number;
  stability: number;
  ins: string[];
  outs: string[];
  debutants: string[];
}

export interface TeamNewsItem {
  team: string;
  injuries: InjuryRecord[];
  injury_count: number;
  injury_severity_total: number;
  ins: string[];
  outs: string[];
  debutants: string[];
  stability: number;
}

export interface RoundNews {
  year: number;
  round_number: number;
  injuries_updated: string | null;
  teams: Record<string, TeamNewsItem>;
}

// ---------------------------------------------------------------------------
// Intelligence Feed
// ---------------------------------------------------------------------------

export type SignalType =
  | "injury"
  | "suspension"
  | "form"
  | "tactical"
  | "selection"
  | "prediction"
  | "general";

export type Sentiment = "positive" | "negative" | "neutral" | "mixed";
export type Direction = "bullish" | "bearish" | "neutral";

export interface IntelSignal {
  id: string;
  source_url: string;
  headline: string;
  summary: string;
  signal_type: SignalType;
  teams: string[];
  players: string[];
  sentiment: Sentiment;
  direction: Record<string, Direction>;
  key_facts: string[];
  relevance_score: number;
  prediction_impact: string;
  published_at: string;
  processed_at: string;
}

export interface IntelFeed {
  signals: IntelSignal[];
  total: number;
  offset: number;
  limit: number;
  breaking_count: number;
  by_type: Record<string, number>;
  by_team: Record<string, number>;
  updated: string | null;
}

export interface IntelSummary {
  total: number;
  breaking: IntelSignal[];
  breaking_count: number;
  top_signals: IntelSignal[];
  by_type: Record<string, number>;
  by_team: Record<string, number>;
  sentiment: Record<string, number>;
  team_direction: Record<string, Record<string, number>>;
  updated: string | null;
}

export interface TeamIntel {
  team: string;
  total_signals: number;
  signals: IntelSignal[];
  bullish: number;
  bearish: number;
  net_sentiment: number;
  by_type: Record<string, number>;
  injuries: IntelSignal[];
  suspensions: IntelSignal[];
  form: IntelSignal[];
  tactical: IntelSignal[];
  updated: string | null;
}
