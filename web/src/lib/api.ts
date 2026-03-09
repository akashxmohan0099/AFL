import type {
  PlayerSearchResult,
  PlayerProfileEnhanced,
  GameLogEntry,
  RoundInfo,
  PlayerPrediction,
  MatchResult,
  MatchDetail,
  OddsComparison,
  PlayerOddsComparison,
  HealthStatus,
  SeasonSummary,
  SeasonMatch,
  UpcomingRound,
  RoundAccuracy,
  MatchComparison,
  PredictionHistorySummary,
  SeasonSchedule,
  VenueInfo,
  VenueDetail,
  MultiBacktestData,
  PlayerDirectoryEntry,
  InjuryList,
  RoundNews,
  IntelFeed,
  IntelSummary,
  TeamIntel,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_KEY = process.env.API_KEY || process.env.NEXT_PUBLIC_API_KEY || "";

async function fetchApi<T>(path: string): Promise<T> {
  const headers: HeadersInit = {};
  if (API_KEY) {
    headers["X-API-Key"] = API_KEY;
  }

  const res = await fetch(`${API_BASE}${path}`, {
    cache: "no-store",
    headers,
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

// Health
export const getHealth = () => fetchApi<HealthStatus>("/api/health");

// Players
export const searchPlayers = (q: string, limit = 20) =>
  fetchApi<PlayerSearchResult[]>(`/api/players/search?q=${encodeURIComponent(q)}&limit=${limit}`);

export const getPlayer = (playerId: string) =>
  fetchApi<PlayerProfileEnhanced>(`/api/players/${encodeURIComponent(playerId)}`);

export const getPlayerGames = (playerId: string, year?: number, limit = 50, offset = 0) => {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (year) params.set("year", String(year));
  return fetchApi<GameLogEntry[]>(`/api/players/${encodeURIComponent(playerId)}/games?${params}`);
};

export const getPlayerPredictions = (playerId: string, year = 2025) =>
  fetchApi<PlayerPrediction[]>(`/api/players/${encodeURIComponent(playerId)}/predictions?year=${year}`);

export const getPlayerDirectory = (year: number) =>
  fetchApi<PlayerDirectoryEntry[]>(`/api/players/directory?year=${year}`);

// Rounds
export const getRounds = (year?: number) => {
  const params = year ? `?year=${year}` : "";
  return fetchApi<RoundInfo[]>(`/api/rounds${params}`);
};

export const getRoundPredictions = (year: number, round: number) =>
  fetchApi<PlayerPrediction[]>(`/api/rounds/${year}/${round}`);

// Matches
export const getMatches = (year: number, round: number) =>
  fetchApi<MatchResult[]>(`/api/matches/${year}/${round}`);

export const getMatchDetail = (matchId: number) =>
  fetchApi<MatchDetail>(`/api/matches/detail/${matchId}`);

// Odds
export const getGameOdds = (year: number, round: number) =>
  fetchApi<OddsComparison[]>(`/api/odds/game/${year}/${round}`);

export const getPlayerOdds = (year: number, round: number) =>
  fetchApi<PlayerOddsComparison[]>(`/api/odds/players/${year}/${round}`);

// Metrics
export const getBacktest = (year: number) =>
  fetchApi<Record<string, unknown>>(`/api/metrics/backtest/${year}`);

export const getMultiBacktest = (year: number) =>
  fetchApi<MultiBacktestData>(`/api/metrics/multi-backtest/${year}`);

// Season endpoints
export const getSeasonSummary = (year: number) =>
  fetchApi<SeasonSummary>(`/api/season/${year}/summary`);

export const getSeasonMatches = (year: number) =>
  fetchApi<SeasonMatch[]>(`/api/season/${year}/matches`);

export const getUpcoming = (year: number) =>
  fetchApi<UpcomingRound>(`/api/season/${year}/upcoming`);

export const getRoundAccuracy = (year: number) =>
  fetchApi<RoundAccuracy[]>(`/api/season/${year}/accuracy`);

export const getMatchComparison =(year: number, matchId: number, roundNumber?: number, homeTeam?: string, awayTeam?: string) => {
  const params = new URLSearchParams();
  if (roundNumber != null) params.set("round_number", String(roundNumber));
  if (homeTeam) params.set("home_team", homeTeam);
  if (awayTeam) params.set("away_team", awayTeam);
  const qs = params.toString();
  return fetchApi<MatchComparison>(`/api/season/${year}/match/${matchId}${qs ? `?${qs}` : ''}`);
};

// Schedule
export const getSeasonSchedule = (year: number) =>
  fetchApi<SeasonSchedule>(`/api/season/${year}/schedule`);

// Prediction History
export const getPredictionHistory = (year: number) =>
  fetchApi<PredictionHistorySummary>(`/api/season/${year}/predictions-history`);

// Venues
export const getVenues = () =>
  fetchApi<VenueInfo[]>(`/api/venues`);

export const getVenueDetail = (name: string) =>
  fetchApi<VenueDetail>(`/api/venues/${encodeURIComponent(name)}`);

// News
export const getInjuries = () =>
  fetchApi<InjuryList>("/api/news/injuries");

export const getRoundNews = (year: number, round: number) =>
  fetchApi<RoundNews>(`/api/news/round/${year}/${round}`);

// Intel feed
export const getIntelFeed = (params?: {
  signal_type?: string;
  team?: string;
  min_relevance?: number;
  limit?: number;
  offset?: number;
}) => {
  const qs = new URLSearchParams();
  if (params?.signal_type) qs.set("signal_type", params.signal_type);
  if (params?.team) qs.set("team", params.team);
  if (params?.min_relevance != null) qs.set("min_relevance", String(params.min_relevance));
  if (params?.limit != null) qs.set("limit", String(params.limit));
  if (params?.offset != null) qs.set("offset", String(params.offset));
  const q = qs.toString();
  return fetchApi<IntelFeed>(`/api/news/intel/feed${q ? `?${q}` : ""}`);
};

export const getIntelSummary = () =>
  fetchApi<IntelSummary>("/api/news/intel/summary");

export const getTeamIntel = (team: string) =>
  fetchApi<TeamIntel>(`/api/news/intel/team/${encodeURIComponent(team)}`);
