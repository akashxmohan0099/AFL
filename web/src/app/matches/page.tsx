"use client";

import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { getSeasonMatches, getSeasonSchedule, getRoundSimulations } from "@/lib/api";
import type { SeasonMatch, SeasonSchedule, ScheduleRound, ScheduleMatch, ScheduleForecast, ScheduleTeamPred, ScheduleMatchPrediction, RoundSimSummary } from "@/lib/types";
import { TEAM_ABBREVS, TEAM_COLORS, CURRENT_YEAR, AVAILABLE_YEARS, displayVenue } from "@/lib/constants";
import { formatDate } from "@/lib/utils";
import { cn } from "@/lib/utils";
import Link from "next/link";

/* ---------- Unified card data shape ---------- */
interface UnifiedMatch {
  home_team: string;
  away_team: string;
  venue?: string;
  date?: string;
  match_id?: number | null;
  round_number: number;
  home_score?: number | null;
  away_score?: number | null;
  home_win_prob?: number | null;
  predicted_winner?: string | null;
  predicted_margin?: number | null;
  correct?: boolean | null;
  home_pred?: ScheduleTeamPred | null;
  away_pred?: ScheduleTeamPred | null;
  home_actual?: { actual_gl?: number; actual_di?: number; actual_mk?: number } | null;
  away_actual?: { actual_gl?: number; actual_di?: number; actual_mk?: number } | null;
  forecast?: ScheduleForecast | null;
  sim?: RoundSimSummary | null;
}

function toUnifiedFromCompleted(m: SeasonMatch): UnifiedMatch {
  return {
    home_team: m.home_team,
    away_team: m.away_team,
    venue: m.venue,
    date: m.date,
    match_id: m.match_id,
    round_number: m.round_number,
    home_score: m.home_score,
    away_score: m.away_score,
    home_win_prob: m.home_win_prob,
    predicted_winner: m.predicted_winner,
    predicted_margin: m.predicted_margin,
    correct: m.correct,
    home_pred: m.home_pred,
    away_pred: m.away_pred,
    home_actual: m.home_actual,
    away_actual: m.away_actual,
  };
}

function toUnifiedFromSchedule(m: ScheduleMatch, roundNumber: number): UnifiedMatch {
  return {
    home_team: m.home_team,
    away_team: m.away_team,
    venue: m.venue,
    date: m.date,
    match_id: m.match_id,
    round_number: roundNumber,
    home_score: m.home_score,
    away_score: m.away_score,
    home_win_prob: m.prediction?.home_win_prob,
    predicted_winner: m.prediction?.predicted_winner,
    predicted_margin: m.prediction?.predicted_margin,
    correct: null,
    home_pred: m.home_pred,
    away_pred: m.away_pred,
    home_actual: null,
    away_actual: null,
    forecast: m.forecast,
  };
}

/* ---------- Single card component ---------- */

function UnifiedMatchCard({ match }: { match: UnifiedMatch }) {
  const homeAbbr = TEAM_ABBREVS[match.home_team] || match.home_team;
  const awayAbbr = TEAM_ABBREVS[match.away_team] || match.away_team;
  const homeColor = TEAM_COLORS[match.home_team]?.primary || "#555";
  const isPlayed = match.home_score != null && match.away_score != null;
  const correct = match.correct;
  const homeProb = match.home_win_prob;

  const hp = match.home_pred;
  const ap = match.away_pred;
  const ha = match.home_actual;
  const aa = match.away_actual;
  const hasPredData = hp != null || ap != null;
  const fc = match.forecast;

  const hasMatchId = match.match_id != null && match.match_id !== 0;
  const matchLink = hasMatchId
    ? `/matches/${match.match_id}?round=${match.round_number}`
    : `/matches/0?round=${match.round_number}&home=${encodeURIComponent(match.home_team)}&away=${encodeURIComponent(match.away_team)}`;

  const cardContent = (
    <Card className={cn(
      "border transition-all duration-150 h-full",
      "group-hover:border-primary/30 group-hover:bg-muted/20",
      correct === true ? "border-emerald-500/25 bg-emerald-500/[0.03]" :
      correct === false ? "border-red-500/25 bg-red-500/[0.03]" :
      "border-border/50"
    )}>
      <CardContent className="pt-4 pb-4 px-4 space-y-3 h-full flex flex-col">
        {/* Team rows with scores or win prob % */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: homeColor }} />
              <span className="text-sm font-bold font-mono">{homeAbbr}</span>
            </div>
            {isPlayed ? (
              <span className="text-base font-bold tabular-nums font-mono">{match.home_score}</span>
            ) : homeProb != null ? (
              <span className="text-sm font-semibold font-mono tabular-nums text-muted-foreground">{Math.round(homeProb * 100)}%</span>
            ) : null}
          </div>
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: TEAM_COLORS[match.away_team]?.primary || "#555" }} />
              <span className="text-sm font-bold font-mono">{awayAbbr}</span>
            </div>
            {isPlayed ? (
              <span className="text-base font-bold tabular-nums font-mono">{match.away_score}</span>
            ) : homeProb != null ? (
              <span className="text-sm font-semibold font-mono tabular-nums text-muted-foreground">{Math.round((1 - homeProb) * 100)}%</span>
            ) : null}
          </div>
        </div>

        {/* Win prob bar */}
        {homeProb != null && (
          <div className="w-full h-1.5 rounded-full bg-muted/50 overflow-hidden flex">
            <div className="h-full rounded-l-full" style={{ width: `${Math.round(homeProb * 100)}%`, backgroundColor: homeColor }} />
          </div>
        )}

        {/* Stat grid — always shown when we have pred data */}
        {hasPredData && (
          <div className="border-t border-border/40 pt-2.5">
            <div className="grid grid-cols-[1fr_auto_auto_auto] gap-x-4 gap-y-1.5 items-center">
              <div />
              <p className="text-[11px] font-mono text-muted-foreground/80 text-center uppercase font-semibold">Goals</p>
              <p className="text-[11px] font-mono text-muted-foreground/80 text-center uppercase font-semibold">Disp</p>
              <p className="text-[11px] font-mono text-muted-foreground/80 text-center uppercase font-semibold">Marks</p>

              {/* Home row */}
              <span className="text-xs font-mono font-semibold">{homeAbbr}</span>
              <StatCell actual={ha?.actual_gl} pred={hp?.pred_gl} isPlayed={isPlayed} isGoals />
              <StatCell actual={ha?.actual_di} pred={hp?.pred_di} isPlayed={isPlayed} />
              <StatCell actual={ha?.actual_mk} pred={hp?.pred_mk} isPlayed={isPlayed} />

              {/* Away row */}
              <span className="text-xs font-mono font-semibold">{awayAbbr}</span>
              <StatCell actual={aa?.actual_gl} pred={ap?.pred_gl} isPlayed={isPlayed} isGoals />
              <StatCell actual={aa?.actual_di} pred={ap?.pred_di} isPlayed={isPlayed} />
              <StatCell actual={aa?.actual_mk} pred={ap?.pred_mk} isPlayed={isPlayed} />
            </div>
            {match.predicted_margin != null && !isPlayed && (
              <p className="text-[11px] font-mono text-muted-foreground/70 mt-1.5 text-right">
                Margin: {Math.abs(match.predicted_margin).toFixed(0)} pts ({TEAM_ABBREVS[match.predicted_winner ?? ""] || match.predicted_winner})
              </p>
            )}
            <p className="text-[11px] font-mono text-muted-foreground/60 mt-1 text-right">
              {isPlayed ? (
                <><span className="font-bold text-foreground/70">Actual</span> / <span className="text-muted-foreground/60">Predicted</span></>
              ) : (
                <span className="text-muted-foreground/60">Predicted team totals</span>
              )}
            </p>
          </div>
        )}

        {/* Simulation summary */}
        {match.sim && (
          <div className="border-t border-border/40 pt-2.5 space-y-2">
            <p className="text-[11px] font-mono text-muted-foreground/70 uppercase tracking-wider font-semibold">
              Monte Carlo <span className="normal-case text-muted-foreground/50 font-normal">({match.sim.n_sims.toLocaleString()} sims)</span>
            </p>
            {/* Sim win prob bar */}
            <div className="w-full h-2 rounded-full bg-muted/50 overflow-hidden flex">
              <div className="h-full rounded-l-full" style={{ width: `${Math.round(match.sim.home_win_pct * 100)}%`, backgroundColor: homeColor }} />
            </div>
            <div className="flex justify-between text-xs font-mono tabular-nums text-foreground/80 font-semibold">
              <span>{homeAbbr} {Math.round(match.sim.home_win_pct * 100)}%</span>
              <span>{Math.round(match.sim.away_win_pct * 100)}% {awayAbbr}</span>
            </div>
            {/* Key stats row */}
            <div className="flex items-center justify-between text-xs font-mono">
              <div className="flex items-center gap-1.5">
                <span className="text-muted-foreground/70">Total</span>
                <span className="tabular-nums font-bold text-foreground/90">{match.sim.avg_total.toFixed(0)}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-muted-foreground/70">Margin</span>
                <span className="tabular-nums font-bold text-foreground/90">
                  {match.sim.avg_margin > 0 ? "+" : ""}{match.sim.avg_margin.toFixed(0)}
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-muted-foreground/70">Range</span>
                <span className="tabular-nums font-semibold text-foreground/80">{match.sim.score_range.home.p25.toFixed(0)}-{match.sim.score_range.home.p75.toFixed(0)}</span>
              </div>
            </div>
            {/* Top scorers */}
            {match.sim.top_scorers.length > 0 && (
              <div className="flex gap-1.5 flex-wrap">
                {match.sim.top_scorers.map((s) => (
                  <span
                    key={s.player}
                    className="text-[11px] font-mono px-2 py-0.5 rounded bg-muted/50 text-foreground/80 font-medium"
                  >
                    <span className="w-2 h-2 rounded-full inline-block mr-1" style={{
                      backgroundColor: TEAM_COLORS[s.team]?.primary || "#555"
                    }} />
                    {s.player.split(",")[0]} {Math.round(s.p_1plus * 100)}%
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Weather forecast */}
        {fc && (
          <div className="border-t border-border/40 pt-2.5 flex items-center gap-2 flex-wrap">
            {fc.temperature_avg != null && (
              <span className={cn(
                "text-xs font-mono font-medium px-2 py-0.5 rounded",
                fc.temperature_avg > 30 ? "bg-red-500/15 text-red-400" :
                fc.temperature_avg < 12 ? "bg-cyan-500/15 text-cyan-400" :
                "bg-muted text-muted-foreground"
              )}>
                {fc.temperature_avg.toFixed(0)}°C
              </span>
            )}
            {fc.is_roofed && (
              <span className="text-xs font-mono font-medium px-2 py-0.5 rounded bg-blue-500/15 text-blue-400">ROOF</span>
            )}
            {fc.is_wet && (
              <span className="text-xs font-mono font-medium px-2 py-0.5 rounded bg-blue-500/15 text-blue-400">WET</span>
            )}
            {fc.wind_speed_avg != null && fc.wind_speed_avg > 20 && (
              <span className="text-xs font-mono font-medium px-2 py-0.5 rounded bg-amber-500/15 text-amber-400">
                {fc.wind_speed_avg.toFixed(0)}km/h
              </span>
            )}
          </div>
        )}

        {/* Footer — pushed to bottom */}
        <div className="flex justify-between items-center pt-1 mt-auto">
          <span className="text-xs text-muted-foreground/80 truncate mr-2">{displayVenue(match.venue)}</span>
          <div className="flex items-center gap-2 shrink-0">
            {match.date && !isPlayed && (
              <span className="text-[11px] text-muted-foreground/70 font-mono">{formatDate(match.date)}</span>
            )}
            {match.predicted_winner && isPlayed && (
              <span className="text-[11px] font-mono font-medium text-muted-foreground/70">
                {TEAM_ABBREVS[match.predicted_winner] || match.predicted_winner}
              </span>
            )}
            {correct != null && (
              <span className={cn(
                "text-[11px] font-mono font-bold px-2 py-0.5 rounded",
                correct ? "text-emerald-400 bg-emerald-400/15" : "text-red-400 bg-red-400/15"
              )} title={correct ? "Winner prediction correct" : "Winner prediction wrong"}>
                {correct ? "Correct" : "Wrong"}
              </span>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );

  return <Link href={matchLink} className="block group h-full">{cardContent}</Link>;
}

function StatCell({ actual, pred, isPlayed, isGoals }: { actual?: number | null; pred?: number | null; isPlayed: boolean; isGoals?: boolean }) {
  return (
    <div className="text-center">
      {isPlayed && actual != null && (
        <span className="text-xs font-mono font-bold tabular-nums text-foreground">{actual}</span>
      )}
      {pred != null && (
        <span className={cn("font-mono tabular-nums", isPlayed ? "text-[11px] text-muted-foreground/60 ml-0.5" : "text-xs text-muted-foreground/90")}>
          {isPlayed ? "/ " : ""}{isGoals ? pred.toFixed(1) : Math.round(pred)}
        </span>
      )}
      {pred == null && !isPlayed && <span className="text-xs text-muted-foreground/40">-</span>}
    </div>
  );
}

function formatUpdatedTime(isoStr: string): string {
  try {
    const d = new Date(isoStr);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffH = Math.floor(diffMs / 3600000);
    if (diffH < 1) return `${Math.floor(diffMs / 60000)}m ago`;
    if (diffH < 24) return `${diffH}h ago`;
    const diffD = Math.floor(diffH / 24);
    return `${diffD}d ago`;
  } catch {
    return isoStr;
  }
}

export default function MatchesPage() {
  const [matches, setMatches] = useState<SeasonMatch[]>([]);
  const [schedule, setSchedule] = useState<SeasonSchedule | null>(null);
  const [simsByRound, setSimsByRound] = useState<Record<number, RoundSimSummary[]>>({});
  const [year, setYear] = useState(CURRENT_YEAR);
  const [selectedRound, setSelectedRound] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  const years = AVAILABLE_YEARS;

  useEffect(() => {
    setLoading(true);
    setSimsByRound({});
    Promise.all([
      getSeasonMatches(year).catch(() => []),
      getSeasonSchedule(year).catch(() => null),
    ])
      .then(([m, s]) => {
        setMatches(m || []);
        setSchedule(s);
        setSelectedRound(null);

        // Find all rounds with prediction data and fetch simulations
        if (s?.rounds) {
          const completedKeys = new Set((m || []).map((cm: SeasonMatch) => `${cm.home_team}|${cm.away_team}|${cm.round_number}`));
          const upcomingRoundNums = new Set<number>();
          for (const r of s.rounds) {
            const hasUnplayed = r.matches.some(
              (rm: ScheduleMatch) => rm.home_score == null && !completedKeys.has(`${rm.home_team}|${rm.away_team}|${r.round_number}`)
            );
            if (hasUnplayed) upcomingRoundNums.add(r.round_number);
          }
          // Fetch simulations for upcoming rounds (max 2)
          const upcomingToSim = [...upcomingRoundNums].sort((a, b) => a - b).slice(0, 2);
          // Also fetch simulations for completed rounds (most recent 3)
          const completedRoundNums = [...new Set((m || []).map((cm: SeasonMatch) => cm.round_number))].sort((a, b) => b - a).slice(0, 3);
          const allRoundsToSim = [...new Set([...upcomingToSim, ...completedRoundNums])];
          Promise.all(
            allRoundsToSim.map((rn) =>
              getRoundSimulations(year, rn)
                .then((sims) => ({ rn, sims }))
                .catch(() => ({ rn, sims: [] as RoundSimSummary[] }))
            )
          ).then((results) => {
            const newSims: Record<number, RoundSimSummary[]> = {};
            for (const { rn, sims } of results) {
              if (sims.length > 0) newSims[rn] = sims;
            }
            setSimsByRound(newSims);
          });
        }
      })
      .finally(() => setLoading(false));
  }, [year]);

  // On-demand: fetch sims for selected round if not already loaded
  useEffect(() => {
    if (selectedRound == null || simsByRound[selectedRound]) return;
    getRoundSimulations(year, selectedRound)
      .then((sims) => {
        if (sims.length > 0) {
          setSimsByRound((prev) => ({ ...prev, [selectedRound]: sims }));
        }
      })
      .catch(() => {});
  }, [selectedRound, year]);

  // Completed match keys (for deduplicating with schedule)
  const completedMatchKeys = new Set(
    matches.map((m) => `${m.home_team}|${m.away_team}|${m.round_number}`)
  );

  // From schedule, find rounds that have any unplayed matches
  const allScheduleRounds = schedule?.rounds || [];
  const upcomingRounds: ScheduleRound[] = [];
  for (const r of allScheduleRounds) {
    const unplayed = r.matches.filter(
      (m) => m.home_score == null && !completedMatchKeys.has(`${m.home_team}|${m.away_team}|${r.round_number}`)
    );
    if (unplayed.length > 0) {
      upcomingRounds.push({ ...r, matches: unplayed });
    }
  }

  // Build schedule prediction lookup so completed MatchCards can inherit prediction data
  const schedulePredLookup = new Map<string, ScheduleMatch>();
  for (const r of allScheduleRounds) {
    for (const m of r.matches) {
      schedulePredLookup.set(`${m.home_team}|${m.away_team}|${r.round_number}`, m);
    }
  }

  // All round numbers for the filter
  const completedRoundNums = [...new Set(matches.map((m) => m.round_number))];
  const allRoundNums = [
    ...completedRoundNums,
    ...upcomingRounds.map((r) => r.round_number),
  ];
  const uniqueRounds = [...new Set(allRoundNums)].sort((a, b) => a - b);

  // Win/loss stats for completed
  const played = matches.filter((m) => m.correct != null);
  const correctCount = played.filter((m) => m.correct).length;
  const accuracy = played.length > 0 ? correctCount / played.length : 0;

  // Group completed matches by round
  const completedGrouped: Record<number, SeasonMatch[]> = {};
  const filteredCompleted = selectedRound != null
    ? matches.filter((m) => m.round_number === selectedRound)
    : matches;
  for (const m of filteredCompleted) {
    if (!completedGrouped[m.round_number]) completedGrouped[m.round_number] = [];
    completedGrouped[m.round_number].push(m);
  }

  // Filter upcoming rounds
  const filteredUpcoming = selectedRound != null
    ? upcomingRounds.filter((r) => r.round_number === selectedRound)
    : upcomingRounds;

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-48" />
        <div className="flex gap-2">
          {[1, 2, 3].map((i) => <Skeleton key={i} className="h-9 w-16" />)}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3, 4, 5, 6].map((i) => <Skeleton key={i} className="h-36" />)}
        </div>
      </div>
    );
  }

  // Build ordered sections using unified matches
  type Section = {
    roundNum: number;
    matches: UnifiedMatch[];
    hasUpcoming: boolean;
    status?: string;
    predictionUpdated?: string;
  };

  const sectionMap: Record<number, Section> = {};

  // Add completed matches (enriched with schedule prediction data + simulation)
  for (const [roundNum, roundMatches] of Object.entries(completedGrouped)) {
    const rn = Number(roundNum);
    if (!sectionMap[rn]) sectionMap[rn] = { roundNum: rn, matches: [], hasUpcoming: false };
    const roundSims = simsByRound[rn] || [];
    for (const m of roundMatches) {
      const u = toUnifiedFromCompleted(m);
      // Enrich with schedule prediction data if the completed card is missing it
      const schedKey = `${m.home_team}|${m.away_team}|${m.round_number}`;
      const sm = schedulePredLookup.get(schedKey);
      if (sm) {
        if (u.home_win_prob == null && sm.prediction?.home_win_prob != null) u.home_win_prob = sm.prediction.home_win_prob;
        if (u.predicted_winner == null && sm.prediction?.predicted_winner) u.predicted_winner = sm.prediction.predicted_winner;
        if (u.predicted_margin == null && sm.prediction?.predicted_margin != null) u.predicted_margin = sm.prediction.predicted_margin;
        if (u.home_pred == null && sm.home_pred) u.home_pred = sm.home_pred;
        if (u.away_pred == null && sm.away_pred) u.away_pred = sm.away_pred;
        if (u.match_id == null && sm.match_id != null) u.match_id = sm.match_id;
      }
      // Attach simulation summary
      const sim = roundSims.find(
        (s) => s.home_team === m.home_team && s.away_team === m.away_team
      );
      if (sim) u.sim = sim;
      sectionMap[rn].matches.push(u);
    }
  }

  // Add upcoming matches (with simulation data)
  for (const r of filteredUpcoming) {
    if (!sectionMap[r.round_number]) sectionMap[r.round_number] = { roundNum: r.round_number, matches: [], hasUpcoming: false };
    sectionMap[r.round_number].hasUpcoming = true;
    sectionMap[r.round_number].status = r.status;
    if (r.prediction_updated) sectionMap[r.round_number].predictionUpdated = r.prediction_updated;
    const roundSims = simsByRound[r.round_number] || [];
    for (const m of r.matches) {
      const u = toUnifiedFromSchedule(m, r.round_number);
      // Attach simulation summary if available
      const sim = roundSims.find(
        (s) => s.home_team === m.home_team && s.away_team === m.away_team
      );
      if (sim) u.sim = sim;
      sectionMap[r.round_number].matches.push(u);
    }
  }

  // Also set prediction_updated for completed-only rounds from schedule
  for (const r of allScheduleRounds) {
    if (sectionMap[r.round_number] && r.prediction_updated && !sectionMap[r.round_number].predictionUpdated) {
      sectionMap[r.round_number].predictionUpdated = r.prediction_updated;
    }
  }

  // Sort: ascending round order (Round 0, 1, 2, ...)
  const sections = Object.values(sectionMap).sort((a, b) => a.roundNum - b.roundNum);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <h1 className="text-xl sm:text-2xl font-bold">Matches</h1>
        <div className="flex items-center gap-2">
          {upcomingRounds.length > 0 && (
            <Badge variant="outline" className="text-xs font-mono">
              {upcomingRounds.reduce((s, r) => s + r.matches.length, 0)} upcoming
            </Badge>
          )}
          {played.length > 0 && (
            <Badge variant="outline" className="text-xs font-mono">
              {correctCount}/{played.length} correct ({(accuracy * 100).toFixed(1)}%)
            </Badge>
          )}
        </div>
      </div>

      {/* Year + Round selector */}
      <div className="flex gap-2 flex-wrap items-center">
        {years.map((y) => (
          <Button key={y} variant={y === year ? "default" : "outline"} size="sm" onClick={() => setYear(y)}>
            {y}
          </Button>
        ))}
        {uniqueRounds.length > 0 && (
          <>
            <span className="mx-2 text-muted-foreground">|</span>
            <Button
              variant={selectedRound === null ? "secondary" : "ghost"}
              size="sm"
              onClick={() => setSelectedRound(null)}
            >
              All
            </Button>
            <select
              className="border border-border rounded-md px-2 py-1.5 text-sm bg-background"
              value={selectedRound ?? ""}
              onChange={(e) => setSelectedRound(e.target.value ? Number(e.target.value) : null)}
            >
              <option value="">Select Round...</option>
              {uniqueRounds.map((r) => (
                <option key={r} value={r}>Round {r}</option>
              ))}
            </select>
          </>
        )}
      </div>

      {sections.length === 0 ? (
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground">No match data available for {year}.</p>
          </CardContent>
        </Card>
      ) : (
        sections.map((section) => {
          const hasCompleted = section.matches.some((m) => m.home_score != null);
          const hasUnplayed = section.matches.some((m) => m.home_score == null);
          return (
            <div key={section.roundNum} className="space-y-4">
              <div className="flex items-center gap-3">
                <h3 className="text-base font-bold text-muted-foreground uppercase tracking-wider">
                  Round {section.roundNum}
                </h3>
                {section.hasUpcoming && (
                  <span className={cn(
                    "text-xs font-mono font-bold px-2.5 py-1 rounded uppercase",
                    section.status === "upcoming"
                      ? "text-primary bg-primary/10"
                      : "text-muted-foreground bg-muted/50"
                  )}>
                    {hasCompleted && hasUnplayed ? "In Progress" : section.status === "upcoming" ? "Next" : "Future"}
                  </span>
                )}
                {section.predictionUpdated && (
                  <span className="text-[11px] font-mono text-muted-foreground/70 ml-auto">
                    Predictions: {formatUpdatedTime(section.predictionUpdated)}
                  </span>
                )}
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 auto-rows-fr">
                {section.matches.map((m, i) => (
                  <UnifiedMatchCard key={`${section.roundNum}-${i}`} match={m} />
                ))}
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}
