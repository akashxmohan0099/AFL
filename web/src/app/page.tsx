"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  getSeasonSummary,
  getSeasonMatches,
  getUpcoming,
  getHealth,
  API_BASE,
} from "@/lib/api";
import type {
  SeasonSummary,
  SeasonMatch,
  UpcomingRound,
  HealthStatus,
} from "@/lib/types";
import { TEAM_ABBREVS, TEAM_COLORS, CURRENT_YEAR, CHART_COLORS, displayVenue } from "@/lib/constants";
import { ArrowRight, ChevronRight } from "lucide-react";

function StatTicker({ label, value, color, suffix = "", tip, subtitle }: { label: string; value: string; color: string; suffix?: string; tip?: string; subtitle?: string }) {
  return (
    <div className="flex items-center gap-3 px-4 py-3 border border-border/50 rounded-lg bg-card/50 cursor-help" title={tip}>
      <div className="w-1 h-8 rounded-full" style={{ backgroundColor: color }} />
      <div>
        <p className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">{label}</p>
        <p className="text-xl font-bold tabular-nums font-mono" style={{ color }}>{value}{suffix}</p>
        {subtitle && <p className="text-[10px] text-muted-foreground/60 mt-0.5">{subtitle}</p>}
      </div>
    </div>
  );
}

function MatchTicker({ match }: { match: SeasonMatch }) {
  const homeAbbr = TEAM_ABBREVS[match.home_team] || match.home_team;
  const awayAbbr = TEAM_ABBREVS[match.away_team] || match.away_team;
  const correct = match.correct;

  return (
    <Link href={`/matches/${match.match_id}${match.round_number != null ? `?round=${match.round_number}` : ''}`} className="block group">
      <div className={`flex items-center gap-3 px-3 py-2.5 rounded-lg border transition-all duration-150 group-hover:bg-muted/30 ${
        correct === true ? "border-emerald-500/30 bg-emerald-500/5" :
        correct === false ? "border-red-500/30 bg-red-500/5" :
        "border-border/50"
      }`}>
        {/* Home */}
        <div className="flex items-center gap-1.5 min-w-[60px]">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: TEAM_COLORS[match.home_team]?.primary || "#555" }} />
          <span className="text-xs font-semibold font-mono">{homeAbbr}</span>
        </div>
        <span className="text-sm font-bold tabular-nums font-mono w-7 text-right">{match.home_score ?? "-"}</span>
        <span className="text-[10px] text-muted-foreground">v</span>
        <span className="text-sm font-bold tabular-nums font-mono w-7">{match.away_score ?? "-"}</span>
        <div className="flex items-center gap-1.5 min-w-[60px]">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: TEAM_COLORS[match.away_team]?.primary || "#555" }} />
          <span className="text-xs font-semibold font-mono">{awayAbbr}</span>
        </div>
        <span className="flex-1" />
        {correct != null && (
          <span className={`text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded ${
            correct ? "text-emerald-400 bg-emerald-400/10" : "text-red-400 bg-red-400/10"
          }`} title={correct ? "We predicted the correct winner" : "We predicted the wrong winner"}>
            {correct ? "Correct" : "Wrong"}
          </span>
        )}
        <ChevronRight className="w-3 h-3 text-muted-foreground/50 group-hover:text-muted-foreground transition-colors" />
      </div>
    </Link>
  );
}

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [summary, setSummary] = useState<SeasonSummary | null>(null);
  const [matches, setMatches] = useState<SeasonMatch[]>([]);
  const [upcoming, setUpcoming] = useState<UpcomingRound | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    setLoading(true);
    Promise.all([
      getHealth().catch(() => null),
      getSeasonSummary(CURRENT_YEAR).catch(() => null),
      getSeasonMatches(CURRENT_YEAR).catch(() => []),
      getUpcoming(CURRENT_YEAR).catch(() => null),
    ])
      .then(([h, s, m, u]) => {
        if (!h) setError("Cannot connect to API server");
        setHealth(h);
        setSummary(s as SeasonSummary | null);
        setMatches((m as SeasonMatch[]) || []);
        setUpcoming(u as UpcomingRound | null);
      })
      .finally(() => setLoading(false));
  }, []);

  if (error && !health) {
    return (
      <div className="space-y-4">
        <h1 className="text-xl font-bold">Dashboard</h1>
        <div className="border border-red-500/30 bg-red-500/5 rounded-lg p-4">
          <p className="text-red-400 text-sm font-medium">{error}</p>
          <p className="text-xs text-muted-foreground mt-2">
            API base: <code className="bg-muted px-1.5 py-0.5 rounded text-[11px] font-mono">{API_BASE}</code>
          </p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="space-y-5">
        <Skeleton className="h-7 w-48" />
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {[1, 2, 3, 4].map((i) => <Skeleton key={i} className="h-20" />)}
        </div>
        <Skeleton className="h-72" />
      </div>
    );
  }

  const acc = summary?.accuracy;
  const completedMatches = matches
    .filter((m) => m.home_score != null)
    .sort((a, b) => (b.round_number ?? 0) - (a.round_number ?? 0))
    .slice(0, 8);
  const played = matches.filter((m) => m.correct != null);
  const correctCount = played.filter((m) => m.correct).length;
  const winRate = played.length > 0 ? (correctCount / played.length) * 100 : 0;

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-bold tracking-tight">{CURRENT_YEAR} Season Overview</h1>
          {summary?.current_round != null && (
            <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
              R{summary.current_round}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 text-[11px] text-muted-foreground font-mono">
          {health?.latest_data && (
            <>
              <span>Data: {health.latest_data}</span>
              <span className="w-px h-3 bg-border" />
            </>
          )}
          <span>{summary?.completed_rounds ?? 0}/{summary?.total_rounds ?? 0} rounds</span>
          <span className="w-px h-3 bg-border" />
          <span>{summary?.total_matches ?? 0} matches</span>
        </div>
      </div>

      {/* Stat Ticker Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatTicker
          label="Goal Accuracy"
          value={acc?.goals_mae != null ? acc.goals_mae.toFixed(2) : "--"}
          color={CHART_COLORS.goals}
          subtitle="Avg error per player (goals)"
          tip="Mean Absolute Error for goal predictions. Lower is better — 0.5 means we're off by half a goal per player on average."
        />
        <StatTicker
          label="Disposal Accuracy"
          value={acc?.disposals_mae != null ? acc.disposals_mae.toFixed(1) : "--"}
          color={CHART_COLORS.disposals}
          subtitle="Avg error per player (disposals)"
          tip="Mean Absolute Error for disposal predictions. Lower is better — measures average prediction error per player."
        />
        <StatTicker
          label="Goal Scorer Calls"
          value={acc?.scorer_accuracy != null ? acc.scorer_accuracy.toFixed(1) : "--"}
          color={CHART_COLORS.marks}
          suffix="%"
          subtitle="Correct 1+ goal predictions"
          tip="How often we correctly predict whether a player will kick at least 1 goal. Higher is better."
        />
        <StatTicker
          label="Match Winner"
          value={played.length > 0 ? winRate.toFixed(1) : acc?.game_winner_accuracy != null ? acc.game_winner_accuracy.toFixed(1) : "--"}
          color={CHART_COLORS.behinds}
          suffix="%"
          subtitle="Correct winner predictions"
          tip="Percentage of matches where we correctly predicted the winning team. Higher is better."
        />
      </div>

      {/* Recent Results */}
      <div>
        <Card className="border-border/50">
          <CardHeader className="pb-2 pt-4 px-4">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Recent Match Results</CardTitle>
              <Link href="/matches" className="text-[10px] text-muted-foreground hover:text-primary font-mono transition-colors flex items-center gap-0.5">
                ALL <ArrowRight className="w-2.5 h-2.5" />
              </Link>
            </div>
          </CardHeader>
          <CardContent className="px-3 pb-3 space-y-1">
            {completedMatches.length > 0 ? (
              completedMatches.map((m) => <MatchTicker key={m.match_id} match={m} />)
            ) : (
              <p className="text-xs text-muted-foreground py-6 text-center">No completed matches yet</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Upcoming Round */}
      {upcoming && upcoming.matches.length > 0 && (
        <Card className="border-border/50">
          <CardHeader className="pb-2 pt-4 px-4">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">
                Upcoming &mdash; Round {upcoming.round_number}
              </CardTitle>
              <Link href="/predictions" className="text-[10px] text-muted-foreground hover:text-primary font-mono transition-colors flex items-center gap-0.5">
                PREDICTIONS <ArrowRight className="w-2.5 h-2.5" />
              </Link>
            </div>
          </CardHeader>
          <CardContent className="px-4 pb-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
              {upcoming.matches.map((m, i) => {
                const homeAbbr = TEAM_ABBREVS[m.home_team] || m.home_team;
                const awayAbbr = TEAM_ABBREVS[m.away_team] || m.away_team;
                const homePreds = upcoming.predictions.filter((p) => p.team === m.home_team);
                const awayPreds = upcoming.predictions.filter((p) => p.team === m.away_team);
                const homeGoals = homePreds.reduce((s, p) => s + (p.predicted_goals ?? 0), 0);
                const awayGoals = awayPreds.reduce((s, p) => s + (p.predicted_goals ?? 0), 0);
                const favored = homeGoals > awayGoals ? "home" : "away";

                return (
                  <div key={i} className="flex items-center justify-between px-3 py-2.5 rounded-lg border border-border/50 bg-card/30 hover:bg-muted/20 transition-colors">
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: TEAM_COLORS[m.home_team]?.primary || "#555" }} />
                        <span className={`text-xs font-mono font-semibold ${favored === "home" ? "text-primary" : ""}`}>{homeAbbr}</span>
                      </div>
                      <span className="text-[10px] text-muted-foreground/50">v</span>
                      <div className="flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: TEAM_COLORS[m.away_team]?.primary || "#555" }} />
                        <span className={`text-xs font-mono font-semibold ${favored === "away" ? "text-primary" : ""}`}>{awayAbbr}</span>
                      </div>
                    </div>
                    <span className="text-[10px] text-muted-foreground truncate ml-2">{displayVenue(m.venue)}</span>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Quick Navigation */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {[
          { href: "/predictions", label: "Predictions", sub: "Next round" },
          { href: "/predictions/history", label: "Pred vs Actual", sub: "Track record" },
          { href: "/players", label: "Players", sub: "Search & stats" },
          { href: "/venues", label: "Venues", sub: "Ground data" },
        ].map((item) => (
          <Link key={item.href} href={item.href} className="group">
            <div className="px-3 py-3 rounded-lg border border-border/50 bg-card/30 hover:border-primary/30 hover:bg-primary/5 transition-all duration-150">
              <p className="text-xs font-medium group-hover:text-primary transition-colors">{item.label}</p>
              <p className="text-[10px] text-muted-foreground">{item.sub}</p>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
