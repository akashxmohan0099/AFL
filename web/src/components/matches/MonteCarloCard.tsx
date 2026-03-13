"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { MatchSimulation, SimPlayer, SimMultiSuggestion } from "@/lib/types";
import { TEAM_ABBREVS } from "@/lib/constants";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Probability bar — horizontal fill with percentage
// ---------------------------------------------------------------------------

function ProbBar({ value, color = "emerald" }: { value: number; color?: string }) {
  const pct = Math.round(value * 100);
  const bgClass =
    pct >= 70
      ? "bg-emerald-500"
      : pct >= 50
        ? "bg-emerald-500/70"
        : pct >= 30
          ? "bg-amber-500/70"
          : "bg-red-500/50";
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-16 h-1.5 rounded-full bg-muted overflow-hidden">
        <div className={cn("h-full rounded-full", bgClass)} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[11px] tabular-nums font-mono text-muted-foreground w-9 text-right">
        {pct}%
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Score distribution row
// ---------------------------------------------------------------------------

function PercentilesRow({ label, pcts, color }: { label: string; pcts: { p10: number; p25: number; p50: number; p75: number; p90: number }; color: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="flex items-center gap-1.5 w-16 shrink-0">
        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
        <span className="text-xs font-medium">{label}</span>
      </span>
      <div className="flex-1 flex items-center gap-0.5">
        {/* Visual range bar */}
        <div className="flex-1 relative h-5">
          <div className="absolute inset-y-0 bg-muted/40 rounded-sm" style={{ left: `${(pcts.p10 / 200) * 100}%`, right: `${100 - (pcts.p90 / 200) * 100}%` }} />
          <div className="absolute inset-y-0 bg-muted rounded-sm" style={{ left: `${(pcts.p25 / 200) * 100}%`, right: `${100 - (pcts.p75 / 200) * 100}%` }} />
          <div className="absolute top-0 bottom-0 w-0.5 rounded-full" style={{ left: `${(pcts.p50 / 200) * 100}%`, backgroundColor: color }} />
        </div>
      </div>
      <div className="flex gap-2 text-[10px] tabular-nums font-mono text-muted-foreground shrink-0">
        <span title="10th percentile">{pcts.p10.toFixed(0)}</span>
        <span className="text-muted-foreground/30">|</span>
        <span className="font-semibold text-foreground" title="Median">{pcts.p50.toFixed(0)}</span>
        <span className="text-muted-foreground/30">|</span>
        <span title="90th percentile">{pcts.p90.toFixed(0)}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Player simulation table
// ---------------------------------------------------------------------------

function PlayerSimTable({
  players,
  teamName,
  teamColor,
  statView,
}: {
  players: SimPlayer[];
  teamName: string;
  teamColor: string;
  statView: "goals" | "disposals" | "marks";
}) {
  const abbr = TEAM_ABBREVS[teamName] || teamName;

  if (statView === "goals") {
    const sorted = [...players].sort((a, b) => b.goals.p_1plus - a.goals.p_1plus);
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-sm">
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: teamColor }} />
            {abbr} — Goals
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs">Player</TableHead>
                  <TableHead className="text-right text-xs">Avg</TableHead>
                  <TableHead className="text-right text-xs">1+</TableHead>
                  <TableHead className="text-right text-xs">2+</TableHead>
                  <TableHead className="text-right text-xs">3+</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sorted.map((p) => (
                  <TableRow key={p.player}>
                    <TableCell className="text-xs font-medium py-1.5">{p.player}</TableCell>
                    <TableCell className="text-right tabular-nums text-xs py-1.5 font-mono">
                      {p.goals.avg.toFixed(2)}
                    </TableCell>
                    <TableCell className="text-right py-1.5"><ProbBar value={p.goals.p_1plus} /></TableCell>
                    <TableCell className="text-right py-1.5"><ProbBar value={p.goals.p_2plus} /></TableCell>
                    <TableCell className="text-right py-1.5"><ProbBar value={p.goals.p_3plus} /></TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (statView === "disposals") {
    const sorted = [...players].sort((a, b) => b.disposals.avg - a.disposals.avg);
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-sm">
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: teamColor }} />
            {abbr} — Disposals
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs">Player</TableHead>
                  <TableHead className="text-right text-xs">Avg</TableHead>
                  <TableHead className="text-right text-xs">15+</TableHead>
                  <TableHead className="text-right text-xs">20+</TableHead>
                  <TableHead className="text-right text-xs">25+</TableHead>
                  <TableHead className="text-right text-xs">30+</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sorted.map((p) => (
                  <TableRow key={p.player}>
                    <TableCell className="text-xs font-medium py-1.5">{p.player}</TableCell>
                    <TableCell className="text-right tabular-nums text-xs py-1.5 font-mono">
                      {p.disposals.avg.toFixed(1)}
                    </TableCell>
                    <TableCell className="text-right py-1.5"><ProbBar value={p.disposals.p_15plus} /></TableCell>
                    <TableCell className="text-right py-1.5"><ProbBar value={p.disposals.p_20plus} /></TableCell>
                    <TableCell className="text-right py-1.5"><ProbBar value={p.disposals.p_25plus} /></TableCell>
                    <TableCell className="text-right py-1.5"><ProbBar value={p.disposals.p_30plus} /></TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Marks
  const sorted = [...players].sort((a, b) => b.marks.avg - a.marks.avg);
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: teamColor }} />
          {abbr} — Marks
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-xs">Player</TableHead>
                <TableHead className="text-right text-xs">Avg</TableHead>
                <TableHead className="text-right text-xs">3+</TableHead>
                <TableHead className="text-right text-xs">5+</TableHead>
                <TableHead className="text-right text-xs">7+</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sorted.map((p) => (
                <TableRow key={p.player}>
                  <TableCell className="text-xs font-medium py-1.5">{p.player}</TableCell>
                  <TableCell className="text-right tabular-nums text-xs py-1.5 font-mono">
                    {p.marks.avg.toFixed(1)}
                  </TableCell>
                  <TableCell className="text-right py-1.5"><ProbBar value={p.marks.p_3plus} /></TableCell>
                  <TableCell className="text-right py-1.5"><ProbBar value={p.marks.p_5plus} /></TableCell>
                  <TableCell className="text-right py-1.5"><ProbBar value={p.marks.p_7plus} /></TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Multi-bet suggestions
// ---------------------------------------------------------------------------

function MultiSuggestionsCard({
  suggestions,
  homeTeam,
  awayTeam,
  homeColor,
  awayColor,
}: {
  suggestions: SimMultiSuggestion[];
  homeTeam: string;
  awayTeam: string;
  homeColor: string;
  awayColor: string;
}) {
  if (suggestions.length === 0) return null;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Correlated Multi-Bet Suggestions</CardTitle>
        <p className="text-[10px] text-muted-foreground/60 font-mono">
          Joint probabilities from {suggestions.length} combinations with positive correlation lift
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        {suggestions.slice(0, 5).map((s, i) => (
          <div key={i} className="border border-border/30 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-muted-foreground">
                #{i + 1} — {s.n_legs} legs
              </span>
              <div className="flex items-center gap-3 text-[10px] font-mono">
                <span>
                  Joint:{" "}
                  <span className="font-bold text-emerald-400">
                    {(s.joint_prob * 100).toFixed(1)}%
                  </span>
                </span>
                <span className="text-muted-foreground/40">vs</span>
                <span>
                  Independent:{" "}
                  <span className="text-muted-foreground">
                    {(s.indep_prob * 100).toFixed(1)}%
                  </span>
                </span>
                {s.correlation_lift !== 1 && (
                  <span
                    className={cn(
                      "px-1.5 py-0.5 rounded text-[9px] font-semibold border",
                      s.correlation_lift > 1
                        ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                        : "bg-red-500/10 text-red-400 border-red-500/30"
                    )}
                  >
                    {s.correlation_lift.toFixed(2)}x
                  </span>
                )}
              </div>
            </div>
            <div className="space-y-1">
              {s.legs.map((leg, j) => {
                const color = leg.team === homeTeam ? homeColor : leg.team === awayTeam ? awayColor : "#888";
                return (
                  <div key={j} className="flex items-center gap-2 text-xs">
                    <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
                    <span className="flex-1">{leg.label}</span>
                    <span className="tabular-nums font-mono text-muted-foreground">
                      {(leg.solo_prob * 100).toFixed(0)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Main export: MonteCarloCard
// ---------------------------------------------------------------------------

export function MonteCarloCard({
  simulation,
  homeTeam,
  awayTeam,
  homeColor,
  awayColor,
}: {
  simulation: MatchSimulation;
  homeTeam: string;
  awayTeam: string;
  homeColor: string;
  awayColor: string;
}) {
  const [statView, setStatView] = useState<"goals" | "disposals" | "marks">("goals");
  const outcomes = simulation.match_outcomes;
  const homeAbbr = TEAM_ABBREVS[homeTeam] || homeTeam;
  const awayAbbr = TEAM_ABBREVS[awayTeam] || awayTeam;

  const homePlayers = simulation.players.filter((p) => p.is_home);
  const awayPlayers = simulation.players.filter((p) => !p.is_home);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold">Monte Carlo Simulation</h2>
          <p className="text-[10px] text-muted-foreground/60 font-mono">
            {simulation.n_sims.toLocaleString()} simulations — correlated player outcomes
          </p>
        </div>
      </div>

      {/* Match outcome summary */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Simulated Match Outcomes</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Win probability */}
          <div className="space-y-1.5">
            <div className="w-full h-6 rounded-full bg-muted overflow-hidden flex">
              <div
                className="h-full flex items-center justify-center text-[10px] font-bold text-white"
                style={{ width: `${Math.round(outcomes.home_win_pct * 100)}%`, backgroundColor: homeColor }}
              >
                {outcomes.home_win_pct >= 0.1 && `${Math.round(outcomes.home_win_pct * 100)}%`}
              </div>
              {outcomes.draw_pct > 0.005 && (
                <div
                  className="h-full bg-muted-foreground/20 flex items-center justify-center text-[9px] font-mono text-muted-foreground"
                  style={{ width: `${Math.round(outcomes.draw_pct * 100)}%` }}
                >
                  {outcomes.draw_pct >= 0.02 && "D"}
                </div>
              )}
              <div
                className="h-full flex items-center justify-center text-[10px] font-bold text-white"
                style={{ width: `${Math.round(outcomes.away_win_pct * 100)}%`, backgroundColor: awayColor }}
              >
                {outcomes.away_win_pct >= 0.1 && `${Math.round(outcomes.away_win_pct * 100)}%`}
              </div>
            </div>
            <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
              <span>{homeAbbr} win</span>
              <span>{awayAbbr} win</span>
            </div>
          </div>

          {/* Score distributions */}
          <div className="space-y-2">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
              Score Range <span className="normal-case text-muted-foreground/40 font-normal ml-1">p10 | median | p90</span>
            </p>
            <PercentilesRow label={homeAbbr} pcts={outcomes.score_distribution.home} color={homeColor} />
            <PercentilesRow label={awayAbbr} pcts={outcomes.score_distribution.away} color={awayColor} />
          </div>

          {/* Key stats */}
          <div className="grid grid-cols-3 gap-3">
            <div className="text-center">
              <p className="text-xs text-muted-foreground">Avg Total</p>
              <p className="text-lg font-bold tabular-nums font-mono">{outcomes.avg_total.toFixed(0)}</p>
            </div>
            <div className="text-center">
              <p className="text-xs text-muted-foreground">Avg Margin</p>
              <p className="text-lg font-bold tabular-nums font-mono">
                {outcomes.avg_margin > 0 ? "+" : ""}{outcomes.avg_margin.toFixed(0)}
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-muted-foreground">Favourite</p>
              <p className="text-lg font-bold">
                {outcomes.home_win_pct > outcomes.away_win_pct ? homeAbbr : awayAbbr}
              </p>
            </div>
          </div>

          {/* Total score brackets */}
          {outcomes.total_brackets.length > 0 && (
            <div>
              <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-2">
                Total Score Probability
              </p>
              <div className="flex flex-wrap gap-1.5">
                {outcomes.total_brackets.map((b) => {
                  const pct = Math.round(b.p_over * 100);
                  const intensity = pct >= 70 ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30" :
                    pct >= 50 ? "bg-amber-500/10 text-amber-400 border-amber-500/30" :
                    pct >= 20 ? "bg-muted text-muted-foreground border-border/30" :
                    "bg-muted/50 text-muted-foreground/50 border-border/20";
                  return (
                    <div
                      key={b.threshold}
                      className={cn("px-2 py-1 rounded border text-[10px] font-mono tabular-nums", intensity)}
                    >
                      {b.threshold}+: {pct}%
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Stat view toggle */}
      <div className="flex items-center gap-1">
        {(["goals", "disposals", "marks"] as const).map((view) => (
          <button
            key={view}
            onClick={() => setStatView(view)}
            className={cn(
              "px-3 py-1 text-xs font-medium border transition-colors",
              view === "goals" ? "rounded-l-md" : view === "marks" ? "rounded-r-md border-l-0" : "border-l-0",
              statView === view
                ? "bg-primary text-primary-foreground border-primary"
                : "bg-muted/50 text-muted-foreground border-border hover:bg-muted"
            )}
          >
            {view.charAt(0).toUpperCase() + view.slice(1)}
          </button>
        ))}
      </div>

      {/* Player simulation tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <PlayerSimTable
          players={homePlayers}
          teamName={homeTeam}
          teamColor={homeColor}
          statView={statView}
        />
        <PlayerSimTable
          players={awayPlayers}
          teamName={awayTeam}
          teamColor={awayColor}
          statView={statView}
        />
      </div>

    </div>
  );
}
