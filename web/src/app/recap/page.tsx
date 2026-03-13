"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  getSeasonMatches,
  getPredictionHistory,
  getRoundAccuracy,
} from "@/lib/api";
import type {
  SeasonMatch,
  PredictionHistoryEntry,
  RoundAccuracy,
} from "@/lib/types";
import { TEAM_COLORS, TEAM_ABBREVS, CURRENT_YEAR, AVAILABLE_YEARS } from "@/lib/constants";
import { ExportButton } from "@/components/ui/export-button";
import { cn } from "@/lib/utils";

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="text-center">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-lg font-bold tabular-nums">{value}</p>
      {sub && <p className="text-[10px] text-muted-foreground">{sub}</p>}
    </div>
  );
}

export default function RecapPage() {
  const [year, setYear] = useState(CURRENT_YEAR);
  const [selectedRound, setSelectedRound] = useState<number | null>(null);
  const [matches, setMatches] = useState<SeasonMatch[]>([]);
  const [history, setHistory] = useState<PredictionHistoryEntry[]>([]);
  const [accuracy, setAccuracy] = useState<RoundAccuracy[]>([]);
  const [loading, setLoading] = useState(true);

  // Load data for selected year
  useEffect(() => {
    setLoading(true);
    Promise.all([
      getSeasonMatches(year).catch(() => [] as SeasonMatch[]),
      getPredictionHistory(year).catch(() => ({ entries: [], summary: {} })),
      getRoundAccuracy(year).catch(() => [] as RoundAccuracy[]),
    ]).then(([m, h, a]) => {
      setMatches(m);
      setHistory(h.entries || []);
      setAccuracy(a);
      // Auto-select latest completed round
      const completedRounds = [...new Set(m.filter((g) => g.home_score != null).map((g) => g.round_number))].sort((a, b) => b - a);
      if (completedRounds.length > 0 && selectedRound === null) {
        setSelectedRound(completedRounds[0]);
      }
      setLoading(false);
    });
  }, [year]); // eslint-disable-line react-hooks/exhaustive-deps

  const completedRounds = [...new Set(matches.filter((g) => g.home_score != null).map((g) => g.round_number))].sort((a, b) => a - b);

  const roundMatches = matches.filter(
    (m) => m.round_number === selectedRound && m.home_score != null
  );

  const roundPredictions = history.filter((p) => p.round === selectedRound);

  const roundAccuracy = accuracy.find((a) => a.round_number === selectedRound);

  // Game winner stats for this round
  const gameResults = roundMatches.map((m) => {
    const actualWinner =
      m.home_score != null && m.away_score != null
        ? m.home_score > m.away_score
          ? m.home_team
          : m.away_score > m.home_score
            ? m.away_team
            : "Draw"
        : null;
    return {
      ...m,
      actual_winner: actualWinner,
      correct: m.predicted_winner != null && m.predicted_winner === actualWinner,
    };
  });

  const winnerCorrect = gameResults.filter((g) => g.correct).length;
  const winnerTotal = gameResults.filter((g) => g.actual_winner != null && g.predicted_winner != null).length;

  // Top performers vs predictions
  const bigBeats = roundPredictions
    .filter((p) => p.actual_goals != null && p.predicted_goals != null)
    .map((p) => ({
      ...p,
      goal_diff: (p.actual_goals ?? 0) - (p.predicted_goals ?? 0),
      disp_diff: (p.actual_disposals ?? 0) - (p.predicted_disposals ?? 0),
    }))
    .sort((a, b) => b.goal_diff - a.goal_diff);

  const exportData = roundPredictions.map((p) => ({
    Player: p.player,
    Team: p.team,
    Opponent: p.opponent,
    "Pred GL": p.predicted_goals?.toFixed(2) ?? "",
    "Act GL": p.actual_goals ?? "",
    "Pred DI": p.predicted_disposals?.toFixed(1) ?? "",
    "Act DI": p.actual_disposals ?? "",
    "Pred MK": p.predicted_marks?.toFixed(1) ?? "",
    "Act MK": p.actual_marks ?? "",
  }));

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-48" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-48" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <h1 className="text-xl sm:text-2xl font-bold">Round Recap</h1>
          <div className="flex gap-1">
            {AVAILABLE_YEARS.slice(0, 3).map((y) => (
              <Button
                key={y}
                variant={y === year ? "default" : "outline"}
                size="sm"
                className="h-7 text-xs px-2.5"
                onClick={() => {
                  setYear(y);
                  setSelectedRound(null);
                }}
              >
                {y}
              </Button>
            ))}
          </div>
        </div>
        {selectedRound && (
          <ExportButton
            data={exportData}
            filename={`recap_R${selectedRound}_${year}`}
          />
        )}
      </div>

      {/* Round selector */}
      {completedRounds.length > 0 ? (
        <div className="flex flex-wrap gap-1">
          {completedRounds.map((r) => (
            <Button
              key={r}
              variant={r === selectedRound ? "default" : "outline"}
              size="sm"
              className="h-7 text-xs px-2.5 min-w-[40px]"
              onClick={() => setSelectedRound(r)}
            >
              R{r}
            </Button>
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="pt-6">
            <p className="text-muted-foreground">No completed rounds for {year}.</p>
          </CardContent>
        </Card>
      )}

      {selectedRound != null && (
        <>
          {/* Summary stats */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">
                Round {selectedRound} Summary
                <Badge variant="outline" className="ml-2 text-xs font-normal">
                  {roundMatches.length} games
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-4">
                <StatCard
                  label="Game Winner"
                  value={winnerTotal > 0 ? `${winnerCorrect}/${winnerTotal}` : "-"}
                  sub={winnerTotal > 0 ? `${((winnerCorrect / winnerTotal) * 100).toFixed(0)}%` : undefined}
                />
                <StatCard
                  label="Goals MAE"
                  value={roundAccuracy?.goals_mae?.toFixed(3) ?? "-"}
                />
                <StatCard
                  label="Disposals MAE"
                  value={roundAccuracy?.disposals_mae?.toFixed(2) ?? "-"}
                />
                <StatCard
                  label="Marks MAE"
                  value={roundAccuracy?.marks_mae?.toFixed(2) ?? "-"}
                />
                <StatCard
                  label="Scorer Acc"
                  value={roundAccuracy?.scorer_accuracy != null ? `${roundAccuracy.scorer_accuracy.toFixed(0)}%` : "-"}
                />
                <StatCard
                  label="Players"
                  value={String(roundPredictions.length)}
                />
              </div>
            </CardContent>
          </Card>

          {/* Game results */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Match Results</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {gameResults.map((m) => (
                  <Link
                    key={m.match_id}
                    href={`/matches/${m.match_id}`}
                    className="block"
                  >
                    <div
                      className={cn(
                        "flex items-center gap-3 p-3 rounded-lg border transition-colors hover:bg-muted/50",
                        m.correct ? "border-emerald-500/20" : "border-red-500/20"
                      )}
                    >
                      {/* Home team */}
                      <div className="flex items-center gap-2 flex-1 justify-end">
                        <span className="text-sm font-medium">
                          {TEAM_ABBREVS[m.home_team] || m.home_team}
                        </span>
                        <span
                          className="w-3 h-3 rounded-full shrink-0"
                          style={{ backgroundColor: TEAM_COLORS[m.home_team]?.primary || "#666" }}
                        />
                      </div>

                      {/* Score */}
                      <div className="flex items-center gap-2 tabular-nums text-sm font-bold min-w-[80px] justify-center">
                        <span>{m.home_score}</span>
                        <span className="text-muted-foreground text-xs">-</span>
                        <span>{m.away_score}</span>
                      </div>

                      {/* Away team */}
                      <div className="flex items-center gap-2 flex-1">
                        <span
                          className="w-3 h-3 rounded-full shrink-0"
                          style={{ backgroundColor: TEAM_COLORS[m.away_team]?.primary || "#666" }}
                        />
                        <span className="text-sm font-medium">
                          {TEAM_ABBREVS[m.away_team] || m.away_team}
                        </span>
                      </div>

                      {/* Prediction result */}
                      <div className="flex items-center gap-2 min-w-[100px] justify-end">
                        {m.home_win_prob != null && (
                          <span className="text-[10px] text-muted-foreground">
                            {(m.home_win_prob * 100).toFixed(0)}% home
                          </span>
                        )}
                        <Badge
                          variant={m.correct ? "default" : "destructive"}
                          className="text-[10px] px-1.5"
                        >
                          {m.correct ? "HIT" : "MISS"}
                        </Badge>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Biggest over/underperformers */}
          {bigBeats.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Over-performers */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base text-emerald-600">
                    Top Over-Performers (Goals)
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Player</TableHead>
                        <TableHead>Team</TableHead>
                        <TableHead className="text-right">Pred</TableHead>
                        <TableHead className="text-right">Actual</TableHead>
                        <TableHead className="text-right">Diff</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {bigBeats.slice(0, 10).map((p, i) => (
                        <TableRow key={`over-${i}`}>
                          <TableCell className="text-sm font-medium">
                            <Link
                              href={`/players/${encodeURIComponent(`${p.player}_${p.team}`)}`}
                              className="hover:text-primary transition-colors"
                            >
                              {p.player}
                            </Link>
                          </TableCell>
                          <TableCell>
                            <span className="flex items-center gap-1.5">
                              <span
                                className="w-2 h-2 rounded-full"
                                style={{ backgroundColor: TEAM_COLORS[p.team]?.primary || "#666" }}
                              />
                              <span className="text-xs">{TEAM_ABBREVS[p.team]}</span>
                            </span>
                          </TableCell>
                          <TableCell className="text-right tabular-nums text-sm">
                            {p.predicted_goals?.toFixed(2)}
                          </TableCell>
                          <TableCell className="text-right tabular-nums text-sm font-medium">
                            {p.actual_goals}
                          </TableCell>
                          <TableCell className="text-right tabular-nums text-sm">
                            <span className={p.goal_diff > 0 ? "text-emerald-600" : p.goal_diff < 0 ? "text-red-500" : ""}>
                              {p.goal_diff > 0 ? "+" : ""}{p.goal_diff.toFixed(2)}
                            </span>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>

              {/* Under-performers */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base text-red-500">
                    Biggest Under-Performers (Goals)
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Player</TableHead>
                        <TableHead>Team</TableHead>
                        <TableHead className="text-right">Pred</TableHead>
                        <TableHead className="text-right">Actual</TableHead>
                        <TableHead className="text-right">Diff</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {[...bigBeats].reverse().slice(0, 10).map((p, i) => (
                        <TableRow key={`under-${i}`}>
                          <TableCell className="text-sm font-medium">
                            <Link
                              href={`/players/${encodeURIComponent(`${p.player}_${p.team}`)}`}
                              className="hover:text-primary transition-colors"
                            >
                              {p.player}
                            </Link>
                          </TableCell>
                          <TableCell>
                            <span className="flex items-center gap-1.5">
                              <span
                                className="w-2 h-2 rounded-full"
                                style={{ backgroundColor: TEAM_COLORS[p.team]?.primary || "#666" }}
                              />
                              <span className="text-xs">{TEAM_ABBREVS[p.team]}</span>
                            </span>
                          </TableCell>
                          <TableCell className="text-right tabular-nums text-sm">
                            {p.predicted_goals?.toFixed(2)}
                          </TableCell>
                          <TableCell className="text-right tabular-nums text-sm font-medium">
                            {p.actual_goals}
                          </TableCell>
                          <TableCell className="text-right tabular-nums text-sm">
                            <span className={p.goal_diff > 0 ? "text-emerald-600" : p.goal_diff < 0 ? "text-red-500" : ""}>
                              {p.goal_diff > 0 ? "+" : ""}{p.goal_diff.toFixed(2)}
                            </span>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </div>
          )}
        </>
      )}
    </div>
  );
}
