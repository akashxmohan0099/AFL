"use client";

import { useEffect, useState } from "react";
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
import { getPredictionHistory } from "@/lib/api";
import type { PredictionHistorySummary } from "@/lib/types";
import { TEAM_ABBREVS, TEAM_COLORS, CURRENT_YEAR, AVAILABLE_YEARS } from "@/lib/constants";
import { cn } from "@/lib/utils";
import Link from "next/link";

function predColor(actual: number, predicted: number, threshold: number): string {
  return Math.abs(actual - predicted) <= threshold ? "text-emerald-400" : "text-red-400";
}

export default function PredictionHistoryPage() {
  const [data, setData] = useState<PredictionHistorySummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [year, setYear] = useState(CURRENT_YEAR);
  const [filterTeam, setFilterTeam] = useState("");
  const [filterRound, setFilterRound] = useState<number | null>(null);
  const [sortCol, setSortCol] = useState<string>("round");
  const [sortAsc, setSortAsc] = useState(true);

  const years = AVAILABLE_YEARS;

  useEffect(() => {
    setLoading(true);
    getPredictionHistory(year)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [year]);

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-64" />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
        <Skeleton className="h-64" />
        <Skeleton className="h-96" />
      </div>
    );
  }

  const entries = data?.entries || [];
  const summary = data?.summary;

  // Unique teams and rounds for filtering
  const teams = [...new Set(entries.map((e) => e.team))].sort();
  const rounds = [...new Set(entries.map((e) => e.round))].sort((a, b) => a - b);

  // Filter
  let filtered = entries;
  if (filterTeam) filtered = filtered.filter((e) => e.team === filterTeam);
  if (filterRound != null) filtered = filtered.filter((e) => e.round === filterRound);

  // Sort
  const handleSort = (col: string) => {
    if (sortCol === col) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(col);
      setSortAsc(col === "round" || col === "player");
    }
  };

  const sortedEntries = [...filtered].sort((a, b) => {
    const av = (a as unknown as Record<string, unknown>)[sortCol];
    const bv = (b as unknown as Record<string, unknown>)[sortCol];
    const va = typeof av === "number" ? av : typeof av === "string" ? av : 0;
    const vb = typeof bv === "number" ? bv : typeof bv === "string" ? bv : 0;
    if (va < vb) return sortAsc ? -1 : 1;
    if (va > vb) return sortAsc ? 1 : -1;
    return 0;
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold">Predictions vs Actuals</h1>
          <Badge variant="secondary" className="text-sm">{year}</Badge>
          {summary && (
            <Badge variant="outline" className="text-sm">
              {summary.total_predictions.toLocaleString()} predictions
            </Badge>
          )}
        </div>
      </div>

      {/* Year selector */}
      <div className="flex gap-2 flex-wrap items-center">
        {years.map((y) => (
          <Button
            key={y}
            variant={y === year ? "default" : "outline"}
            size="sm"
            onClick={() => { setYear(y); setFilterTeam(""); setFilterRound(null); }}
          >
            {y}
          </Button>
        ))}
        {teams.length > 0 && (
          <>
            <span className="mx-2 text-muted-foreground">|</span>
            <select
              className="border border-border rounded-md px-2 py-1.5 text-sm bg-background"
              value={filterTeam}
              onChange={(e) => setFilterTeam(e.target.value)}
            >
              <option value="">All Teams</option>
              {teams.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </>
        )}
        {rounds.length > 0 && (
          <select
            className="border border-border rounded-md px-2 py-1.5 text-sm bg-background"
            value={filterRound ?? ""}
            onChange={(e) => setFilterRound(e.target.value ? Number(e.target.value) : null)}
          >
            <option value="">All Rounds</option>
            {rounds.map((r) => (
              <option key={r} value={r}>Round {r}</option>
            ))}
          </select>
        )}
      </div>

      {/* Summary Cards */}
      {summary && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="cursor-help" title="Mean Absolute Error for goal predictions. Lower is better. A MAE of 0.5 means on average we're off by half a goal per player.">
            <CardContent className="pt-5 pb-4">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Goals MAE</p>
              <p className="text-2xl font-bold mt-1 tabular-nums">{summary.goals_mae.toFixed(3)}</p>
            </CardContent>
          </Card>
          <Card className="cursor-help" title="Mean Absolute Error for disposal predictions. Lower is better. A MAE of 6 means on average we're off by 6 disposals per player.">
            <CardContent className="pt-5 pb-4">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Disposals MAE</p>
              <p className="text-2xl font-bold mt-1 tabular-nums">{summary.disposals_mae.toFixed(2)}</p>
            </CardContent>
          </Card>
          <Card className="cursor-help" title="Mean Absolute Error for marks predictions. Lower is better. A MAE of 1.5 means on average we're off by 1.5 marks per player.">
            <CardContent className="pt-5 pb-4">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Marks MAE</p>
              <p className="text-2xl font-bold mt-1 tabular-nums">{summary.marks_mae.toFixed(2)}</p>
            </CardContent>
          </Card>
          <Card className="cursor-help" title="How often we correctly predict whether a player will score at least 1 goal. Higher is better.">
            <CardContent className="pt-5 pb-4">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Scorer Accuracy</p>
              <p className="text-2xl font-bold mt-1 tabular-nums">{summary.scorer_accuracy.toFixed(1)}%</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Full table */}
      {filtered.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle className="text-base cursor-help" title="Every player prediction our model made compared to what actually happened. Green = within 1, Yellow = within 2, Red = off by 3+.">
              All Predictions ({filtered.length.toLocaleString()})
            </CardTitle>
            <p className="text-[10px] text-muted-foreground/50 font-mono mt-1">
              Format: <span className="font-bold text-foreground/70">Actual</span> <span className="text-muted-foreground/40">/</span> <span className="text-muted-foreground/60">Predicted</span>
            </p>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    {[
                      { key: "round", label: "Rd", tip: "Round number" },
                      { key: "player", label: "Player", tip: "Player name" },
                      { key: "team", label: "Team", tip: "Player's team" },
                      { key: "opponent", label: "Opp", tip: "Opposition team" },
                      { key: "venue", label: "Venue", tip: "Match venue" },
                      { key: "actual_goals", label: "Goals", tip: "Goals scored (actual) vs model prediction" },
                      { key: "actual_disposals", label: "Disposals", tip: "Total disposals (actual) vs model prediction" },
                      { key: "actual_marks", label: "Marks", tip: "Marks taken (actual) vs model prediction" },
                    ].map((c) => (
                      <TableHead
                        key={c.key}
                        className={cn(
                          "cursor-pointer hover:bg-muted/50 select-none",
                          c.key.startsWith("actual") ? "text-right" : ""
                        )}
                        onClick={() => handleSort(c.key)}
                        title={c.tip}
                      >
                        {c.label}
                        {sortCol === c.key && (sortAsc ? " \u25B2" : " \u25BC")}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sortedEntries.slice(0, 200).map((e, i) => {
                    return (
                    <TableRow key={i}>
                      <TableCell className="tabular-nums">{e.round}</TableCell>
                      <TableCell className="font-medium text-sm">
                        <Link href={`/players/${encodeURIComponent(`${e.player}_${e.team}`)}`} className="hover:text-primary transition-colors">{e.player}</Link>
                      </TableCell>
                      <TableCell>
                        <span className="flex items-center gap-1.5">
                          <span
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: TEAM_COLORS[e.team]?.primary || "#666" }}
                          />
                          {TEAM_ABBREVS[e.team] || e.team}
                        </span>
                      </TableCell>
                      <TableCell>{TEAM_ABBREVS[e.opponent] || e.opponent}</TableCell>
                      <TableCell className="text-xs truncate max-w-[100px]">{e.venue}</TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-baseline gap-1 justify-end">
                          <span className={cn("font-bold tabular-nums", predColor(e.actual_goals, e.predicted_goals, 1))}>
                            {e.actual_goals}
                          </span>
                          <span className="text-[10px] text-muted-foreground/40">/</span>
                          <span className="text-[11px] tabular-nums text-muted-foreground/60 font-mono">
                            {e.predicted_goals.toFixed(2)}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-baseline gap-1 justify-end">
                          <span className={cn("font-bold tabular-nums", predColor(e.actual_disposals, e.predicted_disposals, 5))}>
                            {e.actual_disposals}
                          </span>
                          <span className="text-[10px] text-muted-foreground/40">/</span>
                          <span className="text-[11px] tabular-nums text-muted-foreground/60 font-mono">
                            {e.predicted_disposals.toFixed(1)}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-baseline gap-1 justify-end">
                          <span className={cn("font-bold tabular-nums", predColor(e.actual_marks, e.predicted_marks, 2))}>
                            {e.actual_marks}
                          </span>
                          <span className="text-[10px] text-muted-foreground/40">/</span>
                          <span className="text-[11px] tabular-nums text-muted-foreground/60 font-mono">
                            {e.predicted_marks.toFixed(1)}
                          </span>
                        </div>
                      </TableCell>
                    </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
              {sortedEntries.length > 200 && (
                <p className="text-xs text-muted-foreground mt-2 text-center">
                  Showing 200 of {sortedEntries.length.toLocaleString()} entries. Use filters to narrow down.
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground">
              No prediction history available for {year}.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
