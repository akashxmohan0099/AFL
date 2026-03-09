"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { getUpcoming } from "@/lib/api";
import type { UpcomingRound, PlayerPrediction } from "@/lib/types";
import { TEAM_ABBREVS, TEAM_COLORS, CURRENT_YEAR } from "@/lib/constants";
import { ArrowRight } from "lucide-react";

interface TeamAggregate {
  team: string;
  opponent: string;
  players: number;
  totalGoals: number;
  totalDisposals: number;
  totalMarks: number;
}

function LeaderboardTable({
  title,
  data,
  valueKey,
  valueLabel,
  decimals,
  extraCols,
}: {
  title: string;
  data: PlayerPrediction[];
  valueKey: keyof PlayerPrediction;
  valueLabel: string;
  decimals: number;
  extraCols?: { key: keyof PlayerPrediction; label: string; format: (v: number) => string }[];
}) {
  const [showAll, setShowAll] = useState(false);

  const sorted = [...data]
    .sort((a, b) => ((b[valueKey] as number) ?? 0) - ((a[valueKey] as number) ?? 0))
    .filter((p) => (p[valueKey] as number) != null);

  const visible = showAll ? sorted : sorted.slice(0, 10);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-8">#</TableHead>
              <TableHead>Player</TableHead>
              <TableHead>Team</TableHead>
              <TableHead>Opp</TableHead>
              <TableHead className="text-right">{valueLabel}</TableHead>
              {extraCols?.map((c) => (
                <TableHead key={c.key as string} className="text-right">{c.label}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {visible.map((p, i) => (
              <TableRow key={`${p.player}-${p.team}`}>
                <TableCell className="text-muted-foreground tabular-nums text-xs">{i + 1}</TableCell>
                <TableCell className="font-medium text-sm">
                  <Link href={`/players/${encodeURIComponent(`${p.player}_${p.team}`)}`} className="hover:text-primary transition-colors">{p.player}</Link>
                </TableCell>
                <TableCell>
                  <span className="flex items-center gap-1.5">
                    <span
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: TEAM_COLORS[p.team]?.primary || "#666" }}
                    />
                    <span className="text-xs">{TEAM_ABBREVS[p.team] || p.team}</span>
                  </span>
                </TableCell>
                <TableCell className="text-xs">{TEAM_ABBREVS[p.opponent] || p.opponent}</TableCell>
                <TableCell className="text-right font-semibold tabular-nums">
                  {((p[valueKey] as number) ?? 0).toFixed(decimals)}
                </TableCell>
                {extraCols?.map((c) => (
                  <TableCell key={c.key as string} className="text-right tabular-nums text-sm">
                    {(p[c.key] as number) != null ? c.format(p[c.key] as number) : "-"}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {sorted.length > 10 && (
          <Button
            variant="ghost"
            size="sm"
            className="w-full mt-2 text-xs text-muted-foreground"
            onClick={() => setShowAll(!showAll)}
          >
            {showAll ? "Show top 10" : `Show all ${sorted.length}`}
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

export default function PredictionsPage() {
  const [upcoming, setUpcoming] = useState<UpcomingRound | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getUpcoming(CURRENT_YEAR)
      .then(setUpcoming)
      .catch(() => setError("Failed to load predictions. Is the API running?"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-48" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-64" />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold">Predictions</h1>
        <Card>
          <CardContent className="pt-6">
            <p className="text-destructive">{error}</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!upcoming || upcoming.predictions.length === 0) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold">Predictions</h1>
        <Card>
          <CardContent className="pt-6">
            <p className="text-muted-foreground">
              No upcoming predictions available for {CURRENT_YEAR}.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const predictions = upcoming.predictions;

  // Team aggregates
  const teamMap = new Map<string, TeamAggregate>();
  for (const p of predictions) {
    const existing = teamMap.get(p.team);
    if (existing) {
      existing.players += 1;
      existing.totalGoals += p.predicted_goals ?? 0;
      existing.totalDisposals += p.predicted_disposals ?? 0;
      existing.totalMarks += p.predicted_marks ?? 0;
    } else {
      teamMap.set(p.team, {
        team: p.team,
        opponent: p.opponent,
        players: 1,
        totalGoals: p.predicted_goals ?? 0,
        totalDisposals: p.predicted_disposals ?? 0,
        totalMarks: p.predicted_marks ?? 0,
      });
    }
  }
  const teamAggregates = [...teamMap.values()].sort((a, b) => b.totalGoals - a.totalGoals);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold">Predictions</h1>
          <Badge variant="secondary" className="text-sm">
            Round {upcoming.round_number}, {upcoming.year}
          </Badge>
          <Badge variant="outline" className="text-sm">
            {predictions.length} players
          </Badge>
        </div>
        <Link
          href="/predictions/history"
          className="text-sm text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
        >
          View history <ArrowRight className="w-3.5 h-3.5" />
        </Link>
      </div>

      {/* Team Aggregates */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Team Predictions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Team</TableHead>
                  <TableHead>vs</TableHead>
                  <TableHead className="text-right">Players</TableHead>
                  <TableHead className="text-right">Pred Goals</TableHead>
                  <TableHead className="text-right">Pred Disposals</TableHead>
                  <TableHead className="text-right">Pred Marks</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {teamAggregates.map((t) => (
                  <TableRow key={t.team}>
                    <TableCell>
                      <span className="flex items-center gap-2">
                        <span
                          className="w-2.5 h-2.5 rounded-full shrink-0"
                          style={{ backgroundColor: TEAM_COLORS[t.team]?.primary || "#666" }}
                        />
                        <span className="font-medium text-sm">{TEAM_ABBREVS[t.team] || t.team}</span>
                      </span>
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {TEAM_ABBREVS[t.opponent] || t.opponent}
                    </TableCell>
                    <TableCell className="text-right tabular-nums text-sm">{t.players}</TableCell>
                    <TableCell className="text-right tabular-nums font-semibold">{t.totalGoals.toFixed(1)}</TableCell>
                    <TableCell className="text-right tabular-nums text-sm">{t.totalDisposals.toFixed(0)}</TableCell>
                    <TableCell className="text-right tabular-nums text-sm">{t.totalMarks.toFixed(0)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Leaderboards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <LeaderboardTable
          title="Top Goal Scorers"
          data={predictions}
          valueKey="predicted_goals"
          valueLabel="Pred GL"
          decimals={2}
          extraCols={[
            { key: "p_scorer", label: "P(1+)", format: (v) => `${(v * 100).toFixed(0)}%` },
            { key: "p_2plus_goals", label: "P(2+)", format: (v) => `${(v * 100).toFixed(0)}%` },
          ]}
        />
        <LeaderboardTable
          title="Top Disposal Getters"
          data={predictions}
          valueKey="predicted_disposals"
          valueLabel="Pred DI"
          decimals={1}
          extraCols={[
            { key: "p_20plus_disp", label: "P(20+)", format: (v) => `${(v * 100).toFixed(0)}%` },
            { key: "p_25plus_disp", label: "P(25+)", format: (v) => `${(v * 100).toFixed(0)}%` },
          ]}
        />
        <LeaderboardTable
          title="Top Mark Takers"
          data={predictions}
          valueKey="predicted_marks"
          valueLabel="Pred MK"
          decimals={1}
          extraCols={[
            { key: "player_role" as keyof PlayerPrediction, label: "Role", format: (v) => String(v) },
          ]}
        />
      </div>
    </div>
  );
}
