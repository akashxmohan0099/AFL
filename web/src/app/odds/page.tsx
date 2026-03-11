"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
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
import { getGameOdds, getPlayerOdds, getRounds } from "@/lib/api";
import type { OddsComparison, PlayerOddsComparison, RoundInfo } from "@/lib/types";
import { TEAM_ABBREVS, TEAM_COLORS, CURRENT_YEAR } from "@/lib/constants";

function EdgeBadge({ edge }: { edge?: number }) {
  if (edge == null)
    return <span className="text-muted-foreground text-xs">-</span>;
  const pct = (edge * 100).toFixed(1);
  const absEdge = Math.abs(edge);
  if (absEdge < 0.02) {
    return (
      <Badge variant="outline" className="tabular-nums text-xs">
        {pct}%
      </Badge>
    );
  }
  return (
    <Badge
      variant={edge > 0 ? "default" : "destructive"}
      className="tabular-nums text-xs"
    >
      {edge > 0 ? "+" : ""}
      {pct}%
    </Badge>
  );
}

function ProbBar({ prob }: { prob: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-2 rounded-full bg-muted overflow-hidden">
        <div
          className="h-full bg-blue-500 rounded-full"
          style={{ width: `${Math.round(prob * 100)}%` }}
        />
      </div>
      <span className="text-xs tabular-nums">
        {(prob * 100).toFixed(1)}%
      </span>
    </div>
  );
}

export default function OddsPage() {
  const [rounds, setRounds] = useState<RoundInfo[]>([]);
  const [year, setYear] = useState(CURRENT_YEAR);
  const [round, setRound] = useState<number | null>(null);
  const [gameOdds, setGameOdds] = useState<OddsComparison[]>([]);
  const [playerOdds, setPlayerOdds] = useState<PlayerOddsComparison[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getRounds()
      .then(setRounds)
      .catch(() => setError("Failed to load odds data. Is the API running?"));
  }, []);

  useEffect(() => {
    if (year && round) {
      setLoading(true);
      Promise.all([
        getGameOdds(year, round).catch(() => []),
        getPlayerOdds(year, round).catch(() => []),
      ])
        .then(([g, p]) => {
          setGameOdds(g || []);
          setPlayerOdds(p || []);
        })
        .finally(() => setLoading(false));
    }
  }, [year, round]);

  const years = [...new Set(rounds.map((r) => r.year))].sort(
    (a, b) => b - a
  );
  const availableRounds = rounds
    .filter((r) => r.year === year)
    .map((r) => r.round_number)
    .sort((a, b) => a - b);

  useEffect(() => {
    if (availableRounds.length === 0) {
      setRound(null);
      return;
    }
    if (round == null || !availableRounds.includes(round)) {
      setRound(availableRounds[0]);
    }
  }, [availableRounds, round]);

  // Sort player odds by absolute edge
  const sortedPlayerOdds = [...playerOdds]
    .sort((a, b) => Math.abs(b.edge ?? 0) - Math.abs(a.edge ?? 0))
    .slice(0, 50);

  if (error) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold">Odds Comparison</h1>
        <Card>
          <CardContent className="pt-6">
            <p className="text-destructive">{error}</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Odds Comparison</h1>

      {/* Filters */}
      <div className="flex gap-2 flex-wrap items-center">
        {years.map((y) => (
          <Button
            key={y}
            variant={y === year ? "default" : "outline"}
            size="sm"
            onClick={() => {
              setYear(y);
              setRound(null);
            }}
          >
            {y}
          </Button>
        ))}
        {availableRounds.length > 0 && (
          <>
            <span className="mx-2 text-muted-foreground">|</span>
            <select
              className="border border-border rounded-md px-2 py-1.5 text-sm bg-background"
              value={round ?? ""}
              onChange={(e) => setRound(Number(e.target.value))}
            >
              {availableRounds.map((r) => (
                <option key={r} value={r}>
                  Round {r}
                </option>
              ))}
            </select>
          </>
        )}
      </div>

      {loading ? (
        <div className="space-y-4">
          <Skeleton className="h-48" />
          <Skeleton className="h-64" />
        </div>
      ) : (
        <>
          {/* Game Odds */}
          <Card>
            <CardHeader>
              <div>
                <CardTitle className="text-base">Match Winner Odds</CardTitle>
                <p className="text-xs text-muted-foreground mt-1">Our predicted win probability vs bookmaker odds. Positive edge = we think the team is more likely to win than the market does.</p>
              </div>
            </CardHeader>
            <CardContent>
              {gameOdds.length === 0 ? (
                <p className="text-muted-foreground text-sm">
                  No game odds data for this round.
                </p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Match</TableHead>
                      <TableHead className="text-right">Our Prediction</TableHead>
                      <TableHead className="text-right">Market Odds</TableHead>
                      <TableHead className="text-right">Home Edge</TableHead>
                      <TableHead className="text-right">Away Edge</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {gameOdds.map((o) => (
                      <TableRow key={o.match_id}>
                        <TableCell className="font-medium">
                          <div className="flex items-center gap-2">
                            <span
                              className="w-2.5 h-2.5 rounded-full"
                              style={{
                                backgroundColor:
                                  TEAM_COLORS[o.home_team]?.primary || "#666",
                              }}
                            />
                            {TEAM_ABBREVS[o.home_team] || o.home_team}
                            <span className="text-muted-foreground">vs</span>
                            <span
                              className="w-2.5 h-2.5 rounded-full"
                              style={{
                                backgroundColor:
                                  TEAM_COLORS[o.away_team]?.primary || "#666",
                              }}
                            />
                            {TEAM_ABBREVS[o.away_team] || o.away_team}
                          </div>
                        </TableCell>
                        <TableCell className="text-right">
                          {o.model_home_prob != null ? (
                            <ProbBar prob={o.model_home_prob} />
                          ) : (
                            "-"
                          )}
                        </TableCell>
                        <TableCell className="text-right">
                          {o.market_home_prob != null ? (
                            <ProbBar prob={o.market_home_prob} />
                          ) : (
                            "-"
                          )}
                        </TableCell>
                        <TableCell className="text-right">
                          <EdgeBadge edge={o.edge_home} />
                        </TableCell>
                        <TableCell className="text-right">
                          <EdgeBadge edge={o.edge_away} />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>

          {/* Player Odds */}
          <Card>
            <CardHeader>
              <div>
                <CardTitle className="text-base">
                  Player Markets{" "}
                  <Badge variant="outline" className="ml-2 text-xs">
                    {playerOdds.length} markets
                  </Badge>
                </CardTitle>
                <p className="text-xs text-muted-foreground mt-1">Our probability vs betting market implied probability for player stat lines. Sorted by biggest disagreement.</p>
              </div>
            </CardHeader>
            <CardContent>
              {playerOdds.length === 0 ? (
                <p className="text-muted-foreground text-sm">
                  No player odds data for this round.
                </p>
              ) : (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Player</TableHead>
                        <TableHead>Team</TableHead>
                        <TableHead>Stat Type</TableHead>
                        <TableHead className="text-right">Line</TableHead>
                        <TableHead className="text-right">Market Prob</TableHead>
                        <TableHead className="text-right">Our Prob</TableHead>
                        <TableHead className="text-right">Edge</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {sortedPlayerOdds.map((o, i) => (
                        <TableRow key={i}>
                          <TableCell className="font-medium">
                            <Link href={`/players/${encodeURIComponent(`${o.player}_${o.team}`)}`} className="hover:text-primary transition-colors">{o.player}</Link>
                          </TableCell>
                          <TableCell>
                            <span className="flex items-center gap-1.5">
                              <span
                                className="w-2 h-2 rounded-full"
                                style={{
                                  backgroundColor:
                                    TEAM_COLORS[o.team]?.primary || "#666",
                                }}
                              />
                              {TEAM_ABBREVS[o.team] || o.team}
                            </span>
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline" className="text-[10px]">
                              {o.market_type}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-right tabular-nums">
                            {o.market_line?.toFixed(1) ?? "-"}
                          </TableCell>
                          <TableCell className="text-right tabular-nums">
                            {o.market_implied_prob != null
                              ? (o.market_implied_prob * 100).toFixed(1) + "%"
                              : "-"}
                          </TableCell>
                          <TableCell className="text-right tabular-nums">
                            {o.model_prob != null
                              ? (o.model_prob * 100).toFixed(1) + "%"
                              : "-"}
                          </TableCell>
                          <TableCell className="text-right">
                            <EdgeBadge edge={o.edge} />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
