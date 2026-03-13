"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ExportButton } from "@/components/ui/export-button";
import { getTeamProfile } from "@/lib/api";
import type {
  TeamProfile,
  TeamProfileTopPlayer,
} from "@/lib/types";
import {
  TEAM_COLORS,
  TEAM_ABBREVS,
  CURRENT_YEAR,
  AVAILABLE_YEARS,
} from "@/lib/constants";
import { cn } from "@/lib/utils";

function FormBadge({ result }: { result: string }) {
  const color =
    result === "W"
      ? "bg-emerald-500/15 text-emerald-600 border-emerald-500/30"
      : result === "L"
        ? "bg-red-500/15 text-red-600 border-red-500/30"
        : "bg-amber-500/15 text-amber-600 border-amber-500/30";
  return (
    <span
      className={cn(
        "inline-flex items-center justify-center w-5 h-5 rounded text-[10px] font-bold border",
        color
      )}
    >
      {result}
    </span>
  );
}

function StatCard({
  label,
  value,
  sub,
  className,
}: {
  label: string;
  value: string | number;
  sub?: string;
  className?: string;
}) {
  return (
    <div className={cn("text-center", className)}>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-lg font-bold tabular-nums">{value}</p>
      {sub && <p className="text-[10px] text-muted-foreground">{sub}</p>}
    </div>
  );
}

function TopPlayersTable({
  title,
  players,
  statLabel,
}: {
  title: string;
  players: TeamProfileTopPlayer[];
  statLabel: string;
}) {
  if (!players || players.length === 0) return null;
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
              <TableHead className="text-right">Games</TableHead>
              <TableHead className="text-right">Total</TableHead>
              <TableHead className="text-right">Avg {statLabel}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {players.map((p, i) => (
              <TableRow key={p.player_id}>
                <TableCell className="text-muted-foreground tabular-nums text-xs">
                  {i + 1}
                </TableCell>
                <TableCell className="font-medium text-sm">
                  <Link
                    href={`/players/${encodeURIComponent(p.player_id)}`}
                    className="hover:text-primary transition-colors"
                  >
                    {p.name}
                  </Link>
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {p.games}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm font-semibold">
                  {p.total}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {p.avg.toFixed(1)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

export default function TeamProfilePage() {
  const params = useParams<{ teamName: string }>();
  const teamName = decodeURIComponent(params.teamName);

  const [year, setYear] = useState(CURRENT_YEAR);
  const [profile, setProfile] = useState<TeamProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getTeamProfile(teamName, year)
      .then(setProfile)
      .catch(() => setError("Failed to load team profile."))
      .finally(() => setLoading(false));
  }, [teamName, year]);

  const teamColor = TEAM_COLORS[teamName]?.primary || "#666";
  const teamAbbrev = TEAM_ABBREVS[teamName] || teamName;

  // Build export data from top players
  const exportData: Record<string, unknown>[] = [];
  if (profile) {
    const addPlayers = (category: string, players: TeamProfileTopPlayer[]) => {
      for (const p of players) {
        exportData.push({
          Category: category,
          Player: p.name,
          Games: p.games,
          Total: p.total,
          Avg: p.avg.toFixed(1),
        });
      }
    };
    addPlayers("Goals", profile.top_goals ?? []);
    addPlayers("Disposals", profile.top_disposals ?? []);
    addPlayers("Marks", profile.top_marks ?? []);
  }

  const exportColumns = [
    { key: "Category", header: "Category" },
    { key: "Player", header: "Player" },
    { key: "Games", header: "Games" },
    { key: "Total", header: "Total" },
    { key: "Avg", header: "Avg" },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span
            className="w-4 h-4 rounded-full shrink-0"
            style={{ backgroundColor: teamColor }}
          />
          <h1 className="text-xl sm:text-2xl font-bold">{teamName}</h1>
          <Badge variant="outline" className="text-xs">
            {teamAbbrev}
          </Badge>
        </div>
        <ExportButton
          data={exportData}
          filename={`${teamName.replace(/\s+/g, "_")}_top_players_${year}`}
          columns={exportColumns}
        />
      </div>

      {/* Year selector */}
      <div className="flex gap-1">
        {AVAILABLE_YEARS.slice(0, 5).map((y) => (
          <Button
            key={y}
            variant={y === year ? "default" : "outline"}
            size="sm"
            className="h-7 text-xs px-2.5"
            onClick={() => setYear(y)}
          >
            {y}
          </Button>
        ))}
      </div>

      {loading ? (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-32" />
            ))}
          </div>
          <Skeleton className="h-48" />
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-64" />
            ))}
          </div>
        </div>
      ) : error ? (
        <Card>
          <CardContent className="pt-6">
            <p className="text-destructive">{error}</p>
          </CardContent>
        </Card>
      ) : !profile ? (
        <Card>
          <CardContent className="pt-6">
            <p className="text-muted-foreground">
              No data available for {teamName} in {year}.
            </p>
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Season Record + Home/Away + Season Averages */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Season Record */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Season Record</CardTitle>
              </CardHeader>
              <CardContent>
                {profile.record ? (
                  <>
                    <div className="grid grid-cols-3 gap-4">
                      <StatCard
                        label="Wins"
                        value={profile.record.wins}
                        className="text-emerald-600"
                      />
                      <StatCard
                        label="Losses"
                        value={profile.record.losses}
                        className="text-red-500"
                      />
                      <StatCard
                        label="Draws"
                        value={profile.record.draws}
                        className="text-muted-foreground"
                      />
                    </div>
                    <div className="mt-4 grid grid-cols-2 gap-4 border-t pt-3">
                      <StatCard label="Points" value={profile.record.points} />
                      <StatCard
                        label="Percentage"
                        value={profile.record.percentage.toFixed(1) + "%"}
                      />
                    </div>
                    <p className="text-[10px] text-muted-foreground text-center mt-3">
                      {profile.record.played} games played
                    </p>
                  </>
                ) : (
                  <p className="text-muted-foreground text-sm">No games played.</p>
                )}
              </CardContent>
            </Card>

            {/* Home / Away Split */}
            {profile.home_away && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Home / Away</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <p className="text-xs font-medium text-muted-foreground mb-1">Home</p>
                      <div className="grid grid-cols-3 gap-2">
                        <StatCard label="Played" value={profile.home_away.home.played} />
                        <StatCard label="Wins" value={profile.home_away.home.wins} className="text-emerald-600" />
                        <StatCard label="Avg Score" value={profile.home_away.home.avg_score?.toFixed(1) ?? "-"} />
                      </div>
                    </div>
                    <div className="border-t pt-3">
                      <p className="text-xs font-medium text-muted-foreground mb-1">Away</p>
                      <div className="grid grid-cols-3 gap-2">
                        <StatCard label="Played" value={profile.home_away.away.played} />
                        <StatCard label="Wins" value={profile.home_away.away.wins} className="text-emerald-600" />
                        <StatCard label="Avg Score" value={profile.home_away.away.avg_score?.toFixed(1) ?? "-"} />
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Season Averages */}
            {profile.season_averages && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Season Averages</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 gap-4">
                    <StatCard label="Avg Score" value={profile.season_averages.avg_score.toFixed(1)} />
                    <StatCard label="Avg Conceded" value={profile.season_averages.avg_conceded.toFixed(1)} />
                    <div className="border-t pt-3">
                      <StatCard
                        label="Avg Margin"
                        value={(profile.season_averages.avg_margin > 0 ? "+" : "") + profile.season_averages.avg_margin.toFixed(1)}
                        className={profile.season_averages.avg_margin > 0 ? "text-emerald-600" : profile.season_averages.avg_margin < 0 ? "text-red-500" : ""}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Recent Form */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">
                Recent Form
                <Badge variant="outline" className="ml-2 text-xs font-normal">
                  Last {profile.recent_form.length} games
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {profile.recent_form.length === 0 ? (
                <p className="text-muted-foreground text-sm">
                  No recent games found.
                </p>
              ) : (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-8">Rd</TableHead>
                        <TableHead className="w-8">Result</TableHead>
                        <TableHead>Opponent</TableHead>
                        <TableHead className="text-right">Score</TableHead>
                        <TableHead className="text-right">Opp</TableHead>
                        <TableHead className="text-right">Margin</TableHead>
                        <TableHead>Venue</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {profile.recent_form.map((g) => (
                        <TableRow key={g.round_number}>
                          <TableCell className="tabular-nums text-xs text-muted-foreground">
                            {g.round_number}
                          </TableCell>
                          <TableCell>
                            <FormBadge result={g.result} />
                          </TableCell>
                          <TableCell>
                            <span className="flex items-center gap-1.5">
                              <span
                                className="w-2 h-2 rounded-full shrink-0"
                                style={{
                                  backgroundColor:
                                    TEAM_COLORS[g.opponent]?.primary || "#666",
                                }}
                              />
                              <span className="text-sm font-medium">
                                {g.opponent}
                              </span>
                            </span>
                          </TableCell>
                          <TableCell className="text-right tabular-nums text-sm font-semibold">
                            {g.score}
                          </TableCell>
                          <TableCell className="text-right tabular-nums text-sm">
                            {g.opp_score}
                          </TableCell>
                          <TableCell className="text-right tabular-nums text-sm">
                            <span
                              className={cn(
                                g.margin > 0
                                  ? "text-emerald-600"
                                  : g.margin < 0
                                    ? "text-red-500"
                                    : "text-muted-foreground"
                              )}
                            >
                              {g.margin > 0 ? "+" : ""}
                              {g.margin}
                            </span>
                          </TableCell>
                          <TableCell className="text-xs text-muted-foreground">
                            {g.venue}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Top Players */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <TopPlayersTable
              title="Top Goal Scorers"
              players={profile.top_goals}
              statLabel="GL"
            />
            <TopPlayersTable
              title="Top Disposal Getters"
              players={profile.top_disposals}
              statLabel="DI"
            />
            <TopPlayersTable
              title="Top Mark Takers"
              players={profile.top_marks}
              statLabel="MK"
            />
          </div>
        </>
      )}
    </div>
  );
}
