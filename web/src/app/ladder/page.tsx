"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
import { Badge } from "@/components/ui/badge";
import { getLadder } from "@/lib/api";
import type { LadderEntry } from "@/lib/types";
import { TEAM_COLORS, TEAM_ABBREVS, CURRENT_YEAR, AVAILABLE_YEARS } from "@/lib/constants";
import { ExportButton } from "@/components/ui/export-button";
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

export default function LadderPage() {
  const [year, setYear] = useState(CURRENT_YEAR);
  const [ladder, setLadder] = useState<LadderEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getLadder(year)
      .then((res) => setLadder(res.ladder))
      .catch(() => setError("Failed to load ladder."))
      .finally(() => setLoading(false));
  }, [year]);

  const exportData = ladder.map((t) => ({
    Position: t.position,
    Team: t.team,
    Played: t.played,
    Wins: t.wins,
    Losses: t.losses,
    Draws: t.draws,
    Points: t.points,
    "Points For": t.points_for,
    "Points Against": t.points_against,
    "Percentage": t.percentage,
    "Avg Margin": t.avg_margin,
    Form: t.form.join(" "),
  }));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold">Ladder</h1>
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
        </div>
        <ExportButton
          data={exportData}
          filename={`afl_ladder_${year}`}
          columns={[
            { key: "Position", header: "#" },
            { key: "Team", header: "Team" },
            { key: "Played", header: "P" },
            { key: "Wins", header: "W" },
            { key: "Losses", header: "L" },
            { key: "Draws", header: "D" },
            { key: "Points", header: "Pts" },
            { key: "Points For", header: "PF" },
            { key: "Points Against", header: "PA" },
            { key: "Percentage", header: "%" },
            { key: "Avg Margin", header: "Avg Margin" },
            { key: "Form", header: "Form" },
          ]}
        />
      </div>

      {loading ? (
        <div className="space-y-2">
          {Array.from({ length: 18 }).map((_, i) => (
            <Skeleton key={i} className="h-10 w-full" />
          ))}
        </div>
      ) : error ? (
        <Card>
          <CardContent className="pt-6">
            <p className="text-destructive">{error}</p>
          </CardContent>
        </Card>
      ) : ladder.length === 0 ? (
        <Card>
          <CardContent className="pt-6">
            <p className="text-muted-foreground">No ladder data available for {year}.</p>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">
              {year} AFL Standings
              <Badge variant="outline" className="ml-2 text-xs font-normal">
                {ladder.length} teams
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-8">#</TableHead>
                    <TableHead>Team</TableHead>
                    <TableHead className="text-center">P</TableHead>
                    <TableHead className="text-center">W</TableHead>
                    <TableHead className="text-center">L</TableHead>
                    <TableHead className="text-center">D</TableHead>
                    <TableHead className="text-center font-semibold">Pts</TableHead>
                    <TableHead className="text-right">PF</TableHead>
                    <TableHead className="text-right">PA</TableHead>
                    <TableHead className="text-right">%</TableHead>
                    <TableHead className="text-right">Avg Mgn</TableHead>
                    <TableHead className="text-center">Form</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {ladder.map((team) => {
                    const inTop8 = team.position <= 8;
                    return (
                      <TableRow
                        key={team.team}
                        className={cn(
                          team.position === 8 && "border-b-2 border-primary/30",
                          inTop8 && "bg-primary/[0.02]"
                        )}
                      >
                        <TableCell className="tabular-nums text-xs text-muted-foreground font-medium">
                          {team.position}
                        </TableCell>
                        <TableCell>
                          <Link href={`/teams/${encodeURIComponent(team.team)}`} className="hover:text-primary transition-colors">
                            <span className="flex items-center gap-2">
                              <span
                                className="w-3 h-3 rounded-full shrink-0"
                                style={{
                                  backgroundColor:
                                    TEAM_COLORS[team.team]?.primary || "#666",
                                }}
                              />
                              <span className="font-medium text-sm">
                                {team.team}
                              </span>
                              <span className="text-xs text-muted-foreground">
                                {TEAM_ABBREVS[team.team]}
                              </span>
                            </span>
                          </Link>
                        </TableCell>
                        <TableCell className="text-center tabular-nums text-sm">
                          {team.played}
                        </TableCell>
                        <TableCell className="text-center tabular-nums text-sm font-medium text-emerald-600">
                          {team.wins}
                        </TableCell>
                        <TableCell className="text-center tabular-nums text-sm text-red-500">
                          {team.losses}
                        </TableCell>
                        <TableCell className="text-center tabular-nums text-sm text-muted-foreground">
                          {team.draws}
                        </TableCell>
                        <TableCell className="text-center tabular-nums text-sm font-bold">
                          {team.points}
                        </TableCell>
                        <TableCell className="text-right tabular-nums text-sm">
                          {team.points_for}
                        </TableCell>
                        <TableCell className="text-right tabular-nums text-sm">
                          {team.points_against}
                        </TableCell>
                        <TableCell className="text-right tabular-nums text-sm font-medium">
                          {team.percentage.toFixed(1)}
                        </TableCell>
                        <TableCell className="text-right tabular-nums text-sm">
                          <span
                            className={cn(
                              team.avg_margin > 0
                                ? "text-emerald-600"
                                : team.avg_margin < 0
                                  ? "text-red-500"
                                  : "text-muted-foreground"
                            )}
                          >
                            {team.avg_margin > 0 ? "+" : ""}
                            {team.avg_margin.toFixed(1)}
                          </span>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center justify-center gap-0.5">
                            {team.form.map((r, i) => (
                              <FormBadge key={i} result={r} />
                            ))}
                          </div>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
            {ladder.length > 0 && (
              <p className="text-[10px] text-muted-foreground mt-3">
                Top 8 qualify for finals. Points: W=4, D=2, L=0. Percentage = (Points For / Points Against) x 100.
              </p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
