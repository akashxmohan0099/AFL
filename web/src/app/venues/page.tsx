"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
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
import { getVenues } from "@/lib/api";
import type { VenueInfo } from "@/lib/types";
import { TEAM_ABBREVS, TEAM_COLORS, displayVenue } from "@/lib/constants";
import { cn } from "@/lib/utils";

type SortKey = "total_games" | "avg_total_score" | "avg_margin" | "avg_temperature" | "pct_wet_games" | "venue";

export default function VenuesPage() {
  const [venues, setVenues] = useState<VenueInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortKey, setSortKey] = useState<SortKey>("total_games");
  const [sortAsc, setSortAsc] = useState(false);
  const [filterTeam, setFilterTeam] = useState("");

  useEffect(() => {
    setLoading(true);
    getVenues()
      .then(setVenues)
      .catch(() => setVenues([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(key === "venue"); }
  };

  const sortIcon = (key: SortKey) => sortKey === key ? (sortAsc ? " \u25B2" : " \u25BC") : "";
  const thCls = "cursor-pointer hover:bg-muted/50 select-none";

  // Get all unique teams for filter
  const allTeams = Array.from(new Set(venues.flatMap((v) => (v.home_teams || []).map((ht) => ht.team)))).sort();

  const filtered = filterTeam
    ? venues.filter((v) => (v.home_teams || []).some((ht) => ht.team === filterTeam))
    : venues;

  const sorted = [...filtered].sort((a, b) => {
    let va: string | number = 0;
    let vb: string | number = 0;
    if (sortKey === "venue") {
      va = displayVenue(a.venue);
      vb = displayVenue(b.venue);
    } else {
      va = (a as unknown as Record<string, number>)[sortKey] ?? 0;
      vb = (b as unknown as Record<string, number>)[sortKey] ?? 0;
    }
    if (va < vb) return sortAsc ? -1 : 1;
    if (va > vb) return sortAsc ? 1 : -1;
    return 0;
  });

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-3 flex-wrap">
        <h1 className="text-xl font-bold tracking-tight">Venues</h1>
        <Badge variant="outline" className="text-xs font-mono">
          {filtered.length} grounds
        </Badge>
      </div>

      {/* Team filter */}
      <div className="flex gap-1.5 flex-wrap items-center">
        <button
          onClick={() => setFilterTeam("")}
          className={cn(
            "px-2.5 py-1 text-xs font-medium rounded-md border transition-colors",
            !filterTeam ? "bg-primary text-primary-foreground border-primary" : "bg-card border-border text-muted-foreground hover:text-foreground"
          )}
        >
          All
        </button>
        {allTeams.map((team) => (
          <button
            key={team}
            onClick={() => setFilterTeam(filterTeam === team ? "" : team)}
            className={cn(
              "px-2 py-1 text-xs font-mono font-medium rounded-md border transition-colors flex items-center gap-1.5",
              filterTeam === team ? "bg-primary/10 text-primary border-primary/30" : "bg-card border-border/50 text-muted-foreground hover:text-foreground"
            )}
          >
            <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: TEAM_COLORS[team]?.primary || "#555" }} />
            {TEAM_ABBREVS[team] || team}
          </button>
        ))}
      </div>

      {venues.length === 0 ? (
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground">No venue data available.</p>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="pt-4">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className={thCls} onClick={() => handleSort("venue")}>
                      Venue{sortIcon("venue")}
                    </TableHead>
                    <TableHead>City</TableHead>
                    <TableHead>Home Team(s)</TableHead>
                    <TableHead className={cn(thCls, "text-right")} onClick={() => handleSort("total_games")}>
                      Games{sortIcon("total_games")}
                    </TableHead>
                    <TableHead className="text-right">Period</TableHead>
                    <TableHead className={cn(thCls, "text-right")} onClick={() => handleSort("avg_total_score")}>
                      Avg Score{sortIcon("avg_total_score")}
                    </TableHead>
                    <TableHead className={cn(thCls, "text-right")} onClick={() => handleSort("avg_margin")}>
                      Avg Margin{sortIcon("avg_margin")}
                    </TableHead>
                    <TableHead className={cn(thCls, "text-right")} onClick={() => handleSort("avg_temperature")}>
                      Avg Temp{sortIcon("avg_temperature")}
                    </TableHead>
                    <TableHead className={cn(thCls, "text-right")} onClick={() => handleSort("pct_wet_games")}>
                      Wet %{sortIcon("pct_wet_games")}
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sorted.map((v) => (
                    <TableRow key={v.venue} className="group">
                      <TableCell>
                        <Link
                          href={`/venues/${encodeURIComponent(v.venue)}`}
                          className="font-semibold text-sm hover:text-primary transition-colors"
                        >
                          {displayVenue(v.venue)}
                          {v.is_roofed && (
                            <Badge variant="outline" className="ml-1.5 text-[9px] px-1 py-0 text-green-400 border-green-400/30">
                              Roof
                            </Badge>
                          )}
                        </Link>
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {v.city || "-"}
                      </TableCell>
                      <TableCell>
                        {v.home_teams && v.home_teams.length > 0 ? (
                          <div className="flex flex-wrap gap-1.5">
                            {v.home_teams.map((ht) => (
                              <span key={ht.team} className="flex items-center gap-1">
                                <span
                                  className="w-2 h-2 rounded-full shrink-0"
                                  style={{ backgroundColor: TEAM_COLORS[ht.team]?.primary || "#555" }}
                                />
                                <span className="text-xs font-mono">
                                  {TEAM_ABBREVS[ht.team] || ht.team}
                                </span>
                              </span>
                            ))}
                          </div>
                        ) : (
                          <span className="text-xs text-muted-foreground">-</span>
                        )}
                      </TableCell>
                      <TableCell className="text-right tabular-nums font-semibold text-sm">
                        {v.total_games}
                      </TableCell>
                      <TableCell className="text-right text-xs text-muted-foreground tabular-nums">
                        {v.year_from && v.year_to
                          ? v.year_from === v.year_to
                            ? `${v.year_from}`
                            : `${v.year_from}-${String(v.year_to).slice(-2)}`
                          : "-"}
                      </TableCell>
                      <TableCell className="text-right tabular-nums text-sm">
                        {v.avg_total_score?.toFixed(0) ?? "-"}
                      </TableCell>
                      <TableCell className="text-right tabular-nums text-sm">
                        {v.avg_margin?.toFixed(1) ?? "-"}
                      </TableCell>
                      <TableCell className="text-right tabular-nums text-sm text-muted-foreground">
                        {v.avg_temperature != null ? `${v.avg_temperature.toFixed(0)}\u00B0C` : "-"}
                      </TableCell>
                      <TableCell className="text-right tabular-nums text-sm text-muted-foreground">
                        {v.pct_wet_games != null ? `${v.pct_wet_games.toFixed(0)}%` : "-"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
