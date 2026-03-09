"use client";

import { useState, useEffect, useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { getPlayerDirectory } from "@/lib/api";
import type { PlayerDirectoryEntry } from "@/lib/types";
import { TEAM_ABBREVS, TEAM_COLORS, CURRENT_YEAR } from "@/lib/constants";
import Link from "next/link";
import { Search, ChevronUp, ChevronDown, Users } from "lucide-react";
import { cn } from "@/lib/utils";

type SortKey = "name" | "team" | "games" | "avg_goals" | "avg_disposals" | "avg_marks" | "avg_tackles" | "avg_kicks" | "avg_handballs" | "avg_hitouts";

const YEARS = Array.from({ length: CURRENT_YEAR - 2014 }, (_, i) => CURRENT_YEAR - i);

function renderSortIcon(
  sort: { key: SortKey; asc: boolean },
  col: SortKey,
) {
  if (sort.key !== col) return null;
  return sort.asc
    ? <ChevronUp className="w-3 h-3 inline ml-0.5" />
    : <ChevronDown className="w-3 h-3 inline ml-0.5" />;
}

export default function PlayersPage() {
  const [players, setPlayers] = useState<PlayerDirectoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [teamFilter, setTeamFilter] = useState<string>("");
  const [sort, setSort] = useState<{ key: SortKey; asc: boolean }>({ key: "avg_disposals", asc: false });
  const [year, setYear] = useState(CURRENT_YEAR);

  useEffect(() => {
    setLoading(true);
    // Try current year, fall back to previous if too few players
    getPlayerDirectory(year)
      .then((data) => {
        if (data.length < 100 && year === CURRENT_YEAR) {
          // Season barely started, fall back
          return getPlayerDirectory(CURRENT_YEAR - 1).then((fallback) => {
            setYear(CURRENT_YEAR - 1);
            return fallback;
          });
        }
        return data;
      })
      .then(setPlayers)
      .catch(() => setPlayers([]))
      .finally(() => setLoading(false));
  }, []);

  const loadYear = (y: number) => {
    setYear(y);
    setLoading(true);
    setTeamFilter("");
    getPlayerDirectory(y)
      .then(setPlayers)
      .catch(() => setPlayers([]))
      .finally(() => setLoading(false));
  };

  const teams = useMemo(() => {
    const t = [...new Set(players.map((p) => p.team))].sort();
    return t;
  }, [players]);

  const filtered = useMemo(() => {
    let list = players;
    if (query.length >= 2) {
      const q = query.toLowerCase();
      list = list.filter((p) => p.name.toLowerCase().includes(q));
    }
    if (teamFilter) {
      list = list.filter((p) => p.team === teamFilter);
    }
    // Sort
    const k = sort.key;
    list = [...list].sort((a, b) => {
      const av = a[k];
      const bv = b[k];
      if (typeof av === "string" && typeof bv === "string") {
        return sort.asc ? av.localeCompare(bv) : bv.localeCompare(av);
      }
      return sort.asc ? (av as number) - (bv as number) : (bv as number) - (av as number);
    });
    return list;
  }, [players, query, teamFilter, sort]);

  const toggleSort = (key: SortKey) => {
    setSort((prev) => ({
      key,
      asc: prev.key === key ? !prev.asc : key === "name" || key === "team",
    }));
  };

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-10 w-80" />
        <Skeleton className="h-[600px]" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-bold tracking-tight">Players</h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            {year} season — {filtered.length} of {players.length} players
          </p>
        </div>
        <select
          value={year}
          onChange={(e) => loadYear(Number(e.target.value))}
          className="text-xs bg-muted border border-border rounded px-2 py-1.5 font-mono"
        >
          {YEARS.map((y) => (
            <option key={y} value={y}>{y}</option>
          ))}
        </select>
      </div>

      {/* Search + Team Filter */}
      <div className="flex flex-wrap gap-3 items-center">
        <div className="relative w-64">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search players..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="pl-9 h-9 text-sm"
          />
        </div>
        <div className="flex flex-wrap gap-1.5">
          <button
            onClick={() => setTeamFilter("")}
            className={cn(
              "px-2 py-1 text-[10px] font-medium rounded border transition-colors",
              !teamFilter
                ? "bg-primary/10 text-primary border-primary/30"
                : "text-muted-foreground border-border/50 hover:bg-muted/30"
            )}
          >
            All
          </button>
          {teams.map((t) => {
            const abbr = TEAM_ABBREVS[t] || t.substring(0, 3).toUpperCase();
            const color = TEAM_COLORS[t]?.primary || "#666";
            const active = teamFilter === t;
            return (
              <button
                key={t}
                onClick={() => setTeamFilter(active ? "" : t)}
                className={cn(
                  "px-2 py-1 text-[10px] font-mono font-semibold rounded border transition-colors",
                  active
                    ? "border-primary/30"
                    : "text-muted-foreground border-border/50 hover:bg-muted/30"
                )}
                style={active ? { backgroundColor: color + "20", color, borderColor: color + "50" } : undefined}
              >
                {abbr}
              </button>
            );
          })}
        </div>
      </div>

      {/* Player Table */}
      {filtered.length === 0 ? (
        <Card>
          <CardContent className="pt-6 text-center">
            <Users className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
            <p className="text-muted-foreground text-sm">No players found</p>
          </CardContent>
        </Card>
      ) : (
        <Card className="border-border/50">
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow className="hover:bg-transparent">
                    <TableHead className="cursor-pointer select-none hover:bg-muted/30 w-[200px]" onClick={() => toggleSort("name")}>
                      Player {renderSortIcon(sort, "name")}
                    </TableHead>
                    <TableHead className="cursor-pointer select-none hover:bg-muted/30" onClick={() => toggleSort("team")}>
                      Team {renderSortIcon(sort, "team")}
                    </TableHead>
                    <TableHead className="text-right cursor-pointer select-none hover:bg-muted/30" onClick={() => toggleSort("games")}>
                      GP {renderSortIcon(sort, "games")}
                    </TableHead>
                    <TableHead className="text-right cursor-pointer select-none hover:bg-muted/30" onClick={() => toggleSort("avg_goals")}>
                      GL {renderSortIcon(sort, "avg_goals")}
                    </TableHead>
                    <TableHead className="text-right cursor-pointer select-none hover:bg-muted/30" onClick={() => toggleSort("avg_disposals")}>
                      DI {renderSortIcon(sort, "avg_disposals")}
                    </TableHead>
                    <TableHead className="text-right cursor-pointer select-none hover:bg-muted/30" onClick={() => toggleSort("avg_marks")}>
                      MK {renderSortIcon(sort, "avg_marks")}
                    </TableHead>
                    <TableHead className="text-right cursor-pointer select-none hover:bg-muted/30" onClick={() => toggleSort("avg_tackles")}>
                      TK {renderSortIcon(sort, "avg_tackles")}
                    </TableHead>
                    <TableHead className="text-right cursor-pointer select-none hover:bg-muted/30" onClick={() => toggleSort("avg_kicks")}>
                      KI {renderSortIcon(sort, "avg_kicks")}
                    </TableHead>
                    <TableHead className="text-right cursor-pointer select-none hover:bg-muted/30" onClick={() => toggleSort("avg_handballs")}>
                      HB {renderSortIcon(sort, "avg_handballs")}
                    </TableHead>
                    <TableHead className="text-right cursor-pointer select-none hover:bg-muted/30" onClick={() => toggleSort("avg_hitouts")}>
                      HO {renderSortIcon(sort, "avg_hitouts")}
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filtered.map((p) => {
                    const color = TEAM_COLORS[p.team]?.primary || "#666";
                    const abbr = TEAM_ABBREVS[p.team] || p.team;
                    return (
                      <TableRow key={p.player_id} className="group">
                        <TableCell>
                          <Link
                            href={`/players/${encodeURIComponent(p.player_id)}`}
                            className="font-medium text-sm hover:text-primary transition-colors"
                          >
                            {p.name}
                          </Link>
                        </TableCell>
                        <TableCell>
                          <span className="flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: color }} />
                            <span className="text-xs font-mono">{abbr}</span>
                          </span>
                        </TableCell>
                        <TableCell className="text-right tabular-nums text-sm">{p.games}</TableCell>
                        <TableCell className="text-right tabular-nums text-sm">{p.avg_goals.toFixed(1)}</TableCell>
                        <TableCell className="text-right tabular-nums text-sm font-medium">{p.avg_disposals.toFixed(1)}</TableCell>
                        <TableCell className="text-right tabular-nums text-sm">{p.avg_marks.toFixed(1)}</TableCell>
                        <TableCell className="text-right tabular-nums text-sm">{p.avg_tackles.toFixed(1)}</TableCell>
                        <TableCell className="text-right tabular-nums text-sm">{p.avg_kicks.toFixed(1)}</TableCell>
                        <TableCell className="text-right tabular-nums text-sm">{p.avg_handballs.toFixed(1)}</TableCell>
                        <TableCell className="text-right tabular-nums text-sm">{p.avg_hitouts.toFixed(1)}</TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
