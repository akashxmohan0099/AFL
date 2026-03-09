"use client";

import { useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { MatchComparisonPlayer, PlayerAdvancedStats } from "@/lib/types";
import { TEAM_ABBREVS } from "@/lib/constants";
import { cn } from "@/lib/utils";

function playerHref(player: string, team: string) {
  return `/players/${encodeURIComponent(`${player}_${team}`)}`;
}

export function predColor(actual: number, predicted: number): string {
  return actual >= predicted ? "text-emerald-400" : "text-red-400";
}

const STAT_KEYS = ["gl", "bh", "di", "mk", "ki", "hb", "tk", "ho", "cp", "up", "if", "cl", "cg", "ff", "fa"] as const;
const STAT_LABELS: Record<string, string> = {
  gl: "GL", bh: "BH", di: "DI", mk: "MK", ki: "KI", hb: "HB", tk: "TK",
  ho: "HO", cp: "CP", up: "UP", if: "IF", cl: "CL", cg: "CG", ff: "FF", fa: "FA",
};

function AdvStatBlock({ label, stats }: { label: string; stats?: PlayerAdvancedStats }) {
  if (!stats) return null;
  const primary = ["gl", "di", "mk", "tk"] as const;
  const secondary = ["ki", "hb", "ho", "cp", "ff"] as const;
  const hasRange = stats.max_gl != null;
  return (
    <div className="px-3 py-2.5 rounded-lg bg-muted/30 min-w-[150px]">
      <p className="text-xs font-mono text-muted-foreground uppercase mb-2 font-semibold tracking-wide">{label} <span className="normal-case font-normal text-muted-foreground/60">({stats.games} games)</span></p>
      {/* Primary stats */}
      <div className="flex gap-3 text-xs font-mono tabular-nums">
        {primary.map((k) => {
          const avg = stats[`avg_${k}`];
          if (avg == null) return null;
          return (
            <span key={k}>
              <span className="text-muted-foreground/60">{STAT_LABELS[k]}</span>{" "}
              <span className="font-bold text-foreground">{k === "gl" ? avg.toFixed(1) : avg.toFixed(0)}</span>
            </span>
          );
        })}
      </div>
      {/* Secondary stats */}
      <div className="flex gap-3 text-[11px] font-mono tabular-nums text-muted-foreground mt-1">
        {secondary.map((k) => {
          const avg = stats[`avg_${k}`];
          if (avg == null) return null;
          return (
            <span key={k}>
              <span className="text-muted-foreground/50">{STAT_LABELS[k]}</span>{" "}
              <span>{avg.toFixed(0)}</span>
            </span>
          );
        })}
      </div>
      {/* Range: Low / Med / High for core stats only */}
      {hasRange && (
        <div className="mt-2 pt-2 border-t border-border/10">
          <table className="text-[11px] font-mono tabular-nums w-full">
            <thead>
              <tr className="text-muted-foreground/50">
                <td className="pr-2"></td>
                {primary.map((k) => stats[`max_${k}`] != null ? <td key={k} className="px-2 text-center font-medium">{STAT_LABELS[k]}</td> : null)}
              </tr>
            </thead>
            <tbody>
              <tr className="text-muted-foreground/60">
                <td className="pr-2 text-muted-foreground/50">Low</td>
                {primary.map((k) => stats[`min_${k}`] != null ? <td key={k} className="px-2 text-center">{stats[`min_${k}`]}</td> : null)}
              </tr>
              <tr className="text-foreground/80 font-semibold">
                <td className="pr-2 text-muted-foreground/50 font-normal">Med</td>
                {primary.map((k) => stats[`med_${k}`] != null ? <td key={k} className="px-2 text-center">{stats[`med_${k}`]}</td> : null)}
              </tr>
              <tr className="text-muted-foreground/60">
                <td className="pr-2 text-muted-foreground/50">High</td>
                {primary.map((k) => stats[`max_${k}`] != null ? <td key={k} className="px-2 text-center">{stats[`max_${k}`]}</td> : null)}
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function StreakBar({ label, values, color }: { label: string; values: number[]; color: string }) {
  if (!values || values.length === 0) return null;
  const avg = values.reduce((a, b) => a + b, 0) / values.length;
  return (
    <div className="flex items-center gap-2.5">
      <span className="text-xs font-mono text-muted-foreground w-11 text-right shrink-0 font-medium">{label}</span>
      <div className="flex gap-1 font-mono tabular-nums text-xs">
        {values.map((v, i) => (
          <span
            key={i}
            className="w-6 h-6 flex items-center justify-center rounded font-semibold"
            style={{
              backgroundColor: `${color}${v === 0 ? "10" : "20"}`,
              color: v === 0 ? "var(--muted-foreground)" : color,
              opacity: v === 0 ? 0.4 : 0.5 + (i / values.length) * 0.5,
            }}
          >
            {v}
          </span>
        ))}
      </div>
      <span className="text-xs font-mono text-muted-foreground tabular-nums">
        avg <span className="font-bold text-foreground/70">{avg.toFixed(1)}</span>
      </span>
    </div>
  );
}

function PlayerAdvancedRow({ p, teamColor, teamColors, isPlayed }: {
  p: MatchComparisonPlayer;
  teamColor: string;
  teamColors?: Record<string, string>;
  isPlayed: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const adv = p.advanced;
  const color = teamColors?.[p.team] || teamColor;

  return (
    <>
      <TableRow
        className="cursor-pointer hover:bg-muted/30"
        onClick={() => setExpanded(!expanded)}
      >
        <TableCell className="font-medium text-sm">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: color }} />
            <Link href={playerHref(p.player, p.team)} className="hover:text-primary transition-colors" onClick={(e) => e.stopPropagation()}>
              {p.player}
              {p.player_role && (
                <span className="block text-[10px] font-mono text-muted-foreground/60 font-normal">{p.player_role}</span>
              )}
            </Link>
          </span>
        </TableCell>
        <TableCell className="text-right text-sm">
          {isPlayed ? (
            <span className={cn("font-bold tabular-nums", p.actual_gl != null && p.predicted_gl != null ? (p.actual_gl >= p.predicted_gl ? "text-emerald-400" : "text-red-400") : "")}>
              {p.actual_gl ?? "-"}<span className="text-muted-foreground/40 text-[10px]"> / </span><span className="text-muted-foreground/60 text-[11px] font-mono">{p.predicted_gl?.toFixed(2)}</span>
            </span>
          ) : (
            <span className="font-semibold tabular-nums">{p.predicted_gl?.toFixed(2) ?? "-"}</span>
          )}
        </TableCell>
        <TableCell className="text-right text-sm">
          {isPlayed ? (
            <span className={cn("font-bold tabular-nums", p.actual_di != null && p.predicted_di != null ? (p.actual_di >= p.predicted_di ? "text-emerald-400" : "text-red-400") : "")}>
              {p.actual_di ?? "-"}<span className="text-muted-foreground/40 text-[10px]"> / </span><span className="text-muted-foreground/60 text-[11px] font-mono">{p.predicted_di?.toFixed(1)}</span>
            </span>
          ) : (
            <span className="font-semibold tabular-nums">{p.predicted_di?.toFixed(1) ?? "-"}</span>
          )}
        </TableCell>
        <TableCell className="text-right text-sm">
          {isPlayed ? (
            <span className={cn("font-bold tabular-nums", p.actual_mk != null && p.predicted_mk != null ? (p.actual_mk >= p.predicted_mk ? "text-emerald-400" : "text-red-400") : "")}>
              {p.actual_mk ?? "-"}<span className="text-muted-foreground/40 text-[10px]"> / </span><span className="text-muted-foreground/60 text-[11px] font-mono">{p.predicted_mk?.toFixed(1)}</span>
            </span>
          ) : (
            <span className="font-semibold tabular-nums">{p.predicted_mk?.toFixed(1) ?? "-"}</span>
          )}
        </TableCell>
        <TableCell className="text-right text-xs text-muted-foreground/70 tabular-nums font-mono">
          {adv?.career?.avg_tk != null ? adv.career.avg_tk.toFixed(0) : "-"}
        </TableCell>
        <TableCell className="text-right text-[10px] text-muted-foreground/50">
          {expanded ? "\u25B2" : "\u25BC"}
        </TableCell>
      </TableRow>
      {expanded && adv && (
        <TableRow>
          <TableCell colSpan={6} className="p-0">
            <div className="px-5 py-4 bg-muted/10 border-b border-border/30 space-y-4">
              {/* Stats blocks row */}
              <div className="flex gap-2.5 flex-wrap">
                <AdvStatBlock label="Last 5" stats={adv.form_5} />
                {adv.season && <AdvStatBlock label="Season" stats={adv.season} />}
                <AdvStatBlock label="Career" stats={adv.career} />
                {adv.venue && <AdvStatBlock label="At Venue" stats={adv.venue} />}
                {adv.opponent && <AdvStatBlock label="vs Opp" stats={adv.opponent} />}
              </div>

              {/* Probabilities + confidence intervals in one row */}
              <div>
                <p className="text-xs font-mono text-muted-foreground uppercase mb-2 font-semibold tracking-wide">Probabilities</p>
                <div className="flex gap-2 flex-wrap items-center">
                  {p.p_scorer != null && (
                    <span className={cn("text-xs font-mono px-2.5 py-1 rounded-md", p.p_scorer >= 0.7 ? "bg-emerald-500/15 text-emerald-400 font-semibold" : "bg-muted/50 text-muted-foreground")}>
                      1+ Goals {(p.p_scorer * 100).toFixed(0)}%
                    </span>
                  )}
                  {p.p_2plus_goals != null && p.p_2plus_goals >= 0.1 && (
                    <span className={cn("text-xs font-mono px-2.5 py-1 rounded-md", p.p_2plus_goals >= 0.5 ? "bg-emerald-500/15 text-emerald-400 font-semibold" : "bg-muted/50 text-muted-foreground")}>
                      2+ Goals {(p.p_2plus_goals * 100).toFixed(0)}%
                    </span>
                  )}
                  {p.p_3plus_goals != null && p.p_3plus_goals >= 0.05 && (
                    <span className={cn("text-xs font-mono px-2.5 py-1 rounded-md", p.p_3plus_goals >= 0.4 ? "bg-emerald-500/15 text-emerald-400 font-semibold" : "bg-muted/50 text-muted-foreground")}>
                      3+ Goals {(p.p_3plus_goals * 100).toFixed(0)}%
                    </span>
                  )}
                  {p.p_15plus_disp != null && p.p_15plus_disp >= 0.3 && (
                    <span className={cn("text-xs font-mono px-2.5 py-1 rounded-md", p.p_15plus_disp >= 0.8 ? "bg-emerald-500/15 text-emerald-400 font-semibold" : "bg-muted/50 text-muted-foreground")}>
                      15+ Disp {(p.p_15plus_disp * 100).toFixed(0)}%
                    </span>
                  )}
                  {p.p_20plus_disp != null && p.p_20plus_disp >= 0.2 && (
                    <span className={cn("text-xs font-mono px-2.5 py-1 rounded-md", p.p_20plus_disp >= 0.7 ? "bg-emerald-500/15 text-emerald-400 font-semibold" : "bg-muted/50 text-muted-foreground")}>
                      20+ Disp {(p.p_20plus_disp * 100).toFixed(0)}%
                    </span>
                  )}
                  {p.p_25plus_disp != null && p.p_25plus_disp >= 0.1 && (
                    <span className={cn("text-xs font-mono px-2.5 py-1 rounded-md", p.p_25plus_disp >= 0.5 ? "bg-emerald-500/15 text-emerald-400 font-semibold" : "bg-muted/50 text-muted-foreground")}>
                      25+ Disp {(p.p_25plus_disp * 100).toFixed(0)}%
                    </span>
                  )}
                  {p.p_30plus_disp != null && p.p_30plus_disp >= 0.05 && (
                    <span className={cn("text-xs font-mono px-2.5 py-1 rounded-md", p.p_30plus_disp >= 0.4 ? "bg-emerald-500/15 text-emerald-400 font-semibold" : "bg-muted/50 text-muted-foreground")}>
                      30+ Disp {(p.p_30plus_disp * 100).toFixed(0)}%
                    </span>
                  )}
                  {p.p_3plus_mk != null && p.p_3plus_mk >= 0.2 && (
                    <span className={cn("text-xs font-mono px-2.5 py-1 rounded-md", p.p_3plus_mk >= 0.7 ? "bg-emerald-500/15 text-emerald-400 font-semibold" : "bg-muted/50 text-muted-foreground")}>
                      3+ Marks {(p.p_3plus_mk * 100).toFixed(0)}%
                    </span>
                  )}
                  {p.p_5plus_mk != null && p.p_5plus_mk >= 0.1 && (
                    <span className={cn("text-xs font-mono px-2.5 py-1 rounded-md", p.p_5plus_mk >= 0.5 ? "bg-emerald-500/15 text-emerald-400 font-semibold" : "bg-muted/50 text-muted-foreground")}>
                      5+ Marks {(p.p_5plus_mk * 100).toFixed(0)}%
                    </span>
                  )}
                  {/* Confidence intervals inline */}
                  {p.conf_gl && (
                    <span className="text-xs font-mono px-2.5 py-1 rounded-md bg-muted/30 text-muted-foreground">
                      Goals {p.conf_gl[0]}–{p.conf_gl[1]}
                    </span>
                  )}
                  {p.conf_di && (
                    <span className="text-xs font-mono px-2.5 py-1 rounded-md bg-muted/30 text-muted-foreground">
                      Disp {p.conf_di[0]}–{p.conf_di[1]}
                    </span>
                  )}
                  {p.conf_mk && (
                    <span className="text-xs font-mono px-2.5 py-1 rounded-md bg-muted/30 text-muted-foreground">
                      Marks {p.conf_mk[0]}–{p.conf_mk[1]}
                    </span>
                  )}
                </div>
              </div>

              {/* Streaks - last 10 games */}
              {(adv.streak_gl || adv.streak_di) && (
                <div className="space-y-2">
                  <p className="text-xs font-mono text-muted-foreground uppercase font-semibold tracking-wide">Last {adv.streak_gl?.length ?? adv.streak_di?.length} Games</p>
                  {adv.streak_gl && <StreakBar label="Goals" values={adv.streak_gl} color="#10b981" />}
                  {adv.streak_di && <StreakBar label="Disp" values={adv.streak_di} color="#6366f1" />}
                  {adv.streak_mk && <StreakBar label="Marks" values={adv.streak_mk} color="#f59e0b" />}
                  {adv.streak_tk && <StreakBar label="Tckl" values={adv.streak_tk} color="#ef4444" />}
                </div>
              )}

              {/* Recent games list */}
              {adv.recent_games && adv.recent_games.length > 0 && (
                <div>
                  <p className="text-xs font-mono text-muted-foreground uppercase mb-2 font-semibold tracking-wide">Recent Games ({adv.recent_games.length})</p>
                  <div className="flex gap-2 flex-wrap">
                    {adv.recent_games.map((g, i) => (
                      <div key={i} className="text-xs font-mono px-2.5 py-1.5 rounded-md bg-card/50 border border-border/30 tabular-nums">
                        <span className="text-muted-foreground">{g.year !== new Date().getFullYear() ? `${g.year} ` : ""}R{g.round}</span>{" "}
                        <span className="text-muted-foreground">{TEAM_ABBREVS[g.opponent] || g.opponent}</span>{" "}
                        <span className="font-bold">{g.gl ?? 0}g</span>{" "}
                        <span className="font-medium">{g.di ?? 0}d</span>{" "}
                        <span className="font-medium">{g.mk ?? 0}m</span>{" "}
                        <span className="text-muted-foreground">{g.tk ?? 0}t</span>
                        {g.ho != null && g.ho > 0 && <span className="text-muted-foreground"> {g.ho}ho</span>}
                        {g.cp != null && g.cp > 0 && <span className="text-muted-foreground"> {g.cp}cp</span>}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Opponent game-by-game history */}
              {adv.opponent_games && adv.opponent_games.length > 0 && (
                <div>
                  <p className="text-xs font-mono text-muted-foreground uppercase mb-2 font-semibold tracking-wide">
                    vs {TEAM_ABBREVS[adv.opponent_games[0].opponent] || adv.opponent_games[0].opponent} History ({adv.opponent_games.length} games)
                  </p>
                  <div className="flex gap-2 flex-wrap">
                    {adv.opponent_games.map((g, i) => (
                      <div key={i} className="text-xs font-mono px-2.5 py-1.5 rounded-md bg-amber-500/5 border border-amber-500/20 tabular-nums">
                        <span className="text-muted-foreground">{g.year} R{g.round}</span>{" "}
                        <span className="text-muted-foreground/70">{g.venue ? (TEAM_ABBREVS[g.venue] || g.venue).slice(0, 8) : ""}</span>{" "}
                        <span className="font-bold">{g.gl ?? 0}g</span>{" "}
                        <span className="font-medium">{g.di ?? 0}d</span>{" "}
                        <span className="font-medium">{g.mk ?? 0}m</span>{" "}
                        <span className="text-muted-foreground">{g.tk ?? 0}t</span>
                        {g.ho != null && g.ho > 0 && <span className="text-muted-foreground"> {g.ho}ho</span>}
                        {g.cp != null && g.cp > 0 && <span className="text-muted-foreground"> {g.cp}cp</span>}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </TableCell>
        </TableRow>
      )}
    </>
  );
}

type SortKey = "predicted_gl" | "predicted_di" | "predicted_mk" | "p_scorer" | "player";

export function PlayerAdvancedTable({
  players,
  teamName,
  teamColor,
  isPlayed,
  teamColors,
}: {
  players: MatchComparisonPlayer[];
  teamName: string;
  teamColor: string;
  isPlayed: boolean;
  teamColors?: Record<string, string>;
}) {
  const [sortKey, setSortKey] = useState<SortKey>("predicted_gl");
  const [sortAsc, setSortAsc] = useState(false);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(key === "player"); }
  };

  const sorted = [...players].sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    const va = typeof av === "number" ? av : typeof av === "string" ? av : 0;
    const vb = typeof bv === "number" ? bv : typeof bv === "string" ? bv : 0;
    if (va < vb) return sortAsc ? -1 : 1;
    if (va > vb) return sortAsc ? 1 : -1;
    return 0;
  });

  const sortIcon = (key: SortKey) => sortKey === key ? (sortAsc ? " \u25B2" : " \u25BC") : "";
  const thCls = "text-right cursor-pointer hover:bg-muted/50 select-none text-[11px]";

  return (
    <Card>
      <CardHeader className="pb-1">
        <CardTitle className="flex items-center gap-2 text-base">
          {teamColors ? (
            Object.entries(teamColors).map(([team, clr]) => (
              <span key={team} className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-full" style={{ backgroundColor: clr }} />
                <span>{team}</span>
              </span>
            ))
          ) : (
            <>
              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: teamColor }} />
              {teamName}
            </>
          )}
          <Badge variant="outline" className="ml-auto text-[9px] px-1.5 py-0">
            Click row to expand
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="cursor-pointer hover:bg-muted/50 select-none text-[11px]" onClick={() => handleSort("player")}>
                  Player{sortIcon("player")}
                </TableHead>
                <TableHead className={thCls} onClick={() => handleSort("predicted_gl")}>
                  Goals{sortIcon("predicted_gl")}
                </TableHead>
                <TableHead className={thCls} onClick={() => handleSort("predicted_di")}>
                  Disp{sortIcon("predicted_di")}
                </TableHead>
                <TableHead className={thCls} onClick={() => handleSort("predicted_mk")}>
                  Marks{sortIcon("predicted_mk")}
                </TableHead>
                <TableHead className="text-right text-[11px]">TK</TableHead>
                <TableHead className="text-right text-[11px] w-8" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {sorted.map((p, i) => (
                <PlayerAdvancedRow
                  key={i}
                  p={p}
                  teamColor={teamColor}
                  teamColors={teamColors}
                  isPlayed={isPlayed}
                />
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}

export function PlayerComparisonTable({
  players,
  teamName,
  teamColor,
  isPlayed,
  showTeamIndicator,
  teamColors,
}: {
  players: MatchComparisonPlayer[];
  teamName: string;
  teamColor: string;
  isPlayed: boolean;
  showTeamIndicator?: boolean;
  teamColors?: Record<string, string>;
}) {
  const [sortKey, setSortKey] = useState<SortKey>("predicted_gl");
  const [sortAsc, setSortAsc] = useState(false);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(key === "player"); }
  };

  const sorted = [...players].sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    const va = typeof av === "number" ? av : typeof av === "string" ? av : 0;
    const vb = typeof bv === "number" ? bv : typeof bv === "string" ? bv : 0;
    if (va < vb) return sortAsc ? -1 : 1;
    if (va > vb) return sortAsc ? 1 : -1;
    return 0;
  });

  const sortIcon = (key: SortKey) => sortKey === key ? (sortAsc ? " \u25B2" : " \u25BC") : "";

  const thCls = "text-right cursor-pointer hover:bg-muted/50 select-none text-[11px]";

  return (
    <Card>
      <CardHeader className="pb-1">
        <CardTitle className="flex items-center gap-2 text-base">
          {showTeamIndicator && teamColors ? (
            <>
              {Object.entries(teamColors).map(([team, color]) => (
                <span key={team} className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
                  <span>{team}</span>
                </span>
              ))}
            </>
          ) : (
            <>
              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: teamColor }} />
              {teamName}
            </>
          )}
        </CardTitle>
        {isPlayed && (
          <p className="text-[10px] text-muted-foreground/50 font-mono mt-1">
            Format: <span className="font-bold text-foreground/70">Actual</span> <span className="text-muted-foreground/40">/</span> <span className="text-muted-foreground/60">Pred</span> <span className="text-muted-foreground/30">[CI]</span>
          </p>
        )}
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="cursor-pointer hover:bg-muted/50 select-none text-[11px]" onClick={() => handleSort("player")}>
                  Player{sortIcon("player")}
                </TableHead>
                <TableHead className={thCls} onClick={() => handleSort("predicted_gl")}>
                  Goals{sortIcon("predicted_gl")}
                </TableHead>
                <TableHead className={thCls} onClick={() => handleSort("p_scorer")}>
                  P(1+){sortIcon("p_scorer")}
                </TableHead>
                <TableHead className={thCls} onClick={() => handleSort("predicted_di")}>
                  Disp{sortIcon("predicted_di")}
                </TableHead>
                <TableHead className={thCls} onClick={() => handleSort("predicted_mk")}>
                  Marks{sortIcon("predicted_mk")}
                </TableHead>
                <TableHead className="text-right text-[11px]">
                  Key Probs
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sorted.map((p, i) => {
                const confGl = p.conf_gl;
                const confDi = p.conf_di;
                const confMk = p.conf_mk;
                const pScorer = p.p_scorer;
                const p2gl = p.p_2plus_goals;
                const p3gl = p.p_3plus_goals;
                const p20di = p.p_20plus_disp;
                const p25di = p.p_25plus_disp;
                const p5mk = p.p_5plus_mk;

                const probs: string[] = [];
                if (p2gl != null && p2gl >= 0.1) probs.push(`2+GL ${(p2gl * 100).toFixed(0)}%`);
                if (p3gl != null && p3gl >= 0.05) probs.push(`3+GL ${(p3gl * 100).toFixed(0)}%`);
                if (p20di != null && p20di >= 0.2) probs.push(`20+DI ${(p20di * 100).toFixed(0)}%`);
                if (p25di != null && p25di >= 0.1) probs.push(`25+DI ${(p25di * 100).toFixed(0)}%`);
                if (p5mk != null && p5mk >= 0.1) probs.push(`5+MK ${(p5mk * 100).toFixed(0)}%`);

                return (
                  <TableRow key={i}>
                    <TableCell className="font-medium text-sm">
                      <span className="flex items-center gap-1.5">
                        {showTeamIndicator && teamColors && (
                          <span
                            className="w-2 h-2 rounded-full shrink-0"
                            style={{ backgroundColor: teamColors[p.team] || "#666" }}
                          />
                        )}
                        <Link href={playerHref(p.player, p.team)} className="hover:text-primary transition-colors">
                          {p.player}
                          {p.player_role && (
                            <span className="block text-[10px] font-mono text-muted-foreground/60 font-normal">{p.player_role}</span>
                          )}
                        </Link>
                      </span>
                    </TableCell>
                    <TableCell className="text-right text-sm">
                      {isPlayed ? (
                        <div className="flex items-baseline gap-1 justify-end">
                          <span className={cn("font-bold tabular-nums", p.actual_gl != null && p.predicted_gl != null ? predColor(p.actual_gl, p.predicted_gl) : "")}>{p.actual_gl ?? ""}</span>
                          {p.predicted_gl != null && <span className="text-[10px] text-muted-foreground/40">/</span>}
                          <span className="text-[11px] tabular-nums text-muted-foreground/60 font-mono">{p.predicted_gl?.toFixed(2) ?? "-"}</span>
                          {confGl && <span className="text-[9px] text-muted-foreground/30 font-mono">[{confGl[0]}-{confGl[1]}]</span>}
                        </div>
                      ) : (
                        <div className="flex items-baseline gap-1 justify-end">
                          <span className="font-semibold tabular-nums text-foreground">{p.predicted_gl?.toFixed(2) ?? "-"}</span>
                          {confGl && <span className="text-[9px] text-muted-foreground/40 font-mono">[{confGl[0]}-{confGl[1]}]</span>}
                        </div>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {pScorer != null ? (
                        <span className={cn("text-[11px] font-mono tabular-nums font-semibold", pScorer >= 0.5 ? "text-emerald-400" : "text-muted-foreground")}>
                          {(pScorer * 100).toFixed(0)}%
                        </span>
                      ) : "-"}
                    </TableCell>
                    <TableCell className="text-right text-sm">
                      {isPlayed ? (
                        <div className="flex items-baseline gap-1 justify-end">
                          <span className={cn("font-bold tabular-nums", p.actual_di != null && p.predicted_di != null ? predColor(p.actual_di, p.predicted_di) : "")}>{p.actual_di ?? ""}</span>
                          {p.predicted_di != null && <span className="text-[10px] text-muted-foreground/40">/</span>}
                          <span className="text-[11px] tabular-nums text-muted-foreground/60 font-mono">{p.predicted_di?.toFixed(1) ?? "-"}</span>
                          {confDi && <span className="text-[9px] text-muted-foreground/30 font-mono">[{confDi[0]}-{confDi[1]}]</span>}
                        </div>
                      ) : (
                        <div className="flex items-baseline gap-1 justify-end">
                          <span className="font-semibold tabular-nums text-foreground">{p.predicted_di?.toFixed(1) ?? "-"}</span>
                          {confDi && <span className="text-[9px] text-muted-foreground/40 font-mono">[{confDi[0]}-{confDi[1]}]</span>}
                        </div>
                      )}
                    </TableCell>
                    <TableCell className="text-right text-sm">
                      {isPlayed ? (
                        <div className="flex items-baseline gap-1 justify-end">
                          <span className={cn("font-bold tabular-nums", p.actual_mk != null && p.predicted_mk != null ? predColor(p.actual_mk, p.predicted_mk) : "")}>{p.actual_mk ?? ""}</span>
                          {p.predicted_mk != null && <span className="text-[10px] text-muted-foreground/40">/</span>}
                          <span className="text-[11px] tabular-nums text-muted-foreground/60 font-mono">{p.predicted_mk?.toFixed(1) ?? "-"}</span>
                          {confMk && <span className="text-[9px] text-muted-foreground/30 font-mono">[{confMk[0]}-{confMk[1]}]</span>}
                        </div>
                      ) : (
                        <div className="flex items-baseline gap-1 justify-end">
                          <span className="font-semibold tabular-nums text-foreground">{p.predicted_mk?.toFixed(1) ?? "-"}</span>
                          {confMk && <span className="text-[9px] text-muted-foreground/40 font-mono">[{confMk[0]}-{confMk[1]}]</span>}
                        </div>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {probs.length > 0 ? (
                        <div className="flex flex-wrap gap-1 justify-end">
                          {probs.map((pr, j) => (
                            <span key={j} className="text-[9px] font-mono px-1 py-0.5 rounded bg-muted/50 text-muted-foreground whitespace-nowrap">
                              {pr}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <span className="text-[10px] text-muted-foreground/30">-</span>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}
