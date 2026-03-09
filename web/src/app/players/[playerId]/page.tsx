"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
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
import { Skeleton } from "@/components/ui/skeleton";
import { getPlayer, getPlayerGames } from "@/lib/api";
import { TEAM_ABBREVS, TEAM_COLORS, CHART_COLORS } from "@/lib/constants";
import { cn, formatDate } from "@/lib/utils";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { Breadcrumb } from "@/components/ui/breadcrumb";

/* eslint-disable @typescript-eslint/no-explicit-any */

const tooltipStyle = {
  backgroundColor: "rgba(15, 15, 25, 0.95)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 6,
  fontSize: 11,
  fontFamily: "monospace",
};

function StatBox({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="text-center px-2 py-2">
      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{label}</p>
      <p className="text-lg font-bold tabular-nums font-mono" style={color ? { color } : undefined}>{value}</p>
      {sub && <p className="text-[10px] text-muted-foreground">{sub}</p>}
    </div>
  );
}

function MiniBar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  return (
    <div className="w-16 h-1.5 bg-muted rounded-full overflow-hidden inline-block ml-1.5 align-middle">
      <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
    </div>
  );
}

export default function PlayerProfilePage() {
  const params = useParams();
  const playerId = decodeURIComponent(params.playerId as string);
  const [profile, setProfile] = useState<any>(null);
  const [games, setGames] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      getPlayer(playerId),
      getPlayerGames(playerId, undefined, 50),
    ])
      .then(([p, g]) => { setProfile(p); setGames(g); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [playerId]);

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <div className="grid grid-cols-4 gap-3">{[1,2,3,4].map(i => <Skeleton key={i} className="h-20" />)}</div>
        <Skeleton className="h-64" />
      </div>
    );
  }

  if (!profile) {
    return <Card><CardContent className="pt-6"><p className="text-muted-foreground">Player not found.</p></CardContent></Card>;
  }

  const teamColor = TEAM_COLORS[profile.team]?.primary || "#666";
  const career = profile.career || {};
  const highs = profile.career_highs || {};
  const consistency = profile.consistency || {};
  const streaks = profile.streaks || {};
  const homeAway = profile.home_away || {};
  const oppSplits = profile.opponent_splits || [];
  const venueSplits = profile.venue_splits || [];
  const quarterScoring = profile.quarter_scoring;
  const pva = profile.predictions_vs_actuals || {};
  const pvaRecords = pva.records || [];
  const pvaMae = pva.mae || {};

  // Season chart data
  const seasonChart = (profile.seasons || []).map((s: any) => ({
    year: s.year,
    Goals: s.GL,
    Disposals: s.DI,
    Marks: s.MK,
    Tackles: s.TK,
  }));

  // Opponent chart — top 8 by avg disposals
  const oppChart = [...oppSplits]
    .filter((o: any) => o.games >= 3)
    .sort((a: any, b: any) => b.avg_di - a.avg_di)
    .slice(0, 10)
    .map((o: any) => ({
      opponent: TEAM_ABBREVS[o.opponent] || o.opponent.substring(0, 10),
      Goals: o.avg_gl,
      Disposals: o.avg_di,
      Marks: o.avg_mk,
      games: o.games,
    }));

  // Consistency radar data
  const radarData = ["gl", "di", "mk", "tk"].map(stat => {
    const c = consistency[stat];
    if (!c) return null;
    return {
      stat: stat.toUpperCase(),
      avg: c.avg,
      ceiling: c.ceiling,
      floor: c.floor,
    };
  }).filter(Boolean);

  // Max values for mini bars in opponent splits
  const maxOppDI = oppSplits.length > 0 ? Math.max(...oppSplits.map((o: any) => o.avg_di || 0)) : 1;
  const maxOppGL = oppSplits.length > 0 ? Math.max(...oppSplits.map((o: any) => o.avg_gl || 0)) : 1;

  return (
    <div className="space-y-5">
      {/* Breadcrumb + Header */}
      <div>
        <Breadcrumb items={[
          { label: "Players", href: "/players" },
          { label: profile.name },
        ]} />
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-2xl font-bold">{profile.name}</h1>
          <Badge style={{ backgroundColor: teamColor, color: "#fff", borderColor: teamColor }}>{profile.team}</Badge>
          <Badge variant="outline">{profile.total_games} games</Badge>
        </div>
      </div>

      {/* ── Key Stats Row ─────────────────────────────────────────── */}
      <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
        <Card className="border-border/40"><CardContent className="p-0">
          <StatBox label="Goals/gm" value={career.avg_gl?.toFixed(2) ?? "--"} sub={`${profile.career_goals} career`} color={CHART_COLORS.goals} />
        </CardContent></Card>
        <Card className="border-border/40"><CardContent className="p-0">
          <StatBox label="Disposals/gm" value={career.avg_di?.toFixed(1) ?? "--"} sub={`SD ${career.std_di?.toFixed(1) ?? "--"}`} color={CHART_COLORS.disposals} />
        </CardContent></Card>
        <Card className="border-border/40"><CardContent className="p-0">
          <StatBox label="Marks/gm" value={career.avg_mk?.toFixed(1) ?? "--"} sub={`SD ${career.std_mk?.toFixed(1) ?? "--"}`} color={CHART_COLORS.marks} />
        </CardContent></Card>
        <Card className="border-border/40"><CardContent className="p-0">
          <StatBox label="Tackles/gm" value={career.avg_tk?.toFixed(1) ?? "--"} sub={`SD ${career.std_tk?.toFixed(1) ?? "--"}`} />
        </CardContent></Card>
        <Card className="border-border/40"><CardContent className="p-0">
          <StatBox label="Kicks/gm" value={career.avg_ki?.toFixed(1) ?? "--"} />
        </CardContent></Card>
        <Card className="border-border/40"><CardContent className="p-0">
          <StatBox label="Handballs/gm" value={career.avg_hb?.toFixed(1) ?? "--"} />
        </CardContent></Card>
      </div>

      {/* ── Career Highs ─────────────────────────────────────────── */}
      {Object.keys(highs).length > 0 && (
        <Card className="border-border/40">
          <CardHeader className="pb-2 pt-4 px-4">
            <CardTitle className="text-sm font-medium">Career Highs</CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-3">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {(["gl", "di", "mk", "tk", "ki", "hb", "ho"] as const).map(stat => {
                const h = highs[stat];
                if (!h) return null;
                const labels: Record<string, string> = { gl: "Goals", di: "Disposals", mk: "Marks", tk: "Tackles", ki: "Kicks", hb: "Handballs", ho: "Hitouts", bh: "Behinds" };
                return (
                  <div key={stat} className="border border-border/30 rounded-lg px-3 py-2">
                    <p className="text-[10px] text-muted-foreground uppercase">{labels[stat] || stat}</p>
                    <p className="text-xl font-bold tabular-nums">{h.value}</p>
                    <p className="text-[10px] text-muted-foreground">
                      vs {TEAM_ABBREVS[h.opponent] || h.opponent} R{h.round} {h.year}
                    </p>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Recent Form ──────────────────────────────────────────── */}
      {profile.recent_form?.length > 0 && (
        <Card className="border-border/40">
          <CardHeader className="pb-2 pt-4 px-4">
            <CardTitle className="text-sm font-medium">Recent Form (Last {profile.recent_form.length})</CardTitle>
          </CardHeader>
          <CardContent className="px-0 pb-2">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="pl-4">Date</TableHead>
                    <TableHead>Opp</TableHead>
                    <TableHead>Venue</TableHead>
                    <TableHead className="text-right">GL</TableHead>
                    <TableHead className="text-right">DI</TableHead>
                    <TableHead className="text-right">MK</TableHead>
                    <TableHead className="text-right">TK</TableHead>
                    <TableHead className="text-right">KI</TableHead>
                    <TableHead className="text-right pr-4">HB</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {profile.recent_form.map((f: any, i: number) => (
                    <TableRow key={i}>
                      <TableCell className="text-xs pl-4">{f.date}</TableCell>
                      <TableCell>
                        <span className="flex items-center gap-1.5">
                          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: TEAM_COLORS[f.opponent]?.primary || "#666" }} />
                          <span className="text-xs font-mono">{TEAM_ABBREVS[f.opponent] || f.opponent}</span>
                          {f.is_home && <span className="text-[9px] text-muted-foreground">(H)</span>}
                        </span>
                      </TableCell>
                      <TableCell className="text-xs truncate max-w-[120px]">{f.venue}</TableCell>
                      <TableCell className="text-right tabular-nums font-medium">{f.GL}</TableCell>
                      <TableCell className="text-right tabular-nums font-medium">{f.DI}</TableCell>
                      <TableCell className="text-right tabular-nums">{f.MK}</TableCell>
                      <TableCell className="text-right tabular-nums">{f.TK}</TableCell>
                      <TableCell className="text-right tabular-nums">{f.KI}</TableCell>
                      <TableCell className="text-right tabular-nums pr-4">{f.HB}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Streaks + Consistency Row ────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Streaks */}
        {Object.keys(streaks).length > 0 && (
          <Card className="border-border/40">
            <CardHeader className="pb-2 pt-4 px-4">
              <CardTitle className="text-sm font-medium">Streaks</CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-3">
              <div className="space-y-2">
                {Object.entries(streaks).map(([key, s]: [string, any]) => {
                  const labels: Record<string, string> = {
                    goals_1plus: "1+ Goals", goals_2plus: "2+ Goals", goals_3plus: "3+ Goals",
                    disp_20plus: "20+ Disposals", disp_25plus: "25+ Disposals", disp_30plus: "30+ Disposals",
                    marks_5plus: "5+ Marks", tackles_4plus: "4+ Tackles",
                  };
                  return (
                    <div key={key} className="flex items-center justify-between text-sm">
                      <span className="text-xs text-muted-foreground w-28">{labels[key] || key}</span>
                      <div className="flex items-center gap-4 text-xs tabular-nums">
                        <span>Current <span className={cn("font-semibold", s.current >= 3 ? "text-emerald-400" : "")}>{s.current}</span></span>
                        <span>Best <span className="font-semibold">{s.longest}</span></span>
                        <span>Rate <span className="font-semibold">{s.hit_rate}%</span></span>
                        <span className="text-muted-foreground">{s.hits}/{s.total}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Consistency / Distribution */}
        {Object.keys(consistency).length > 0 && (
          <Card className="border-border/40">
            <CardHeader className="pb-2 pt-4 px-4">
              <CardTitle className="text-sm font-medium">Consistency (Last 2 Seasons)</CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-3">
              <div className="space-y-3">
                {(["gl", "di", "mk", "tk"] as const).map(stat => {
                  const c = consistency[stat];
                  if (!c) return null;
                  const labels: Record<string, string> = { gl: "Goals", di: "Disposals", mk: "Marks", tk: "Tackles" };
                  const colors: Record<string, string> = { gl: CHART_COLORS.goals, di: CHART_COLORS.disposals, mk: CHART_COLORS.marks, tk: "#94a3b8" };
                  return (
                    <div key={stat}>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-medium" style={{ color: colors[stat] }}>{labels[stat]}</span>
                        <span className="text-[10px] text-muted-foreground">{c.games} games</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs tabular-nums">
                        <span className="text-muted-foreground w-14">Floor {c.floor}</span>
                        <div className="flex-1 h-2 bg-muted rounded-full relative overflow-hidden">
                          {/* IQR range */}
                          <div className="absolute h-full rounded-full opacity-40"
                            style={{ left: `${(c.p25 / c.max) * 100}%`, width: `${((c.p75 - c.p25) / c.max) * 100}%`, backgroundColor: colors[stat] }} />
                          {/* Median line */}
                          <div className="absolute h-full w-0.5"
                            style={{ left: `${(c.median / c.max) * 100}%`, backgroundColor: colors[stat] }} />
                        </div>
                        <span className="text-muted-foreground w-16 text-right">Ceiling {c.ceiling}</span>
                      </div>
                      <div className="flex items-center gap-3 mt-0.5 text-[10px] text-muted-foreground">
                        <span>Avg {c.avg}</span>
                        <span>Med {c.median}</span>
                        <span>SD {c.std}</span>
                        <span>Range {c.min}-{c.max}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* ── Home vs Away + Quarter Scoring ────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Home vs Away */}
        {Object.keys(homeAway).length > 0 && (
          <Card className="border-border/40">
            <CardHeader className="pb-2 pt-4 px-4">
              <CardTitle className="text-sm font-medium">Home vs Away</CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-3">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead></TableHead>
                    <TableHead className="text-right">GP</TableHead>
                    <TableHead className="text-right">GL</TableHead>
                    <TableHead className="text-right">DI</TableHead>
                    <TableHead className="text-right">MK</TableHead>
                    <TableHead className="text-right">TK</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {(["home", "away"] as const).map(loc => {
                    const d = homeAway[loc];
                    if (!d) return null;
                    return (
                      <TableRow key={loc}>
                        <TableCell className="font-medium text-xs uppercase">{loc}</TableCell>
                        <TableCell className="text-right tabular-nums">{d.games}</TableCell>
                        <TableCell className="text-right tabular-nums">{d.avg_gl?.toFixed(2)}</TableCell>
                        <TableCell className="text-right tabular-nums">{d.avg_di?.toFixed(1)}</TableCell>
                        <TableCell className="text-right tabular-nums">{d.avg_mk?.toFixed(1)}</TableCell>
                        <TableCell className="text-right tabular-nums">{d.avg_tk?.toFixed(1)}</TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        )}

        {/* Quarter Scoring */}
        {quarterScoring && (
          <Card className="border-border/40">
            <CardHeader className="pb-2 pt-4 px-4">
              <CardTitle className="text-sm font-medium">Goals by Quarter</CardTitle>
              <p className="text-[10px] text-muted-foreground">{quarterScoring.games} games</p>
            </CardHeader>
            <CardContent className="px-4 pb-3">
              <div className="grid grid-cols-4 gap-3">
                {(["q1", "q2", "q3", "q4"] as const).map((q, i) => {
                  const val = quarterScoring[q] ?? 0;
                  const max = Math.max(quarterScoring.q1, quarterScoring.q2, quarterScoring.q3, quarterScoring.q4, 0.01);
                  const pct = (val / max) * 100;
                  return (
                    <div key={q} className="text-center">
                      <div className="h-24 flex items-end justify-center mb-1">
                        <div
                          className="w-8 rounded-t"
                          style={{ height: `${Math.max(pct, 5)}%`, backgroundColor: CHART_COLORS.goals, opacity: 0.5 + (i * 0.15) }}
                        />
                      </div>
                      <p className="text-sm font-bold tabular-nums">{val.toFixed(2)}</p>
                      <p className="text-[10px] text-muted-foreground">Q{i + 1}</p>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* ── Season Averages Chart ────────────────────────────────── */}
      {seasonChart.length > 0 && (
        <Card className="border-border/40">
          <CardHeader className="pb-2 pt-4 px-4">
            <CardTitle className="text-sm font-medium">Season Averages</CardTitle>
          </CardHeader>
          <CardContent className="px-2 pb-3">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={seasonChart}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="year" tick={{ fontSize: 11, fontFamily: "monospace" }} stroke="rgba(255,255,255,0.1)" />
                <YAxis tick={{ fontSize: 11 }} stroke="rgba(255,255,255,0.1)" />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend />
                <Bar dataKey="Disposals" fill={CHART_COLORS.disposals} radius={[2, 2, 0, 0]} />
                <Bar dataKey="Goals" fill={CHART_COLORS.goals} radius={[2, 2, 0, 0]} />
                <Bar dataKey="Marks" fill={CHART_COLORS.marks} radius={[2, 2, 0, 0]} />
                <Bar dataKey="Tackles" fill="#94a3b8" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* ── Season Breakdown Table ───────────────────────────────── */}
      {(profile.seasons || []).length > 0 && (
        <Card className="border-border/40">
          <CardHeader className="pb-2 pt-4 px-4">
            <CardTitle className="text-sm font-medium">Season Breakdown</CardTitle>
          </CardHeader>
          <CardContent className="px-0 pb-2">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="pl-4">Year</TableHead>
                    <TableHead className="text-right">GP</TableHead>
                    <TableHead className="text-right">GL</TableHead>
                    <TableHead className="text-right">BH</TableHead>
                    <TableHead className="text-right">DI</TableHead>
                    <TableHead className="text-right">MK</TableHead>
                    <TableHead className="text-right">KI</TableHead>
                    <TableHead className="text-right">HB</TableHead>
                    <TableHead className="text-right">TK</TableHead>
                    <TableHead className="text-right">HO</TableHead>
                    <TableHead className="text-right">CP</TableHead>
                    <TableHead className="text-right">IF</TableHead>
                    <TableHead className="text-right pr-4">CL</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {(profile.seasons || []).map((s: any) => (
                    <TableRow key={s.year}>
                      <TableCell className="font-medium pl-4">{s.year}</TableCell>
                      <TableCell className="text-right tabular-nums">{s.games}</TableCell>
                      <TableCell className="text-right tabular-nums">{s.GL?.toFixed(1)}</TableCell>
                      <TableCell className="text-right tabular-nums">{s.BH?.toFixed(1)}</TableCell>
                      <TableCell className="text-right tabular-nums font-medium">{s.DI?.toFixed(1)}</TableCell>
                      <TableCell className="text-right tabular-nums">{s.MK?.toFixed(1)}</TableCell>
                      <TableCell className="text-right tabular-nums">{s.KI?.toFixed(1)}</TableCell>
                      <TableCell className="text-right tabular-nums">{s.HB?.toFixed(1)}</TableCell>
                      <TableCell className="text-right tabular-nums">{s.TK?.toFixed(1)}</TableCell>
                      <TableCell className="text-right tabular-nums">{s.HO?.toFixed(1)}</TableCell>
                      <TableCell className="text-right tabular-nums">{s.CP?.toFixed(1) ?? "-"}</TableCell>
                      <TableCell className="text-right tabular-nums">{s.IF?.toFixed(1) ?? "-"}</TableCell>
                      <TableCell className="text-right tabular-nums pr-4">{s.CL?.toFixed(1) ?? "-"}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Opponent Splits ──────────────────────────────────────── */}
      {oppSplits.length > 0 && (
        <Card className="border-border/40">
          <CardHeader className="pb-2 pt-4 px-4">
            <CardTitle className="text-sm font-medium">Performance vs Opponents</CardTitle>
            <p className="text-[10px] text-muted-foreground">Minimum 2 games</p>
          </CardHeader>
          <CardContent className="px-0 pb-2">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="pl-4">Opponent</TableHead>
                    <TableHead className="text-right">GP</TableHead>
                    <TableHead className="text-right">GL</TableHead>
                    <TableHead className="text-right">DI</TableHead>
                    <TableHead className="text-right">MK</TableHead>
                    <TableHead className="text-right">TK</TableHead>
                    <TableHead className="text-right pr-4">KI</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {[...oppSplits].sort((a: any, b: any) => b.avg_di - a.avg_di).map((o: any) => (
                    <TableRow key={o.opponent}>
                      <TableCell className="pl-4">
                        <span className="flex items-center gap-1.5">
                          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: TEAM_COLORS[o.opponent]?.primary || "#666" }} />
                          <span className="text-xs font-mono">{TEAM_ABBREVS[o.opponent] || o.opponent}</span>
                        </span>
                      </TableCell>
                      <TableCell className="text-right tabular-nums text-muted-foreground">{o.games}</TableCell>
                      <TableCell className="text-right tabular-nums">
                        {o.avg_gl?.toFixed(2)}
                        <MiniBar value={o.avg_gl || 0} max={maxOppGL} color={CHART_COLORS.goals} />
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {o.avg_di?.toFixed(1)}
                        <MiniBar value={o.avg_di || 0} max={maxOppDI} color={CHART_COLORS.disposals} />
                      </TableCell>
                      <TableCell className="text-right tabular-nums">{o.avg_mk?.toFixed(1)}</TableCell>
                      <TableCell className="text-right tabular-nums">{o.avg_tk?.toFixed(1)}</TableCell>
                      <TableCell className="text-right tabular-nums pr-4">{o.avg_ki?.toFixed(1)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Venue Splits ─────────────────────────────────────────── */}
      {venueSplits.length > 0 && (
        <Card className="border-border/40">
          <CardHeader className="pb-2 pt-4 px-4">
            <CardTitle className="text-sm font-medium">Venue Splits</CardTitle>
          </CardHeader>
          <CardContent className="px-0 pb-2">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="pl-4">Venue</TableHead>
                    <TableHead className="text-right">GP</TableHead>
                    <TableHead className="text-right">GL</TableHead>
                    <TableHead className="text-right">DI</TableHead>
                    <TableHead className="text-right">MK</TableHead>
                    <TableHead className="text-right">TK</TableHead>
                    <TableHead className="text-right pr-4">KI</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {venueSplits.map((v: any) => (
                    <TableRow key={v.venue}>
                      <TableCell className="text-xs pl-4">{v.venue}</TableCell>
                      <TableCell className="text-right tabular-nums text-muted-foreground">{v.games}</TableCell>
                      <TableCell className="text-right tabular-nums">{v.avg_gl?.toFixed(2)}</TableCell>
                      <TableCell className="text-right tabular-nums">{v.avg_di?.toFixed(1)}</TableCell>
                      <TableCell className="text-right tabular-nums">{v.avg_mk?.toFixed(1)}</TableCell>
                      <TableCell className="text-right tabular-nums">{v.avg_tk?.toFixed(1)}</TableCell>
                      <TableCell className="text-right tabular-nums pr-4">{v.avg_ki?.toFixed(1)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Predictions vs Actuals ───────────────────────────────── */}
      {pvaRecords.length > 0 && (
        <Card className="border-border/40">
          <CardHeader className="pb-2 pt-4 px-4">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Predictions vs Actuals ({pvaRecords.length})</CardTitle>
              {Object.keys(pvaMae).length > 0 && (
                <div className="flex gap-2">
                  {pvaMae.goals != null && <Badge variant="outline" className="text-[10px] tabular-nums">GL MAE {pvaMae.goals}</Badge>}
                  {pvaMae.disposals != null && <Badge variant="outline" className="text-[10px] tabular-nums">DI MAE {pvaMae.disposals}</Badge>}
                  {pvaMae.marks != null && <Badge variant="outline" className="text-[10px] tabular-nums">MK MAE {pvaMae.marks}</Badge>}
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent className="px-0 pb-2">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="pl-4">Year</TableHead>
                    <TableHead>Rd</TableHead>
                    <TableHead className="text-right">GL A/P</TableHead>
                    <TableHead className="text-right">DI A/P</TableHead>
                    <TableHead className="text-right pr-4">MK A/P</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {pvaRecords.slice(-20).map((p: any, i: number) => (
                    <TableRow key={i}>
                      <TableCell className="tabular-nums pl-4">{p.year}</TableCell>
                      <TableCell className="tabular-nums">{p.round ?? "-"}</TableCell>
                      <TableCell className="text-right tabular-nums">
                        {p.actual_goals != null && p.predicted_goals != null ? (
                          <span className={cn(Math.abs(p.actual_goals - p.predicted_goals) <= 1 ? "text-emerald-400" : Math.abs(p.actual_goals - p.predicted_goals) <= 2 ? "text-yellow-400" : "text-red-400")}>
                            {p.actual_goals}/{p.predicted_goals.toFixed(1)}
                          </span>
                        ) : "-"}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {p.actual_disposals != null && p.predicted_disposals != null ? (
                          <span className={cn(Math.abs(p.actual_disposals - p.predicted_disposals) <= 3 ? "text-emerald-400" : Math.abs(p.actual_disposals - p.predicted_disposals) <= 6 ? "text-yellow-400" : "text-red-400")}>
                            {p.actual_disposals}/{p.predicted_disposals.toFixed(0)}
                          </span>
                        ) : "-"}
                      </TableCell>
                      <TableCell className="text-right tabular-nums pr-4">
                        {p.actual_marks != null && p.predicted_marks != null ? (
                          <span className={cn(Math.abs(p.actual_marks - p.predicted_marks) <= 1 ? "text-emerald-400" : Math.abs(p.actual_marks - p.predicted_marks) <= 3 ? "text-yellow-400" : "text-red-400")}>
                            {p.actual_marks}/{p.predicted_marks.toFixed(0)}
                          </span>
                        ) : "-"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Full Game Log ────────────────────────────────────────── */}
      {games.length > 0 && (
        <Card className="border-border/40">
          <CardHeader className="pb-2 pt-4 px-4">
            <CardTitle className="text-sm font-medium">Game Log (Last {games.length})</CardTitle>
          </CardHeader>
          <CardContent className="px-0 pb-2">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="pl-4">Date</TableHead>
                    <TableHead>Rd</TableHead>
                    <TableHead>Opp</TableHead>
                    <TableHead>Venue</TableHead>
                    <TableHead className="text-right">GL</TableHead>
                    <TableHead className="text-right">DI</TableHead>
                    <TableHead className="text-right">MK</TableHead>
                    <TableHead className="text-right">KI</TableHead>
                    <TableHead className="text-right">HB</TableHead>
                    <TableHead className="text-right pr-4">TK</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {games.map((g: any) => (
                    <TableRow key={g.match_id}>
                      <TableCell className="text-xs pl-4">{g.date}</TableCell>
                      <TableCell className="tabular-nums">{g.round_number}</TableCell>
                      <TableCell>
                        <span className="flex items-center gap-1.5">
                          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: TEAM_COLORS[g.opponent]?.primary || "#666" }} />
                          <span className="text-xs font-mono">{TEAM_ABBREVS[g.opponent] || g.opponent}</span>
                        </span>
                      </TableCell>
                      <TableCell className="text-xs truncate max-w-[100px]">{g.venue}</TableCell>
                      <TableCell className="text-right tabular-nums">{g.GL}</TableCell>
                      <TableCell className="text-right tabular-nums">{g.DI}</TableCell>
                      <TableCell className="text-right tabular-nums">{g.MK}</TableCell>
                      <TableCell className="text-right tabular-nums">{g.KI}</TableCell>
                      <TableCell className="text-right tabular-nums">{g.HB}</TableCell>
                      <TableCell className="text-right tabular-nums pr-4">{g.TK}</TableCell>
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
