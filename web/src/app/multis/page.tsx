"use client";

import { useEffect, useState } from "react";
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
import { getMultiBacktest } from "@/lib/api";
import type {
  MultiBacktestData,
  MultiCombo,
  MultiLeg,
} from "@/lib/types";
import { TEAM_ABBREVS } from "@/lib/constants";
import { cn } from "@/lib/utils";

const AVAILABLE_YEARS = [2025, 2024, 2023, 2022, 2021];

const TIER_ORDER = ["tier_85", "tier_80", "tier_75", "tier_70"];

const TIER_META: Record<string, { label: string; color: string; bg: string; tip: string }> = {
  tier_85: { label: "85% Confidence", color: "text-green-300",    bg: "bg-green-500/10 border-green-500/20", tip: "Every leg has 85%+ model probability. Safest tier — fewer combos but highest hit rate." },
  tier_80: { label: "80% Confidence", color: "text-emerald-400",  bg: "bg-emerald-500/10 border-emerald-500/20", tip: "Every leg has 80%+ model probability. Good balance of safety and volume." },
  tier_75: { label: "75% Confidence", color: "text-blue-400",     bg: "bg-blue-500/10 border-blue-500/20", tip: "Every leg has 75%+ model probability. More combos available but lower hit rate." },
  tier_70: { label: "70% Confidence", color: "text-indigo-400",   bg: "bg-indigo-500/10 border-indigo-500/20", tip: "Every leg has 70%+ model probability. Highest volume but riskiest tier." },
};

function legLabel(legType: string): string {
  const STATIC: Record<string, string> = {
    goals_1plus: "1+ Goals",
    goals_2plus: "2+ Goals",
    goals_3plus: "3+ Goals",
    disp_15plus: "15+ Disp",
    disp_20plus: "20+ Disp",
    disp_25plus: "25+ Disp",
    disp_30plus: "30+ Disp",
  };
  if (STATIC[legType]) return STATIC[legType];

  // Dynamic: team_total_80 -> "80+ Team", match_total_160 -> "160+ Match"
  const teamMatch = legType.match(/^team_total_(\d+)$/);
  if (teamMatch) return `${teamMatch[1]}+ Team`;

  const matchMatch = legType.match(/^match_total_(\d+)$/);
  if (matchMatch) return `${matchMatch[1]}+ Match`;

  return legType;
}

function pct(v: number): string {
  return (v * 100).toFixed(1) + "%";
}

function isTeamOrMatchLeg(legType: string): boolean {
  return legType.startsWith("team_total_") || legType.startsWith("match_total_");
}

function ComboCard({ combo }: { combo: MultiCombo }) {
  const meta = TIER_META[combo.tier] ?? TIER_META.tier_70;
  const [expanded, setExpanded] = useState(false);

  return (
    <div className={cn("border rounded-lg p-3 space-y-2", meta.bg)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Badge
            variant="outline"
            className={cn("text-[10px] uppercase tracking-wider font-semibold", meta.color)}
          >
            {combo.tier_label || meta.label}
          </Badge>
          <span className="text-xs text-muted-foreground">
            {combo.n_legs} legs
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground tabular-nums">
            {pct(combo.combo_predicted_prob)}
          </span>
          {combo.combo_hit === true && (
            <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30 text-[10px]">
              HIT
            </Badge>
          )}
          {combo.combo_hit === false && (
            <Badge className="bg-red-500/20 text-red-400 border-red-500/30 text-[10px]">
              MISS
            </Badge>
          )}
        </div>
      </div>

      <div className="space-y-1.5">
        {combo.legs.map((leg, i) => (
          <LegRow key={i} leg={leg} showReason={expanded} />
        ))}
      </div>

      {combo.legs.some((l) => l.reason) && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-[10px] text-muted-foreground hover:text-foreground transition-colors"
        >
          {expanded ? "Hide reasoning" : "Show reasoning"}
        </button>
      )}
    </div>
  );
}

function LegRow({ leg, showReason }: { leg: MultiLeg; showReason: boolean }) {
  const isTeamMatch = isTeamOrMatchLeg(leg.leg_type);
  const playerName = leg.player && leg.player.includes(", ")
    ? leg.player.split(", ").reverse().join(" ")
    : leg.player;
  const teamAbbrev = TEAM_ABBREVS[leg.team] || leg.team;

  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-1.5 min-w-0">
          {leg.hit === true && <span className="text-emerald-400 shrink-0">&#10003;</span>}
          {leg.hit === false && <span className="text-red-400 shrink-0">&#10007;</span>}
          {leg.hit == null && <span className="text-muted-foreground shrink-0">-</span>}
          <span className="truncate text-foreground/90">
            {isTeamMatch ? (
              <span className="font-medium">{leg.team}</span>
            ) : leg.player ? (
              <>
                <span className="font-medium">{playerName}</span>
                <span className="text-muted-foreground ml-1">({teamAbbrev})</span>
              </>
            ) : (
              <span className="font-medium">{leg.label}</span>
            )}
          </span>
          <Badge variant="outline" className="text-[9px] px-1 py-0 shrink-0">
            {legLabel(leg.leg_type)}
          </Badge>
        </div>
        <span className="tabular-nums text-muted-foreground ml-2 shrink-0">
          {pct(leg.prob)}
        </span>
      </div>
      {showReason && leg.reason && (
        <p className="text-[10px] text-muted-foreground/80 ml-5 italic leading-tight">
          {leg.reason}
        </p>
      )}
    </div>
  );
}

export default function MultisPage() {
  const [year, setYear] = useState(AVAILABLE_YEARS[0]);
  const [data, setData] = useState<MultiBacktestData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedRound, setSelectedRound] = useState<string>("all");
  const [selectedTier, setSelectedTier] = useState<string>("all");
  const [selectedResult, setSelectedResult] = useState<string>("all");

  useEffect(() => {
    setLoading(true);
    setData(null);
    setSelectedRound("all");
    setSelectedTier("all");
    setSelectedResult("all");
    getMultiBacktest(year)
      .then((d) => {
        if (!d.error) setData(d);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [year]);

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-48" />
        <div className="grid grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => <Skeleton key={i} className="h-28" />)}
        </div>
        <Skeleton className="h-96" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="space-y-6">
        <div className="flex items-start justify-between">
          <h1 className="text-xl sm:text-2xl font-bold">Multi-Bet Backtest</h1>
          <div className="flex items-center gap-2">
            {AVAILABLE_YEARS.map((y) => (
              <button
                key={y}
                onClick={() => setYear(y)}
                className={cn(
                  "px-3 py-1.5 rounded text-sm font-medium transition-colors",
                  y === year
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-muted-foreground hover:text-foreground"
                )}
              >
                {y}
              </button>
            ))}
          </div>
        </div>
        <Card>
          <CardContent className="pt-6">
            <p className="text-muted-foreground text-sm">
              No multi-bet backtest data found for {year}. Run{" "}
              <code className="text-xs bg-muted px-1 py-0.5 rounded">
                python3 multi_backtest.py --year {year}
              </code>{" "}
              to generate.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const { summary, leg_type_stats, failure_analysis, calibration, rounds } = data;

  const tierDefs = data.tier_definitions;
  const detectedTiers: string[] = [];
  for (const t of TIER_ORDER) {
    if (summary[t as keyof typeof summary]) {
      detectedTiers.push(t);
    }
  }

  const tierCards = detectedTiers.map((t) => {
    const stats = (summary[t as keyof typeof summary] || {}) as Record<string, number>;
    const meta = TIER_META[t] ?? TIER_META.tier_70;
    const def = tierDefs?.[t];
    return {
      tier: t,
      label: def?.label || (stats as Record<string, unknown>).label as string || meta.label,
      desc: def?.desc || (stats as Record<string, unknown>).desc as string || "",
      n_combos: stats.n_combos || 0,
      n_hits: stats.n_hits || 0,
      combo_hit_rate: stats.combo_hit_rate || 0,
      avg_predicted_prob: stats.avg_predicted_prob || 0,
      avg_legs: stats.avg_legs || 0,
      meta,
    };
  });

  // Collect all combos
  const allCombos: (MultiCombo & { round_number: string })[] = [];
  for (const [rnd, combos] of Object.entries(rounds)) {
    for (const c of combos) {
      allCombos.push({ ...c, round_number: rnd });
    }
  }

  const roundNumbers = Object.keys(rounds).sort((a, b) => Number(a) - Number(b));
  const uniqueTiers = [...new Set(allCombos.map((c) => c.tier))];

  // Filter combos
  const filtered = allCombos.filter((c) => {
    if (selectedRound !== "all" && c.round_number !== selectedRound) return false;
    if (selectedTier !== "all" && c.tier !== selectedTier) return false;
    if (selectedResult === "hit" && c.combo_hit !== true) return false;
    if (selectedResult === "miss" && c.combo_hit !== false) return false;
    return true;
  });

  // Sort by round then tier priority
  const tierOrderMap: Record<string, number> = {};
  TIER_ORDER.forEach((t, i) => { tierOrderMap[t] = i; });

  filtered.sort((a, b) => {
    const rndDiff = Number(a.round_number) - Number(b.round_number);
    if (rndDiff !== 0) return rndDiff;
    return (tierOrderMap[a.tier] ?? 99) - (tierOrderMap[b.tier] ?? 99);
  });

  const legTypeEntries = Object.entries(leg_type_stats).sort(
    (a, b) => b[1].n_used - a[1].n_used
  );

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold">Multi-Bet Backtest</h1>
          <p className="text-sm text-muted-foreground mt-1">
            {year} season &middot; {summary.overall.total_combos} combos across {roundNumbers.length} rounds
          </p>
        </div>
        <div className="flex items-center gap-2">
          {AVAILABLE_YEARS.map((y) => (
            <button
              key={y}
              onClick={() => setYear(y)}
              className={cn(
                "px-3 py-1.5 rounded text-sm font-medium transition-colors",
                y === year
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground hover:text-foreground"
              )}
            >
              {y}
            </button>
          ))}
        </div>
      </div>

      {/* Tier summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        {tierCards.map((t) => (
          <Card key={t.tier} className={cn("border cursor-help", t.meta.bg)} title={t.meta.tip}>
            <CardContent className="pt-4 pb-3">
              <p className={cn("text-[10px] font-semibold uppercase tracking-wider", t.meta.color)}>
                {t.label}
              </p>
              {t.desc && (
                <p className="text-[9px] text-muted-foreground/60 mt-0.5 leading-tight">{t.desc}</p>
              )}
              <p className="text-2xl font-bold mt-1 tabular-nums">
                {pct(t.combo_hit_rate)}
              </p>
              <div className="flex gap-3 mt-1 text-[10px] text-muted-foreground">
                <span>{t.n_hits}/{t.n_combos} hit</span>
                <span>{t.avg_legs.toFixed(1)} legs</span>
              </div>
            </CardContent>
          </Card>
        ))}
        <Card>
          <CardContent className="pt-4 pb-3">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Overall
            </p>
            <p className="text-2xl font-bold mt-1 tabular-nums">
              {pct(summary.overall.overall_hit_rate)}
            </p>
            <div className="flex gap-3 mt-1 text-[10px] text-muted-foreground">
              <span>{summary.overall.total_hits}/{summary.overall.total_combos} hit</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Leg type performance + Failure analysis side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Leg Type Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Leg Type</TableHead>
                  <TableHead className="text-right">Used</TableHead>
                  <TableHead className="text-right">Hit Rate</TableHead>
                  <TableHead className="text-right">Weakest Link</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {legTypeEntries.map(([lt, s]) => (
                  <TableRow key={lt}>
                    <TableCell className="font-medium text-sm">
                      {legLabel(lt)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{s.n_used}</TableCell>
                    <TableCell className="text-right tabular-nums">
                      <span className={cn(s.hit_rate >= 0.75 ? "text-emerald-400" : s.hit_rate < 0.5 ? "text-red-400" : "")}>
                        {pct(s.hit_rate)}
                      </span>
                    </TableCell>
                    <TableCell className="text-right tabular-nums text-muted-foreground">
                      {s.n_weakest_link > 0 ? s.n_weakest_link : "-"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Failure Analysis</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs text-muted-foreground">Total Misses</p>
                  <p className="text-xl font-bold tabular-nums">
                    {failure_analysis.total_misses}
                    <span className="text-sm text-muted-foreground font-normal ml-1">
                      / {failure_analysis.total_combos}
                    </span>
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Single-Leg Failures</p>
                  <p className="text-xl font-bold tabular-nums">
                    {pct(failure_analysis.single_leg_failures_pct)}
                  </p>
                  <p className="text-[10px] text-muted-foreground">
                    {failure_analysis.single_leg_failures} combos lost to 1 bad leg
                  </p>
                </div>
              </div>
              {failure_analysis.most_failed_leg_type && (
                <div>
                  <p className="text-xs text-muted-foreground">Most Common Failure</p>
                  <p className="text-sm font-semibold text-red-400">
                    {legLabel(failure_analysis.most_failed_leg_type)}
                    <span className="text-muted-foreground font-normal ml-1">
                      ({failure_analysis.failure_counts[failure_analysis.most_failed_leg_type]} times)
                    </span>
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Combo Probability Calibration</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Prob Bucket</TableHead>
                    <TableHead className="text-right">N</TableHead>
                    <TableHead className="text-right">Predicted</TableHead>
                    <TableHead className="text-right">Actual</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.entries(calibration).sort().map(([bucket, cal]) => (
                    <TableRow key={bucket}>
                      <TableCell className="text-sm">{bucket}</TableCell>
                      <TableCell className="text-right tabular-nums">{cal.n}</TableCell>
                      <TableCell className="text-right tabular-nums">{pct(cal.avg_predicted)}</TableCell>
                      <TableCell className="text-right tabular-nums">
                        <span className={cn(
                          Math.abs(cal.actual_hit_rate - cal.avg_predicted) < 0.05
                            ? "text-emerald-400"
                            : Math.abs(cal.actual_hit_rate - cal.avg_predicted) > 0.10
                              ? "text-amber-400"
                              : ""
                        )}>
                          {pct(cal.actual_hit_rate)}
                        </span>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* All combos */}
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between flex-wrap gap-3">
            <CardTitle className="text-base">
              All Combinations ({filtered.length})
            </CardTitle>
            <div className="flex gap-2 flex-wrap">
              <select
                value={selectedRound}
                onChange={(e) => setSelectedRound(e.target.value)}
                className="text-xs bg-muted border border-border rounded px-2 py-1"
              >
                <option value="all">All Rounds</option>
                {roundNumbers.map((r) => (
                  <option key={r} value={r}>Round {r}</option>
                ))}
              </select>
              <select
                value={selectedTier}
                onChange={(e) => setSelectedTier(e.target.value)}
                className="text-xs bg-muted border border-border rounded px-2 py-1"
              >
                <option value="all">All Tiers</option>
                {uniqueTiers.map((t) => {
                  const m = TIER_META[t];
                  return (
                    <option key={t} value={t}>{m?.label || t}</option>
                  );
                })}
              </select>
              <select
                value={selectedResult}
                onChange={(e) => setSelectedResult(e.target.value)}
                className="text-xs bg-muted border border-border rounded px-2 py-1"
              >
                <option value="all">All Results</option>
                <option value="hit">Hits Only</option>
                <option value="miss">Misses Only</option>
              </select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {filtered.length === 0 ? (
            <p className="text-sm text-muted-foreground">No combos match your filters.</p>
          ) : (
            <div className="space-y-6">
              {(() => {
                const grouped: Record<string, typeof filtered> = {};
                for (const c of filtered) {
                  if (!grouped[c.round_number]) grouped[c.round_number] = [];
                  grouped[c.round_number].push(c);
                }
                return Object.entries(grouped)
                  .sort(([a], [b]) => Number(a) - Number(b))
                  .map(([rnd, combos]) => (
                    <div key={rnd}>
                      <h3 className="text-sm font-semibold text-muted-foreground mb-2">
                        Round {rnd}
                        <span className="text-xs font-normal ml-2">
                          {combos.filter((c) => c.combo_hit === true).length}/{combos.length} hit
                        </span>
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
                        {combos.map((c, i) => (
                          <ComboCard key={`${rnd}-${i}`} combo={c} />
                        ))}
                      </div>
                    </div>
                  ));
              })()}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
