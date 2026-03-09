"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { getSeasonSchedule } from "@/lib/api";
import type { SeasonSchedule, ScheduleRound } from "@/lib/types";
import { TEAM_ABBREVS, TEAM_COLORS, CURRENT_YEAR, AVAILABLE_YEARS, displayVenue } from "@/lib/constants";
import { cn, formatDate } from "@/lib/utils";
import Link from "next/link";

function WeatherBadge({ forecast }: { forecast: NonNullable<import("@/lib/types").ScheduleForecast> }) {
  const temp = forecast.temperature_avg;
  const rain = forecast.precipitation_total;
  const wind = forecast.wind_speed_avg;
  const wet = forecast.is_wet;
  const roofed = forecast.is_roofed;

  return (
    <div className="flex items-center gap-2 flex-wrap">
      {roofed && (
        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400">ROOF</span>
      )}
      {temp != null && (
        <span className={cn(
          "text-[10px] font-mono px-1.5 py-0.5 rounded",
          temp > 30 ? "bg-red-500/10 text-red-400" :
          temp < 12 ? "bg-cyan-500/10 text-cyan-400" :
          "bg-muted text-muted-foreground"
        )}>
          {temp.toFixed(0)}°C
        </span>
      )}
      {wet && (
        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400">WET</span>
      )}
      {!wet && rain != null && rain > 0 && (
        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
          {rain.toFixed(1)}mm
        </span>
      )}
      {wind != null && wind > 20 && (
        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400">
          {wind.toFixed(0)}km/h
        </span>
      )}
      {forecast.weather_difficulty_score != null && forecast.weather_difficulty_score > 0.5 && (
        <span
          className={cn(
            "text-[10px] font-mono px-1.5 py-0.5 rounded cursor-help",
            forecast.weather_difficulty_score > 0.7 ? "bg-red-500/10 text-red-400" : "bg-amber-500/10 text-amber-400"
          )}
          title="Weather difficulty (0-1). Combines rain, wind, and temperature extremes. Above 0.7 = tough conditions that may reduce scoring."
        >
          DIFF {forecast.weather_difficulty_score.toFixed(1)}
        </span>
      )}
    </div>
  );
}

function RoundCard({ round }: { round: ScheduleRound }) {
  const statusColor = round.status === "completed"
    ? "text-emerald-400 bg-emerald-400/10"
    : round.status === "in_progress"
    ? "text-amber-400 bg-amber-400/10"
    : round.status === "upcoming"
    ? "text-primary bg-primary/10"
    : "text-muted-foreground bg-muted/50";

  return (
    <Card className="border-border/50">
      <CardHeader className="pb-2 pt-4 px-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium font-mono">
            Round {round.round_number}
          </CardTitle>
          <span className={cn("text-[10px] font-mono font-semibold px-2 py-0.5 rounded uppercase", statusColor)}>
            {round.status}
          </span>
        </div>
      </CardHeader>
      <CardContent className="px-3 pb-3 space-y-1">
        {round.matches.map((m, i) => {
          const homeAbbr = TEAM_ABBREVS[m.home_team] || m.home_team;
          const awayAbbr = TEAM_ABBREVS[m.away_team] || m.away_team;
          const isPlayed = m.home_score != null && m.away_score != null;

          const matchHref = m.match_id
            ? `/matches/${m.match_id}?round=${round.round_number}`
            : `/matches/0?round=${round.round_number}&home=${encodeURIComponent(m.home_team)}&away=${encodeURIComponent(m.away_team)}`;

          return (
            <Link key={i} href={matchHref} className="block group">
              <div className="flex items-center gap-2 px-2.5 py-2 rounded-md border border-border/30 bg-card/30 group-hover:bg-muted/30 group-hover:border-border/50 transition-colors">
                {/* Teams + scores */}
                <div className="flex items-center gap-1.5 min-w-[52px]">
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: TEAM_COLORS[m.home_team]?.primary || "#555" }} />
                  <span className="text-xs font-mono font-semibold">{homeAbbr}</span>
                </div>
                {isPlayed ? (
                  <>
                    <span className="text-sm font-bold tabular-nums font-mono w-6 text-right">{m.home_score}</span>
                    <span className="text-[10px] text-muted-foreground/50">-</span>
                    <span className="text-sm font-bold tabular-nums font-mono w-6">{m.away_score}</span>
                  </>
                ) : (
                  <>
                    <span className="text-[10px] text-muted-foreground/50 w-[52px] text-center">vs</span>
                  </>
                )}
                <div className="flex items-center gap-1.5 min-w-[52px]">
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: TEAM_COLORS[m.away_team]?.primary || "#555" }} />
                  <span className="text-xs font-mono font-semibold">{awayAbbr}</span>
                </div>

                <span className="flex-1" />

                {/* Weather forecast */}
                {m.forecast && <WeatherBadge forecast={m.forecast} />}

                {/* Venue + date */}
                <div className="text-right shrink-0">
                  {m.venue && <p className="text-[10px] text-muted-foreground truncate max-w-[160px]">{displayVenue(m.venue)}</p>}
                  {m.date && <p className="text-[9px] text-muted-foreground/50 font-mono">{formatDate(m.date)}</p>}
                </div>
              </div>
            </Link>
          );
        })}
      </CardContent>
    </Card>
  );
}

export default function SchedulePage() {
  const [schedule, setSchedule] = useState<SeasonSchedule | null>(null);
  const [loading, setLoading] = useState(true);
  const [year, setYear] = useState(CURRENT_YEAR);
  const [filter, setFilter] = useState<"all" | "completed" | "in_progress" | "upcoming" | "future">("all");

  const years = AVAILABLE_YEARS;

  useEffect(() => {
    setLoading(true);
    getSeasonSchedule(year)
      .then(setSchedule)
      .catch(() => setSchedule(null))
      .finally(() => setLoading(false));
  }, [year]);

  if (loading) {
    return (
      <div className="space-y-5">
        <Skeleton className="h-8 w-48" />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => <Skeleton key={i} className="h-64" />)}
        </div>
      </div>
    );
  }

  const rounds = schedule?.rounds || [];
  const completed = rounds.filter((r) => r.status === "completed").length;
  const upcoming = rounds.find((r) => r.status === "upcoming");

  const filtered = filter === "all" ? rounds : rounds.filter((r) => r.status === filter);

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-bold tracking-tight">{year} Season Schedule</h1>
          <Badge variant="outline" className="text-xs font-mono">
            {completed}/{rounds.length} rounds
          </Badge>
          {upcoming && (
            <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
              Next: R{upcoming.round_number}
            </span>
          )}
        </div>
      </div>

      {/* Year + Filter */}
      <div className="flex gap-2 flex-wrap items-center">
        {years.map((y) => (
          <Button key={y} variant={y === year ? "default" : "outline"} size="sm" onClick={() => { setYear(y); setFilter("all"); }}>
            {y}
          </Button>
        ))}
        <span className="mx-1 w-px h-5 bg-border" />
        {(["all", "completed", "in_progress", "upcoming", "future"] as const).map((f) => (
          <Button
            key={f}
            variant={f === filter ? "secondary" : "ghost"}
            size="sm"
            className="text-xs capitalize"
            onClick={() => setFilter(f)}
          >
            {f}
          </Button>
        ))}
      </div>

      {/* Rounds grid */}
      {filtered.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {filtered.map((r) => (
            <RoundCard key={r.round_number} round={r} />
          ))}
        </div>
      ) : (
        <Card className="border-border/50">
          <CardContent className="py-8 text-center">
            <p className="text-sm text-muted-foreground">No rounds found for this filter.</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
