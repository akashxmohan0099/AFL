"use client";

import { useState } from "react";
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
import type { MatchContext, MatchComparison, MatchWeather, WeatherImpactPlayer, WeatherSummaryInfo, TotalScoreBracket, TeamStatMatchup, ScoringAverageEntry } from "@/lib/types";
import { TEAM_ABBREVS, TEAM_COLORS, CURRENT_YEAR } from "@/lib/constants";
import { cn } from "@/lib/utils";

export function StatRange({ min, median, max }: { min?: number | null; median?: number | null; max?: number | null }) {
  if (min == null && median == null && max == null) return null;
  return (
    <p className="text-[9px] font-mono tabular-nums text-muted-foreground/60 leading-tight">
      <span className="text-red-400/70">{min ?? "\u2014"}</span>
      <span className="text-muted-foreground/30"> · </span>
      <span>{median ?? "\u2014"}</span>
      <span className="text-muted-foreground/30"> · </span>
      <span className="text-emerald-400/70">{max ?? "\u2014"}</span>
    </p>
  );
}

export function RecordBadge({ wins, losses, draws }: { wins: number; losses: number; draws?: number }) {
  const parts = [`${wins}W`, `${losses}L`];
  if (draws && draws > 0) parts.push(`${draws}D`);
  return <span className="text-xs font-mono tabular-nums font-semibold">{parts.join("-")}</span>;
}

export function MatchContextCard({
  context,
  homeTeam,
  awayTeam,
  homeColor,
  awayColor,
  weather,
  weatherSummary,
  weatherImpact,
  comparison,
}: {
  context: MatchContext;
  homeTeam: string;
  awayTeam: string;
  homeColor: string;
  awayColor: string;
  weather?: MatchWeather | null;
  weatherSummary?: WeatherSummaryInfo | null;
  weatherImpact?: WeatherImpactPlayer[];
  comparison?: MatchComparison | null;
}) {
  const homeAbbr = TEAM_ABBREVS[homeTeam] || homeTeam;
  const awayAbbr = TEAM_ABBREVS[awayTeam] || awayTeam;
  const [showWeatherTable, setShowWeatherTable] = useState(false);
  const [showAllBrackets, setShowAllBrackets] = useState(false);
  const [showAllTeamBrackets, setShowAllTeamBrackets] = useState(false);
  const [contextView, setContextView] = useState<"default" | "advanced">("default");

  // Season records — show both current and last season
  const homeSeason = context.home_team_season;
  const awaySeason = context.away_team_season;
  const homeLastSeason = context.home_team_last_season;
  const awayLastSeason = context.away_team_last_season;
  const hasAnySeasonData = (homeSeason?.played ?? 0) > 0 || (awaySeason?.played ?? 0) > 0 || (homeLastSeason?.played ?? 0) > 0 || (awayLastSeason?.played ?? 0) > 0;

  const homeVenue = context.home_team_venue;
  const awayVenue = context.away_team_venue;
  const h2hHome = context.h2h_home;
  const h2hAway = context.h2h_away;
  const ground = context.ground_stats;

  const conditions = weatherSummary?.conditions || [];
  const condLabel = conditions.map((c) => c.charAt(0).toUpperCase() + c.slice(1)).join(", ");

  // Weather impact by team
  const homeImpact = weatherImpact?.filter((p) => p.team === homeTeam) || [];
  const awayImpact = weatherImpact?.filter((p) => p.team === awayTeam) || [];
  const homeFavored = homeImpact.filter((p) => p.di_diff > 0).length;
  const awayFavored = awayImpact.filter((p) => p.di_diff > 0).length;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Match Context</CardTitle>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setContextView("default")}
              className={cn(
                "px-2.5 py-1 text-[10px] font-medium rounded-l-md border transition-colors",
                contextView === "default"
                  ? "bg-primary text-primary-foreground border-primary"
                  : "bg-muted/50 text-muted-foreground border-border hover:bg-muted"
              )}
            >
              Default
            </button>
            <button
              onClick={() => setContextView("advanced")}
              className={cn(
                "px-2.5 py-1 text-[10px] font-medium rounded-r-md border border-l-0 transition-colors",
                contextView === "advanced"
                  ? "bg-primary text-primary-foreground border-primary"
                  : "bg-muted/50 text-muted-foreground border-border hover:bg-muted"
              )}
            >
              Advanced
            </button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Row 1: Game info + Weather */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {context.day_of_week && (
            <div>
              <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Day</p>
              <p className="text-sm font-bold mt-0.5">{context.day_of_week}</p>
            </div>
          )}
          {context.time && (
            <div>
              <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Kickoff</p>
              <p className="text-sm font-bold mt-0.5">
                {context.time}
                {context.day_night && (
                  <span className={cn(
                    "ml-1.5 text-[10px] font-mono px-1.5 py-0.5 rounded",
                    context.day_night === "Night" ? "bg-indigo-500/10 text-indigo-400" : "bg-amber-500/10 text-amber-400"
                  )}>
                    {context.day_night}
                  </span>
                )}
              </p>
            </div>
          )}
          {weather && (
            <>
              <div>
                <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Game Temp</p>
                <p className="text-sm font-bold tabular-nums mt-0.5">
                  {weather.temperature_avg.toFixed(1)}°C
                  {conditions.length > 0 && conditions.map((c) => (
                    <span key={c} className={cn(
                      "ml-1.5 text-[10px] font-mono px-1.5 py-0.5 rounded uppercase",
                      c === "wet" ? "bg-blue-500/10 text-blue-400" :
                      c === "hot" ? "bg-red-500/10 text-red-400" :
                      c === "cold" ? "bg-cyan-500/10 text-cyan-400" :
                      "bg-muted text-muted-foreground"
                    )}>
                      {c}
                    </span>
                  ))}
                </p>
              </div>
              <div>
                <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Wind / Rain</p>
                <p className="text-sm font-bold tabular-nums mt-0.5">
                  {weather.wind_speed_avg.toFixed(0)} km/h &middot; {weather.precipitation_total.toFixed(1)} mm
                </p>
              </div>
            </>
          )}
        </div>

        {/* Row 2: Home advantage indicator */}
        <div className="border-t border-border/30 pt-4">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-3">
            Home Advantage
          </p>
          <div className="flex items-center gap-3 mb-1">
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: homeColor }} />
            <span className="text-xs font-semibold">{homeAbbr}</span>
            <Badge variant="outline" className="text-[10px] px-1.5 py-0">HOME</Badge>
          </div>
          <div className="flex items-center gap-3">
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: awayColor }} />
            <span className="text-xs font-semibold">{awayAbbr}</span>
            <Badge variant="outline" className="text-[10px] px-1.5 py-0 text-muted-foreground">AWAY</Badge>
          </div>
        </div>

        {/* Section: Scoring Averages */}
        {(context.home_scoring_averages || context.away_scoring_averages) && (
          <div className="border-t border-border/30 pt-4">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-3">Scoring Averages</p>
            <div className="grid grid-cols-2 gap-4">
              {[
                { abbr: homeAbbr, color: homeColor, avgs: context.home_scoring_averages },
                { abbr: awayAbbr, color: awayColor, avgs: context.away_scoring_averages },
              ].map(({ abbr, color, avgs }) => {
                if (!avgs) return null;
                const rows: { label: string; data: ScoringAverageEntry | null | undefined }[] = [
                  { label: "Season", data: avgs.season },
                  { label: "Last 5", data: avgs.last_5 },
                  { label: "Last 10", data: avgs.last_10 },
                  { label: "vs Opp", data: avgs.vs_opponent },
                  { label: "At Venue", data: avgs.at_venue },
                ];
                return (
                  <div key={abbr} className="px-3 py-2.5 rounded-lg border border-border/30 bg-card/30">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                      <span className="text-xs font-semibold font-mono">{abbr}</span>
                    </div>
                    <table className="w-full text-[11px] font-mono tabular-nums">
                      <thead>
                        <tr className="text-muted-foreground/60">
                          <th className="text-left font-normal pb-1"></th>
                          <th className="text-right font-normal pb-1">For</th>
                          <th className="text-right font-normal pb-1">Agst</th>
                          <th className="text-right font-normal pb-1 text-muted-foreground/40">G</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map(({ label, data }) => {
                          if (!data) return null;
                          return (
                            <tr key={label}>
                              <td className="text-muted-foreground pr-2 py-0.5">{label}</td>
                              <td className="text-right font-semibold text-foreground/80">{data.scored}</td>
                              <td className="text-right text-muted-foreground">{data.conceded}</td>
                              <td className="text-right text-muted-foreground/40">{data.games}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Section: Season Records — both current + last season per team */}
        {hasAnySeasonData && (
          <div className="border-t border-border/30 pt-4">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-3">
              Season Record
            </p>
            <div className="grid grid-cols-2 gap-4">
              {/* Home team */}
              {[
                { abbr: homeAbbr, color: homeColor, season: homeSeason, lastSeason: homeLastSeason },
                { abbr: awayAbbr, color: awayColor, season: awaySeason, lastSeason: awayLastSeason },
              ].map(({ abbr, color, season, lastSeason }) => (
                <div key={abbr} className="px-3 py-2.5 rounded-lg border border-border/30 bg-card/30 space-y-2.5">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                    <span className="text-xs font-semibold font-mono">{abbr}</span>
                  </div>
                  {/* Current season */}
                  <div>
                    <p className="text-[9px] font-mono text-muted-foreground/50 uppercase mb-1">{CURRENT_YEAR}</p>
                    {(season?.played ?? 0) > 0 ? (
                      <>
                        <div className="flex items-center gap-2 mb-0.5">
                          <RecordBadge wins={season!.wins} losses={season!.losses} draws={season!.draws} />
                        </div>
                        <div className="flex gap-4 text-[11px] flex-wrap">
                          {season!.home_record && (
                            <div>
                              <span className="text-muted-foreground">Home: </span>
                              <span className="font-mono font-semibold">{season!.home_record}</span>
                            </div>
                          )}
                          {season!.away_record && (
                            <div>
                              <span className="text-muted-foreground">Away: </span>
                              <span className="font-mono font-semibold">{season!.away_record}</span>
                            </div>
                          )}
                          {season!.avg_score != null && (
                            <div>
                              <span className="text-muted-foreground">Avg: </span>
                              <span className="font-mono font-semibold">{season!.avg_score}</span>
                              <span className="text-muted-foreground">-</span>
                              <span className="font-mono font-semibold">{season!.avg_conceded}</span>
                            </div>
                          )}
                        </div>
                        <StatRange min={season!.min_score} median={season!.median_score} max={season!.max_score} />
                      </>
                    ) : (
                      <p className="text-[11px] text-muted-foreground/40 italic">No games yet</p>
                    )}
                  </div>
                  {/* Last season */}
                  {lastSeason && (lastSeason.played ?? 0) > 0 && (
                    <div className="border-t border-border/20 pt-2">
                      <p className="text-[9px] font-mono text-muted-foreground/50 uppercase mb-1">{CURRENT_YEAR - 1}</p>
                      <div className="flex items-center gap-2 mb-0.5">
                        <RecordBadge wins={lastSeason.wins} losses={lastSeason.losses} draws={lastSeason.draws} />
                      </div>
                      <div className="flex gap-4 text-[11px] flex-wrap">
                        {lastSeason.home_record && (
                          <div>
                            <span className="text-muted-foreground">Home: </span>
                            <span className="font-mono font-semibold">{lastSeason.home_record}</span>
                          </div>
                        )}
                        {lastSeason.away_record && (
                          <div>
                            <span className="text-muted-foreground">Away: </span>
                            <span className="font-mono font-semibold">{lastSeason.away_record}</span>
                          </div>
                        )}
                        {lastSeason.avg_score != null && (
                          <div>
                            <span className="text-muted-foreground">Avg: </span>
                            <span className="font-mono font-semibold">{lastSeason.avg_score}</span>
                            <span className="text-muted-foreground">-</span>
                            <span className="font-mono font-semibold">{lastSeason.avg_conceded}</span>
                          </div>
                        )}
                      </div>
                      <StatRange min={lastSeason.min_score} median={lastSeason.median_score} max={lastSeason.max_score} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Section: Recent Form */}
        {(context.home_recent_form || context.away_recent_form) && (
          <div className="border-t border-border/30 pt-4">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-3">Recent Form (Last 5)</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[
                { label: homeAbbr, color: homeColor, form: context.home_recent_form },
                { label: awayAbbr, color: awayColor, form: context.away_recent_form },
              ].map(({ label, color, form }) => form && form.length > 0 && (
                <div key={label} className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                    <span className="text-xs font-semibold">{label}</span>
                    <div className="flex gap-0.5 ml-auto">
                      {form.map((g, i) => (
                        <span
                          key={i}
                          className={cn(
                            "w-5 h-5 flex items-center justify-center rounded text-[10px] font-bold",
                            g.result === "W" ? "bg-emerald-500/20 text-emerald-400" :
                            g.result === "L" ? "bg-red-500/20 text-red-400" :
                            "bg-muted text-muted-foreground"
                          )}
                        >
                          {g.result}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="space-y-1">
                    {form.map((g, i) => (
                      <div key={i} className="flex items-center text-[10px] font-mono gap-2">
                        <span className={cn("w-4 font-bold", g.result === "W" ? "text-emerald-400" : g.result === "L" ? "text-red-400" : "text-muted-foreground")}>{g.result}</span>
                        <span className="text-muted-foreground w-16 truncate">{TEAM_ABBREVS[g.opponent] || g.opponent}</span>
                        <span className="tabular-nums font-semibold">{g.score}-{g.opp_score}</span>
                        {g.margin != null && (
                          <span className={cn("tabular-nums", g.margin > 0 ? "text-emerald-400/60" : "text-red-400/60")}>
                            ({g.margin > 0 ? "+" : ""}{g.margin})
                          </span>
                        )}
                        {g.is_home && <span className="text-[8px] text-muted-foreground/40">H</span>}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Section: Head to Head */}
        {h2hHome && h2hHome.played > 0 && (
          <div className="border-t border-border/30 pt-4">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-3">
              Head to Head <span className="normal-case">(last 5 years · {h2hHome.played} games)</span>
            </p>
            <div className="px-3 py-2.5 rounded-lg border border-border/30 bg-card/30 space-y-2">
              {/* Record line */}
              <div className="flex items-center gap-2 flex-wrap">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: homeColor }} />
                <span className="text-[11px] font-mono font-semibold">{homeAbbr}</span>
                <span className="text-xs font-mono font-bold tabular-nums">{h2hHome.wins}</span>
                <span className="text-[10px] text-muted-foreground/50">-</span>
                <span className="text-xs font-mono font-bold tabular-nums">{h2hHome.losses}</span>
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: awayColor }} />
                <span className="text-[11px] font-mono font-semibold">{awayAbbr}</span>
              </div>
              {/* Stats line */}
              <div className="flex gap-4 text-[10px] text-muted-foreground flex-wrap">
                {h2hHome.avg_score != null && h2hHome.avg_conceded != null && (
                  <span>
                    Avg score{" "}
                    <span className="font-mono font-semibold" style={{ color: homeColor }}>{h2hHome.avg_score}</span>
                    <span className="text-muted-foreground/40"> - </span>
                    <span className="font-mono font-semibold" style={{ color: awayColor }}>{h2hHome.avg_conceded}</span>
                    {h2hHome.median_score != null && (
                      <span className="text-muted-foreground/50"> (med {h2hHome.median_score})</span>
                    )}
                  </span>
                )}
                {h2hHome.avg_margin != null && (
                  <span>
                    {homeAbbr} margin{" "}
                    <span className={cn("font-mono font-semibold", (h2hHome.avg_margin ?? 0) > 0 ? "text-emerald-400" : "text-red-400")}>
                      {(h2hHome.avg_margin ?? 0) > 0 ? "+" : ""}{h2hHome.avg_margin}
                    </span>
                  </span>
                )}
                {(h2hHome.at_venue_played ?? 0) > 0 && (
                  <span>
                    At venue{" "}
                    <span className="font-mono font-semibold" style={{ color: homeColor }}>{h2hHome.at_venue_avg_score}</span>
                    <span className="text-muted-foreground/40"> - </span>
                    <span className="font-mono font-semibold" style={{ color: awayColor }}>{h2hAway?.at_venue_avg_score ?? "-"}</span>
                    <span className="text-muted-foreground/50"> ({h2hHome.at_venue_played}g)</span>
                  </span>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Section: Venue — Ground Stats */}
        {(ground || (homeVenue && homeVenue.played > 0)) && (
          <div className="border-t border-border/30 pt-4">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-3">
              {context.venue_display || "Venue"} — Scoring Averages
            </p>
            {ground && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                {ground.avg_total_season != null && (
                  <div className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                    <p className="text-[10px] text-muted-foreground">Ground Avg (Season)</p>
                    <p className="text-lg font-bold tabular-nums">{ground.avg_total_season}</p>
                    {ground.median_total_season != null && (
                      <p className="text-[9px] font-mono tabular-nums text-muted-foreground/60">med {ground.median_total_season}</p>
                    )}
                    <p className="text-[10px] text-muted-foreground">total pts · {ground.total_games_season} games</p>
                  </div>
                )}
                {ground.avg_total_5y != null && (
                  <div className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                    <p className="text-[10px] text-muted-foreground">Ground Avg (5yr)</p>
                    <p className="text-lg font-bold tabular-nums">{ground.avg_total_5y}</p>
                    {ground.median_total_5y != null && (
                      <p className="text-[9px] font-mono tabular-nums text-muted-foreground/60">med {ground.median_total_5y}</p>
                    )}
                    <p className="text-[10px] text-muted-foreground">total pts · {ground.total_games_5y} games</p>
                  </div>
                )}
                {(ground.highest_total_5y != null || ground.lowest_total_5y != null) && (
                  <div className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                    <p className="text-[10px] text-muted-foreground">Hi / Lo (5yr)</p>
                    <p className="text-base font-bold tabular-nums">
                      <span className="text-emerald-400">{ground.highest_total_5y ?? "\u2014"}</span>
                      <span className="text-muted-foreground/40 mx-1 text-sm">/</span>
                      <span className="text-red-400">{ground.lowest_total_5y ?? "\u2014"}</span>
                    </p>
                    <p className="text-[10px] text-muted-foreground">total pts</p>
                  </div>
                )}
                {ground.last_5_avg_total != null && (
                  <div className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                    <p className="text-[10px] text-muted-foreground">Last 5 Games</p>
                    <p className="text-lg font-bold tabular-nums">{ground.last_5_avg_total}</p>
                    {ground.last_5_median_total != null && (
                      <p className="text-[9px] font-mono tabular-nums text-muted-foreground/60">med {ground.last_5_median_total}</p>
                    )}
                    <p className="text-[10px] text-muted-foreground">
                      avg pts
                      {ground.last_5_highest != null && ground.last_5_lowest != null && (
                        <span> · <span className="text-red-400/80">{ground.last_5_lowest}</span><span className="text-muted-foreground/50">/</span><span className="text-emerald-400/80">{ground.last_5_highest}</span></span>
                      )}
                    </p>
                  </div>
                )}
              </div>
            )}
            {!ground && (
              <div className="grid grid-cols-2 gap-3 mb-3">
                {homeVenue && homeVenue.played > 0 && (
                  <div className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                    <div className="flex items-center gap-1.5 mb-1">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: homeColor }} />
                      <p className="text-[10px] text-muted-foreground">{homeAbbr} at this ground</p>
                    </div>
                    <p className="text-lg font-bold tabular-nums">{homeVenue.avg_score}<span className="text-sm text-muted-foreground">-{homeVenue.avg_conceded}</span></p>
                    <p className="text-[10px] text-muted-foreground">{homeVenue.played} games (5yr)</p>
                  </div>
                )}
                {awayVenue && awayVenue.played > 0 && (
                  <div className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                    <div className="flex items-center gap-1.5 mb-1">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: awayColor }} />
                      <p className="text-[10px] text-muted-foreground">{awayAbbr} at this ground</p>
                    </div>
                    <p className="text-lg font-bold tabular-nums">{awayVenue.avg_score}<span className="text-sm text-muted-foreground">-{awayVenue.avg_conceded}</span></p>
                    <p className="text-[10px] text-muted-foreground">{awayVenue.played} games (5yr)</p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Section: Venue — Team Record at Venue */}
        {((homeVenue && homeVenue.played > 0) || (awayVenue && awayVenue.played > 0)) && (
          <div className="border-t border-border/30 pt-4">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-3">
              Team Record at {context.venue_display || "This Venue"} <span className="normal-case">(last 5 years)</span>
            </p>
            <div className="space-y-2.5">
              {homeVenue && homeVenue.played > 0 && (
                <div className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: homeColor }} />
                    <span className="text-[11px] font-mono font-semibold">{homeAbbr}</span>
                    <RecordBadge wins={homeVenue.wins} losses={homeVenue.losses} />
                    <span className="text-[10px] text-muted-foreground">
                      ({homeVenue.played} games)
                    </span>
                  </div>
                  <div className="flex gap-3 text-[10px] text-muted-foreground ml-4">
                    <span>Avg <span className="font-mono font-semibold text-foreground/80">{homeVenue.avg_score}-{homeVenue.avg_conceded}</span>{homeVenue.median_score != null && <span className="text-muted-foreground/50"> (med {homeVenue.median_score})</span>}</span>
                    {homeVenue.avg_margin != null && (
                      <span>Margin <span className={cn("font-mono font-semibold", (homeVenue.avg_margin ?? 0) > 0 ? "text-emerald-400" : "text-red-400")}>{(homeVenue.avg_margin ?? 0) > 0 ? "+" : ""}{homeVenue.avg_margin}</span></span>
                    )}
                    {homeVenue.season_avg_score != null && (
                      <span>This yr <span className="font-mono font-semibold text-foreground/80">{homeVenue.season_avg_score}</span> <span className="text-muted-foreground/50">({homeVenue.season_played}g)</span></span>
                    )}
                  </div>
                </div>
              )}
              {awayVenue && awayVenue.played > 0 && (
                <div className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: awayColor }} />
                    <span className="text-[11px] font-mono font-semibold">{awayAbbr}</span>
                    <RecordBadge wins={awayVenue.wins} losses={awayVenue.losses} />
                    <span className="text-[10px] text-muted-foreground">
                      ({awayVenue.played} games)
                    </span>
                  </div>
                  <div className="flex gap-3 text-[10px] text-muted-foreground ml-4">
                    <span>Avg <span className="font-mono font-semibold text-foreground/80">{awayVenue.avg_score}-{awayVenue.avg_conceded}</span>{awayVenue.median_score != null && <span className="text-muted-foreground/50"> (med {awayVenue.median_score})</span>}</span>
                    {awayVenue.avg_margin != null && (
                      <span>Margin <span className={cn("font-mono font-semibold", (awayVenue.avg_margin ?? 0) > 0 ? "text-emerald-400" : "text-red-400")}>{(awayVenue.avg_margin ?? 0) > 0 ? "+" : ""}{awayVenue.avg_margin}</span></span>
                    )}
                    {awayVenue.season_avg_score != null && (
                      <span>This yr <span className="font-mono font-semibold text-foreground/80">{awayVenue.season_avg_score}</span> <span className="text-muted-foreground/50">({awayVenue.season_played}g)</span></span>
                    )}
                  </div>
                </div>
              )}
              {awayVenue && awayVenue.played === 0 && (
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: awayColor }} />
                  <span className="text-[11px] font-mono font-semibold w-8">{awayAbbr}</span>
                  <span className="text-[10px] text-muted-foreground italic">No games at this venue</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Section: Venue — Total Score Distribution */}
        {ground?.total_score_distribution && ground.total_score_distribution.brackets.length > 0 && (() => {
          const dist = ground.total_score_distribution!;
          const brackets = showAllBrackets ? dist.brackets : dist.brackets.slice(0, 7);
          return (
            <div className="border-t border-border/30 pt-4">
              <div className="flex items-center justify-between mb-3">
                <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
                  Total — Match Score Brackets
                </p>
                <p className="text-[9px] text-muted-foreground/50 font-mono">
                  based on {dist.sample_size} games (5yr) at this ground
                </p>
              </div>
              <div className="rounded-lg border border-border/30 overflow-hidden">
                {/* Header */}
                <div className="grid grid-cols-[1fr_auto_auto] border-b border-border/30 bg-muted/20">
                  <div className="px-3 py-1.5" />
                  <div className="px-4 py-1.5 text-[10px] font-medium text-muted-foreground text-right w-28">or More</div>
                  <div className="px-4 py-1.5 text-[10px] font-medium text-muted-foreground text-right w-28">Under</div>
                </div>
                {/* Rows */}
                {brackets.map((b: TotalScoreBracket, i: number) => {
                  const overColor = b.p_over >= 70 ? "text-emerald-400" : b.p_over >= 40 ? "text-foreground" : "text-red-400";
                  const underColor = b.p_under >= 70 ? "text-emerald-400" : b.p_under >= 40 ? "text-foreground" : "text-red-400";
                  return (
                    <div
                      key={b.threshold}
                      className={cn(
                        "grid grid-cols-[1fr_auto_auto] border-b border-border/20 last:border-0",
                        i % 2 === 0 ? "bg-background" : "bg-muted/10"
                      )}
                    >
                      <div className="px-3 py-2.5 text-xs font-semibold">{b.threshold} Points</div>
                      <div className={cn("px-4 py-2.5 text-sm font-bold tabular-nums text-right w-28 font-mono", overColor)}>
                        {b.p_over.toFixed(1)}%
                      </div>
                      <div className={cn("px-4 py-2.5 text-sm font-bold tabular-nums text-right w-28 font-mono", underColor)}>
                        {b.p_under.toFixed(1)}%
                      </div>
                    </div>
                  );
                })}
              </div>
              {dist.brackets.length > 7 && (
                <button
                  className="w-full mt-2 py-2 text-[11px] text-muted-foreground hover:text-foreground font-mono flex items-center justify-center gap-1 transition-colors"
                  onClick={() => setShowAllBrackets((v) => !v)}
                >
                  {showAllBrackets ? "Show less \u2227" : `Show more \u2228`}
                </button>
              )}
            </div>
          );
        })()}

        {/* Row 4c: Team Score Distribution */}
        {context.team_score_distribution && (() => {
          const tsd = context.team_score_distribution!;
          const home = tsd.home;
          const away = tsd.away;
          if (!home && !away) return null;

          // Align brackets by threshold
          const allThresholds = Array.from(new Set([
            ...(home?.brackets.map(b => b.threshold) ?? []),
            ...(away?.brackets.map(b => b.threshold) ?? []),
          ])).sort((a, b) => a - b);

          const homeLookup = Object.fromEntries(home?.brackets.map(b => [b.threshold, b]) ?? []);
          const awayLookup = Object.fromEntries(away?.brackets.map(b => [b.threshold, b]) ?? []);

          const rows = showAllTeamBrackets ? allThresholds : allThresholds.slice(0, 6);

          const homeAbbr2 = TEAM_ABBREVS[tsd.home_team] || tsd.home_team;
          const awayAbbr2 = TEAM_ABBREVS[tsd.away_team] || tsd.away_team;

          return (
            <div className="border-t border-border/30 pt-4">
              <div className="flex items-center justify-between mb-3">
                <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
                  Team Totals
                </p>
                <p className="text-[9px] text-muted-foreground/50 font-mono">
                  all venues (5yr)
                  {home && <span> · {homeAbbr2} {home.sample_size}g</span>}
                  {away && <span> · {awayAbbr2} {away.sample_size}g</span>}
                </p>
              </div>
              <div className="rounded-lg border border-border/30 overflow-hidden">
                {/* Team header */}
                <div className="grid grid-cols-[1fr_1fr_1fr_1fr_1fr] border-b border-border/40 bg-muted/30">
                  <div />
                  <div className="col-span-2 text-center py-2 text-[10px] font-semibold text-foreground/80 border-r border-border/30">
                    {tsd.home_team}
                    {home && <span className="text-muted-foreground font-normal ml-1">(avg {home.avg_score}{home.median_score != null ? ` · med ${home.median_score}` : ""})</span>}
                  </div>
                  <div className="col-span-2 text-center py-2 text-[10px] font-semibold text-foreground/80">
                    {tsd.away_team}
                    {away && <span className="text-muted-foreground font-normal ml-1">(avg {away.avg_score}{away.median_score != null ? ` · med ${away.median_score}` : ""})</span>}
                  </div>
                </div>
                {/* Sub-header */}
                <div className="grid grid-cols-[1fr_1fr_1fr_1fr_1fr] border-b border-border/30 bg-muted/10">
                  <div className="px-3 py-1" />
                  <div className="py-1 text-[9px] text-muted-foreground text-center">or More</div>
                  <div className="py-1 text-[9px] text-muted-foreground text-center border-r border-border/30">Under</div>
                  <div className="py-1 text-[9px] text-muted-foreground text-center">or More</div>
                  <div className="py-1 text-[9px] text-muted-foreground text-center">Under</div>
                </div>
                {/* Rows */}
                {rows.map((t, i) => {
                  const hb = homeLookup[t];
                  const ab = awayLookup[t];
                  const pColor = (p: number) =>
                    p >= 70 ? "text-emerald-400" : p >= 40 ? "text-foreground" : "text-red-400/80";
                  return (
                    <div
                      key={t}
                      className={cn(
                        "grid grid-cols-[1fr_1fr_1fr_1fr_1fr] border-b border-border/20 last:border-0",
                        i % 2 === 0 ? "bg-background" : "bg-muted/10"
                      )}
                    >
                      <div className="px-3 py-2.5 text-xs font-semibold">{t} Points</div>
                      <div className={cn("py-2.5 text-sm font-bold tabular-nums text-center font-mono", hb ? pColor(hb.p_over) : "text-muted-foreground/30")}>
                        {hb ? `${hb.p_over.toFixed(1)}%` : "\u2014"}
                      </div>
                      <div className={cn("py-2.5 text-sm font-bold tabular-nums text-center font-mono border-r border-border/20", hb ? pColor(hb.p_under) : "text-muted-foreground/30")}>
                        {hb ? `${hb.p_under.toFixed(1)}%` : "\u2014"}
                      </div>
                      <div className={cn("py-2.5 text-sm font-bold tabular-nums text-center font-mono", ab ? pColor(ab.p_over) : "text-muted-foreground/30")}>
                        {ab ? `${ab.p_over.toFixed(1)}%` : "\u2014"}
                      </div>
                      <div className={cn("py-2.5 text-sm font-bold tabular-nums text-center font-mono", ab ? pColor(ab.p_under) : "text-muted-foreground/30")}>
                        {ab ? `${ab.p_under.toFixed(1)}%` : "\u2014"}
                      </div>
                    </div>
                  );
                })}
              </div>
              {allThresholds.length > 6 && (
                <button
                  className="w-full mt-2 py-2 text-[11px] text-muted-foreground hover:text-foreground font-mono flex items-center justify-center gap-1 transition-colors"
                  onClick={() => setShowAllTeamBrackets(v => !v)}
                >
                  {showAllTeamBrackets ? "Show less \u2227" : "Show more \u2228"}
                </button>
              )}
            </div>
          );
        })()}

        {/* Section: Weather Impact on Players */}
        {weatherImpact && weatherImpact.length > 0 && (
          <div className="border-t border-border/30 pt-4">
            <div className="flex items-center justify-between mb-3">
              <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
                Player Impact in {condLabel} Conditions
              </p>
              <p className="text-[9px] text-muted-foreground/40 font-mono">Based on historical stats in similar conditions</p>
              <button
                onClick={() => setShowWeatherTable(!showWeatherTable)}
                className="text-[11px] font-mono text-primary hover:text-primary/80 transition-colors px-2 py-1 rounded hover:bg-primary/5"
              >
                {showWeatherTable ? "Hide Details" : "View All Players"}
              </button>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="px-3 py-2.5 rounded-lg border border-border/30 bg-card/30">
                <div className="flex items-center gap-2 mb-2">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: homeColor }} />
                  <span className="text-xs font-semibold font-mono">{homeAbbr}</span>
                  <span className="text-[10px] text-muted-foreground">({homeImpact.length} players)</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                    <span className="text-sm font-bold tabular-nums">{homeFavored}</span>
                    <span className="text-[10px] text-muted-foreground">favored</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
                    <span className="text-sm font-bold tabular-nums">{homeImpact.length - homeFavored}</span>
                    <span className="text-[10px] text-muted-foreground">hindered</span>
                  </div>
                </div>
              </div>
              <div className="px-3 py-2.5 rounded-lg border border-border/30 bg-card/30">
                <div className="flex items-center gap-2 mb-2">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: awayColor }} />
                  <span className="text-xs font-semibold font-mono">{awayAbbr}</span>
                  <span className="text-[10px] text-muted-foreground">({awayImpact.length} players)</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                    <span className="text-sm font-bold tabular-nums">{awayFavored}</span>
                    <span className="text-[10px] text-muted-foreground">favored</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
                    <span className="text-sm font-bold tabular-nums">{awayImpact.length - awayFavored}</span>
                    <span className="text-[10px] text-muted-foreground">hindered</span>
                  </div>
                </div>
              </div>
            </div>

            {showWeatherTable && (
              <div className="mt-3 overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Player</TableHead>
                      <TableHead className="text-right">Goals Diff</TableHead>
                      <TableHead className="text-right">Disp Diff</TableHead>
                      <TableHead className="text-right">Marks Diff</TableHead>
                      <TableHead className="text-right">Sample</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {weatherImpact.map((p, i) => (
                      <TableRow key={i}>
                        <TableCell className="font-medium text-sm">
                          <span className="flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: TEAM_COLORS[p.team]?.primary || "#555" }} />
                            {p.player}
                          </span>
                        </TableCell>
                        <TableCell className="text-right">
                          <span className={cn("tabular-nums font-mono text-sm font-bold", p.gl_diff > 0 ? "text-emerald-400" : p.gl_diff < 0 ? "text-red-400" : "text-muted-foreground")}>
                            {p.gl_diff > 0 ? "+" : ""}{p.gl_diff.toFixed(2)}
                          </span>
                        </TableCell>
                        <TableCell className="text-right">
                          <span className={cn("tabular-nums font-mono text-sm font-bold", p.di_diff > 0 ? "text-emerald-400" : p.di_diff < 0 ? "text-red-400" : "text-muted-foreground")}>
                            {p.di_diff > 0 ? "+" : ""}{p.di_diff.toFixed(1)}
                          </span>
                        </TableCell>
                        <TableCell className="text-right">
                          <span className={cn("tabular-nums font-mono text-sm font-bold", p.mk_diff > 0 ? "text-emerald-400" : p.mk_diff < 0 ? "text-red-400" : "text-muted-foreground")}>
                            {p.mk_diff > 0 ? "+" : ""}{p.mk_diff.toFixed(1)}
                          </span>
                        </TableCell>
                        <TableCell className="text-right text-xs text-muted-foreground tabular-nums">
                          {p.condition_games}/{p.total_games}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </div>
        )}

        {/* Rest Days */}
        {(context.home_rest_days != null || context.away_rest_days != null) && (
          <div className="border-t border-border/30 pt-4">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-3">Rest Days</p>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: homeAbbr, color: homeColor, days: context.home_rest_days },
                { label: awayAbbr, color: awayColor, days: context.away_rest_days },
              ].map(({ label, color, days }) => days != null && (
                <div key={label} className="px-3 py-2 rounded-lg border border-border/30 bg-card/30 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                  <span className="text-xs font-semibold">{label}</span>
                  <span className={cn("text-lg font-bold tabular-nums font-mono ml-auto", days <= 5 ? "text-red-400" : days >= 8 ? "text-emerald-400" : "text-foreground")}>
                    {days}
                  </span>
                  <span className="text-[10px] text-muted-foreground">days</span>
                </div>
              ))}
            </div>

            {/* Historical Rest Days Impact */}
            {(context.home_rest_impact || context.away_rest_impact) && (
              <div className="mt-3">
                <p className="text-[9px] font-medium text-muted-foreground/60 uppercase tracking-wider mb-2">Historical Impact (last 3 years)</p>
                <div className="rounded-lg border border-border/30 overflow-hidden">
                  <div className="grid grid-cols-[auto_1fr_1fr_1fr] border-b border-border/30 bg-muted/20">
                    <div className="px-3 py-1.5 w-16" />
                    {(["short", "normal", "extended"] as const).map(bucket => (
                      <div key={bucket} className="px-2 py-1.5 text-[9px] font-medium text-muted-foreground text-center">
                        {bucket === "short" ? "Short (\u22645d)" : bucket === "normal" ? "Normal (6-7d)" : "Extended (8d+)"}
                      </div>
                    ))}
                  </div>
                  {[
                    { label: homeAbbr, color: homeColor, impact: context.home_rest_impact },
                    { label: awayAbbr, color: awayColor, impact: context.away_rest_impact },
                  ].map(({ label, color, impact }) => impact && (
                    <div key={label} className="grid grid-cols-[auto_1fr_1fr_1fr] border-b border-border/20 last:border-0">
                      <div className="px-3 py-2 w-16 flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                        <span className="text-[10px] font-semibold">{label}</span>
                      </div>
                      {(["short", "normal", "extended"] as const).map(bucket => {
                        const b = impact[bucket];
                        if (!b) return <div key={bucket} className="px-2 py-2 text-center text-[9px] text-muted-foreground/30">\u2014</div>;
                        const winPct = b.played > 0 ? (b.wins / b.played * 100) : 0;
                        return (
                          <div key={bucket} className="px-2 py-2 text-center">
                            <span className="text-[10px] font-bold font-mono tabular-nums">{b.wins}W-{b.losses}L</span>
                            <span className={cn("text-[9px] font-mono ml-1", winPct >= 55 ? "text-emerald-400/70" : winPct < 45 ? "text-red-400/70" : "text-muted-foreground/50")}>
                              {winPct.toFixed(0)}%
                            </span>
                            <p className="text-[8px] font-mono text-muted-foreground/40 tabular-nums">{b.avg_score}-{b.avg_conceded}</p>
                          </div>
                        );
                      })}
                    </div>
                  ))}
                </div>

                {/* League-wide comparison */}
                {context.league_rest_impact && (
                  <div className="mt-2 flex gap-2">
                    {(["short", "normal", "extended"] as const).map(bucket => {
                      const b = context.league_rest_impact?.[bucket];
                      if (!b) return null;
                      return (
                        <div key={bucket} className="flex-1 px-2 py-1.5 rounded border border-border/20 bg-muted/10 text-center">
                          <p className="text-[8px] text-muted-foreground/50 uppercase">{bucket} rest (league)</p>
                          <p className="text-[10px] font-mono font-bold tabular-nums">
                            {b.win_pct}% <span className="text-muted-foreground/40 font-normal">win</span>
                          </p>
                          <p className="text-[8px] font-mono text-muted-foreground/40 tabular-nums">avg {b.avg_score} pts · {b.played}g</p>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Section: Season Stat Matchup */}
        {(context.home_stats || context.away_stats) && (
          <div className="border-t border-border/30 pt-4">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-3">Season Stat Matchup</p>
            <div className="rounded-lg border border-border/30 overflow-hidden">
              <div className="grid grid-cols-[1fr_auto_auto_auto] border-b border-border/30 bg-muted/20">
                <div className="px-3 py-1.5 text-[10px] text-muted-foreground">Stat</div>
                <div className="px-3 py-1.5 text-[10px] text-muted-foreground text-right w-20">
                  <span className="inline-flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: homeColor }} />{homeAbbr}</span>
                </div>
                <div className="px-3 py-1.5 text-[10px] text-muted-foreground text-right w-20">
                  <span className="inline-flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: awayColor }} />{awayAbbr}</span>
                </div>
                <div className="px-3 py-1.5 text-[10px] text-muted-foreground text-center w-14">Edge</div>
              </div>
              {[
                { label: "Disposals", key: "di" },
                { label: "Marks", key: "mk" },
                { label: "Tackles", key: "tk" },
                { label: "Clearances", key: "cl" },
                { label: "Contested Poss", key: "cp" },
                { label: "Inside 50s", key: "if" },
                { label: "Rebound 50s", key: "rb" },
                { label: "Goals", key: "gl" },
              ].map(({ label, key }, i) => {
                const hv = context.home_stats?.[`avg_${key}` as keyof TeamStatMatchup] as number | undefined;
                const av = context.away_stats?.[`avg_${key}` as keyof TeamStatMatchup] as number | undefined;
                if (hv == null && av == null) return null;
                const diff = hv != null && av != null ? hv - av : 0;
                const hMin = context.home_stats?.[`min_${key}` as keyof TeamStatMatchup] as number | undefined;
                const hMed = context.home_stats?.[`median_${key}` as keyof TeamStatMatchup] as number | undefined;
                const hMax = context.home_stats?.[`max_${key}` as keyof TeamStatMatchup] as number | undefined;
                const aMin = context.away_stats?.[`min_${key}` as keyof TeamStatMatchup] as number | undefined;
                const aMed = context.away_stats?.[`median_${key}` as keyof TeamStatMatchup] as number | undefined;
                const aMax = context.away_stats?.[`max_${key}` as keyof TeamStatMatchup] as number | undefined;
                return (
                  <div
                    key={key}
                    className={cn("grid grid-cols-[1fr_auto_auto_auto] border-b border-border/20 last:border-0", i % 2 === 0 ? "bg-background" : "bg-muted/10")}
                  >
                    <div className="px-3 py-2 text-xs">{label}</div>
                    <div className={cn("px-3 py-2 text-right w-20", diff > 0 ? "text-emerald-400" : "text-foreground")}>
                      <span className="text-sm font-bold font-mono tabular-nums">{hv?.toFixed(1) ?? "\u2014"}</span>
                      {hMin != null && <p className="text-[8px] font-mono tabular-nums text-muted-foreground/50">{hMin}\u2013{hMed}\u2013{hMax}</p>}
                    </div>
                    <div className={cn("px-3 py-2 text-right w-20", diff < 0 ? "text-emerald-400" : "text-foreground")}>
                      <span className="text-sm font-bold font-mono tabular-nums">{av?.toFixed(1) ?? "\u2014"}</span>
                      {aMin != null && <p className="text-[8px] font-mono tabular-nums text-muted-foreground/50">{aMin}\u2013{aMed}\u2013{aMax}</p>}
                    </div>
                    <div className="px-3 py-2 text-center w-14">
                      {diff !== 0 && (
                        <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: diff > 0 ? homeColor : awayColor }} />
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* === ADVANCED VIEW SECTIONS === */}
        {contextView === "advanced" && (
          <>
            {/* Quarter Scoring */}
            {(context.home_quarters || context.away_quarters) && (
              <div className="border-t border-border/30 pt-4">
                <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-3">Avg Scoring by Quarter</p>
                <div className="rounded-lg border border-border/30 overflow-hidden">
                  <div className="grid grid-cols-[auto_1fr_1fr_1fr_1fr] border-b border-border/30 bg-muted/20">
                    <div className="px-3 py-1.5 w-20" />
                    {["Q1", "Q2", "Q3", "Q4"].map(q => (
                      <div key={q} className="px-3 py-1.5 text-[10px] font-medium text-muted-foreground text-center">{q}</div>
                    ))}
                  </div>
                  {[
                    { label: homeAbbr, color: homeColor, q: context.home_quarters },
                    { label: awayAbbr, color: awayColor, q: context.away_quarters },
                  ].map(({ label, color, q }) => q && (
                    <div key={label} className="grid grid-cols-[auto_1fr_1fr_1fr_1fr] border-b border-border/20 last:border-0">
                      <div className="px-3 py-2 w-20 flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                        <span className="text-xs font-semibold">{label}</span>
                      </div>
                      {[q.q1, q.q2, q.q3, q.q4].map((qd, i) => (
                        <div key={i} className="px-2 py-2 text-center">
                          <span className="text-sm font-bold font-mono tabular-nums">{qd?.avg_points?.toFixed(1) ?? "\u2014"}</span>
                          <span className="text-[9px] text-muted-foreground/50 font-mono block">
                            {qd ? `${qd.avg_goals.toFixed(1)}g ${qd.avg_behinds.toFixed(1)}b` : ""}
                          </span>
                          {qd?.min_points != null && (
                            <span className="text-[8px] font-mono tabular-nums text-muted-foreground/40 block">
                              {qd.min_points}\u2013{qd.median_points}\u2013{qd.max_points}
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Coaches */}
            {comparison?.coaches && (comparison.coaches.home || comparison.coaches.away) && (
              <div className="border-t border-border/30 pt-4">
                <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-3">Coaches</p>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: homeAbbr, color: homeColor, coach: comparison.coaches.home },
                    { label: awayAbbr, color: awayColor, coach: comparison.coaches.away },
                  ].map(({ label, color, coach }) => coach && (
                    <div key={label} className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                        <span className="text-xs font-semibold">{label}</span>
                      </div>
                      <p className="text-sm font-bold">{coach.name}</p>
                      <div className="flex gap-3 text-[10px] text-muted-foreground mt-1">
                        {coach.wins != null && coach.losses != null && (
                          <span className="font-mono">{coach.wins}W-{coach.losses}L</span>
                        )}
                        {coach.win_pct != null && (
                          <span className="font-mono">{coach.win_pct.toFixed(1)}%</span>
                        )}
                        {coach.career_games != null && (
                          <span>{coach.career_games} games</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Umpires */}
            {comparison?.umpires && comparison.umpires.length > 0 && (
              <div className="border-t border-border/30 pt-4">
                <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-3">Umpires</p>
                <div className="flex gap-2 flex-wrap">
                  {comparison.umpires.map((u, i) => (
                    <div key={i} className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                      <p className="text-xs font-semibold">{u.name}</p>
                      {u.career_games != null && (
                        <p className="text-[10px] text-muted-foreground font-mono">{u.career_games} career games</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Odds */}
            {comparison?.odds && Object.keys(comparison.odds).length > 0 && (
              <div className="border-t border-border/30 pt-4">
                <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-3">Market Odds</p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {comparison.odds.market_home_implied_prob != null && (
                    <div className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                      <p className="text-[10px] text-muted-foreground">{homeAbbr} Market Prob</p>
                      <p className="text-lg font-bold font-mono tabular-nums">{(comparison.odds.market_home_implied_prob * 100).toFixed(1)}%</p>
                    </div>
                  )}
                  {comparison.odds.market_away_implied_prob != null && (
                    <div className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                      <p className="text-[10px] text-muted-foreground">{awayAbbr} Market Prob</p>
                      <p className="text-lg font-bold font-mono tabular-nums">{(comparison.odds.market_away_implied_prob * 100).toFixed(1)}%</p>
                    </div>
                  )}
                  {comparison.odds.market_handicap != null && (
                    <div className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                      <p className="text-[10px] text-muted-foreground">Handicap</p>
                      <p className="text-lg font-bold font-mono tabular-nums">{comparison.odds.market_handicap > 0 ? "+" : ""}{comparison.odds.market_handicap.toFixed(1)}</p>
                    </div>
                  )}
                  {comparison.odds.market_total_score != null && (
                    <div className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                      <p className="text-[10px] text-muted-foreground">Total Score Line</p>
                      <p className="text-lg font-bold font-mono tabular-nums">{comparison.odds.market_total_score.toFixed(1)}</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Advanced Team Stats (FootyWire) */}
            {comparison?.advanced_stats && (comparison.advanced_stats.home || comparison.advanced_stats.away) && (
              <div className="border-t border-border/30 pt-4">
                <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-3">
                  Advanced Stats <span className="normal-case text-muted-foreground/40">(FootyWire)</span>
                </p>
                <div className="rounded-lg border border-border/30 overflow-hidden">
                  <div className="grid grid-cols-[1fr_auto_auto] border-b border-border/30 bg-muted/20">
                    <div className="px-3 py-1.5 text-[10px] text-muted-foreground">Stat</div>
                    <div className="px-3 py-1.5 text-[10px] text-muted-foreground text-right w-20">
                      <span className="inline-flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: homeColor }} />{homeAbbr}</span>
                    </div>
                    <div className="px-3 py-1.5 text-[10px] text-muted-foreground text-right w-20">
                      <span className="inline-flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: awayColor }} />{awayAbbr}</span>
                    </div>
                  </div>
                  {[
                    { label: "Eff. Disposals", key: "ed" as const },
                    { label: "Disp. Eff %", key: "de_pct" as const },
                    { label: "Centre CL", key: "ccl" as const },
                    { label: "Stoppage CL", key: "scl" as const },
                    { label: "Turnovers", key: "to" as const },
                    { label: "Metres Gained", key: "mg" as const },
                    { label: "Score Involvements", key: "si" as const },
                    { label: "Intercepts", key: "itc" as const },
                    { label: "Tackle in 50", key: "t5" as const },
                    { label: "TOG %", key: "tog_pct" as const },
                  ].map(({ label, key }, i) => {
                    const hv = comparison.advanced_stats?.home?.[key];
                    const av = comparison.advanced_stats?.away?.[key];
                    if (hv == null && av == null) return null;
                    return (
                      <div key={key} className={cn("grid grid-cols-[1fr_auto_auto] border-b border-border/20 last:border-0", i % 2 === 0 ? "bg-background" : "bg-muted/10")}>
                        <div className="px-3 py-2 text-xs">{label}</div>
                        <div className="px-3 py-2 text-sm font-bold font-mono tabular-nums text-right w-20">{hv?.toFixed(1) ?? "\u2014"}</div>
                        <div className="px-3 py-2 text-sm font-bold font-mono tabular-nums text-right w-20">{av?.toFixed(1) ?? "\u2014"}</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Attendance Stats */}
            {comparison?.attendance_stats && comparison.attendance_stats.avg_attendance != null && (
              <div className="border-t border-border/30 pt-4">
                <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-3">Venue Attendance</p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                    <p className="text-[10px] text-muted-foreground">Avg</p>
                    <p className="text-lg font-bold tabular-nums">{comparison.attendance_stats.avg_attendance?.toLocaleString()}</p>
                    {comparison.attendance_stats.median_attendance != null && (
                      <p className="text-[9px] font-mono tabular-nums text-muted-foreground/60">med {comparison.attendance_stats.median_attendance.toLocaleString()}</p>
                    )}
                  </div>
                  {comparison.attendance_stats.max_attendance != null && (
                    <div className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                      <p className="text-[10px] text-muted-foreground">Max</p>
                      <p className="text-lg font-bold tabular-nums">{comparison.attendance_stats.max_attendance.toLocaleString()}</p>
                    </div>
                  )}
                  {comparison.attendance_stats.last_5_avg != null && (
                    <div className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                      <p className="text-[10px] text-muted-foreground">Last 5 Avg</p>
                      <p className="text-lg font-bold tabular-nums">{comparison.attendance_stats.last_5_avg.toLocaleString()}</p>
                    </div>
                  )}
                  {comparison.attendance_stats.total_games != null && (
                    <div className="px-3 py-2 rounded-lg border border-border/30 bg-card/30">
                      <p className="text-[10px] text-muted-foreground">Games (3yr)</p>
                      <p className="text-lg font-bold tabular-nums">{comparison.attendance_stats.total_games}</p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
