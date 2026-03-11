"use client";

import { useEffect, useState } from "react";
import { useParams, useSearchParams } from "next/navigation";
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
import { getMatchDetail, getMatchComparison, getMatchSimulation } from "@/lib/api";
import type { MatchDetail, MatchComparison, MatchComparisonPlayer, MatchSimulation } from "@/lib/types";
import { TEAM_ABBREVS, TEAM_COLORS, CURRENT_YEAR, displayVenue } from "@/lib/constants";
import { cn, formatDate } from "@/lib/utils";
import Link from "next/link";
import { Breadcrumb } from "@/components/ui/breadcrumb";
import { MatchContextCard } from "@/components/matches/MatchContextCard";
import { BettingMarketsCard } from "@/components/matches/BettingMarkets";
import { PlayerComparisonTable, PlayerAdvancedTable, predColor } from "@/components/matches/PlayerComparison";
import { MonteCarloCard } from "@/components/matches/MonteCarloCard";
import { ExportButton } from "@/components/ui/export-button";

function DiffCell({ actual, predicted, decimals = 1 }: { actual?: number; predicted?: number; decimals?: number }) {
  if (actual == null && predicted == null)
    return <span className="text-muted-foreground">-</span>;
  const color = (actual != null && predicted != null)
    ? predColor(actual, predicted)
    : "";
  return (
    <div className="flex items-baseline gap-1.5 justify-end">
      <span className={cn("font-bold tabular-nums text-sm", color)}>
        {actual ?? "-"}
      </span>
      <span className="text-[10px] text-muted-foreground/40 font-mono">/</span>
      <span className="text-[11px] tabular-nums text-muted-foreground/60 font-mono">
        {predicted != null ? predicted.toFixed(decimals) : "-"}
      </span>
    </div>
  );
}

export default function MatchDetailPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const matchId = Number(params.matchId);
  const roundNumber = searchParams.get('round') ? Number(searchParams.get('round')) : undefined;
  const homeTeamParam = searchParams.get('home') || undefined;
  const awayTeamParam = searchParams.get('away') || undefined;
  const [match, setMatch] = useState<MatchDetail | null>(null);
  const [comparison, setComparison] = useState<MatchComparison | null>(null);
  const [simulation, setSimulation] = useState<MatchSimulation | null>(null);
  const [simLoading, setSimLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<"post-match" | "prediction">("post-match");
  const [teamView, setTeamView] = useState<"split" | "combined">("split");
  const [playerView, setPlayerView] = useState<"standard" | "advanced">("standard");

  useEffect(() => {
    setLoading(true);
    Promise.all([
      matchId ? getMatchDetail(matchId).catch(() => null) : Promise.resolve(null),
      getMatchComparison(CURRENT_YEAR, matchId, roundNumber, homeTeamParam, awayTeamParam).catch(() =>
        getMatchComparison(CURRENT_YEAR - 1, matchId, roundNumber, homeTeamParam, awayTeamParam).catch(() =>
          getMatchComparison(CURRENT_YEAR - 2, matchId, roundNumber, homeTeamParam, awayTeamParam).catch(() => null)
        )
      ),
    ])
      .then(([m, c]) => {
        setMatch(m);
        setComparison(c);
      })
      .finally(() => setLoading(false));
  }, [matchId, roundNumber, homeTeamParam, awayTeamParam]);

  // Lazy-load Monte Carlo simulation
  useEffect(() => {
    if (!matchId) return;
    setSimLoading(true);
    getMatchSimulation(matchId)
      .then(setSimulation)
      .catch(() => setSimulation(null))
      .finally(() => setSimLoading(false));
  }, [matchId]);

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-48 w-full" />
        <Skeleton className="h-24 w-full" />
        <div className="grid grid-cols-2 gap-4">
          <Skeleton className="h-96" />
          <Skeleton className="h-96" />
        </div>
      </div>
    );
  }

  if (!match && !comparison) {
    if (homeTeamParam && awayTeamParam) {
      const hAbbr = TEAM_ABBREVS[homeTeamParam] || homeTeamParam;
      const aAbbr = TEAM_ABBREVS[awayTeamParam] || awayTeamParam;
      const hColor = TEAM_COLORS[homeTeamParam]?.primary || "#3b82f6";
      const aColor = TEAM_COLORS[awayTeamParam]?.primary || "#ef4444";
      return (
        <div className="space-y-4">
          <Card>
            <CardContent className="pt-6 pb-6">
              <div className="flex items-center justify-center gap-6">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full" style={{ backgroundColor: hColor }} />
                  <span className="text-lg font-bold font-mono">{hAbbr}</span>
                </div>
                <span className="text-muted-foreground text-sm">vs</span>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full" style={{ backgroundColor: aColor }} />
                  <span className="text-lg font-bold font-mono">{aAbbr}</span>
                </div>
              </div>
              {roundNumber != null && (
                <p className="text-center text-sm text-muted-foreground mt-2">Round {roundNumber}</p>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6 pb-6">
              <p className="text-center text-muted-foreground">Predictions not yet available for this match.</p>
              <p className="text-center text-xs text-muted-foreground/60 mt-1">Player predictions will appear here once they are generated.</p>
            </CardContent>
          </Card>
        </div>
      );
    }
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-muted-foreground">Match not found (ID: {matchId}).</p>
        </CardContent>
      </Card>
    );
  }

  const data = comparison || match;
  const homeTeam = data!.home_team;
  const awayTeam = data!.away_team;
  const homeAbbr = TEAM_ABBREVS[homeTeam] || homeTeam;
  const awayAbbr = TEAM_ABBREVS[awayTeam] || awayTeam;
  const homeColor = TEAM_COLORS[homeTeam]?.primary || "#3b82f6";
  const awayColor = TEAM_COLORS[awayTeam]?.primary || "#ef4444";
  const homeScore = data!.home_score;
  const awayScore = data!.away_score;
  const isPlayed = homeScore != null && awayScore != null;
  const showAsPlayed = isPlayed && viewMode === "post-match";
  const homeWinProb =
    comparison?.game_prediction?.home_win_prob ??
    (match?.home_win_prob ?? null);
  const predCorrect =
    comparison?.game_prediction?.correct ?? (match?.correct ?? null);
  const predWinner =
    comparison?.game_prediction?.predicted_winner ??
    (match?.predicted_winner ?? null);
  const weather = comparison?.weather;
  const weatherSummary = comparison?.weather_summary;
  const weatherImpact = comparison?.weather_impact;
  const attendance = comparison?.attendance;
  const matchContext = comparison?.match_context;

  // Get players for comparison table
  const compPlayers = comparison?.players || [];
  const homePlayers = compPlayers.filter((p) => p.is_home);
  const awayPlayers = compPlayers.filter((p) => !p.is_home);
  const hasComparison = compPlayers.length > 0;


  // Summary stats
  const computeTeamSummary = (players: MatchComparisonPlayer[]) => ({
    actualGoals: players.reduce((s, p) => s + (p.actual_gl ?? 0), 0),
    predGoals: players.reduce((s, p) => s + (p.predicted_gl ?? 0), 0),
    actualDisp: players.reduce((s, p) => s + (p.actual_di ?? 0), 0),
    predDisp: players.reduce((s, p) => s + (p.predicted_di ?? 0), 0),
    actualMarks: players.reduce((s, p) => s + (p.actual_mk ?? 0), 0),
    predMarks: players.reduce((s, p) => s + (p.predicted_mk ?? 0), 0),
  });

  const homeSummary = computeTeamSummary(homePlayers);
  const awaySummary = computeTeamSummary(awayPlayers);

  const playerExportData = compPlayers.map((p) => ({
    Player: p.player,
    Team: p.team,
    "Predicted GL": p.predicted_gl?.toFixed(2) ?? "",
    "Actual GL": p.actual_gl != null ? String(p.actual_gl) : "",
    "Predicted DI": p.predicted_di?.toFixed(1) ?? "",
    "Actual DI": p.actual_di != null ? String(p.actual_di) : "",
    "Predicted MK": p.predicted_mk?.toFixed(1) ?? "",
    "Actual MK": p.actual_mk != null ? String(p.actual_mk) : "",
  }));

  const playerExportColumns = [
    { key: "Player", header: "Player" },
    { key: "Team", header: "Team" },
    { key: "Predicted GL", header: "Predicted GL" },
    { key: "Actual GL", header: "Actual GL" },
    { key: "Predicted DI", header: "Predicted DI" },
    { key: "Actual DI", header: "Actual DI" },
    { key: "Predicted MK", header: "Predicted MK" },
    { key: "Actual MK", header: "Actual MK" },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <Breadcrumb items={[
          { label: "Matches", href: "/matches" },
          { label: `${homeAbbr} vs ${awayAbbr}` },
        ]} />
        <h1 className="text-2xl font-bold">
          {homeAbbr} vs {awayAbbr}
        </h1>
        {!showAsPlayed && hasComparison && (
          <p className="text-sm font-medium text-primary mt-1">
            {isPlayed ? "Showing pre-match predictions" : "Pre-match predictions"}
          </p>
        )}
        <div className="flex gap-2 mt-1.5 flex-wrap">
          {data!.round_number != null && (
            <Badge variant="outline">
              Round {data!.round_number}
              {data!.year ? `, ${data!.year}` : ""}
            </Badge>
          )}
          {data!.venue && <Badge variant="outline">{displayVenue(data!.venue)}</Badge>}
          {data!.date && (
            <Badge variant="outline">
              {formatDate(data!.date)}
              {matchContext?.time && ` · ${matchContext.time}`}
              {matchContext?.day_night && ` (${matchContext.day_night})`}
            </Badge>
          )}
          {attendance != null && (
            <Badge variant="outline">{attendance.toLocaleString()} attendance</Badge>
          )}
        </div>
        {isPlayed && hasComparison && (
          <div className="flex items-center gap-1 mt-2">
            <button
              onClick={() => setViewMode("prediction")}
              className={cn(
                "px-3 py-1 text-xs font-medium rounded-l-md border transition-colors",
                viewMode === "prediction"
                  ? "bg-primary text-primary-foreground border-primary"
                  : "bg-muted/50 text-muted-foreground border-border hover:bg-muted"
              )}
            >
              Prediction
            </button>
            <button
              onClick={() => setViewMode("post-match")}
              className={cn(
                "px-3 py-1 text-xs font-medium rounded-r-md border border-l-0 transition-colors",
                viewMode === "post-match"
                  ? "bg-primary text-primary-foreground border-primary"
                  : "bg-muted/50 text-muted-foreground border-border hover:bg-muted"
              )}
            >
              Post-Match
            </button>
          </div>
        )}
      </div>

      {/* Score + Win Probability Card */}
      <Card>
        <CardContent className="pt-5 pb-5">
          <div className="flex justify-center items-center gap-10 text-center">
            <div>
              <div className="flex items-center gap-2 justify-center mb-1">
                <span
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: homeColor }}
                />
                <p className="font-bold text-lg">{homeAbbr}</p>
              </div>
              <p className="text-4xl font-bold tabular-nums">
                {showAsPlayed ? homeScore : "-"}
              </p>
            </div>
            <div className="text-center">
              <span className="text-2xl text-muted-foreground font-light">vs</span>
            </div>
            <div>
              <div className="flex items-center gap-2 justify-center mb-1">
                <span
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: awayColor }}
                />
                <p className="font-bold text-lg">{awayAbbr}</p>
              </div>
              <p className="text-4xl font-bold tabular-nums">
                {showAsPlayed ? awayScore : "-"}
              </p>
            </div>
          </div>

          {/* Win Probability Bar */}
          {homeWinProb != null && (
            <div className="mt-5 space-y-1.5 max-w-lg mx-auto">
              <p className="text-[10px] text-center text-muted-foreground/60 font-mono mb-1">Pre-match win probability</p>
              <div className="w-full h-5 rounded-full bg-muted overflow-hidden flex">
                <div
                  className="h-full transition-all"
                  style={{
                    width: `${Math.round(homeWinProb * 100)}%`,
                    backgroundColor: homeColor,
                  }}
                />
                <div
                  className="h-full transition-all"
                  style={{
                    width: `${Math.round((1 - homeWinProb) * 100)}%`,
                    backgroundColor: awayColor,
                  }}
                />
              </div>
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{homeAbbr} {Math.round(homeWinProb * 100)}%</span>
                {predWinner && (
                  <span className="flex items-center gap-1">
                    Pick: {TEAM_ABBREVS[predWinner] || predWinner}
                    {(() => {
                      const edge = Math.abs(homeWinProb - 0.5) * 100;
                      const conf = edge >= 20 ? { label: "High", cls: "bg-emerald-500/15 text-emerald-600 border-emerald-500/30" }
                        : edge >= 10 ? { label: "Medium", cls: "bg-amber-500/15 text-amber-600 border-amber-500/30" }
                        : { label: "Low", cls: "bg-red-500/15 text-red-500 border-red-500/30" };
                      return (
                        <span className={`text-[9px] px-1.5 py-0.5 rounded border font-semibold ${conf.cls}`}>
                          {conf.label}
                        </span>
                      );
                    })()}
                    {showAsPlayed && predCorrect != null && (
                      <Badge
                        variant={predCorrect ? "default" : "destructive"}
                        className="text-[10px] px-1 py-0 ml-1"
                      >
                        {predCorrect ? "Correct" : "Wrong"}
                      </Badge>
                    )}
                  </span>
                )}
                <span>{awayAbbr} {Math.round((1 - homeWinProb) * 100)}%</span>
              </div>
            </div>
          )}

          {/* Margin comparison */}
          {showAsPlayed && predWinner && (
            <div className="mt-4 text-center">
              <p className="text-sm text-muted-foreground">
                Actual margin:{" "}
                <span className="font-semibold text-foreground tabular-nums">
                  {Math.abs(homeScore! - awayScore!)} pts
                </span>
                {" "}({homeScore! > awayScore! ? homeAbbr : awayAbbr} win)
              </p>
            </div>
          )}
          {!showAsPlayed && comparison?.game_prediction?.predicted_margin != null && (
            <div className="mt-4 text-center">
              <p className="text-sm text-muted-foreground">
                Predicted margin:{" "}
                <span className="font-semibold text-foreground tabular-nums">
                  {Math.abs(comparison.game_prediction.predicted_margin).toFixed(0)} pts
                </span>
                {predWinner && (
                  <> ({TEAM_ABBREVS[predWinner] || predWinner} win)</>
                )}
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Match Context (weather, records, venue history) */}
      {matchContext && (
        <MatchContextCard
          context={matchContext}
          homeTeam={homeTeam}
          awayTeam={awayTeam}
          homeColor={homeColor}
          awayColor={awayColor}
          weather={weather}
          weatherSummary={weatherSummary}
          weatherImpact={weatherImpact}
          comparison={comparison}
        />
      )}

      {/* Team Summary Stats */}
      {hasComparison && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">
              {showAsPlayed ? "Team Totals" : "Predicted Team Totals"}
            </CardTitle>
            {showAsPlayed && (
              <p className="text-[10px] text-muted-foreground/50 font-mono mt-1">
                <span className="font-bold text-foreground/70">Actual</span> <span className="text-muted-foreground/40">/</span> <span className="text-muted-foreground/60">Predicted</span>
              </p>
            )}
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Team</TableHead>
                    <TableHead className="text-right">Goals</TableHead>
                    <TableHead className="text-right">Disposals</TableHead>
                    <TableHead className="text-right">Marks</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TableRow>
                    <TableCell className="font-medium">
                      <span className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: homeColor }} />
                        {homeAbbr}
                      </span>
                    </TableCell>
                    <TableCell className="text-right">
                      {showAsPlayed ? (
                        <DiffCell actual={homeSummary.actualGoals} predicted={homeSummary.predGoals} decimals={1} />
                      ) : (
                        <span className="font-semibold tabular-nums">{homeSummary.predGoals.toFixed(1)}</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {showAsPlayed ? (
                        <DiffCell actual={homeSummary.actualDisp} predicted={homeSummary.predDisp} />
                      ) : (
                        <span className="font-semibold tabular-nums">{Math.round(homeSummary.predDisp)}</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {showAsPlayed ? (
                        <DiffCell actual={homeSummary.actualMarks} predicted={homeSummary.predMarks} />
                      ) : (
                        <span className="font-semibold tabular-nums">{Math.round(homeSummary.predMarks)}</span>
                      )}
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="font-medium">
                      <span className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: awayColor }} />
                        {awayAbbr}
                      </span>
                    </TableCell>
                    <TableCell className="text-right">
                      {showAsPlayed ? (
                        <DiffCell actual={awaySummary.actualGoals} predicted={awaySummary.predGoals} decimals={1} />
                      ) : (
                        <span className="font-semibold tabular-nums">{awaySummary.predGoals.toFixed(1)}</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {showAsPlayed ? (
                        <DiffCell actual={awaySummary.actualDisp} predicted={awaySummary.predDisp} />
                      ) : (
                        <span className="font-semibold tabular-nums">{Math.round(awaySummary.predDisp)}</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {showAsPlayed ? (
                        <DiffCell actual={awaySummary.actualMarks} predicted={awaySummary.predMarks} />
                      ) : (
                        <span className="font-semibold tabular-nums">{Math.round(awaySummary.predMarks)}</span>
                      )}
                    </TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Player Comparison Tables */}
      {hasComparison ? (
        <>
          {/* View toggles */}
          <div className="flex items-center gap-3 flex-wrap">
            <ExportButton
              data={playerExportData}
              filename={`match_${homeAbbr}_vs_${awayAbbr}_R${data!.round_number ?? ""}_${data!.year ?? ""}`}
              columns={playerExportColumns}
            />
            {/* Team grouping toggle */}
            <div className="flex items-center gap-1">
              <button
                onClick={() => setTeamView("split")}
                className={cn(
                  "px-3 py-1 text-xs font-medium rounded-l-md border transition-colors",
                  teamView === "split"
                    ? "bg-primary text-primary-foreground border-primary"
                    : "bg-muted/50 text-muted-foreground border-border hover:bg-muted"
                )}
              >
                By Team
              </button>
              <button
                onClick={() => setTeamView("combined")}
                className={cn(
                  "px-3 py-1 text-xs font-medium rounded-r-md border border-l-0 transition-colors",
                  teamView === "combined"
                    ? "bg-primary text-primary-foreground border-primary"
                    : "bg-muted/50 text-muted-foreground border-border hover:bg-muted"
                )}
              >
                Both Teams
              </button>
            </div>
            {/* Detail level toggle */}
            <div className="flex items-center gap-1">
              <button
                onClick={() => setPlayerView("standard")}
                className={cn(
                  "px-3 py-1 text-xs font-medium rounded-l-md border transition-colors",
                  playerView === "standard"
                    ? "bg-primary text-primary-foreground border-primary"
                    : "bg-muted/50 text-muted-foreground border-border hover:bg-muted"
                )}
              >
                Standard
              </button>
              <button
                onClick={() => setPlayerView("advanced")}
                className={cn(
                  "px-3 py-1 text-xs font-medium rounded-r-md border border-l-0 transition-colors",
                  playerView === "advanced"
                    ? "bg-primary text-primary-foreground border-primary"
                    : "bg-muted/50 text-muted-foreground border-border hover:bg-muted"
                )}
              >
                Advanced
              </button>
            </div>
          </div>

          {playerView === "standard" ? (
            // Standard view
            teamView === "split" ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <PlayerComparisonTable
                  players={homePlayers}
                  teamName={homeTeam}
                  teamColor={homeColor}
                  isPlayed={showAsPlayed}
                />
                <PlayerComparisonTable
                  players={awayPlayers}
                  teamName={awayTeam}
                  teamColor={awayColor}
                  isPlayed={showAsPlayed}
                />
              </div>
            ) : (
              <PlayerComparisonTable
                players={compPlayers}
                teamName="Both Teams"
                teamColor={homeColor}
                isPlayed={showAsPlayed}
                showTeamIndicator
                teamColors={{ [homeTeam]: homeColor, [awayTeam]: awayColor }}
              />
            )
          ) : (
            // Advanced view
            teamView === "split" ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <PlayerAdvancedTable
                  players={homePlayers}
                  teamName={homeTeam}
                  teamColor={homeColor}
                  isPlayed={showAsPlayed}
                />
                <PlayerAdvancedTable
                  players={awayPlayers}
                  teamName={awayTeam}
                  teamColor={awayColor}
                  isPlayed={showAsPlayed}
                />
              </div>
            ) : (
              <PlayerAdvancedTable
                players={compPlayers}
                teamName="Both Teams"
                teamColor={homeColor}
                isPlayed={showAsPlayed}
                teamColors={{ [homeTeam]: homeColor, [awayTeam]: awayColor }}
              />
            )
          )}
        </>
      ) : match ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <span className="w-3 h-3 rounded-full" style={{ backgroundColor: homeColor }} />
                {homeTeam}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Player</TableHead>
                    <TableHead className="text-right">Goals</TableHead>
                    <TableHead className="text-right">1+ Goal %</TableHead>
                    <TableHead className="text-right">Disposals</TableHead>
                    <TableHead className="text-right">Marks</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {[...(match.home_players || [])]
                    .sort((a, b) => (b.predicted_goals ?? 0) - (a.predicted_goals ?? 0))
                    .map((p, i) => (
                      <TableRow key={i}>
                        <TableCell className="font-medium text-sm">
                          <Link href={`/players/${encodeURIComponent(`${p.player}_${p.team}`)}`} className="hover:text-primary transition-colors">{p.player}</Link>
                        </TableCell>
                        <TableCell className="text-right tabular-nums">
                          {p.predicted_goals?.toFixed(2) ?? "-"}
                        </TableCell>
                        <TableCell className="text-right tabular-nums">
                          {p.p_scorer != null ? (p.p_scorer * 100).toFixed(0) + "%" : "-"}
                        </TableCell>
                        <TableCell className="text-right tabular-nums">
                          {p.predicted_disposals?.toFixed(1) ?? "-"}
                        </TableCell>
                        <TableCell className="text-right tabular-nums">
                          {p.predicted_marks?.toFixed(1) ?? "-"}
                        </TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <span className="w-3 h-3 rounded-full" style={{ backgroundColor: awayColor }} />
                {awayTeam}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Player</TableHead>
                    <TableHead className="text-right">Goals</TableHead>
                    <TableHead className="text-right">1+ Goal %</TableHead>
                    <TableHead className="text-right">Disposals</TableHead>
                    <TableHead className="text-right">Marks</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {[...(match.away_players || [])]
                    .sort((a, b) => (b.predicted_goals ?? 0) - (a.predicted_goals ?? 0))
                    .map((p, i) => (
                      <TableRow key={i}>
                        <TableCell className="font-medium text-sm">
                          <Link href={`/players/${encodeURIComponent(`${p.player}_${p.team}`)}`} className="hover:text-primary transition-colors">{p.player}</Link>
                        </TableCell>
                        <TableCell className="text-right tabular-nums">
                          {p.predicted_goals?.toFixed(2) ?? "-"}
                        </TableCell>
                        <TableCell className="text-right tabular-nums">
                          {p.p_scorer != null ? (p.p_scorer * 100).toFixed(0) + "%" : "-"}
                        </TableCell>
                        <TableCell className="text-right tabular-nums">
                          {p.predicted_disposals?.toFixed(1) ?? "-"}
                        </TableCell>
                        <TableCell className="text-right tabular-nums">
                          {p.predicted_marks?.toFixed(1) ?? "-"}
                        </TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </div>
      ) : (
        <Card>
          <CardContent className="pt-6 pb-6">
            <p className="text-center text-muted-foreground">Player predictions not yet available for this match.</p>
            <p className="text-center text-xs text-muted-foreground/60 mt-1">Player-level predictions will appear here once they are generated.</p>
          </CardContent>
        </Card>
      )}

      {/* Betting-style Probability Markets */}
      {hasComparison && !showAsPlayed && (
        <BettingMarketsCard
          players={compPlayers}
          homeTeam={homeTeam}
          awayTeam={awayTeam}
          homeColor={homeColor}
          awayColor={awayColor}
          homeWinProb={homeWinProb}
        />
      )}

      {/* Monte Carlo Simulation */}
      {simulation && (
        <MonteCarloCard
          simulation={simulation}
          homeTeam={homeTeam}
          awayTeam={awayTeam}
          homeColor={homeColor}
          awayColor={awayColor}
        />
      )}
      {simLoading && !simulation && (
        <Card>
          <CardContent className="pt-6 pb-6">
            <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
              <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
              Running Monte Carlo simulation...
            </div>
          </CardContent>
        </Card>
      )}

      {/* Color Legend */}
      {hasComparison && showAsPlayed && (
        <div className="flex items-center gap-6 text-xs text-muted-foreground justify-center">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-emerald-500/20 border border-emerald-500/40" />
            Met or exceeded prediction
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-red-500/20 border border-red-500/40" />
            Below prediction
          </span>
        </div>
      )}
    </div>
  );
}
