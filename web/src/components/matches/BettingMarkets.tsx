"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { MatchComparisonPlayer } from "@/lib/types";
import { TEAM_ABBREVS } from "@/lib/constants";
import { cn } from "@/lib/utils";

interface MarketEntry {
  player: string;
  team: string;
  prob: number;
}

function MarketSection({
  title,
  entries,
  homeTeam,
  homeColor,
  awayColor,
  limit = 8,
}: {
  title: string;
  entries: MarketEntry[];
  homeTeam: string;
  homeColor: string;
  awayColor: string;
  limit?: number;
}) {
  const [showAll, setShowAll] = useState(false);
  const sorted = [...entries].sort((a, b) => b.prob - a.prob);
  const visible = showAll ? sorted : sorted.slice(0, limit);
  if (sorted.length === 0) return null;

  return (
    <div>
      <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-2">
        {title}
        <span className="text-muted-foreground/40 normal-case ml-2 font-normal">
          {sorted.length} players
        </span>
      </p>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-1.5">
        {visible.map((e) => {
          const color = e.team === homeTeam ? homeColor : awayColor;
          const abbr = TEAM_ABBREVS[e.team] || e.team;
          const pct = (e.prob * 100);
          const bgIntensity = pct >= 70 ? "bg-emerald-500/10" : pct >= 50 ? "bg-emerald-500/5" : "bg-card/50";
          const textColor = pct >= 70 ? "text-emerald-400" : pct >= 50 ? "text-foreground" : "text-muted-foreground";
          return (
            <div
              key={`${e.player}-${e.team}`}
              className={cn("flex items-center gap-2 px-2.5 py-2 rounded-lg border border-border/20", bgIntensity)}
            >
              <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
              <div className="flex-1 min-w-0">
                <p className="text-[11px] font-medium truncate">{e.player}</p>
                <p className="text-[9px] text-muted-foreground/50 font-mono">{abbr}</p>
              </div>
              <span className={cn("text-sm font-bold font-mono tabular-nums shrink-0", textColor)}>
                {pct.toFixed(0)}%
              </span>
            </div>
          );
        })}
      </div>
      {sorted.length > limit && (
        <button
          className="mt-1.5 py-1 text-[11px] text-muted-foreground hover:text-foreground font-mono flex items-center gap-1 transition-colors"
          onClick={() => setShowAll(v => !v)}
        >
          {showAll ? "Show less \u2227" : `Show all ${sorted.length} \u2228`}
        </button>
      )}
    </div>
  );
}

export function BettingMarketsCard({
  players,
  homeTeam,
  awayTeam,
  homeColor,
  awayColor,
  homeWinProb,
}: {
  players: MatchComparisonPlayer[];
  homeTeam: string;
  awayTeam: string;
  homeColor: string;
  awayColor: string;
  homeWinProb?: number | null;
}) {
  const homeAbbr = TEAM_ABBREVS[homeTeam] || homeTeam;
  const awayAbbr = TEAM_ABBREVS[awayTeam] || awayTeam;

  // Build market entries from player predictions
  const anytimeGoalScorer: MarketEntry[] = players
    .filter(p => p.p_scorer != null && p.p_scorer > 0.05)
    .map(p => ({ player: p.player, team: p.team, prob: p.p_scorer! }));

  const twoGoals: MarketEntry[] = players
    .filter(p => p.p_2plus_goals != null && p.p_2plus_goals > 0.05)
    .map(p => ({ player: p.player, team: p.team, prob: p.p_2plus_goals! }));

  const threeGoals: MarketEntry[] = players
    .filter(p => p.p_3plus_goals != null && p.p_3plus_goals > 0.03)
    .map(p => ({ player: p.player, team: p.team, prob: p.p_3plus_goals! }));

  const disp15: MarketEntry[] = players
    .filter(p => p.p_15plus_disp != null && p.p_15plus_disp > 0.1)
    .map(p => ({ player: p.player, team: p.team, prob: p.p_15plus_disp! }));

  const disp20: MarketEntry[] = players
    .filter(p => p.p_20plus_disp != null && p.p_20plus_disp > 0.1)
    .map(p => ({ player: p.player, team: p.team, prob: p.p_20plus_disp! }));

  const disp25: MarketEntry[] = players
    .filter(p => p.p_25plus_disp != null && p.p_25plus_disp > 0.05)
    .map(p => ({ player: p.player, team: p.team, prob: p.p_25plus_disp! }));

  const disp30: MarketEntry[] = players
    .filter(p => p.p_30plus_disp != null && p.p_30plus_disp > 0.03)
    .map(p => ({ player: p.player, team: p.team, prob: p.p_30plus_disp! }));

  const marks5: MarketEntry[] = players
    .filter(p => p.p_5plus_mk != null && p.p_5plus_mk > 0.05)
    .map(p => ({ player: p.player, team: p.team, prob: p.p_5plus_mk! }));

  const marks3: MarketEntry[] = players
    .filter(p => p.p_3plus_mk != null && p.p_3plus_mk > 0.05)
    .map(p => ({ player: p.player, team: p.team, prob: p.p_3plus_mk! }));

  const hasAnyMarket = anytimeGoalScorer.length > 0 || twoGoals.length > 0 || threeGoals.length > 0
    || disp15.length > 0 || disp20.length > 0 || disp25.length > 0 || disp30.length > 0
    || marks5.length > 0 || marks3.length > 0;

  const tabs = [
    { id: "goals", label: "Goal Scorer", show: anytimeGoalScorer.length > 0 },
    { id: "multigoals", label: "Multi Goals", show: twoGoals.length > 0 || threeGoals.length > 0 },
    { id: "disposals", label: "Disposals", show: disp15.length > 0 || disp20.length > 0 },
    { id: "marks", label: "Marks", show: marks3.length > 0 || marks5.length > 0 },
  ].filter(t => t.show);

  const [activeTab, setActiveTab] = useState<string>(tabs[0]?.id ?? "goals");

  if (!hasAnyMarket) return null;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center gap-2">
          Player Probabilities
          <span className="text-[10px] font-normal text-muted-foreground/50 font-mono">Model-derived predictions</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Match Result Summary */}
        {homeWinProb != null && (
          <div className="flex gap-2">
            <div className={cn("flex-1 px-3 py-2.5 rounded-lg border border-border/30 text-center", homeWinProb >= 0.5 ? "bg-emerald-500/5" : "bg-card/30")}>
              <p className="text-[9px] text-muted-foreground uppercase tracking-wider">Match Winner</p>
              <div className="flex items-center justify-center gap-2 mt-1">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: homeColor }} />
                <span className="text-xs font-semibold">{homeAbbr}</span>
                <span className={cn("text-lg font-bold font-mono tabular-nums", homeWinProb >= 0.5 ? "text-emerald-400" : "text-muted-foreground")}>
                  {(homeWinProb * 100).toFixed(0)}%
                </span>
              </div>
            </div>
            <div className={cn("flex-1 px-3 py-2.5 rounded-lg border border-border/30 text-center", homeWinProb < 0.5 ? "bg-emerald-500/5" : "bg-card/30")}>
              <p className="text-[9px] text-muted-foreground uppercase tracking-wider">Match Winner</p>
              <div className="flex items-center justify-center gap-2 mt-1">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: awayColor }} />
                <span className="text-xs font-semibold">{awayAbbr}</span>
                <span className={cn("text-lg font-bold font-mono tabular-nums", homeWinProb < 0.5 ? "text-emerald-400" : "text-muted-foreground")}>
                  {((1 - homeWinProb) * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Tab navigation */}
        <div className="flex gap-1 border-b border-border/30 pb-0">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                "px-3 py-1.5 text-xs font-medium border-b-2 transition-colors -mb-px",
                activeTab === tab.id
                  ? "border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              )}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {activeTab === "goals" && (
          <MarketSection
            title="Anytime Goal Scorer"
            entries={anytimeGoalScorer}
            homeTeam={homeTeam}
            homeColor={homeColor}
            awayColor={awayColor}
          />
        )}

        {activeTab === "multigoals" && (
          <div className="space-y-4">
            {twoGoals.length > 0 && (
              <MarketSection
                title="2+ Goals"
                entries={twoGoals}
                homeTeam={homeTeam}
                homeColor={homeColor}
                awayColor={awayColor}
              />
            )}
            {threeGoals.length > 0 && (
              <MarketSection
                title="3+ Goals"
                entries={threeGoals}
                homeTeam={homeTeam}
                homeColor={homeColor}
                awayColor={awayColor}
              />
            )}
          </div>
        )}

        {activeTab === "disposals" && (
          <div className="space-y-4">
            {disp15.length > 0 && (
              <MarketSection
                title="15+ Disposals"
                entries={disp15}
                homeTeam={homeTeam}
                homeColor={homeColor}
                awayColor={awayColor}
              />
            )}
            {disp20.length > 0 && (
              <MarketSection
                title="20+ Disposals"
                entries={disp20}
                homeTeam={homeTeam}
                homeColor={homeColor}
                awayColor={awayColor}
              />
            )}
            {disp25.length > 0 && (
              <MarketSection
                title="25+ Disposals"
                entries={disp25}
                homeTeam={homeTeam}
                homeColor={homeColor}
                awayColor={awayColor}
              />
            )}
            {disp30.length > 0 && (
              <MarketSection
                title="30+ Disposals"
                entries={disp30}
                homeTeam={homeTeam}
                homeColor={homeColor}
                awayColor={awayColor}
              />
            )}
          </div>
        )}

        {activeTab === "marks" && (
          <div className="space-y-4">
            {marks3.length > 0 && (
              <MarketSection
                title="3+ Marks"
                entries={marks3}
                homeTeam={homeTeam}
                homeColor={homeColor}
                awayColor={awayColor}
              />
            )}
            {marks5.length > 0 && (
              <MarketSection
                title="5+ Marks"
                entries={marks5}
                homeTeam={homeTeam}
                homeColor={homeColor}
                awayColor={awayColor}
              />
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
