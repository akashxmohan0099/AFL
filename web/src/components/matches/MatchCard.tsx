"use client";

import { Card, CardContent } from "@/components/ui/card";
import type { MatchResult } from "@/lib/types";
import type { SeasonMatch } from "@/lib/types";
import { TEAM_COLORS, TEAM_ABBREVS } from "@/lib/constants";
import Link from "next/link";
import { cn } from "@/lib/utils";

function MiniStat({ label, pred, actual, color }: { label: string; pred?: number | null; actual?: number | null; color: string }) {
  if (pred == null && actual == null) return null;
  const diff = (pred != null && actual != null) ? actual - pred : null;
  const close = diff != null && Math.abs(diff) <= (label === "GL" ? 2 : 15);
  return (
    <div className="text-center">
      <p className="text-[9px] font-mono text-muted-foreground/60 uppercase">{label}</p>
      <p className="text-[11px] font-mono font-semibold tabular-nums" style={{ color: actual != null ? color : undefined }}>
        {actual != null ? actual : "-"}
      </p>
      {pred != null && (
        <p className={cn("text-[9px] font-mono tabular-nums", close ? "text-emerald-400/70" : "text-muted-foreground/50")}>
          {typeof pred === "number" ? (label === "GL" ? pred.toFixed(1) : Math.round(pred)) : pred}
        </p>
      )}
    </div>
  );
}

export function MatchCard({ match }: { match: MatchResult | SeasonMatch }) {
  const homeAbbr = TEAM_ABBREVS[match.home_team] || match.home_team;
  const awayAbbr = TEAM_ABBREVS[match.away_team] || match.away_team;
  const isPlayed = match.home_score != null && match.away_score != null;
  const correct = "correct" in match ? match.correct : undefined;
  const sm = match as SeasonMatch;

  // Combine pred + actual for each team
  const homePredGl = sm.home_pred?.pred_gl;
  const homePredDi = sm.home_pred?.pred_di;
  const homePredMk = sm.home_pred?.pred_mk;
  const homeActualGl = sm.home_actual?.actual_gl;
  const homeActualDi = sm.home_actual?.actual_di;
  const homeActualMk = sm.home_actual?.actual_mk;
  const awayPredGl = sm.away_pred?.pred_gl;
  const awayPredDi = sm.away_pred?.pred_di;
  const awayPredMk = sm.away_pred?.pred_mk;
  const awayActualGl = sm.away_actual?.actual_gl;
  const awayActualDi = sm.away_actual?.actual_di;
  const awayActualMk = sm.away_actual?.actual_mk;

  const hasPredData = homePredGl != null || awayPredGl != null;

  const hasMatchId = match.match_id != null && match.match_id !== 0;
  const roundParam = "round_number" in match && match.round_number != null ? match.round_number : '';
  const matchLink = hasMatchId
    ? `/matches/${match.match_id}${roundParam ? `?round=${roundParam}` : ''}`
    : `/matches/0?round=${roundParam}&home=${encodeURIComponent(match.home_team)}&away=${encodeURIComponent(match.away_team)}`;

  const card = (
      <Card className={cn(
        "border transition-all duration-150 group-hover:border-primary/30 group-hover:bg-muted/20",
        correct === true ? "border-emerald-500/25 bg-emerald-500/[0.03]" :
        correct === false ? "border-red-500/25 bg-red-500/[0.03]" :
        "border-border/50"
      )}>
        <CardContent className="pt-3 pb-3 px-3 space-y-2">
          {/* Score rows */}
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: TEAM_COLORS[match.home_team]?.primary || "#555" }} />
                <span className="text-xs font-semibold font-mono">{homeAbbr}</span>
              </div>
              {isPlayed && <span className="text-sm font-bold tabular-nums font-mono">{match.home_score}</span>}
            </div>
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: TEAM_COLORS[match.away_team]?.primary || "#555" }} />
                <span className="text-xs font-semibold font-mono">{awayAbbr}</span>
              </div>
              {isPlayed && <span className="text-sm font-bold tabular-nums font-mono">{match.away_score}</span>}
            </div>
          </div>

          {/* Win prob bar */}
          {match.home_win_prob != null && (
            <div className="w-full h-1 rounded-full bg-muted/50 overflow-hidden flex">
              <div className="h-full rounded-l-full" style={{ width: `${Math.round(match.home_win_prob * 100)}%`, backgroundColor: TEAM_COLORS[match.home_team]?.primary || "#6366f1" }} />
            </div>
          )}

          {/* Pred vs Actual stats grid */}
          {hasPredData && (
            <div className="border-t border-border/30 pt-2">
              <div className="grid grid-cols-[1fr_auto_auto_auto] gap-x-3 gap-y-1 items-center">
                {/* Header */}
                <div />
                <p className="text-[8px] font-mono text-muted-foreground/50 text-center uppercase" title="Team goals: actual (bold) vs predicted (light)">Goals</p>
                <p className="text-[8px] font-mono text-muted-foreground/50 text-center uppercase" title="Team disposals: actual (bold) vs predicted (light)">Disp</p>
                <p className="text-[8px] font-mono text-muted-foreground/50 text-center uppercase" title="Team marks: actual (bold) vs predicted (light)">Marks</p>

                {/* Home row */}
                <span className="text-[10px] font-mono font-medium">{homeAbbr}</span>
                <div className="text-center">
                  {isPlayed && <span className="text-[10px] font-mono font-bold tabular-nums text-foreground">{homeActualGl ?? "-"}</span>}
                  {homePredGl != null && <span className={cn("font-mono tabular-nums", isPlayed ? "text-[9px] text-muted-foreground/40 ml-0.5" : "text-[10px] text-muted-foreground")}>{isPlayed ? "/ " : ""}{homePredGl.toFixed(1)}</span>}
                </div>
                <div className="text-center">
                  {isPlayed && <span className="text-[10px] font-mono font-bold tabular-nums text-foreground">{homeActualDi ?? "-"}</span>}
                  {homePredDi != null && <span className={cn("font-mono tabular-nums", isPlayed ? "text-[9px] text-muted-foreground/40 ml-0.5" : "text-[10px] text-muted-foreground")}>{isPlayed ? "/ " : ""}{Math.round(homePredDi)}</span>}
                </div>
                <div className="text-center">
                  {isPlayed && <span className="text-[10px] font-mono font-bold tabular-nums text-foreground">{homeActualMk ?? "-"}</span>}
                  {homePredMk != null && <span className={cn("font-mono tabular-nums", isPlayed ? "text-[9px] text-muted-foreground/40 ml-0.5" : "text-[10px] text-muted-foreground")}>{isPlayed ? "/ " : ""}{Math.round(homePredMk)}</span>}
                </div>

                {/* Away row */}
                <span className="text-[10px] font-mono font-medium">{awayAbbr}</span>
                <div className="text-center">
                  {isPlayed && <span className="text-[10px] font-mono font-bold tabular-nums text-foreground">{awayActualGl ?? "-"}</span>}
                  {awayPredGl != null && <span className={cn("font-mono tabular-nums", isPlayed ? "text-[9px] text-muted-foreground/40 ml-0.5" : "text-[10px] text-muted-foreground")}>{isPlayed ? "/ " : ""}{awayPredGl.toFixed(1)}</span>}
                </div>
                <div className="text-center">
                  {isPlayed && <span className="text-[10px] font-mono font-bold tabular-nums text-foreground">{awayActualDi ?? "-"}</span>}
                  {awayPredDi != null && <span className={cn("font-mono tabular-nums", isPlayed ? "text-[9px] text-muted-foreground/40 ml-0.5" : "text-[10px] text-muted-foreground")}>{isPlayed ? "/ " : ""}{Math.round(awayPredDi)}</span>}
                </div>
                <div className="text-center">
                  {isPlayed && <span className="text-[10px] font-mono font-bold tabular-nums text-foreground">{awayActualMk ?? "-"}</span>}
                  {awayPredMk != null && <span className={cn("font-mono tabular-nums", isPlayed ? "text-[9px] text-muted-foreground/40 ml-0.5" : "text-[10px] text-muted-foreground")}>{isPlayed ? "/ " : ""}{Math.round(awayPredMk)}</span>}
                </div>
              </div>
              <p className="text-[7px] font-mono text-muted-foreground/30 mt-1.5 text-right">
                {isPlayed ? (
                  <><span className="font-bold text-foreground/40">Actual</span> / <span className="text-muted-foreground/40">Predicted</span></>
                ) : (
                  <span className="text-muted-foreground/40">Predicted</span>
                )}
              </p>
            </div>
          )}

          {/* Footer */}
          <div className="flex justify-between items-center pt-0.5">
            <span className="text-[10px] text-muted-foreground/60 truncate mr-2">{match.venue}</span>
            <div className="flex items-center gap-1.5 shrink-0">
              {"predicted_winner" in match && match.predicted_winner && (
                <span className="text-[9px] font-mono text-muted-foreground/50">
                  {TEAM_ABBREVS[match.predicted_winner as string] || match.predicted_winner}
                </span>
              )}
              {correct != null && (
                <span className={cn(
                  "text-[9px] font-mono font-bold px-1.5 py-0.5 rounded",
                  correct ? "text-emerald-400 bg-emerald-400/10" : "text-red-400 bg-red-400/10"
                )}>
                  {correct ? "HIT" : "MISS"}
                </span>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
  );

  return <Link href={matchLink} className="block group">{card}</Link>;
}
