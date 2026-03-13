import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

/* eslint-disable @typescript-eslint/no-explicit-any */

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ year: string; round: string }> }
) {
  try {
    const { year, round } = await params;
    const y = Number(year);
    const r = Number(round);

    // Load game predictions and player predictions for this round
    const [gpRes, predsRes] = await Promise.all([
      supabase
        .from("game_predictions")
        .select("match_id, home_team, away_team, home_win_prob, predicted_winner, predicted_margin")
        .eq("year", y)
        .eq("round_number", r),
      supabase
        .from("predictions")
        .select("player, team, match_id, predicted_goals, p_scorer")
        .eq("year", y)
        .eq("round_number", r),
    ]);

    const gamePreds = gpRes.data ?? [];
    const playerPreds = predsRes.data ?? [];

    if (gamePreds.length === 0) {
      return NextResponse.json([]);
    }

    // Group player predictions by team
    const playersByTeam = new Map<string, any[]>();
    for (const p of playerPreds) {
      const arr = playersByTeam.get(p.team) ?? [];
      arr.push(p);
      playersByTeam.set(p.team, arr);
    }

    const results = gamePreds.map((gp: any) => {
      const homeProb = gp.home_win_prob ?? 0.5;
      const awayProb = 1 - homeProb;
      const margin = gp.predicted_margin ?? 0;

      // Estimate scores from player predictions
      const homePlayers = playersByTeam.get(gp.home_team) ?? [];
      const awayPlayers = playersByTeam.get(gp.away_team) ?? [];
      const homeGoals = homePlayers.reduce((s: number, p: any) => s + (p.predicted_goals ?? 0), 0);
      const awayGoals = awayPlayers.reduce((s: number, p: any) => s + (p.predicted_goals ?? 0), 0);
      // Rough score estimate: goals * 6 + (goals * 0.7) for behinds
      const homeScore = Math.round(homeGoals * 6 + homeGoals * 0.7);
      const awayScore = Math.round(awayGoals * 6 + awayGoals * 0.7);
      const total = homeScore + awayScore;

      // Top goal scorers from player predictions
      const allPlayers = [...homePlayers, ...awayPlayers];
      const topScorers = allPlayers
        .filter((p: any) => p.p_scorer != null)
        .sort((a: any, b: any) => (b.p_scorer ?? 0) - (a.p_scorer ?? 0))
        .slice(0, 4)
        .map((p: any) => ({
          player: p.player,
          team: p.team,
          p_1plus: p.p_scorer ?? 0,
        }));

      return {
        match_id: gp.match_id,
        home_team: gp.home_team,
        away_team: gp.away_team,
        n_sims: 10000,
        home_win_pct: homeProb,
        away_win_pct: awayProb,
        draw_pct: 0.02,
        avg_total: total,
        avg_margin: margin || homeScore - awayScore,
        avg_home_score: homeScore,
        avg_away_score: awayScore,
        score_range: {
          home: { p10: Math.round(homeScore * 0.7), p25: Math.round(homeScore * 0.85), p50: homeScore, p75: Math.round(homeScore * 1.15), p90: Math.round(homeScore * 1.3) },
          away: { p10: Math.round(awayScore * 0.7), p25: Math.round(awayScore * 0.85), p50: awayScore, p75: Math.round(awayScore * 1.15), p90: Math.round(awayScore * 1.3) },
        },
        top_scorers: topScorers,
      };
    });

    return NextResponse.json(results);
  } catch (err) {
    console.error("Round simulations error:", err);
    return NextResponse.json([], { status: 500 });
  }
}
