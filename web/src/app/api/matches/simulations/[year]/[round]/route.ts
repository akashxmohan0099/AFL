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

    // Load game predictions, MC simulations, and player predictions in parallel
    const [gpRes, mcRes, predsRes] = await Promise.all([
      supabase
        .from("game_predictions")
        .select("match_id, home_team, away_team, home_win_prob, predicted_winner, predicted_margin")
        .eq("year", y)
        .eq("round_number", r),
      supabase
        .from("mc_simulations")
        .select("player, team, opponent, predicted_goals, predicted_disposals, mc_p_1plus_goals, mc_p_2plus_goals, mc_p_3plus_goals")
        .eq("year", y)
        .eq("round_number", r),
      supabase
        .from("predictions")
        .select("player, team, match_id, predicted_goals, p_scorer")
        .eq("year", y)
        .eq("round_number", r),
    ]);

    const gamePreds = gpRes.data ?? [];
    const mcRows = mcRes.data ?? [];
    const playerPreds = predsRes.data ?? [];

    if (gamePreds.length === 0) {
      return NextResponse.json([]);
    }

    // Index MC rows by team
    const mcByTeam = new Map<string, any[]>();
    for (const row of mcRows) {
      const arr = mcByTeam.get(row.team) ?? [];
      arr.push(row);
      mcByTeam.set(row.team, arr);
    }

    // Index player predictions by team
    const playersByTeam = new Map<string, any[]>();
    for (const p of playerPreds) {
      const arr = playersByTeam.get(p.team) ?? [];
      arr.push(p);
      playersByTeam.set(p.team, arr);
    }

    // Compute average raw goal total across all games for relative scaling.
    // Per-player predicted_goals are calibrated for prop markets and inflate ~2x,
    // but the ratio BETWEEN games carries signal about which will be higher-scoring.
    const AVG_TOTAL = 165;
    let sumRawTotal = 0;
    let gamesWithPreds = 0;
    for (const gp of gamePreds) {
      const hp = playersByTeam.get(gp.home_team) ?? [];
      const ap = playersByTeam.get(gp.away_team) ?? [];
      const raw = hp.reduce((s: number, p: any) => s + (p.predicted_goals ?? 0), 0)
                + ap.reduce((s: number, p: any) => s + (p.predicted_goals ?? 0), 0);
      if (raw > 0) { sumRawTotal += raw; gamesWithPreds++; }
    }
    const avgRawTotal = gamesWithPreds > 0 ? sumRawTotal / gamesWithPreds : 0;

    const results = gamePreds.map((gp: any) => {
      const homeProb = gp.home_win_prob ?? 0.5;
      const awayProb = 1 - homeProb;
      const margin = gp.predicted_margin ?? 0;

      const homePlayers = playersByTeam.get(gp.home_team) ?? [];
      const awayPlayers = playersByTeam.get(gp.away_team) ?? [];

      // Check if MC data exists for this match
      const homeMC = mcByTeam.get(gp.home_team) ?? [];
      const awayMC = mcByTeam.get(gp.away_team) ?? [];
      const hasMC = homeMC.length > 0 || awayMC.length > 0;

      // Game-specific total: use raw goal sum ratio for per-game variation.
      // The ratio between games reflects which matchups are higher/lower scoring.
      const homeRawGoals = homePlayers.reduce((s: number, p: any) => s + (p.predicted_goals ?? 0), 0);
      const awayRawGoals = awayPlayers.reduce((s: number, p: any) => s + (p.predicted_goals ?? 0), 0);
      const rawTotal = homeRawGoals + awayRawGoals;
      const scaleFactor = avgRawTotal > 0
        ? Math.max(0.70, Math.min(1.40, rawTotal / avgRawTotal))
        : 1;
      const gameTotal = Math.round(AVG_TOTAL * scaleFactor);
      const homeScore = Math.round(gameTotal / 2 + margin / 2);
      const awayScore = Math.round(gameTotal / 2 - margin / 2);
      const total = homeScore + awayScore;

      // Score ranges: wider spread for MC (real variance), narrower for estimates
      const hmul = hasMC
        ? { p25: 0.78, p75: 1.22 }
        : { p25: 0.85, p75: 1.15 };

      // Top goal scorers: use MC probabilities when available, otherwise deterministic p_scorer
      let topScorers: { player: string; team: string; p_1plus: number }[];
      if (hasMC) {
        const allMC = [...homeMC, ...awayMC];
        topScorers = allMC
          .filter((p: any) => p.mc_p_1plus_goals != null)
          .sort((a: any, b: any) => (b.mc_p_1plus_goals ?? 0) - (a.mc_p_1plus_goals ?? 0))
          .slice(0, 4)
          .map((p: any) => ({
            player: p.player,
            team: p.team,
            p_1plus: p.mc_p_1plus_goals ?? 0,
          }));
      } else {
        const allPlayers = [...homePlayers, ...awayPlayers];
        topScorers = allPlayers
          .filter((p: any) => p.p_scorer != null)
          .sort((a: any, b: any) => (b.p_scorer ?? 0) - (a.p_scorer ?? 0))
          .slice(0, 4)
          .map((p: any) => ({
            player: p.player,
            team: p.team,
            p_1plus: p.p_scorer ?? 0,
          }));
      }

      return {
        match_id: gp.match_id,
        home_team: gp.home_team,
        away_team: gp.away_team,
        n_sims: hasMC ? 10000 : 0,
        has_mc: hasMC,
        home_win_pct: homeProb,
        away_win_pct: awayProb,
        draw_pct: 0.02,
        avg_total: total,
        avg_margin: margin || homeScore - awayScore,
        avg_home_score: homeScore,
        avg_away_score: awayScore,
        score_range: {
          home: { p10: Math.round(homeScore * 0.7), p25: Math.round(homeScore * hmul.p25), p50: homeScore, p75: Math.round(homeScore * hmul.p75), p90: Math.round(homeScore * 1.3) },
          away: { p10: Math.round(awayScore * 0.7), p25: Math.round(awayScore * hmul.p25), p50: awayScore, p75: Math.round(awayScore * hmul.p75), p90: Math.round(awayScore * 1.3) },
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
