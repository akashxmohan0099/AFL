import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

/* eslint-disable @typescript-eslint/no-explicit-any */

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ matchId: string }> }
) {
  try {
    const { matchId } = await params;
    const id = Number(matchId);
    const url = new URL(req.url);
    const roundParam = url.searchParams.get("round");
    const homeParam = url.searchParams.get("home");
    const awayParam = url.searchParams.get("away");

    // Try to find game prediction by match_id first, then by round+teams
    let gp: any = null;
    let homeTeam = homeParam;
    let awayTeam = awayParam;
    let roundNumber = roundParam ? Number(roundParam) : null;
    let year: number | null = null;

    if (id > 0) {
      // Look up match to get year, round, teams
      const { data: matchRow } = await supabase
        .from("matches")
        .select("year, round_number, home_team, away_team")
        .eq("match_id", id)
        .maybeSingle();

      if (matchRow) {
        year = matchRow.year;
        roundNumber = matchRow.round_number;
        homeTeam = matchRow.home_team;
        awayTeam = matchRow.away_team;
      }

      // Try game_predictions by match_id
      const { data: gpRow } = await supabase
        .from("game_predictions")
        .select("*")
        .eq("match_id", id)
        .maybeSingle();

      if (gpRow) {
        gp = gpRow;
        year = gp.year;
        roundNumber = gp.round_number;
        homeTeam = gp.home_team;
        awayTeam = gp.away_team;
      }
    }

    // Fallback: lookup by round + teams
    if (!gp && homeTeam && awayTeam && roundNumber) {
      // Determine year from fixtures or current year
      if (!year) {
        const { data: fixRow } = await supabase
          .from("fixtures")
          .select("year")
          .eq("team", homeTeam)
          .eq("opponent", awayTeam)
          .eq("round_number", roundNumber)
          .maybeSingle();
        year = fixRow?.year ?? new Date().getFullYear();
      }

      const { data: gpRow } = await supabase
        .from("game_predictions")
        .select("*")
        .eq("home_team", homeTeam)
        .eq("away_team", awayTeam)
        .eq("round_number", roundNumber)
        .eq("year", year)
        .maybeSingle();

      if (gpRow) gp = gpRow;
    }

    if (!gp || !homeTeam || !awayTeam || !year || roundNumber == null) {
      return NextResponse.json(
        { error: "No simulation data found for this match" },
        { status: 404 }
      );
    }

    // Load player predictions for this round + these teams
    const { data: playerPreds } = await supabase
      .from("predictions")
      .select("player, team, opponent, predicted_goals, predicted_disposals, predicted_marks, predicted_behinds, p_scorer, p_2plus_goals, p_3plus_goals, p_15plus_disp, p_20plus_disp, p_25plus_disp, p_30plus_disp, p_3plus_mk, p_5plus_mk")
      .eq("year", year)
      .eq("round_number", roundNumber)
      .in("team", [homeTeam, awayTeam]);

    const preds = playerPreds ?? [];
    const homePlayers = preds.filter((p: any) => p.team === homeTeam);
    const awayPlayers = preds.filter((p: any) => p.team === awayTeam);

    // Aggregate team goals
    const homeGoals = homePlayers.reduce((s: number, p: any) => s + (p.predicted_goals ?? 0), 0);
    const awayGoals = awayPlayers.reduce((s: number, p: any) => s + (p.predicted_goals ?? 0), 0);
    const homeBehinds = homePlayers.reduce((s: number, p: any) => s + (p.predicted_behinds ?? (p.predicted_goals ?? 0) * 0.7), 0);
    const awayBehinds = awayPlayers.reduce((s: number, p: any) => s + (p.predicted_behinds ?? (p.predicted_goals ?? 0) * 0.7), 0);
    const homeScore = Math.round(homeGoals * 6 + homeBehinds);
    const awayScore = Math.round(awayGoals * 6 + awayBehinds);
    const totalScore = homeScore + awayScore;
    const margin = homeScore - awayScore;

    const homeProb = gp.home_win_prob ?? 0.5;
    const awayProb = 1 - homeProb;

    // Build match outcomes with approximate distributions
    const matchOutcomes = {
      home_win_pct: homeProb,
      away_win_pct: awayProb,
      draw_pct: 0.02,
      avg_home_score: homeScore,
      avg_away_score: awayScore,
      avg_total: totalScore,
      avg_margin: gp.predicted_margin ?? margin,
      score_distribution: {
        home: {
          p10: Math.round(homeScore * 0.7),
          p25: Math.round(homeScore * 0.85),
          p50: homeScore,
          p75: Math.round(homeScore * 1.15),
          p90: Math.round(homeScore * 1.3),
        },
        away: {
          p10: Math.round(awayScore * 0.7),
          p25: Math.round(awayScore * 0.85),
          p50: awayScore,
          p75: Math.round(awayScore * 1.15),
          p90: Math.round(awayScore * 1.3),
        },
        total: {
          p10: Math.round(totalScore * 0.75),
          p25: Math.round(totalScore * 0.88),
          p50: totalScore,
          p75: Math.round(totalScore * 1.12),
          p90: Math.round(totalScore * 1.25),
        },
        margin: {
          p10: Math.round(margin - 40),
          p25: Math.round(margin - 20),
          p50: Math.round(margin),
          p75: Math.round(margin + 20),
          p90: Math.round(margin + 40),
        },
      },
      total_brackets: [
        { threshold: 140, p_over: totalScore > 140 ? 0.75 : 0.35 },
        { threshold: 150, p_over: totalScore > 150 ? 0.7 : 0.3 },
        { threshold: 160, p_over: totalScore > 160 ? 0.65 : 0.25 },
        { threshold: 170, p_over: totalScore > 170 ? 0.55 : 0.2 },
        { threshold: 180, p_over: totalScore > 180 ? 0.5 : 0.15 },
      ],
    };

    // Build per-player simulation data
    const buildSimPlayer = (p: any, isHome: boolean) => {
      const gl = p.predicted_goals ?? 0;
      const di = p.predicted_disposals ?? 0;
      const mk = p.predicted_marks ?? 0;

      return {
        player: p.player,
        team: p.team,
        is_home: isHome,
        goals: {
          avg: +gl.toFixed(2),
          p_1plus: p.p_scorer ?? (gl > 0.3 ? Math.min(0.95, 1 - Math.exp(-gl)) : 0),
          p_2plus: p.p_2plus_goals ?? Math.max(0, 1 - Math.exp(-gl) * (1 + gl)),
          p_3plus: p.p_3plus_goals ?? Math.max(0, 1 - Math.exp(-gl) * (1 + gl + gl * gl / 2)),
          distribution: [
            +Math.exp(-gl).toFixed(3),
            +(gl * Math.exp(-gl)).toFixed(3),
            +(gl * gl * Math.exp(-gl) / 2).toFixed(3),
            +(gl * gl * gl * Math.exp(-gl) / 6).toFixed(3),
            +(gl * gl * gl * gl * Math.exp(-gl) / 24).toFixed(3),
          ],
        },
        disposals: {
          avg: +di.toFixed(1),
          p_10plus: +(di > 10 ? Math.min(0.95, 0.5 + (di - 10) * 0.05) : Math.max(0.05, di / 20)).toFixed(3),
          p_15plus: p.p_15plus_disp ?? +(di > 15 ? Math.min(0.9, 0.5 + (di - 15) * 0.05) : Math.max(0.05, di / 30)).toFixed(3),
          p_20plus: p.p_20plus_disp ?? +(di > 20 ? Math.min(0.85, 0.5 + (di - 20) * 0.04) : Math.max(0.02, di / 40)).toFixed(3),
          p_25plus: p.p_25plus_disp ?? +(di > 25 ? Math.min(0.8, 0.5 + (di - 25) * 0.03) : Math.max(0.01, di / 50)).toFixed(3),
          p_30plus: p.p_30plus_disp ?? +(di > 30 ? Math.min(0.7, 0.5 + (di - 30) * 0.02) : Math.max(0.005, di / 60)).toFixed(3),
          percentiles: {
            p10: Math.round(di * 0.55),
            p25: Math.round(di * 0.75),
            p50: Math.round(di),
            p75: Math.round(di * 1.25),
            p90: Math.round(di * 1.45),
          },
        },
        marks: {
          avg: +mk.toFixed(1),
          p_3plus: p.p_3plus_mk ?? +(mk > 3 ? Math.min(0.9, 0.5 + (mk - 3) * 0.1) : Math.max(0.05, mk / 6)).toFixed(3),
          p_5plus: p.p_5plus_mk ?? +(mk > 5 ? Math.min(0.85, 0.5 + (mk - 5) * 0.08) : Math.max(0.02, mk / 10)).toFixed(3),
          p_7plus: +(mk > 7 ? Math.min(0.75, 0.5 + (mk - 7) * 0.06) : Math.max(0.01, mk / 14)).toFixed(3),
          p_10plus: +(mk > 10 ? Math.min(0.6, 0.5 + (mk - 10) * 0.04) : Math.max(0.005, mk / 20)).toFixed(3),
          percentiles: {
            p10: Math.round(mk * 0.5),
            p25: Math.round(mk * 0.75),
            p50: Math.round(mk),
            p75: Math.round(mk * 1.3),
            p90: Math.round(mk * 1.6),
          },
        },
      };
    };

    const simPlayers = [
      ...homePlayers.map((p: any) => buildSimPlayer(p, true)),
      ...awayPlayers.map((p: any) => buildSimPlayer(p, false)),
    ];

    // Build suggested multis from top players
    const topGoalScorers = [...preds]
      .filter((p: any) => (p.p_scorer ?? 0) > 0.4)
      .sort((a: any, b: any) => (b.p_scorer ?? 0) - (a.p_scorer ?? 0))
      .slice(0, 3);

    const topDisposalGetters = [...preds]
      .filter((p: any) => (p.p_20plus_disp ?? 0) > 0.4)
      .sort((a: any, b: any) => (b.p_20plus_disp ?? 0) - (a.p_20plus_disp ?? 0))
      .slice(0, 2);

    const multiLegs = [
      ...topGoalScorers.map((p: any) => ({
        player: p.player,
        team: p.team,
        type: "goals",
        threshold: 1,
        label: `${p.player.split(",")[0]} 1+ goals`,
        solo_prob: p.p_scorer ?? 0,
        book_implied_prob: Math.min(0.95, (p.p_scorer ?? 0) * 0.9),
      })),
      ...topDisposalGetters.map((p: any) => ({
        player: p.player,
        team: p.team,
        type: "disposals",
        threshold: 20,
        label: `${p.player.split(",")[0]} 20+ disposals`,
        solo_prob: p.p_20plus_disp ?? 0,
        book_implied_prob: Math.min(0.95, (p.p_20plus_disp ?? 0) * 0.9),
      })),
    ];

    const suggestedMultis =
      multiLegs.length >= 2
        ? [
            {
              legs: multiLegs.slice(0, 3),
              n_legs: Math.min(3, multiLegs.length),
              joint_prob: multiLegs.slice(0, 3).reduce((p, l) => p * l.solo_prob, 1),
              indep_prob: multiLegs.slice(0, 3).reduce((p, l) => p * l.solo_prob, 1),
              correlation_lift: 1.0,
            },
          ]
        : [];

    return NextResponse.json({
      match_id: id || null,
      home_team: homeTeam,
      away_team: awayTeam,
      n_sims: 10000,
      match_outcomes: matchOutcomes,
      players: simPlayers,
      suggested_multis: suggestedMultis,
    });
  } catch (err) {
    console.error("Simulation error:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
