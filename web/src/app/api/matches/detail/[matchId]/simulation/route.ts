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

    // Resolve match details
    let homeTeam = homeParam;
    let awayTeam = awayParam;
    let roundNumber = roundParam ? Number(roundParam) : null;
    let year: number | null = null;
    let matchIdResolved = id > 0 ? id : null;

    if (id > 0) {
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
    }

    // Determine year from fixtures if not found
    if (!year && homeTeam && awayTeam && roundNumber) {
      const { data: fixRow } = await supabase
        .from("fixtures")
        .select("year")
        .eq("team", homeTeam)
        .eq("opponent", awayTeam)
        .eq("round_number", roundNumber)
        .maybeSingle();
      year = fixRow?.year ?? new Date().getFullYear();
    }

    if (!homeTeam || !awayTeam || !year || roundNumber == null) {
      return NextResponse.json(
        { error: "Could not resolve match details" },
        { status: 404 }
      );
    }

    // Load game prediction for win prob + margin
    const { data: gpRow } = await supabase
      .from("game_predictions")
      .select("home_win_prob, predicted_margin, predicted_winner")
      .eq("year", year)
      .eq("home_team", homeTeam)
      .eq("away_team", awayTeam)
      .eq("round_number", roundNumber)
      .maybeSingle();

    const homeProb = gpRow?.home_win_prob ?? 0.5;
    const awayProb = 1 - homeProb;
    const predMargin = gpRow?.predicted_margin ?? 0;

    // Try to load REAL Monte Carlo data from mc_simulations table
    const { data: mcRows } = await supabase
      .from("mc_simulations")
      .select("*")
      .eq("year", year)
      .eq("round_number", roundNumber)
      .in("team", [homeTeam, awayTeam]);

    const hasRealMC = mcRows && mcRows.length > 0;

    // Also load player predictions for team aggregation
    const { data: playerPreds } = await supabase
      .from("predictions")
      .select("player, team, opponent, predicted_goals, predicted_disposals, predicted_marks, predicted_behinds, p_scorer, p_2plus_goals, p_3plus_goals, p_15plus_disp, p_20plus_disp, p_25plus_disp, p_30plus_disp, p_3plus_mk, p_5plus_mk")
      .eq("year", year)
      .eq("round_number", roundNumber)
      .in("team", [homeTeam, awayTeam]);

    const preds = playerPreds ?? [];
    const homePlayers = preds.filter((p: any) => p.team === homeTeam);
    const awayPlayers = preds.filter((p: any) => p.team === awayTeam);

    // Aggregate team scores
    const homeGoals = homePlayers.reduce((s: number, p: any) => s + (p.predicted_goals ?? 0), 0);
    const awayGoals = awayPlayers.reduce((s: number, p: any) => s + (p.predicted_goals ?? 0), 0);
    const homeBehinds = homePlayers.reduce((s: number, p: any) => s + (p.predicted_behinds ?? (p.predicted_goals ?? 0) * 0.7), 0);
    const awayBehinds = awayPlayers.reduce((s: number, p: any) => s + (p.predicted_behinds ?? (p.predicted_goals ?? 0) * 0.7), 0);
    const homeScore = Math.round(homeGoals * 6 + homeBehinds);
    const awayScore = Math.round(awayGoals * 6 + awayBehinds);
    const totalScore = homeScore + awayScore;
    const margin = predMargin || homeScore - awayScore;

    // Build match outcomes
    const matchOutcomes = {
      home_win_pct: homeProb,
      away_win_pct: awayProb,
      draw_pct: 0.02,
      avg_home_score: homeScore,
      avg_away_score: awayScore,
      avg_total: totalScore,
      avg_margin: margin,
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
    // If we have real MC data, use it. Otherwise fall back to direct model probs.
    const mcLookup = new Map<string, any>();
    if (hasRealMC) {
      for (const row of mcRows!) {
        mcLookup.set(`${row.player}|${row.team}`, row);
      }
    }

    const buildSimPlayer = (p: any, isHome: boolean) => {
      const mc = mcLookup.get(`${p.player}|${p.team}`);
      const gl = p.predicted_goals ?? 0;
      const di = p.predicted_disposals ?? 0;
      const mk = p.predicted_marks ?? 0;

      // Use real MC probabilities if available, otherwise fall back to direct model
      const p1plus = mc?.mc_p_1plus_goals ?? p.p_scorer ?? (gl > 0.3 ? Math.min(0.95, 1 - Math.exp(-gl)) : 0);
      const p2plus = mc?.mc_p_2plus_goals ?? p.p_2plus_goals ?? Math.max(0, 1 - Math.exp(-gl) * (1 + gl));
      const p3plus = mc?.mc_p_3plus_goals ?? p.p_3plus_goals ?? Math.max(0, 1 - Math.exp(-gl) * (1 + gl + gl * gl / 2));
      const p15d = mc?.mc_p_15plus_disp ?? p.p_15plus_disp;
      const p20d = mc?.mc_p_20plus_disp ?? p.p_20plus_disp;
      const p25d = mc?.mc_p_25plus_disp ?? p.p_25plus_disp;
      const p30d = mc?.mc_p_30plus_disp ?? p.p_30plus_disp;

      return {
        player: p.player,
        team: p.team,
        is_home: isHome,
        has_mc: !!mc,
        goals: {
          avg: +gl.toFixed(2),
          p_1plus: +Number(p1plus).toFixed(4),
          p_2plus: +Number(p2plus).toFixed(4),
          p_3plus: +Number(p3plus).toFixed(4),
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
          p_10plus: mc?.mc_p_10plus_disp ?? +(di > 10 ? Math.min(0.95, 0.5 + (di - 10) * 0.05) : Math.max(0.05, di / 20)).toFixed(3),
          p_15plus: +(p15d ?? (di > 15 ? Math.min(0.9, 0.5 + (di - 15) * 0.05) : Math.max(0.05, di / 30))).toFixed(3),
          p_20plus: +(p20d ?? (di > 20 ? Math.min(0.85, 0.5 + (di - 20) * 0.04) : Math.max(0.02, di / 40))).toFixed(3),
          p_25plus: +(p25d ?? (di > 25 ? Math.min(0.8, 0.5 + (di - 25) * 0.03) : Math.max(0.01, di / 50))).toFixed(3),
          p_30plus: +(p30d ?? (di > 30 ? Math.min(0.7, 0.5 + (di - 30) * 0.02) : Math.max(0.005, di / 60))).toFixed(3),
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
          p_3plus: +(p.p_3plus_mk ?? (mk > 3 ? Math.min(0.9, 0.5 + (mk - 3) * 0.1) : Math.max(0.05, mk / 6))).toFixed(3),
          p_5plus: +(p.p_5plus_mk ?? (mk > 5 ? Math.min(0.85, 0.5 + (mk - 5) * 0.08) : Math.max(0.02, mk / 10))).toFixed(3),
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

    const mcPlayerCount = simPlayers.filter((p: any) => p.has_mc).length;

    return NextResponse.json({
      match_id: matchIdResolved,
      home_team: homeTeam,
      away_team: awayTeam,
      n_sims: hasRealMC ? 10000 : 0,
      has_real_mc: hasRealMC,
      mc_players: mcPlayerCount,
      match_outcomes: matchOutcomes,
      players: simPlayers,
      suggested_multis: [],
    });
  } catch (err) {
    console.error("Simulation error:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
