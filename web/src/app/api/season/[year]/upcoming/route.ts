import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

/* eslint-disable @typescript-eslint/no-explicit-any */

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ year: string }> }
) {
  try {
    const { year } = await params;
    const y = Number(year);

    // Load fixtures, completed matches, game predictions, and player predictions
    const [fixRes, matchesRes, gpRes] = await Promise.all([
      supabase
        .from("fixtures")
        .select("team, opponent, venue, date, round_number")
        .eq("year", y)
        .eq("is_home", true)
        .order("round_number", { ascending: true }),
      supabase
        .from("matches")
        .select("home_team, away_team, round_number, home_score, away_score")
        .eq("year", y),
      supabase
        .from("game_predictions")
        .select("home_team, away_team, round_number, home_win_prob, predicted_winner, predicted_margin")
        .eq("year", y),
    ]);

    const fixtures = fixRes.data ?? [];
    const matches = matchesRes.data ?? [];
    const gamePreds = gpRes.data ?? [];

    // Build set of completed match keys
    const completedKeys = new Set<string>();
    for (const m of matches) {
      if (m.home_score != null && m.away_score != null) {
        completedKeys.add(`${m.home_team}|${m.away_team}|${m.round_number}`);
      }
    }

    // Find the first round with unplayed fixtures
    let upcomingRound: number | null = null;
    for (const f of fixtures) {
      const key = `${f.team}|${f.opponent}|${f.round_number}`;
      if (!completedKeys.has(key)) {
        upcomingRound = f.round_number;
        break;
      }
    }

    if (upcomingRound == null) {
      return NextResponse.json({
        year: y,
        round_number: null,
        matches: [],
        predictions: [],
        game_predictions: [],
      });
    }

    // Get unplayed fixtures for the upcoming round
    const roundFixtures = fixtures.filter((f: any) => {
      if (f.round_number !== upcomingRound) return false;
      const key = `${f.team}|${f.opponent}|${f.round_number}`;
      return !completedKeys.has(key);
    });

    // Game predictions lookup
    const gpLookup = new Map<string, any>();
    for (const gp of gamePreds) {
      gpLookup.set(`${gp.home_team}_${gp.away_team}_${gp.round_number}`, gp);
    }

    const fixtureMatches = roundFixtures.map((f: any) => {
      const gp = gpLookup.get(`${f.team}_${f.opponent}_${f.round_number}`);
      return {
        home_team: f.team,
        away_team: f.opponent,
        venue: f.venue ?? null,
        date: f.date ?? null,
        home_win_prob: gp?.home_win_prob != null ? +Number(gp.home_win_prob).toFixed(4) : null,
        predicted_winner: gp?.predicted_winner ?? null,
        predicted_margin: gp?.predicted_margin ?? null,
      };
    });

    // Load player predictions for the upcoming round
    const { data: preds } = await supabase
      .from("predictions")
      .select("player, team, opponent, predicted_goals, predicted_disposals, predicted_marks, p_scorer")
      .eq("year", y)
      .eq("round_number", upcomingRound);

    const predictions = (preds ?? []).map((p: any) => ({
      player: p.player,
      team: p.team,
      opponent: p.opponent,
      predicted_goals: p.predicted_goals ?? null,
      predicted_disposals: p.predicted_disposals ?? null,
      predicted_marks: p.predicted_marks ?? null,
      p_scorer: p.p_scorer ?? null,
    }));

    return NextResponse.json({
      year: y,
      round_number: upcomingRound,
      matches: fixtureMatches,
      predictions,
    });
  } catch (err) {
    console.error("Upcoming error:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
