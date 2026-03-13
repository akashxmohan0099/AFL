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

    // Load completed matches, fixtures, game predictions, and player predictions
    const [matchesRes, fixRes, gpRes, predsRes] = await Promise.all([
      supabase
        .from("matches")
        .select("match_id, home_team, away_team, date, round_number, venue, home_score, away_score")
        .eq("year", y)
        .order("round_number", { ascending: true }),
      supabase
        .from("fixtures")
        .select("team, opponent, venue, date, round_number")
        .eq("year", y)
        .eq("is_home", true),
      supabase
        .from("game_predictions")
        .select("match_id, home_team, away_team, round_number, home_win_prob, predicted_winner, predicted_margin")
        .eq("year", y),
      supabase
        .from("predictions")
        .select("team, match_id, round_number, predicted_goals, predicted_disposals")
        .eq("year", y),
    ]);

    const matches = matchesRes.data ?? [];
    const fixtures = fixRes.data ?? [];
    const gamePreds = gpRes.data ?? [];
    const playerPreds = predsRes.data ?? [];

    // Build set of completed match keys
    const completedKeys = new Set<string>();
    for (const m of matches) {
      completedKeys.add(`${m.home_team}|${m.away_team}|${m.round_number}`);
    }

    // Build game prediction lookup
    const gpByMatchId = new Map<number, any>();
    const gpByTeams = new Map<string, any>();
    for (const gp of gamePreds) {
      if (gp.match_id) gpByMatchId.set(gp.match_id, gp);
      gpByTeams.set(`${gp.home_team}_${gp.away_team}_${gp.round_number}`, gp);
    }

    // Aggregate player predictions per team per round
    const teamPredAgg = new Map<string, { goals: number; disposals: number }>();
    for (const p of playerPreds) {
      const key = `${p.team}|${p.match_id ?? p.round_number}`;
      const existing = teamPredAgg.get(key) ?? { goals: 0, disposals: 0 };
      existing.goals += p.predicted_goals ?? 0;
      existing.disposals += p.predicted_disposals ?? 0;
      teamPredAgg.set(key, existing);
    }

    // Build results from completed matches
    const results: any[] = matches.map((m: any) => {
      const gp =
        gpByMatchId.get(m.match_id) ??
        gpByTeams.get(`${m.home_team}_${m.away_team}_${m.round_number}`);

      const homeWinProb = gp?.home_win_prob != null ? +Number(gp.home_win_prob).toFixed(4) : null;
      const predictedWinner =
        gp?.predicted_winner ??
        (homeWinProb != null
          ? homeWinProb > 0.5
            ? m.home_team
            : m.away_team
          : null);

      let correct: boolean | null = null;
      if (predictedWinner && m.home_score != null && m.away_score != null) {
        const actualWinner =
          m.home_score > m.away_score
            ? m.home_team
            : m.away_score > m.home_score
              ? m.away_team
              : "Draw";
        correct = predictedWinner === actualWinner;
      }

      const homeKey1 = `${m.home_team}|${m.match_id}`;
      const homeKey2 = `${m.home_team}|${m.round_number}`;
      const awayKey1 = `${m.away_team}|${m.match_id}`;
      const awayKey2 = `${m.away_team}|${m.round_number}`;
      const homePred = teamPredAgg.get(homeKey1) ?? teamPredAgg.get(homeKey2);
      const awayPred = teamPredAgg.get(awayKey1) ?? teamPredAgg.get(awayKey2);

      return {
        match_id: m.match_id,
        home_team: m.home_team,
        away_team: m.away_team,
        date: m.date ? String(m.date).slice(0, 10) : null,
        round_number: m.round_number,
        venue: m.venue,
        home_score: m.home_score,
        away_score: m.away_score,
        home_win_prob: homeWinProb,
        predicted_winner: predictedWinner,
        predicted_margin: gp?.predicted_margin ?? null,
        correct,
        home_predicted_goals: homePred ? +homePred.goals.toFixed(1) : null,
        away_predicted_goals: awayPred ? +awayPred.goals.toFixed(1) : null,
        home_predicted_disposals: homePred ? +homePred.disposals.toFixed(1) : null,
        away_predicted_disposals: awayPred ? +awayPred.disposals.toFixed(1) : null,
      };
    });

    // Add upcoming fixtures that don't have completed matches
    for (const f of fixtures) {
      const key = `${f.team}|${f.opponent}|${f.round_number}`;
      if (completedKeys.has(key)) continue;

      const gp = gpByTeams.get(`${f.team}_${f.opponent}_${f.round_number}`);
      const homeWinProb = gp?.home_win_prob != null ? +Number(gp.home_win_prob).toFixed(4) : null;
      const predictedWinner =
        gp?.predicted_winner ??
        (homeWinProb != null ? (homeWinProb > 0.5 ? f.team : f.opponent) : null);

      const homePred = teamPredAgg.get(`${f.team}|${f.round_number}`);
      const awayPred = teamPredAgg.get(`${f.opponent}|${f.round_number}`);

      results.push({
        match_id: null,
        home_team: f.team,
        away_team: f.opponent,
        date: f.date ?? null,
        round_number: f.round_number,
        venue: f.venue,
        home_score: null,
        away_score: null,
        home_win_prob: homeWinProb,
        predicted_winner: predictedWinner,
        predicted_margin: gp?.predicted_margin ?? null,
        correct: null,
        home_predicted_goals: homePred ? +homePred.goals.toFixed(1) : null,
        away_predicted_goals: awayPred ? +awayPred.goals.toFixed(1) : null,
        home_predicted_disposals: homePred ? +homePred.disposals.toFixed(1) : null,
        away_predicted_disposals: awayPred ? +awayPred.disposals.toFixed(1) : null,
      });
    }

    // Sort by round number
    results.sort((a, b) => a.round_number - b.round_number);

    return NextResponse.json(results);
  } catch (err) {
    console.error("Season matches error:", err);
    return NextResponse.json([], { status: 500 });
  }
}
