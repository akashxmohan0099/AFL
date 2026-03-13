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

    // Load matches, game predictions, and aggregated player predictions in parallel
    const [matchesRes, gpRes, predsRes] = await Promise.all([
      supabase
        .from("matches")
        .select("match_id, home_team, away_team, date, round_number, venue, home_score, away_score")
        .eq("year", y)
        .order("round_number", { ascending: true }),
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
    if (matches.length === 0) {
      return NextResponse.json([]);
    }

    // Build game prediction lookup
    const gpByMatchId = new Map<number, any>();
    const gpByTeams = new Map<string, any>();
    for (const gp of gpRes.data ?? []) {
      if (gp.match_id) gpByMatchId.set(gp.match_id, gp);
      gpByTeams.set(`${gp.home_team}_${gp.away_team}_${gp.round_number}`, gp);
    }

    // Aggregate player predictions per team per match
    const teamPredAgg = new Map<string, { goals: number; disposals: number }>();
    for (const p of predsRes.data ?? []) {
      const key = `${p.team}|${p.match_id ?? p.round_number}`;
      const existing = teamPredAgg.get(key) ?? { goals: 0, disposals: 0 };
      existing.goals += p.predicted_goals ?? 0;
      existing.disposals += p.predicted_disposals ?? 0;
      teamPredAgg.set(key, existing);
    }

    const results = matches.map((m: any) => {
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

      // Aggregated team predictions
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
        correct,
        home_predicted_goals: homePred ? +homePred.goals.toFixed(1) : null,
        away_predicted_goals: awayPred ? +awayPred.goals.toFixed(1) : null,
        home_predicted_disposals: homePred ? +homePred.disposals.toFixed(1) : null,
        away_predicted_disposals: awayPred ? +awayPred.disposals.toFixed(1) : null,
      };
    });

    return NextResponse.json(results);
  } catch (err) {
    console.error("Season matches error:", err);
    return NextResponse.json([], { status: 500 });
  }
}
