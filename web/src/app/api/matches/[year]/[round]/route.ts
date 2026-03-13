import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ year: string; round: string }> }
) {
  try {
    const { year, round } = await params;
    const y = Number(year);
    const r = Number(round);

    // Load matches and game predictions in parallel
    const [matchesRes, gpRes] = await Promise.all([
      supabase
        .from("matches")
        .select("match_id, home_team, away_team, venue, date, year, round_number, home_score, away_score")
        .eq("year", y)
        .eq("round_number", r),
      supabase
        .from("game_predictions")
        .select("*")
        .eq("year", y)
        .eq("round_number", r),
    ]);

    if (matchesRes.error) {
      console.error("Matches query error:", matchesRes.error);
      return NextResponse.json([], { status: 500 });
    }

    const matches = matchesRes.data ?? [];
    if (matches.length === 0) {
      return NextResponse.json([]);
    }

    // Build a lookup for game predictions by match_id or home_team+away_team
    const gpByMatchId = new Map<number, Record<string, unknown>>();
    const gpByTeams = new Map<string, Record<string, unknown>>();
    for (const gp of gpRes.data ?? []) {
      if (gp.match_id) gpByMatchId.set(gp.match_id, gp);
      gpByTeams.set(`${gp.home_team}_${gp.away_team}`, gp);
    }

    const results = matches.map((m) => {
      const gp = gpByMatchId.get(m.match_id) ?? gpByTeams.get(`${m.home_team}_${m.away_team}`);

      const homeWinProb = gp?.home_win_prob != null ? Number(gp.home_win_prob) : null;
      const awayWinProb = homeWinProb != null ? +(1 - homeWinProb).toFixed(4) : null;
      const predictedWinner =
        homeWinProb != null
          ? homeWinProb > 0.5
            ? m.home_team
            : m.away_team
          : null;

      // Determine correctness
      let correct: boolean | null = null;
      if (
        predictedWinner &&
        m.home_score != null &&
        m.away_score != null
      ) {
        const actualWinner =
          m.home_score > m.away_score
            ? m.home_team
            : m.away_score > m.home_score
              ? m.away_team
              : "Draw";
        correct = predictedWinner === actualWinner;
      }

      return {
        match_id: m.match_id,
        home_team: m.home_team,
        away_team: m.away_team,
        venue: m.venue,
        date: m.date ? String(m.date).slice(0, 10) : null,
        year: m.year,
        round_number: m.round_number,
        home_score: m.home_score,
        away_score: m.away_score,
        home_win_prob: homeWinProb != null ? +homeWinProb.toFixed(4) : null,
        away_win_prob: awayWinProb,
        predicted_winner: predictedWinner,
        correct,
      };
    });

    return NextResponse.json(results);
  } catch (err) {
    console.error("Matches error:", err);
    return NextResponse.json([], { status: 500 });
  }
}
