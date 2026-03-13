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

    // Load matches, game predictions, and player predictions in parallel
    const [matchesRes, gpRes, predsRes] = await Promise.all([
      supabase
        .from("matches")
        .select("match_id, home_team, away_team, venue, date, round_number, home_score, away_score")
        .eq("year", y)
        .order("round_number", { ascending: true }),
      supabase
        .from("game_predictions")
        .select("match_id, home_team, away_team, round_number, home_win_prob, predicted_winner")
        .eq("year", y),
      supabase
        .from("predictions")
        .select("team, match_id, round_number, predicted_goals")
        .eq("year", y),
    ]);

    const matches = matchesRes.data ?? [];
    const gamePreds = gpRes.data ?? [];
    const playerPreds = predsRes.data ?? [];

    // Build game prediction lookup
    const gpByMatchId = new Map<number, any>();
    const gpByTeamRound = new Map<string, any>();
    for (const gp of gamePreds) {
      if (gp.match_id) gpByMatchId.set(gp.match_id, gp);
      gpByTeamRound.set(`${gp.home_team}_${gp.away_team}_${gp.round_number}`, gp);
    }

    // Aggregate predicted goals per team per match/round
    const teamGoals = new Map<string, number>();
    for (const p of playerPreds) {
      const key = `${p.team}|${p.match_id ?? p.round_number}`;
      teamGoals.set(key, (teamGoals.get(key) ?? 0) + (p.predicted_goals ?? 0));
    }

    // Group matches by round
    const roundMap = new Map<number, any[]>();
    for (const m of matches) {
      const arr = roundMap.get(m.round_number) ?? [];
      arr.push(m);
      roundMap.set(m.round_number, arr);
    }

    // Determine round status
    const rounds = [...roundMap.keys()].sort((a, b) => a - b).map((rn) => {
      const roundMatches = roundMap.get(rn)!;
      const playedCount = roundMatches.filter(
        (m: any) => m.home_score != null && m.away_score != null
      ).length;

      let status: "completed" | "in_progress" | "upcoming" | "future";
      if (playedCount === roundMatches.length) {
        status = "completed";
      } else if (playedCount > 0) {
        status = "in_progress";
      } else {
        // Check if predictions exist
        const hasPreds = gamePreds.some((gp: any) => gp.round_number === rn);
        status = hasPreds ? "upcoming" : "future";
      }

      const matchEntries = roundMatches.map((m: any) => {
        const gp =
          gpByMatchId.get(m.match_id) ??
          gpByTeamRound.get(`${m.home_team}_${m.away_team}_${rn}`);

        const homeWinProb = gp?.home_win_prob != null ? +Number(gp.home_win_prob).toFixed(4) : null;
        const awayWinProb = homeWinProb != null ? +(1 - homeWinProb).toFixed(4) : null;
        const predictedWinner =
          gp?.predicted_winner ??
          (homeWinProb != null
            ? homeWinProb > 0.5
              ? m.home_team
              : m.away_team
            : null);

        const homeGoalKey1 = `${m.home_team}|${m.match_id}`;
        const homeGoalKey2 = `${m.home_team}|${rn}`;
        const awayGoalKey1 = `${m.away_team}|${m.match_id}`;
        const awayGoalKey2 = `${m.away_team}|${rn}`;

        const homePredGoals = teamGoals.get(homeGoalKey1) ?? teamGoals.get(homeGoalKey2) ?? null;
        const awayPredGoals = teamGoals.get(awayGoalKey1) ?? teamGoals.get(awayGoalKey2) ?? null;

        return {
          home_team: m.home_team,
          away_team: m.away_team,
          venue: m.venue ?? null,
          date: m.date ? String(m.date).slice(0, 10) : null,
          match_id: m.match_id,
          home_score: m.home_score ?? null,
          away_score: m.away_score ?? null,
          home_win_prob: homeWinProb,
          away_win_prob: awayWinProb,
          predicted_winner: predictedWinner,
          home_predicted_goals: homePredGoals != null ? +homePredGoals.toFixed(1) : null,
          away_predicted_goals: awayPredGoals != null ? +awayPredGoals.toFixed(1) : null,
        };
      });

      return {
        round_number: rn,
        status,
        matches: matchEntries,
      };
    });

    return NextResponse.json({ year: y, rounds });
  } catch (err) {
    console.error("Season schedule error:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
