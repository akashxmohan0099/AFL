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

    // Load matches, predictions, game predictions, and player actuals
    const [matchesRes, predsRes, gpRes] = await Promise.all([
      supabase
        .from("matches")
        .select("match_id, home_team, away_team, round_number, home_score, away_score")
        .eq("year", y),
      supabase
        .from("predictions")
        .select("player, team, match_id, round_number, predicted_goals, predicted_disposals, predicted_marks, p_scorer")
        .eq("year", y),
      supabase
        .from("game_predictions")
        .select("match_id, home_team, away_team, round_number, home_win_prob")
        .eq("year", y),
    ]);

    const matches = matchesRes.data ?? [];
    const predictions = predsRes.data ?? [];
    const gamePreds = gpRes.data ?? [];

    // Find completed matches
    const completedMatches = matches.filter(
      (m: any) => m.home_score != null && m.away_score != null
    );
    const completedMatchIds = completedMatches.map((m: any) => m.match_id);

    if (completedMatchIds.length === 0) {
      return NextResponse.json([]);
    }

    // Load player_games for completed matches
    // Supabase .in() has a limit, so batch if needed
    const batchSize = 500;
    const allPlayerGames: any[] = [];
    for (let i = 0; i < completedMatchIds.length; i += batchSize) {
      const batch = completedMatchIds.slice(i, i + batchSize);
      const { data } = await supabase
        .from("player_games")
        .select("player, team, match_id, gl, di, mk")
        .in("match_id", batch);
      if (data) allPlayerGames.push(...data);
    }

    // Build actuals map
    const actualsMap = new Map<string, any>();
    for (const pg of allPlayerGames) {
      actualsMap.set(`${pg.player}|${pg.team}|${pg.match_id}`, pg);
    }

    // Build match lookup
    const matchMap = new Map<number, any>();
    for (const m of completedMatches) {
      matchMap.set(m.match_id, m);
    }

    // Group predictions by round
    const predsByRound = new Map<number, any[]>();
    for (const p of predictions) {
      if (!completedMatchIds.includes(p.match_id)) continue;
      const arr = predsByRound.get(p.round_number) ?? [];
      arr.push(p);
      predsByRound.set(p.round_number, arr);
    }

    // Group game predictions by round
    const gpByRound = new Map<number, any[]>();
    for (const gp of gamePreds) {
      const arr = gpByRound.get(gp.round_number) ?? [];
      arr.push(gp);
      gpByRound.set(gp.round_number, arr);
    }

    // Compute per-round accuracy
    const completedRounds = new Set(completedMatches.map((m: any) => m.round_number));
    const results: any[] = [];

    for (const rn of [...completedRounds].sort((a, b) => a - b)) {
      const roundPreds = predsByRound.get(rn) ?? [];
      const roundGps = gpByRound.get(rn) ?? [];

      const goalErrors: number[] = [];
      const dispErrors: number[] = [];
      const markErrors: number[] = [];
      let scorerCorrect = 0;
      let scorerTotal = 0;

      for (const pred of roundPreds) {
        const key = `${pred.player}|${pred.team}|${pred.match_id}`;
        const actual = actualsMap.get(key);
        if (!actual) continue;

        if (pred.predicted_goals != null && actual.gl != null) {
          goalErrors.push(Math.abs(pred.predicted_goals - actual.gl));
        }
        if (pred.predicted_disposals != null && actual.di != null) {
          dispErrors.push(Math.abs(pred.predicted_disposals - actual.di));
        }
        if (pred.predicted_marks != null && actual.mk != null) {
          markErrors.push(Math.abs(pred.predicted_marks - actual.mk));
        }
        if (pred.p_scorer != null && actual.gl != null) {
          const predictedScorer = pred.p_scorer >= 0.5;
          const actuallyScored = actual.gl >= 1;
          if (predictedScorer === actuallyScored) scorerCorrect++;
          scorerTotal++;
        }
      }

      // Game winner accuracy for the round
      let gwCorrect = 0;
      let gwTotal = 0;
      for (const gp of roundGps) {
        if (gp.home_win_prob == null) continue;
        const match =
          matchMap.get(gp.match_id) ??
          [...matchMap.values()].find(
            (m: any) =>
              m.home_team === gp.home_team &&
              m.away_team === gp.away_team &&
              m.round_number === rn
          );
        if (!match) continue;

        const predictedWinner =
          gp.home_win_prob > 0.5 ? match.home_team : match.away_team;
        const actualWinner =
          match.home_score > match.away_score
            ? match.home_team
            : match.away_score > match.home_score
              ? match.away_team
              : "Draw";

        gwTotal++;
        if (predictedWinner === actualWinner) gwCorrect++;
      }

      results.push({
        round_number: rn,
        n_players: goalErrors.length || dispErrors.length || markErrors.length,
        goals_mae:
          goalErrors.length > 0
            ? +(goalErrors.reduce((a, b) => a + b, 0) / goalErrors.length).toFixed(3)
            : null,
        disposals_mae:
          dispErrors.length > 0
            ? +(dispErrors.reduce((a, b) => a + b, 0) / dispErrors.length).toFixed(3)
            : null,
        marks_mae:
          markErrors.length > 0
            ? +(markErrors.reduce((a, b) => a + b, 0) / markErrors.length).toFixed(3)
            : null,
        scorer_accuracy:
          scorerTotal > 0 ? +(scorerCorrect / scorerTotal).toFixed(4) : null,
        game_winner_accuracy:
          gwTotal > 0 ? +(gwCorrect / gwTotal).toFixed(4) : null,
      });
    }

    return NextResponse.json(results);
  } catch (err) {
    console.error("Round accuracy error:", err);
    return NextResponse.json([], { status: 500 });
  }
}
