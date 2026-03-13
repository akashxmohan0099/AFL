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

    // Load matches, predictions, and game predictions in parallel
    const [matchesRes, predsRes, gpRes] = await Promise.all([
      supabase
        .from("matches")
        .select("match_id, home_team, away_team, round_number, home_score, away_score")
        .eq("year", y),
      supabase
        .from("predictions")
        .select("player, team, round_number, match_id, predicted_goals, predicted_disposals, predicted_marks, p_scorer")
        .eq("year", y),
      supabase
        .from("game_predictions")
        .select("match_id, home_team, away_team, round_number, home_win_prob")
        .eq("year", y),
    ]);

    const matches = matchesRes.data ?? [];
    const predictions = predsRes.data ?? [];
    const gamePreds = gpRes.data ?? [];

    const totalMatches = matches.length;
    const roundNumbers = [...new Set(matches.map((m: any) => m.round_number))].sort(
      (a, b) => a - b
    );
    const totalRounds = roundNumbers.length > 0 ? Math.max(...roundNumbers) : 0;

    // Compute completed rounds: rounds where all matches have scores
    const matchesByRound = new Map<number, any[]>();
    for (const m of matches) {
      const arr = matchesByRound.get(m.round_number) ?? [];
      arr.push(m);
      matchesByRound.set(m.round_number, arr);
    }

    let completedRounds = 0;
    let currentRound = 1;
    const sortedRounds = [...matchesByRound.keys()].sort((a, b) => a - b);

    for (const rn of sortedRounds) {
      const roundMatches = matchesByRound.get(rn)!;
      const allPlayed = roundMatches.every(
        (m: any) => m.home_score != null && m.away_score != null
      );
      if (allPlayed) {
        completedRounds = rn;
      } else {
        currentRound = rn;
        break;
      }
    }

    if (completedRounds > 0 && currentRound <= completedRounds) {
      currentRound = completedRounds + 1;
    }

    // Load player_games for completed matches to compute accuracy
    const completedMatchIds = matches
      .filter((m: any) => m.home_score != null && m.away_score != null)
      .map((m: any) => m.match_id);

    let goalsMae: number | null = null;
    let disposalsMae: number | null = null;
    let marksMae: number | null = null;
    let scorerAccuracy: number | null = null;

    if (completedMatchIds.length > 0 && predictions.length > 0) {
      // Load actual player stats for completed matches
      const pgRes = await supabase
        .from("player_games")
        .select("player, team, match_id, gl, di, mk")
        .in("match_id", completedMatchIds.slice(0, 500));

      const playerGames = pgRes.data ?? [];
      const actualsMap = new Map<string, any>();
      for (const pg of playerGames) {
        actualsMap.set(`${pg.player}|${pg.team}|${pg.match_id}`, pg);
      }

      let goalErrors: number[] = [];
      let dispErrors: number[] = [];
      let markErrors: number[] = [];
      let scorerCorrect = 0;
      let scorerTotal = 0;

      for (const pred of predictions) {
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

      if (goalErrors.length > 0) {
        goalsMae = +(goalErrors.reduce((a, b) => a + b, 0) / goalErrors.length).toFixed(3);
      }
      if (dispErrors.length > 0) {
        disposalsMae = +(dispErrors.reduce((a, b) => a + b, 0) / dispErrors.length).toFixed(3);
      }
      if (markErrors.length > 0) {
        marksMae = +(markErrors.reduce((a, b) => a + b, 0) / markErrors.length).toFixed(3);
      }
      if (scorerTotal > 0) {
        scorerAccuracy = +(scorerCorrect / scorerTotal).toFixed(4);
      }
    }

    // Game winner accuracy
    let gameWinnerAccuracy: number | null = null;
    let gameWinnerCorrect = 0;
    let gameWinnerTotal = 0;

    const completedMatchMap = new Map<number, any>();
    for (const m of matches) {
      if (m.home_score != null && m.away_score != null) {
        completedMatchMap.set(m.match_id, m);
      }
    }

    for (const gp of gamePreds) {
      const match =
        completedMatchMap.get(gp.match_id) ??
        [...completedMatchMap.values()].find(
          (m: any) => m.home_team === gp.home_team && m.away_team === gp.away_team
        );
      if (!match || gp.home_win_prob == null) continue;

      const predictedWinner =
        gp.home_win_prob > 0.5 ? match.home_team : match.away_team;
      const actualWinner =
        match.home_score > match.away_score
          ? match.home_team
          : match.away_score > match.home_score
            ? match.away_team
            : "Draw";

      gameWinnerTotal++;
      if (predictedWinner === actualWinner) gameWinnerCorrect++;
    }

    if (gameWinnerTotal > 0) {
      gameWinnerAccuracy = +(gameWinnerCorrect / gameWinnerTotal).toFixed(4);
    }

    const result = {
      year: y,
      total_matches: totalMatches,
      total_rounds: totalRounds,
      completed_rounds: completedRounds,
      current_round: currentRound,
      rounds_list: sortedRounds,
      accuracy: {
        goals_mae: goalsMae,
        disposals_mae: disposalsMae,
        marks_mae: marksMae,
        scorer_accuracy: scorerAccuracy,
        game_winner_accuracy: gameWinnerAccuracy,
        game_winner_correct: gameWinnerCorrect,
        game_winner_total: gameWinnerTotal,
      },
    };

    return NextResponse.json(result);
  } catch (err) {
    console.error("Season summary error:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
