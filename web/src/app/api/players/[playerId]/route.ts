import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

// ---- Helpers ----

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function computePercentiles(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    p10: round2(percentile(sorted, 10)),
    p25: round2(percentile(sorted, 25)),
    median: round2(percentile(sorted, 50)),
    p75: round2(percentile(sorted, 75)),
    p90: round2(percentile(sorted, 90)),
  };
}

function round2(v: number): number {
  return Math.round(v * 100) / 100;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type GameRow = Record<string, any>;

const STAT_COLS = [
  "gl", "bh", "di", "mk", "ki", "hb", "tk", "ho",
  "cp", "up", "cl", "cg", "ff", "fa", "cm", "mi",
  "one_pct", "rb", "bo", "ga",
] as const;

function computeAverages(rows: GameRow[]) {
  const result: Record<string, number> = {};
  for (const col of STAT_COLS) {
    const vals = rows.map((r) => Number(r[col] ?? 0));
    result[col] = round2(mean(vals));
  }
  return result;
}

function computeHighs(rows: GameRow[]) {
  const highs: Record<string, { value: number; match_id: number; opponent: string; date: string; venue: string }> = {};
  for (const col of STAT_COLS) {
    let maxVal = -1;
    let maxRow: GameRow | null = null;
    for (const r of rows) {
      const v = Number(r[col] ?? 0);
      if (v > maxVal) {
        maxVal = v;
        maxRow = r;
      }
    }
    if (maxRow) {
      highs[col] = {
        value: maxVal,
        match_id: maxRow.match_id,
        opponent: maxRow.opponent,
        date: maxRow.date,
        venue: maxRow.venue,
      };
    }
  }
  return highs;
}

function computeStreaks(rows: GameRow[]) {
  // rows should be sorted by date DESC (most recent first)
  const thresholds: Record<string, { stat: string; min: number }[]> = {
    goals: [
      { stat: "gl", min: 1 },
      { stat: "gl", min: 2 },
      { stat: "gl", min: 3 },
    ],
    disposals: [
      { stat: "di", min: 20 },
      { stat: "di", min: 25 },
      { stat: "di", min: 30 },
    ],
    marks: [{ stat: "mk", min: 5 }],
    tackles: [{ stat: "tk", min: 4 }],
  };

  const result: Record<string, Record<string, number>> = {};

  for (const [category, checks] of Object.entries(thresholds)) {
    result[category] = {};
    for (const { stat, min } of checks) {
      let streak = 0;
      for (const row of rows) {
        if (Number(row[stat] ?? 0) >= min) {
          streak++;
        } else {
          break;
        }
      }
      result[category][`${min}+`] = streak;
    }
  }

  return result;
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ playerId: string }> }
) {
  try {
    const { playerId } = await params;
    const decodedPlayerId = decodeURIComponent(playerId);

    // Fetch all player_games rows for this player
    // The `if` column is a reserved word — use quoted select
    const { data: games, error } = await supabase
      .from("player_games")
      .select(
        `match_id, player_id, player, team, opponent, venue, date, year, round_number,
         is_home, gl, bh, di, mk, ki, hb, tk, ho, cp, up, "if", cl, cg, ff, fa, cm, mi,
         one_pct, rb, bo, ga, q1_goals, q2_goals, q3_goals, q4_goals,
         is_finals, age_years, career_games_pre, career_goals_pre`
      )
      .eq("player_id", decodedPlayerId)
      .order("date", { ascending: false });

    if (error) {
      console.error("Player profile error:", error);
      return NextResponse.json({ error: "Failed to fetch player data" }, { status: 500 });
    }

    if (!games || games.length === 0) {
      return NextResponse.json({ error: "Player not found" }, { status: 404 });
    }

    const latestGame = games[0];
    const playerName = latestGame.player;
    const playerTeam = latestGame.team;
    const totalGames = games.length;

    // ---- Career Stats ----
    const careerAvgs = computeAverages(games);
    const careerHighs = computeHighs(games);

    // ---- Season Breakdown ----
    const byYear = new Map<number, GameRow[]>();
    for (const g of games) {
      const yr = g.year;
      if (!byYear.has(yr)) byYear.set(yr, []);
      byYear.get(yr)!.push(g);
    }

    const seasons = Array.from(byYear.entries())
      .map(([year, rows]) => ({
        year,
        games: rows.length,
        ...computeAverages(rows),
      }))
      .sort((a, b) => b.year - a.year);

    // ---- Recent Form (last 10) ----
    const recentGames = games.slice(0, 10);
    const recentForm = recentGames.map((g) => ({
      date: g.date,
      opponent: g.opponent,
      venue: g.venue,
      round_number: g.round_number,
      year: g.year,
      is_home: g.is_home,
      gl: g.gl,
      bh: g.bh,
      di: g.di,
      mk: g.mk,
      ki: g.ki,
      hb: g.hb,
      tk: g.tk,
      ho: g.ho,
    }));

    // ---- Home vs Away Splits ----
    const homeGames = games.filter((g) => g.is_home);
    const awayGames = games.filter((g) => !g.is_home);
    const homeAwaySplits = {
      home: { games: homeGames.length, ...computeAverages(homeGames) },
      away: { games: awayGames.length, ...computeAverages(awayGames) },
    };

    // ---- Opponent Splits (min 2 games) ----
    const byOpponent = new Map<string, GameRow[]>();
    for (const g of games) {
      if (!byOpponent.has(g.opponent)) byOpponent.set(g.opponent, []);
      byOpponent.get(g.opponent)!.push(g);
    }
    const opponentSplits = Array.from(byOpponent.entries())
      .filter(([, rows]) => rows.length >= 2)
      .map(([opponent, rows]) => ({
        opponent,
        games: rows.length,
        ...computeAverages(rows),
      }))
      .sort((a, b) => b.games - a.games);

    // ---- Venue Splits ----
    const byVenue = new Map<string, GameRow[]>();
    for (const g of games) {
      if (!byVenue.has(g.venue)) byVenue.set(g.venue, []);
      byVenue.get(g.venue)!.push(g);
    }
    const venueSplits = Array.from(byVenue.entries())
      .map(([venue, rows]) => ({
        venue,
        games: rows.length,
        ...computeAverages(rows),
      }))
      .sort((a, b) => b.games - a.games);

    // ---- Streaks ----
    const streaks = computeStreaks(games); // games already sorted date DESC

    // ---- Consistency (last 2 seasons) ----
    const currentYear = latestGame.year;
    const last2Seasons = games.filter(
      (g) => g.year === currentYear || g.year === currentYear - 1
    );
    const consistency: Record<string, ReturnType<typeof computePercentiles>> = {};
    for (const stat of ["gl", "di", "mk", "tk"] as const) {
      const vals = last2Seasons.map((g) => Number(g[stat] ?? 0));
      if (vals.length > 0) {
        consistency[stat] = computePercentiles(vals);
      }
    }

    // ---- Quarter Scoring ----
    const qGames = games.filter(
      (g) =>
        g.q1_goals != null &&
        g.q2_goals != null &&
        g.q3_goals != null &&
        g.q4_goals != null
    );
    const quarterScoring = {
      q1: round2(mean(qGames.map((g) => Number(g.q1_goals ?? 0)))),
      q2: round2(mean(qGames.map((g) => Number(g.q2_goals ?? 0)))),
      q3: round2(mean(qGames.map((g) => Number(g.q3_goals ?? 0)))),
      q4: round2(mean(qGames.map((g) => Number(g.q4_goals ?? 0)))),
      games: qGames.length,
    };

    // ---- Predictions vs Actuals ----
    let predictionsVsActuals: Record<string, unknown>[] = [];
    try {
      // Get predictions for this player
      const { data: predictions } = await supabase
        .from("predictions")
        .select("*")
        .eq("player", playerName)
        .eq("team", playerTeam);

      // Get outcomes for this player
      const { data: outcomes } = await supabase
        .from("outcomes")
        .select("*")
        .eq("player", playerName)
        .eq("team", playerTeam);

      if (predictions && outcomes) {
        // Join on year + round_number + player + team
        const outcomeMap = new Map<string, GameRow>();
        for (const o of outcomes) {
          const key = `${o.year}_${o.round_number}_${o.player}_${o.team}`;
          outcomeMap.set(key, o);
        }

        predictionsVsActuals = predictions
          .map((p) => {
            const key = `${p.year}_${p.round_number}_${p.player}_${p.team}`;
            const o = outcomeMap.get(key);
            if (!o) return null;
            return {
              year: p.year,
              round_number: p.round_number,
              opponent: p.opponent ?? o.opponent,
              venue: p.venue ?? o.venue,
              predicted_goals: p.predicted_goals,
              actual_goals: o.actual_goals ?? o.gl,
              predicted_disposals: p.predicted_disposals,
              actual_disposals: o.actual_disposals ?? o.di,
              predicted_marks: p.predicted_marks,
              actual_marks: o.actual_marks ?? o.mk,
            };
          })
          .filter(Boolean) as Record<string, unknown>[];
      }
    } catch {
      // predictions/outcomes tables may not exist yet
    }

    // ---- Prediction MAE ----
    let predictionMae = { goals: 0, disposals: 0, marks: 0 };
    if (predictionsVsActuals.length > 0) {
      const pva = predictionsVsActuals as {
        predicted_goals?: number;
        actual_goals?: number;
        predicted_disposals?: number;
        actual_disposals?: number;
        predicted_marks?: number;
        actual_marks?: number;
      }[];
      const goalErrors = pva
        .filter((p) => p.predicted_goals != null && p.actual_goals != null)
        .map((p) => Math.abs((p.predicted_goals ?? 0) - (p.actual_goals ?? 0)));
      const dispErrors = pva
        .filter((p) => p.predicted_disposals != null && p.actual_disposals != null)
        .map((p) => Math.abs((p.predicted_disposals ?? 0) - (p.actual_disposals ?? 0)));
      const markErrors = pva
        .filter((p) => p.predicted_marks != null && p.actual_marks != null)
        .map((p) => Math.abs((p.predicted_marks ?? 0) - (p.actual_marks ?? 0)));

      predictionMae = {
        goals: round2(mean(goalErrors)),
        disposals: round2(mean(dispErrors)),
        marks: round2(mean(markErrors)),
      };
    }

    // ---- Build Response ----
    const profile = {
      player_id: decodedPlayerId,
      name: playerName,
      team: playerTeam,
      total_games: totalGames,
      career_goals: games.reduce((sum, g) => sum + Number(g.gl ?? 0), 0),
      career_goal_avg: round2(mean(games.map((g) => Number(g.gl ?? 0)))),
      career_averages: careerAvgs,
      career_highs: careerHighs,
      seasons,
      recent_form: recentForm,
      home_away_splits: homeAwaySplits,
      opponent_splits: opponentSplits,
      venue_splits: venueSplits,
      streaks,
      consistency,
      quarter_scoring: quarterScoring,
      predictions_vs_actuals: predictionsVsActuals,
      prediction_mae: predictionMae,
    };

    return NextResponse.json(profile);
  } catch (err) {
    console.error("Player profile error:", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
