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

    // Load predictions for the year
    const { data: predictions, error: predErr } = await supabase
      .from("predictions")
      .select(
        "player, team, opponent, match_id, round_number, venue, predicted_goals, predicted_disposals, predicted_marks, p_scorer"
      )
      .eq("year", y);

    if (predErr) {
      console.error("Predictions history query error:", predErr);
      return NextResponse.json(
        { error: "Failed to load predictions" },
        { status: 500 }
      );
    }

    const preds = predictions ?? [];
    if (preds.length === 0) {
      return NextResponse.json({
        summary: {
          total_predictions: 0,
          goals_mae: null,
          disposals_mae: null,
          marks_mae: null,
          scorer_accuracy: null,
        },
        entries: [],
      });
    }

    // Get all unique match_ids from predictions
    const matchIds = [...new Set(preds.map((p: any) => p.match_id).filter(Boolean))];

    // Load player actuals for those matches (batch to stay under .in() limit)
    const batchSize = 500;
    const allActuals: any[] = [];
    for (let i = 0; i < matchIds.length; i += batchSize) {
      const batch = matchIds.slice(i, i + batchSize);
      const { data } = await supabase
        .from("player_games")
        .select("player, team, match_id, gl, di, mk")
        .in("match_id", batch);
      if (data) allActuals.push(...data);
    }

    // Build actuals lookup
    const actualsMap = new Map<string, any>();
    for (const pg of allActuals) {
      actualsMap.set(`${pg.player}|${pg.team}|${pg.match_id}`, pg);
    }

    // Build entries and compute MAE
    const entries: any[] = [];
    let goalErrors: number[] = [];
    let dispErrors: number[] = [];
    let markErrors: number[] = [];
    let scorerCorrect = 0;
    let scorerTotal = 0;

    for (const pred of preds) {
      const key = `${pred.player}|${pred.team}|${pred.match_id}`;
      const actual = actualsMap.get(key);

      // Only include entries where we have actual outcomes
      if (!actual) continue;

      const entry: any = {
        round: pred.round_number,
        player: pred.player,
        team: pred.team,
        opponent: pred.opponent ?? "",
        venue: pred.venue ?? "",
        match_id: pred.match_id,
        predicted_goals: pred.predicted_goals ?? null,
        actual_goals: actual.gl ?? null,
        predicted_disposals: pred.predicted_disposals ?? null,
        actual_disposals: actual.di ?? null,
        predicted_marks: pred.predicted_marks ?? null,
        actual_marks: actual.mk ?? null,
        p_scorer: pred.p_scorer ?? null,
        actually_scored: actual.gl != null ? actual.gl >= 1 : null,
      };

      entries.push(entry);

      // Accumulate errors
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

    const summary = {
      total_predictions: entries.length,
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
    };

    return NextResponse.json({ summary, entries });
  } catch (err) {
    console.error("Predictions history error:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
