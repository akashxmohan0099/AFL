import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ year: string }> }
) {
  try {
  const { year } = await params;
  const y = Number(year);

  // Load predictions and outcomes for the year
  const [predRes, outRes] = await Promise.all([
    supabase
      .from("predictions")
      .select("round_number, player, team, match_id, predicted_goals, predicted_disposals, predicted_marks, p_scorer, p_1plus_goals")
      .eq("year", y),
    supabase
      .from("outcomes")
      .select("round_number, player, team, match_id, actual_goals, actual_disposals, actual_marks")
      .eq("year", y),
  ]);

  const preds = predRes.data ?? [];
  const outs = outRes.data ?? [];

  if (!preds.length || !outs.length) {
    return NextResponse.json({
      year: y,
      n_predictions: 0,
      rounds: 0,
      rounds_with_data: 0,
    });
  }

  // Join on (round_number, player, team)
  const outMap = new Map(
    outs.map((o) => [`${o.round_number}_${o.player}_${o.team}`, o])
  );

  let goalErrors = 0, dispErrors = 0, markErrors = 0;
  let goalCount = 0, dispCount = 0, markCount = 0;
  let scorerCorrect = 0, scorerTotal = 0;
  const roundsSet = new Set<number>();

  for (const p of preds) {
    const key = `${p.round_number}_${p.player}_${p.team}`;
    const o = outMap.get(key);
    if (!o) continue;

    roundsSet.add(p.round_number);

    if (p.predicted_goals != null && o.actual_goals != null) {
      goalErrors += Math.abs(p.predicted_goals - o.actual_goals);
      goalCount++;

      // Scorer accuracy: did we predict 1+ goal correctly?
      const predScorer = (p.p_1plus_goals ?? p.p_scorer ?? 0) >= 0.5;
      const actualScorer = o.actual_goals >= 1;
      if (predScorer === actualScorer) scorerCorrect++;
      scorerTotal++;
    }
    if (p.predicted_disposals != null && o.actual_disposals != null) {
      dispErrors += Math.abs(p.predicted_disposals - o.actual_disposals);
      dispCount++;
    }
    if (p.predicted_marks != null && o.actual_marks != null) {
      markErrors += Math.abs(p.predicted_marks - o.actual_marks);
      markCount++;
    }
  }

  return NextResponse.json({
    year: y,
    n_predictions: preds.length,
    rounds: roundsSet.size,
    rounds_with_data: roundsSet.size,
    goals_mae: goalCount > 0 ? +(goalErrors / goalCount).toFixed(4) : null,
    disposals_mae: dispCount > 0 ? +(dispErrors / dispCount).toFixed(2) : null,
    marks_mae: markCount > 0 ? +(markErrors / markCount).toFixed(2) : null,
    scorer_accuracy: scorerTotal > 0 ? +((scorerCorrect / scorerTotal) * 100).toFixed(1) : null,
  });
  } catch (err) {
    console.error("route error:", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
