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

    const { data: matches } = await supabase
      .from("matches")
      .select("match_id")
      .eq("year", y)
      .eq("round_number", r);

    if (!matches?.length) return NextResponse.json([]);

    const matchIds = matches.map((m) => m.match_id);

    const [poRes, predRes] = await Promise.all([
      supabase.from("player_odds").select("*").in("match_id", matchIds),
      supabase
        .from("predictions")
        .select(
          "player, team, predicted_goals, predicted_disposals, p_scorer, p_1plus_goals, p_2plus_goals, p_3plus_goals, p_15plus_disp, p_20plus_disp, p_25plus_disp, p_30plus_disp"
        )
        .eq("year", y)
        .eq("round_number", r),
    ]);

    const playerOdds = poRes.data ?? [];
    const preds = predRes.data ?? [];

    // Build a lookup by normalized player name
    const predMap = new Map(
      preds.map((p) => [`${p.player}_${p.team}`.toLowerCase(), p])
    );

    const result = playerOdds.map((po) => {
      const key = `${po.player}_`.toLowerCase();
      // Try exact match first, then fuzzy
      let pred = predMap.get(key);
      if (!pred) {
        const poName = po.player.toLowerCase();
        for (const [k, v] of predMap) {
          if (k.includes(poName) || poName.includes(k.split("_")[0])) {
            pred = v;
            break;
          }
        }
      }

      const entries: Array<Record<string, unknown>> = [];

      if (po.market_disposal_line != null) {
        const line = po.market_disposal_line;
        let modelProb = null;
        if (pred) {
          if (line <= 12.5) modelProb = pred.p_15plus_disp;
          else if (line <= 17.5) modelProb = pred.p_20plus_disp;
          else if (line <= 22.5) modelProb = pred.p_25plus_disp;
          else if (line <= 27.5) modelProb = pred.p_30plus_disp;
          else modelProb = pred.p_30plus_disp;
        }

        entries.push({
          player: po.player,
          team: pred?.team ?? "",
          market_type: "disposals",
          market_line: line,
          market_price: po.market_disposal_over_price,
          market_implied_prob: po.market_disposal_implied_over,
          model_prob: modelProb,
          edge:
            modelProb != null && po.market_disposal_implied_over != null
              ? +(modelProb - po.market_disposal_implied_over).toFixed(4)
              : null,
        });
      }

      if (po.market_fgs_price != null) {
        entries.push({
          player: po.player,
          team: pred?.team ?? "",
          market_type: "first_goal",
          market_line: null,
          market_price: po.market_fgs_price,
          market_implied_prob: po.market_fgs_implied_prob,
          model_prob: pred?.p_scorer ?? null,
          edge:
            pred?.p_scorer != null && po.market_fgs_implied_prob != null
              ? +(pred.p_scorer - po.market_fgs_implied_prob).toFixed(4)
              : null,
        });
      }

      if (po.market_2goals_price != null) {
        entries.push({
          player: po.player,
          team: pred?.team ?? "",
          market_type: "2plus_goals",
          market_line: null,
          market_price: po.market_2goals_price,
          market_implied_prob: po.market_2goals_implied_prob,
          model_prob: pred?.p_2plus_goals ?? null,
          edge:
            pred?.p_2plus_goals != null && po.market_2goals_implied_prob != null
              ? +(pred.p_2plus_goals - po.market_2goals_implied_prob).toFixed(4)
              : null,
        });
      }

      return entries;
    });

    return NextResponse.json(result.flat());
  } catch (err) {
    console.error("route error:", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
