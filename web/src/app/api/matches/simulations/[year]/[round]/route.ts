import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

/* eslint-disable @typescript-eslint/no-explicit-any */

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ year: string; round: string }> }
) {
  try {
    const { year, round } = await params;
    const y = Number(year);
    const r = Number(round);

    const { data, error } = await supabase
      .from("simulations")
      .select("*")
      .eq("year", y)
      .eq("round_number", r);

    if (error) {
      console.error("Round simulations query error:", error);
      return NextResponse.json([], { status: 500 });
    }

    if (!data || data.length === 0) {
      return NextResponse.json([]);
    }

    const results = data.map((sim: any) => {
      const outcomes = sim.match_outcomes ?? {};
      const players: any[] = sim.players ?? [];

      // Extract top 3 goal scorers from player data
      const topScorers = [...players]
        .sort((a, b) => {
          const aGoalPct = a.goals?.p_1plus ?? 0;
          const bGoalPct = b.goals?.p_1plus ?? 0;
          return bGoalPct - aGoalPct;
        })
        .slice(0, 3)
        .map((p) => ({
          player: p.player,
          team: p.team,
          goal_pct: p.goals?.p_1plus ?? 0,
        }));

      return {
        match_id: sim.match_id,
        home_team: sim.home_team,
        away_team: sim.away_team,
        n_sims: sim.n_sims ?? 0,
        home_win_pct: outcomes.home_win_pct ?? null,
        away_win_pct: outcomes.away_win_pct ?? null,
        draw_pct: outcomes.draw_pct ?? null,
        avg_total: outcomes.avg_total ?? null,
        avg_margin: outcomes.avg_margin ?? null,
        avg_home_score: outcomes.avg_home_score ?? null,
        avg_away_score: outcomes.avg_away_score ?? null,
        score_range: {
          home: outcomes.score_distribution?.home ?? null,
          away: outcomes.score_distribution?.away ?? null,
        },
        top_scorers: topScorers,
      };
    });

    return NextResponse.json(results);
  } catch (err) {
    console.error("Round simulations error:", err);
    return NextResponse.json([], { status: 500 });
  }
}
