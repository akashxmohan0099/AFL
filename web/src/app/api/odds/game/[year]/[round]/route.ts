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

    // Get matches for this round
    const { data: matches } = await supabase
      .from("matches")
      .select("match_id, home_team, away_team, venue, date, home_score, away_score")
      .eq("year", y)
      .eq("round_number", r);

    if (!matches?.length) return NextResponse.json([]);

    const matchIds = matches.map((m) => m.match_id);

    // Load odds and game predictions in parallel
    const [oddsRes, gpRes] = await Promise.all([
      supabase.from("odds").select("*").in("match_id", matchIds),
      supabase
        .from("game_predictions")
        .select("*")
        .eq("year", y)
        .eq("round_number", r),
    ]);

    const oddsMap = new Map((oddsRes.data ?? []).map((o) => [o.match_id, o]));
    const gpMap = new Map(
      (gpRes.data ?? []).map((g) => [
        g.match_id ?? `${g.home_team}_${g.away_team}`,
        g,
      ])
    );

    const result = matches.map((m) => {
      const odds = oddsMap.get(m.match_id);
      const gp =
        gpMap.get(m.match_id) ?? gpMap.get(`${m.home_team}_${m.away_team}`);

      const modelHome = gp?.home_win_prob ?? null;
      const modelAway = gp?.away_win_prob ?? null;
      const marketHome = odds?.market_home_implied_prob ?? null;
      const marketAway = odds?.market_away_implied_prob ?? null;

      return {
        match_id: m.match_id,
        home_team: m.home_team,
        away_team: m.away_team,
        venue: m.venue,
        date: m.date,
        home_score: m.home_score,
        away_score: m.away_score,
        model_home_prob: modelHome,
        model_away_prob: modelAway,
        market_home_prob: marketHome,
        market_away_prob: marketAway,
        edge_home:
          modelHome != null && marketHome != null
            ? +(modelHome - marketHome).toFixed(4)
            : null,
        edge_away:
          modelAway != null && marketAway != null
            ? +(modelAway - marketAway).toFixed(4)
            : null,
        predicted_margin: gp?.predicted_margin ?? null,
        predicted_winner: gp?.predicted_winner ?? null,
      };
    });

    return NextResponse.json(result);
  } catch (err) {
    console.error("route error:", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
