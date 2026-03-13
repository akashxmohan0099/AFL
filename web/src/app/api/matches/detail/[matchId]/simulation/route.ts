import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ matchId: string }> }
) {
  try {
    const { matchId } = await params;
    const id = Number(matchId);

    const { data, error } = await supabase
      .from("simulations")
      .select("*")
      .eq("match_id", id)
      .limit(1)
      .maybeSingle();

    if (error) {
      console.error("Simulation query error:", error);
      return NextResponse.json(
        { error: "Failed to load simulation" },
        { status: 500 }
      );
    }

    if (!data) {
      return NextResponse.json(
        { error: "No simulation found for this match" },
        { status: 404 }
      );
    }

    // The simulations table stores match_outcomes, players, and
    // suggested_multis as JSONB columns
    const result = {
      match_id: data.match_id,
      home_team: data.home_team,
      away_team: data.away_team,
      n_sims: data.n_sims ?? 0,
      match_outcomes: data.match_outcomes ?? null,
      players: data.players ?? [],
      suggested_multis: data.suggested_multis ?? [],
    };

    return NextResponse.json(result);
  } catch (err) {
    console.error("Simulation error:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
