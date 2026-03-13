import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ playerId: string }> }
) {
  try {
    const { playerId } = await params;
    const decodedPlayerId = decodeURIComponent(playerId);
    const { searchParams } = new URL(request.url);

    const year = searchParams.get("year")
      ? parseInt(searchParams.get("year")!, 10)
      : null;
    const limit = Math.min(parseInt(searchParams.get("limit") ?? "50", 10), 200);
    const offset = parseInt(searchParams.get("offset") ?? "0", 10);

    let query = supabase
      .from("player_games")
      .select(
        "match_id, player_id, player, team, opponent, venue, date, year, round_number, is_home, gl, bh, di, mk, ki, hb, tk, ho"
      )
      .eq("player_id", decodedPlayerId)
      .order("date", { ascending: false })
      .range(offset, offset + limit - 1);

    if (year) {
      query = query.eq("year", year);
    }

    const { data, error } = await query;

    if (error) {
      console.error("Player games error:", error);
      return NextResponse.json([]);
    }

    if (!data) {
      return NextResponse.json([]);
    }

    const entries = data.map((g) => ({
      match_id: g.match_id,
      date: g.date,
      year: g.year,
      round_number: g.round_number,
      team: g.team,
      opponent: g.opponent,
      venue: g.venue,
      is_home: g.is_home,
      GL: g.gl,
      BH: g.bh,
      DI: g.di,
      MK: g.mk,
      KI: g.ki,
      HB: g.hb,
      TK: g.tk,
      HO: g.ho,
    }));

    return NextResponse.json(entries);
  } catch (err) {
    console.error("Player games error:", err);
    return NextResponse.json([]);
  }
}
