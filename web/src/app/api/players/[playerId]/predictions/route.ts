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
    const year = parseInt(searchParams.get("year") ?? "2026", 10);

    // First, look up the player's current name and team from player_games
    const { data: playerInfo, error: playerError } = await supabase
      .from("player_games")
      .select("player, team")
      .eq("player_id", decodedPlayerId)
      .order("date", { ascending: false })
      .limit(1)
      .single();

    if (playerError || !playerInfo) {
      return NextResponse.json([]);
    }

    const { player: playerName, team: playerTeam } = playerInfo;

    // Query predictions for this player/team/year
    const { data: predictions, error: predError } = await supabase
      .from("predictions")
      .select("*")
      .eq("player", playerName)
      .eq("team", playerTeam)
      .eq("year", year)
      .order("round_number", { ascending: true });

    if (predError) {
      console.error("Player predictions error:", predError);
      return NextResponse.json([]);
    }

    return NextResponse.json(predictions ?? []);
  } catch (err) {
    console.error("Player predictions error:", err);
    return NextResponse.json([]);
  }
}
