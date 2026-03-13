import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const q = searchParams.get("q")?.trim() ?? "";
    const limit = Math.min(parseInt(searchParams.get("limit") ?? "20", 10), 100);

    if (!q) {
      return NextResponse.json([]);
    }

    // Build ilike patterns:
    // "cripps" → %cripps%
    // "patrick cripps" → also try %cripps, patrick% (Surname, First format)
    const parts = q.split(/\s+/);
    const patterns: string[] = [`%${q}%`];

    // If multiple words, also search "Last, First" format
    if (parts.length >= 2) {
      // Try "LastWord, everything else" pattern
      const last = parts[parts.length - 1];
      const first = parts.slice(0, -1).join(" ");
      patterns.push(`%${last}, ${first}%`);
      // Also try "FirstWord, rest" for when user types "Surname First"
      const surname = parts[0];
      const given = parts.slice(1).join(" ");
      patterns.push(`%${surname}, ${given}%`);
    }

    // Query player_games with ilike on all patterns (OR)
    // Fetch a broader set, then group in JS
    let query = supabase
      .from("player_games")
      .select("player_id, player, team, date");

    // Build OR filter for all patterns
    const orClauses = patterns.map((p) => `player.ilike.${p}`).join(",");
    query = query.or(orClauses);

    const { data, error } = await query.order("date", { ascending: false });

    if (error) {
      console.error("Player search error:", error);
      return NextResponse.json([]);
    }

    if (!data || data.length === 0) {
      return NextResponse.json([]);
    }

    // Group by player_id: get latest name/team and total games
    const playerMap = new Map<
      string,
      { player_id: string; name: string; team: string; total_games: number; latest_date: string }
    >();

    for (const row of data) {
      const existing = playerMap.get(row.player_id);
      if (!existing) {
        playerMap.set(row.player_id, {
          player_id: row.player_id,
          name: row.player,
          team: row.team,
          total_games: 1,
          latest_date: row.date,
        });
      } else {
        existing.total_games++;
        // Keep the latest entry for name/team
        if (row.date > existing.latest_date) {
          existing.name = row.player;
          existing.team = row.team;
          existing.latest_date = row.date;
        }
      }
    }

    // Score: exact match on player name gets 100, partial gets 80
    const qLower = q.toLowerCase();
    const results = Array.from(playerMap.values())
      .map((p) => {
        const nameLower = p.name.toLowerCase();
        let score = 80;
        if (nameLower === qLower) score = 100;
        else if (nameLower.startsWith(qLower) || nameLower.includes(`, ${qLower}`)) score = 95;
        return {
          player_id: p.player_id,
          name: p.name,
          team: p.team,
          total_games: p.total_games,
          score,
        };
      })
      .sort((a, b) => b.score - a.score || b.total_games - a.total_games)
      .slice(0, limit);

    return NextResponse.json(results);
  } catch (err) {
    console.error("Player search error:", err);
    return NextResponse.json([]);
  }
}
