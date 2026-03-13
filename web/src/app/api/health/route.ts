import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET() {
  try {
    const [pgRes, mRes, expRes] = await Promise.all([
      supabase.from("player_games").select("match_id", { count: "exact", head: true }),
      supabase.from("matches").select("match_id", { count: "exact", head: true }),
      supabase.from("experiments").select("id", { count: "exact", head: true }),
    ]);

    const pgCount = pgRes.count ?? 0;
    const mCount = mRes.count ?? 0;
    const expCount = expRes.count ?? 0;
    const status = pgCount > 0 && mCount > 0 ? "ok" : "degraded";

    const result: Record<string, unknown> = {
      status,
      player_games: pgCount,
      matches: mCount,
      experiments: expCount,
      cache_loaded: true,
    };

    if (mCount > 0) {
      const { data } = await supabase
        .from("matches")
        .select("date")
        .order("date", { ascending: false })
        .limit(1)
        .single();
      if (data?.date) {
        result.latest_data = String(data.date).slice(0, 10);
      }
    }

    return NextResponse.json(result);
  } catch (err) {
    console.error("Health check error:", err);
    return NextResponse.json(
      { status: "error", error: "Failed to connect to database" },
      { status: 500 }
    );
  }
}
