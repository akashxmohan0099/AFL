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

    // Load injuries and team lists
    const [injRes, tlRes] = await Promise.all([
      supabase.from("news_injuries").select("*").order("severity", { ascending: false }),
      supabase
        .from("news_team_lists")
        .select("data")
        .eq("year", y)
        .eq("round_number", r)
        .limit(1)
        .single(),
    ]);

    const injuries = injRes.data ?? [];
    const teamListData = tlRes.data?.data ?? null;

    // Group injuries by team
    const injByTeam: Record<string, typeof injuries> = {};
    for (const inj of injuries) {
      if (!injByTeam[inj.team]) injByTeam[inj.team] = [];
      injByTeam[inj.team].push(inj);
    }

    return NextResponse.json({
      year: y,
      round_number: r,
      injuries: injByTeam,
      team_lists: teamListData,
    });
  } catch (err) {
    console.error("route error:", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
