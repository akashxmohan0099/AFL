import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET() {
  const { data: rows } = await supabase
    .from("news_injuries")
    .select("*")
    .order("team")
    .order("severity", { ascending: false });

  if (!rows?.length) {
    return NextResponse.json({ teams: {}, total: 0, updated: null });
  }

  const teams: Record<string, Array<{
    player: string;
    injury: string;
    severity: number;
    severity_label: string;
    estimated_return: string;
  }>> = {};

  for (const r of rows) {
    if (!teams[r.team]) teams[r.team] = [];
    teams[r.team].push({
      player: r.player,
      injury: r.injury,
      severity: r.severity,
      severity_label: r.severity_label,
      estimated_return: r.estimated_return,
    });
  }

  return NextResponse.json({
    teams,
    total: rows.length,
    updated: rows[0]?.updated_at ?? null,
  });
}
