import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET(req: NextRequest) {
  const sp = req.nextUrl.searchParams;
  const signalType = sp.get("signal_type") ?? "";
  const team = sp.get("team") ?? "";
  const minRelevance = Number(sp.get("min_relevance") ?? 0);
  const limit = Number(sp.get("limit") ?? 50);
  const offset = Number(sp.get("offset") ?? 0);

  const { data: row } = await supabase
    .from("news_intel")
    .select("data, updated_at")
    .order("id", { ascending: false })
    .limit(1)
    .single();

  if (!row?.data) {
    return NextResponse.json({
      signals: [],
      total: 0,
      offset,
      limit,
      breaking_count: 0,
      by_type: {},
      by_team: {},
      updated: null,
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let signals: any[] = row.data.signals ?? row.data ?? [];

  if (signalType) {
    signals = signals.filter(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (s: any) => s.signal_type?.toLowerCase() === signalType.toLowerCase()
    );
  }
  if (team) {
    signals = signals.filter(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (s: any) =>
        (s.team ?? s.entity ?? "").toLowerCase().includes(team.toLowerCase())
    );
  }
  if (minRelevance > 0) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    signals = signals.filter((s: any) => (s.relevance_score ?? 0) >= minRelevance);
  }

  const total = signals.length;

  // Counts
  const byType: Record<string, number> = {};
  const byTeam: Record<string, number> = {};
  let breakingCount = 0;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  for (const s of signals) {
    const t = s.signal_type ?? "other";
    byType[t] = (byType[t] ?? 0) + 1;
    const tm = s.team ?? s.entity ?? "";
    if (tm) byTeam[tm] = (byTeam[tm] ?? 0) + 1;
    if ((s.relevance_score ?? 0) >= 0.7) breakingCount++;
  }

  const paged = signals.slice(offset, offset + limit);

  return NextResponse.json({
    signals: paged,
    total,
    offset,
    limit,
    breaking_count: breakingCount,
    by_type: byType,
    by_team: byTeam,
    updated: row.updated_at,
  });
}
