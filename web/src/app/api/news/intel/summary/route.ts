import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET() {
  try {
  const { data: row } = await supabase
    .from("news_intel")
    .select("data, updated_at")
    .order("id", { ascending: false })
    .limit(1)
    .single();

  if (!row?.data) {
    return NextResponse.json({
      total: 0,
      breaking: [],
      breaking_count: 0,
      top_signals: [],
      by_type: {},
      by_team: {},
      sentiment: { bullish: 0, bearish: 0, neutral: 0 },
      team_direction: {},
      updated: null,
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const signals: any[] = row.data.signals ?? row.data ?? [];
  const total = signals.length;

  const byType: Record<string, number> = {};
  const byTeam: Record<string, number> = {};
  const sentiment = { bullish: 0, bearish: 0, neutral: 0 };
  const teamDirection: Record<string, { bullish: number; bearish: number; neutral: number }> = {};
  const breaking: typeof signals = [];

  for (const s of signals) {
    const t = s.signal_type ?? "other";
    byType[t] = (byType[t] ?? 0) + 1;
    const tm = s.team ?? s.entity ?? "";
    if (tm) {
      byTeam[tm] = (byTeam[tm] ?? 0) + 1;
      if (!teamDirection[tm]) teamDirection[tm] = { bullish: 0, bearish: 0, neutral: 0 };
    }

    const sent = (s.sentiment ?? "neutral").toLowerCase();
    if (sent === "bullish" || sent === "positive") {
      sentiment.bullish++;
      if (tm) teamDirection[tm].bullish++;
    } else if (sent === "bearish" || sent === "negative") {
      sentiment.bearish++;
      if (tm) teamDirection[tm].bearish++;
    } else {
      sentiment.neutral++;
      if (tm) teamDirection[tm].neutral++;
    }

    if ((s.relevance_score ?? 0) >= 0.7) breaking.push(s);
  }

  const topSignals = [...signals]
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    .sort((a: any, b: any) => (b.relevance_score ?? 0) - (a.relevance_score ?? 0))
    .slice(0, 10);

  return NextResponse.json({
    total,
    breaking,
    breaking_count: breaking.length,
    top_signals: topSignals,
    by_type: byType,
    by_team: byTeam,
    sentiment,
    team_direction: teamDirection,
    updated: row.updated_at,
  });
  } catch (err) {
    console.error("route error:", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
