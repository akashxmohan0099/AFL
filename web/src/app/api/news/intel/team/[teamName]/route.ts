import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ teamName: string }> }
) {
  try {
  const { teamName } = await params;
  const team = decodeURIComponent(teamName);

  const { data: row } = await supabase
    .from("news_intel")
    .select("data, updated_at")
    .order("id", { ascending: false })
    .limit(1)
    .single();

  if (!row?.data) {
    return NextResponse.json({
      team,
      total_signals: 0,
      signals: [],
      bullish: 0,
      bearish: 0,
      net_sentiment: 0,
      by_type: {},
      updated: null,
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const allSignals: any[] = row.data.signals ?? row.data ?? [];
  const signals = allSignals.filter(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (s: any) =>
      (s.team ?? s.entity ?? "").toLowerCase().includes(team.toLowerCase())
  );

  let bullish = 0;
  let bearish = 0;
  const byType: Record<string, number> = {};
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const injuries: any[] = [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const suspensions: any[] = [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const form: any[] = [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const tactical: any[] = [];

  for (const s of signals) {
    const t = s.signal_type ?? "other";
    byType[t] = (byType[t] ?? 0) + 1;

    const sent = (s.sentiment ?? "neutral").toLowerCase();
    if (sent === "bullish" || sent === "positive") bullish++;
    else if (sent === "bearish" || sent === "negative") bearish++;

    if (t === "injury") injuries.push(s);
    else if (t === "suspension") suspensions.push(s);
    else if (t === "form") form.push(s);
    else if (t === "tactical") tactical.push(s);
  }

  return NextResponse.json({
    team,
    total_signals: signals.length,
    signals,
    bullish,
    bearish,
    net_sentiment: bullish - bearish,
    by_type: byType,
    injuries,
    suspensions,
    form,
    tactical,
    updated: row.updated_at,
  });
  } catch (err) {
    console.error("route error:", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
