import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ venueName: string }> }
) {
  try {
  const { venueName } = await params;
  const venue = decodeURIComponent(venueName);

  const { data: matches } = await supabase
    .from("matches")
    .select("match_id, home_team, away_team, year, round_number, date, home_score, away_score, total_score, margin, attendance")
    .eq("venue", venue)
    .order("date", { ascending: false });

  if (!matches?.length) {
    return NextResponse.json({ error: "Venue not found" }, { status: 404 });
  }

  const matchIds = matches.map((m) => m.match_id);

  // Load weather and player stats in parallel
  const [wxRes, pgRes] = await Promise.all([
    supabase
      .from("weather")
      .select("temperature_avg, wind_speed_avg, precipitation_total, humidity_avg, is_wet, is_roofed")
      .in("match_id", matchIds),
    supabase
      .from("player_games")
      .select("player_id, player, team, gl, di, match_id")
      .eq("venue", venue)
      .gte("year", Math.max(...matches.map((m) => m.year)) - 5),
  ]);

  const wx = wxRes.data ?? [];
  const pg = pgRes.data ?? [];

  // Weather aggregate
  const temps = wx.map((w) => w.temperature_avg).filter((t): t is number => t != null);
  const winds = wx.map((w) => w.wind_speed_avg).filter((w): w is number => w != null);
  const humid = wx.map((w) => w.humidity_avg).filter((h): h is number => h != null);
  const wetCount = wx.filter((w) => w.is_wet).length;
  const roofed = wx.some((w) => w.is_roofed);

  const avg = (arr: number[]) =>
    arr.length > 0 ? +(arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(1) : null;

  // Top performers
  const playerStats = new Map<string, { player: string; team: string; games: number; totalGL: number; totalDI: number }>();
  for (const p of pg) {
    const key = p.player_id;
    const existing = playerStats.get(key) ?? {
      player: p.player,
      team: p.team,
      games: 0,
      totalGL: 0,
      totalDI: 0,
    };
    existing.games++;
    existing.totalGL += p.gl ?? 0;
    existing.totalDI += p.di ?? 0;
    playerStats.set(key, existing);
  }

  const qualified = Array.from(playerStats.values()).filter((p) => p.games >= 5);

  const topGoals = [...qualified]
    .sort((a, b) => b.totalGL / b.games - a.totalGL / a.games)
    .slice(0, 10)
    .map((p) => ({
      player: p.player,
      team: p.team,
      games: p.games,
      total_goals: p.totalGL,
      avg_goals: +(p.totalGL / p.games).toFixed(2),
    }));

  const topDisposals = [...qualified]
    .sort((a, b) => b.totalDI / b.games - a.totalDI / a.games)
    .slice(0, 10)
    .map((p) => ({
      player: p.player,
      team: p.team,
      games: p.games,
      total_disposals: p.totalDI,
      avg_disposals: +(p.totalDI / p.games).toFixed(1),
    }));

  const scores = matches.map((m) => m.total_score).filter((s): s is number => s != null);
  const margins = matches.map((m) => m.margin).filter((m): m is number => m != null);

  return NextResponse.json({
    venue,
    total_games: matches.length,
    avg_total_score: avg(scores),
    avg_margin: avg(margins.map(Math.abs)),
    is_roofed: roofed,
    weather: {
      avg_temperature: avg(temps),
      avg_wind_speed: avg(winds),
      avg_humidity: avg(humid),
      pct_wet: wx.length > 0 ? +((wetCount / wx.length) * 100).toFixed(1) : null,
    },
    top_goal_scorers: topGoals,
    top_disposal_getters: topDisposals,
    recent_matches: matches.slice(0, 20).map((m) => ({
      match_id: m.match_id,
      year: m.year,
      round_number: m.round_number,
      date: m.date,
      home_team: m.home_team,
      away_team: m.away_team,
      home_score: m.home_score,
      away_score: m.away_score,
      attendance: m.attendance,
    })),
  });
  } catch (err) {
    console.error("route error:", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
