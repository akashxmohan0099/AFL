import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET() {
  try {
  // Fetch matches with weather join
  const { data: matches } = await supabase
    .from("matches")
    .select("match_id, venue, year, total_score, margin, attendance, home_team");

  if (!matches?.length) return NextResponse.json([]);

  const { data: weatherRows } = await supabase
    .from("weather")
    .select("match_id, temperature_avg, precipitation_total, is_wet, is_roofed");

  const weatherMap = new Map(
    (weatherRows ?? []).map((w) => [w.match_id, w])
  );

  // Fetch home team counts
  const { data: tmRows } = await supabase
    .from("team_matches")
    .select("venue, team, is_home")
    .eq("is_home", true);

  // Group by venue
  const venueMap = new Map<string, {
    matches: typeof matches;
    homeTeams: Map<string, number>;
  }>();

  for (const m of matches) {
    if (!m.venue) continue;
    if (!venueMap.has(m.venue)) {
      venueMap.set(m.venue, { matches: [], homeTeams: new Map() });
    }
    venueMap.get(m.venue)!.matches.push(m);
  }

  for (const tm of tmRows ?? []) {
    if (!tm.venue) continue;
    const entry = venueMap.get(tm.venue);
    if (entry) {
      entry.homeTeams.set(tm.team, (entry.homeTeams.get(tm.team) ?? 0) + 1);
    }
  }

  const result = Array.from(venueMap.entries()).map(([venue, { matches: vm, homeTeams }]) => {
    const years = vm.map((m) => m.year).filter(Boolean);
    const scores = vm.map((m) => m.total_score).filter((s): s is number => s != null);
    const margins = vm.map((m) => m.margin).filter((m): m is number => m != null);

    const wxMatches = vm.map((m) => weatherMap.get(m.match_id)).filter(Boolean);
    const temps = wxMatches.map((w) => w!.temperature_avg).filter((t): t is number => t != null);
    const precip = wxMatches.map((w) => w!.precipitation_total).filter((p): p is number => p != null);
    const wetCount = wxMatches.filter((w) => w!.is_wet).length;
    const roofed = wxMatches.some((w) => w!.is_roofed);

    return {
      venue,
      total_games: vm.length,
      year_from: years.length > 0 ? Math.min(...years) : null,
      year_to: years.length > 0 ? Math.max(...years) : null,
      is_roofed: roofed,
      home_teams: Array.from(homeTeams.entries())
        .map(([team, home_games]) => ({ team, home_games }))
        .sort((a, b) => b.home_games - a.home_games),
      avg_total_score:
        scores.length > 0
          ? +(scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1)
          : null,
      avg_margin:
        margins.length > 0
          ? +(
              margins.map(Math.abs).reduce((a, b) => a + b, 0) / margins.length
            ).toFixed(1)
          : null,
      avg_temperature:
        temps.length > 0
          ? +(temps.reduce((a, b) => a + b, 0) / temps.length).toFixed(1)
          : null,
      avg_precipitation:
        precip.length > 0
          ? +(precip.reduce((a, b) => a + b, 0) / precip.length).toFixed(2)
          : null,
      pct_wet_games:
        wxMatches.length > 0
          ? +((wetCount / wxMatches.length) * 100).toFixed(1)
          : null,
    };
  });

  result.sort((a, b) => b.total_games - a.total_games);
  return NextResponse.json(result);
  } catch (err) {
    console.error("route error:", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
