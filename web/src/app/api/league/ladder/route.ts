import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET(req: NextRequest) {
  const year = Number(req.nextUrl.searchParams.get("year") ?? new Date().getFullYear());

  const { data: rows } = await supabase
    .from("team_matches")
    .select("team, score, opp_score, margin, result, round_number, date, is_home")
    .eq("year", year)
    .order("date", { ascending: true });

  if (!rows?.length) return NextResponse.json({ year, ladder: [] });

  // Group by team
  const teams = new Map<string, typeof rows>();
  for (const r of rows) {
    if (!teams.has(r.team)) teams.set(r.team, []);
    teams.get(r.team)!.push(r);
  }

  const ladder = Array.from(teams.entries()).map(([team, games]) => {
    const wins = games.filter((g) => g.result === "W").length;
    const losses = games.filter((g) => g.result === "L").length;
    const draws = games.filter((g) => g.result === "D").length;
    const pointsFor = games.reduce((s, g) => s + (g.score ?? 0), 0);
    const pointsAgainst = games.reduce((s, g) => s + (g.opp_score ?? 0), 0);
    const percentage = pointsAgainst > 0 ? (pointsFor / pointsAgainst) * 100 : 0;
    const form = games
      .slice(-5)
      .map((g) => g.result ?? "");
    const avgMargin =
      games.length > 0
        ? games.reduce((s, g) => s + (g.margin ?? 0), 0) / games.length
        : 0;

    return {
      team,
      played: games.length,
      wins,
      losses,
      draws,
      points: wins * 4 + draws * 2,
      points_for: pointsFor,
      points_against: pointsAgainst,
      percentage: +percentage.toFixed(1),
      form,
      avg_margin: +avgMargin.toFixed(1),
    };
  });

  ladder.sort((a, b) => b.points - a.points || b.percentage - a.percentage);
  ladder.forEach((t, i) => ((t as Record<string, unknown>).position = i + 1));

  return NextResponse.json({ year, ladder });
}
