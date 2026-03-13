import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ teamName: string }> }
) {
  try {
  const { teamName } = await params;
  const team = decodeURIComponent(teamName);
  const year = Number(req.nextUrl.searchParams.get("year") ?? new Date().getFullYear());

  // Team matches for the season
  const { data: games } = await supabase
    .from("team_matches")
    .select("*")
    .eq("team", team)
    .eq("year", year)
    .order("date", { ascending: true });

  if (!games?.length) {
    return NextResponse.json({ error: "No data" }, { status: 404 });
  }

  const wins = games.filter((g) => g.result === "W").length;
  const losses = games.filter((g) => g.result === "L").length;
  const draws = games.filter((g) => g.result === "D").length;
  const pointsFor = games.reduce((s, g) => s + (g.score ?? 0), 0);
  const pointsAgainst = games.reduce((s, g) => s + (g.opp_score ?? 0), 0);
  const percentage = pointsAgainst > 0 ? (pointsFor / pointsAgainst) * 100 : 0;

  // Recent form
  const recentForm = games.slice(-5).map((g) => ({
    round_number: g.round_number,
    opponent: g.opponent,
    score: g.score,
    opp_score: g.opp_score,
    result: g.result,
    margin: g.margin,
    venue: g.venue,
    is_home: g.is_home,
  }));

  // Home/away split
  const homeGames = games.filter((g) => g.is_home);
  const awayGames = games.filter((g) => !g.is_home);
  const splitStats = (arr: typeof games) => ({
    played: arr.length,
    wins: arr.filter((g) => g.result === "W").length,
    losses: arr.filter((g) => g.result === "L").length,
    draws: arr.filter((g) => g.result === "D").length,
    avg_score: arr.length > 0 ? +(arr.reduce((s, g) => s + (g.score ?? 0), 0) / arr.length).toFixed(1) : 0,
    avg_conceded: arr.length > 0 ? +(arr.reduce((s, g) => s + (g.opp_score ?? 0), 0) / arr.length).toFixed(1) : 0,
    avg_margin: arr.length > 0 ? +(arr.reduce((s, g) => s + (g.margin ?? 0), 0) / arr.length).toFixed(1) : 0,
  });

  // Top players
  const { data: pg } = await supabase
    .from("player_games")
    .select("player_id, player, gl, di, mk")
    .eq("team", team)
    .eq("year", year);

  const playerMap = new Map<string, { name: string; games: number; gl: number; di: number; mk: number }>();
  for (const p of pg ?? []) {
    const existing = playerMap.get(p.player_id) ?? { name: p.player, games: 0, gl: 0, di: 0, mk: 0 };
    existing.games++;
    existing.gl += p.gl ?? 0;
    existing.di += p.di ?? 0;
    existing.mk += p.mk ?? 0;
    playerMap.set(p.player_id, existing);
  }

  const players = Array.from(playerMap.entries()).map(([pid, s]) => ({ player_id: pid, ...s }));

  const topGoals = [...players].sort((a, b) => b.gl - a.gl).slice(0, 5).map((p) => ({
    player_id: p.player_id, name: p.name, games: p.games, total: p.gl, avg: +(p.gl / p.games).toFixed(2),
  }));
  const topDisp = [...players].sort((a, b) => b.di - a.di).slice(0, 5).map((p) => ({
    player_id: p.player_id, name: p.name, games: p.games, total: p.di, avg: +(p.di / p.games).toFixed(1),
  }));
  const topMarks = [...players].sort((a, b) => b.mk - a.mk).slice(0, 5).map((p) => ({
    player_id: p.player_id, name: p.name, games: p.games, total: p.mk, avg: +(p.mk / p.games).toFixed(1),
  }));

  const avgScore = +(pointsFor / games.length).toFixed(1);
  const avgConceded = +(pointsAgainst / games.length).toFixed(1);
  const restDays = games.map((g) => g.rest_days).filter((d): d is number => d != null);

  return NextResponse.json({
    team,
    year,
    record: {
      played: games.length,
      wins,
      losses,
      draws,
      points: wins * 4 + draws * 2,
      points_for: pointsFor,
      points_against: pointsAgainst,
      percentage: +percentage.toFixed(1),
    },
    recent_form: recentForm,
    top_goals: topGoals,
    top_disposals: topDisp,
    top_marks: topMarks,
    season_averages: {
      avg_score: avgScore,
      avg_conceded: avgConceded,
      avg_margin: +(avgScore - avgConceded).toFixed(1),
      avg_rest_days: restDays.length > 0 ? +(restDays.reduce((a, b) => a + b, 0) / restDays.length).toFixed(1) : null,
    },
    home_away: {
      home: splitStats(homeGames),
      away: splitStats(awayGames),
    },
  });
  } catch (err) {
    console.error("route error:", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
