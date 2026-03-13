import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

/* eslint-disable @typescript-eslint/no-explicit-any */

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ matchId: string }> }
) {
  try {
    const { matchId } = await params;
    const id = Number(matchId);

    // 1. Load the match
    const { data: matchRow, error: matchErr } = await supabase
      .from("matches")
      .select("*")
      .eq("match_id", id)
      .single();

    if (matchErr || !matchRow) {
      return NextResponse.json({ error: "Match not found" }, { status: 404 });
    }

    const m = matchRow;
    const year = m.year;
    const roundNum = m.round_number;

    // 2. Load related data in parallel
    const [tmRes, pgRes, predRes, gpRes] = await Promise.all([
      // team_matches for both teams
      supabase
        .from("team_matches")
        .select("*")
        .eq("match_id", id),
      // player_games for this match
      supabase
        .from("player_games")
        .select("*")
        .eq("match_id", id),
      // predictions for this round (filter by team afterward)
      supabase
        .from("predictions")
        .select("*")
        .eq("year", year)
        .eq("round_number", roundNum),
      // game predictions
      supabase
        .from("game_predictions")
        .select("*")
        .eq("year", year)
        .eq("round_number", roundNum),
    ]);

    // Build team stats
    const teamMatchRows = tmRes.data ?? [];
    const statCols = ["gl", "bh", "di", "mk", "tk", "cp", "if", "cl", "rb"];

    function buildTeamStats(teamName: string) {
      const row = teamMatchRows.find((r: any) => r.team === teamName);
      if (!row) return null;
      const stats: Record<string, number | null> = {
        score: row.score ?? null,
        opp_score: row.opp_score ?? null,
      };
      for (const col of statCols) {
        stats[col] = row[col] ?? null;
      }
      return stats;
    }

    // Build game prediction data
    const gpRows = gpRes.data ?? [];
    const gp =
      gpRows.find((g: any) => g.match_id === id) ??
      gpRows.find(
        (g: any) => g.home_team === m.home_team && g.away_team === m.away_team
      );

    const homeWinProb = gp?.home_win_prob != null ? +Number(gp.home_win_prob).toFixed(4) : null;
    const awayWinProb = homeWinProb != null ? +(1 - homeWinProb).toFixed(4) : null;
    const predictedWinner =
      homeWinProb != null
        ? homeWinProb > 0.5
          ? m.home_team
          : m.away_team
        : null;

    // Build player actuals map from player_games
    const playerGameRows = pgRes.data ?? [];
    const actualsMap = new Map<string, Record<string, any>>();
    for (const row of playerGameRows) {
      const key = `${row.player}|${row.team}`;
      actualsMap.set(key, row);
    }

    // Build predictions map
    const predRows = predRes.data ?? [];
    const predsByPlayerTeam = new Map<string, Record<string, any>>();
    for (const p of predRows) {
      // Match by match_id if available, otherwise by team membership
      const matchIdMatches = p.match_id === id;
      const teamMatches =
        (p.team === m.home_team || p.team === m.away_team) &&
        (p.opponent === m.home_team || p.opponent === m.away_team);
      if (matchIdMatches || teamMatches) {
        const key = `${p.player}|${p.team}`;
        predsByPlayerTeam.set(key, p);
      }
    }

    // Build player lists for each side
    function buildPlayerList(teamName: string) {
      const players: Record<string, any>[] = [];
      const seen = new Set<string>();

      // First add predictions (with merged actuals)
      for (const [key, pred] of predsByPlayerTeam) {
        if (pred.team !== teamName) continue;
        const actual = actualsMap.get(key);
        const entry: Record<string, any> = {
          player: pred.player,
          team: pred.team,
          opponent: pred.opponent ?? "",
          // Predicted values
          predicted_goals: pred.predicted_goals ?? null,
          predicted_disposals: pred.predicted_disposals ?? null,
          predicted_marks: pred.predicted_marks ?? null,
          p_scorer: pred.p_scorer ?? null,
          p_2plus_goals: pred.p_2plus_goals ?? null,
          p_3plus_goals: pred.p_3plus_goals ?? null,
          p_15plus_disp: pred.p_15plus_disp ?? null,
          p_20plus_disp: pred.p_20plus_disp ?? null,
          p_25plus_disp: pred.p_25plus_disp ?? null,
          p_30plus_disp: pred.p_30plus_disp ?? null,
          p_3plus_mk: pred.p_3plus_mk ?? null,
          p_5plus_mk: pred.p_5plus_mk ?? null,
        };
        // Merge actual stats (uppercase keys for frontend)
        if (actual) {
          entry.GL = actual.gl ?? null;
          entry.BH = actual.bh ?? null;
          entry.DI = actual.di ?? null;
          entry.MK = actual.mk ?? null;
          entry.KI = actual.ki ?? null;
          entry.HB = actual.hb ?? null;
          entry.TK = actual.tk ?? null;
          entry.HO = actual.ho ?? null;
        }
        seen.add(key);
        players.push(entry);
      }

      // Then add any players with actuals but no predictions
      for (const [key, actual] of actualsMap) {
        if (seen.has(key)) continue;
        if (actual.team !== teamName) continue;
        players.push({
          player: actual.player,
          team: actual.team,
          opponent:
            actual.team === m.home_team ? m.away_team : m.home_team,
          GL: actual.gl ?? null,
          BH: actual.bh ?? null,
          DI: actual.di ?? null,
          MK: actual.mk ?? null,
          KI: actual.ki ?? null,
          HB: actual.hb ?? null,
          TK: actual.tk ?? null,
          HO: actual.ho ?? null,
        });
      }

      return players;
    }

    const result = {
      match_id: m.match_id,
      home_team: m.home_team,
      away_team: m.away_team,
      venue: m.venue ?? "",
      date: m.date ? String(m.date).slice(0, 10) : null,
      year: m.year,
      round_number: m.round_number,
      home_score: m.home_score ?? null,
      away_score: m.away_score ?? null,
      is_finals: m.is_finals ?? false,
      home_team_stats: buildTeamStats(m.home_team),
      away_team_stats: buildTeamStats(m.away_team),
      home_win_prob: homeWinProb,
      away_win_prob: awayWinProb,
      predicted_winner: predictedWinner,
      home_players: buildPlayerList(m.home_team),
      away_players: buildPlayerList(m.away_team),
    };

    return NextResponse.json(result);
  } catch (err) {
    console.error("Match detail error:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
