import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

/* eslint-disable @typescript-eslint/no-explicit-any */

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ year: string; matchId: string }> }
) {
  try {
    const { year, matchId } = await params;
    const y = Number(year);
    const id = Number(matchId);
    const { searchParams } = new URL(request.url);
    const roundNumber = searchParams.get("round_number")
      ? Number(searchParams.get("round_number"))
      : null;
    const homeTeamParam = searchParams.get("home_team");
    const awayTeamParam = searchParams.get("away_team");

    // 1. Load the match
    let matchRow: any = null;

    // Try by match_id first
    if (id) {
      const { data } = await supabase
        .from("matches")
        .select("*")
        .eq("match_id", id)
        .eq("year", y)
        .maybeSingle();
      matchRow = data;
    }

    // Fallback: lookup by home_team + away_team + round_number
    if (!matchRow && homeTeamParam && awayTeamParam && roundNumber) {
      const { data } = await supabase
        .from("matches")
        .select("*")
        .eq("year", y)
        .eq("round_number", roundNumber)
        .eq("home_team", homeTeamParam)
        .eq("away_team", awayTeamParam)
        .maybeSingle();
      matchRow = data;
    }

    const homeTeam = matchRow?.home_team ?? homeTeamParam ?? "";
    const awayTeam = matchRow?.away_team ?? awayTeamParam ?? "";
    const rn = matchRow?.round_number ?? roundNumber;
    const matchId_actual = matchRow?.match_id ?? id;

    // 2. Load predictions and game predictions
    const predQuery = supabase
      .from("predictions")
      .select("*")
      .eq("year", y);

    if (rn) {
      predQuery.eq("round_number", rn);
    }

    const [predsRes, gpRes] = await Promise.all([
      predQuery,
      supabase
        .from("game_predictions")
        .select("*")
        .eq("year", y)
        .eq("round_number", rn ?? 0),
    ]);

    const allPreds = predsRes.data ?? [];
    const gamePreds = gpRes.data ?? [];

    // Filter predictions to this match
    const matchPreds = allPreds.filter((p: any) => {
      if (p.match_id === matchId_actual) return true;
      return (
        (p.team === homeTeam || p.team === awayTeam) &&
        (p.opponent === homeTeam || p.opponent === awayTeam)
      );
    });

    // Get game prediction
    const gp =
      gamePreds.find((g: any) => g.match_id === matchId_actual) ??
      gamePreds.find(
        (g: any) => g.home_team === homeTeam && g.away_team === awayTeam
      );

    // 3. Load actual player stats if match has been played
    let playerActuals: any[] = [];
    const isPlayed =
      matchRow?.home_score != null && matchRow?.away_score != null;

    if (isPlayed && matchId_actual) {
      const { data } = await supabase
        .from("player_games")
        .select("*")
        .eq("match_id", matchId_actual);
      playerActuals = data ?? [];
    }

    // Build actuals lookup
    const actualsMap = new Map<string, any>();
    for (const pg of playerActuals) {
      actualsMap.set(`${pg.player}|${pg.team}`, pg);
    }

    // Build player-by-player comparison
    const players: any[] = [];
    const seen = new Set<string>();

    for (const pred of matchPreds) {
      const key = `${pred.player}|${pred.team}`;
      if (seen.has(key)) continue;
      seen.add(key);

      const actual = actualsMap.get(key);
      const isHome = pred.team === homeTeam;

      const entry: any = {
        player: pred.player,
        team: pred.team,
        is_home: isHome,
        // Predictions
        predicted_gl: pred.predicted_goals ?? null,
        predicted_di: pred.predicted_disposals ?? null,
        predicted_mk: pred.predicted_marks ?? null,
        predicted_bh: pred.predicted_behinds ?? null,
        p_scorer: pred.p_scorer ?? null,
        player_role: pred.player_role ?? null,
        career_goal_avg: pred.career_goal_avg ?? null,
        p_2plus_goals: pred.p_2plus_goals ?? null,
        p_3plus_goals: pred.p_3plus_goals ?? null,
        p_15plus_disp: pred.p_15plus_disp ?? null,
        p_20plus_disp: pred.p_20plus_disp ?? null,
        p_25plus_disp: pred.p_25plus_disp ?? null,
        p_30plus_disp: pred.p_30plus_disp ?? null,
        p_3plus_mk: pred.p_3plus_mk ?? null,
        p_5plus_mk: pred.p_5plus_mk ?? null,
      };

      // Confidence intervals
      if (pred.conf_lower_gl != null && pred.conf_upper_gl != null) {
        entry.conf_gl = [pred.conf_lower_gl, pred.conf_upper_gl];
      }
      if (pred.conf_lower_di != null && pred.conf_upper_di != null) {
        entry.conf_di = [pred.conf_lower_di, pred.conf_upper_di];
      }

      // Actual stats
      if (actual) {
        entry.actual_gl = actual.gl ?? null;
        entry.actual_bh = actual.bh ?? null;
        entry.actual_di = actual.di ?? null;
        entry.actual_mk = actual.mk ?? null;
        entry.actual_ki = actual.ki ?? null;
        entry.actual_hb = actual.hb ?? null;
        entry.actual_tk = actual.tk ?? null;
        entry.actual_ho = actual.ho ?? null;
        entry.actual_cp = actual.cp ?? null;
        entry.actual_up = actual.up ?? null;
        entry.actual_if = actual["if"] ?? null;
        entry.actual_cl = actual.cl ?? null;
        entry.actual_cg = actual.cg ?? null;
        entry.actual_ff = actual.ff ?? null;
        entry.actual_fa = actual.fa ?? null;
      }

      players.push(entry);
    }

    // Add players with actuals but no predictions
    for (const [key, actual] of actualsMap) {
      if (seen.has(key)) continue;
      const isHome = actual.team === homeTeam;
      players.push({
        player: actual.player,
        team: actual.team,
        is_home: isHome,
        actual_gl: actual.gl ?? null,
        actual_bh: actual.bh ?? null,
        actual_di: actual.di ?? null,
        actual_mk: actual.mk ?? null,
        actual_ki: actual.ki ?? null,
        actual_hb: actual.hb ?? null,
        actual_tk: actual.tk ?? null,
        actual_ho: actual.ho ?? null,
        actual_cp: actual.cp ?? null,
        actual_up: actual.up ?? null,
        actual_if: actual["if"] ?? null,
        actual_cl: actual.cl ?? null,
        actual_cg: actual.cg ?? null,
        actual_ff: actual.ff ?? null,
        actual_fa: actual.fa ?? null,
      });
    }

    // Build game prediction info
    const gamePrediction = gp
      ? {
          predicted_winner:
            gp.predicted_winner ??
            (gp.home_win_prob > 0.5 ? homeTeam : awayTeam),
          home_win_prob: gp.home_win_prob != null ? +Number(gp.home_win_prob).toFixed(4) : null,
          predicted_margin: gp.predicted_margin ?? null,
          correct:
            isPlayed && gp.home_win_prob != null
              ? (gp.home_win_prob > 0.5 ? homeTeam : awayTeam) ===
                (matchRow.home_score > matchRow.away_score
                  ? homeTeam
                  : matchRow.away_score > matchRow.home_score
                    ? awayTeam
                    : "Draw")
              : null,
        }
      : null;

    const result = {
      match_id: matchId_actual,
      year: y,
      round_number: rn,
      date: matchRow?.date ? String(matchRow.date).slice(0, 10) : null,
      venue: matchRow?.venue ?? null,
      home_team: homeTeam,
      away_team: awayTeam,
      home_score: matchRow?.home_score ?? null,
      away_score: matchRow?.away_score ?? null,
      game_prediction: gamePrediction,
      players,
    };

    return NextResponse.json(result);
  } catch (err) {
    console.error("Match comparison error:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
