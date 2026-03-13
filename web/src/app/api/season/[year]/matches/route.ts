import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

/* eslint-disable @typescript-eslint/no-explicit-any */

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ year: string }> }
) {
  try {
    const { year } = await params;
    const y = Number(year);

    // Load completed matches, fixtures, game predictions, player predictions, and team stats
    const [matchesRes, fixRes, gpRes, predsRes, tmRes] = await Promise.all([
      supabase
        .from("matches")
        .select("match_id, home_team, away_team, date, round_number, venue, home_score, away_score")
        .eq("year", y)
        .order("round_number", { ascending: true }),
      supabase
        .from("fixtures")
        .select("team, opponent, venue, date, round_number")
        .eq("year", y)
        .eq("is_home", true),
      supabase
        .from("game_predictions")
        .select("match_id, home_team, away_team, round_number, home_win_prob, predicted_winner, predicted_margin")
        .eq("year", y),
      supabase
        .from("predictions")
        .select("team, match_id, round_number, predicted_goals, predicted_disposals, predicted_marks")
        .eq("year", y),
      supabase
        .from("team_matches")
        .select("match_id, team, gl, di, mk")
        .eq("year", y),
    ]);

    const matches = matchesRes.data ?? [];
    const fixtures = fixRes.data ?? [];
    const gamePreds = gpRes.data ?? [];
    const playerPreds = predsRes.data ?? [];
    const teamMatches = tmRes.data ?? [];

    // Build set of completed match keys
    const completedKeys = new Set<string>();
    for (const m of matches) {
      completedKeys.add(`${m.home_team}|${m.away_team}|${m.round_number}`);
    }

    // Build game prediction lookup
    const gpByMatchId = new Map<number, any>();
    const gpByTeams = new Map<string, any>();
    for (const gp of gamePreds) {
      if (gp.match_id) gpByMatchId.set(gp.match_id, gp);
      gpByTeams.set(`${gp.home_team}_${gp.away_team}_${gp.round_number}`, gp);
    }

    // Aggregate player predictions per team per match/round
    const teamPredAgg = new Map<string, { goals: number; disposals: number; marks: number }>();
    for (const p of playerPreds) {
      const key = `${p.team}|${p.match_id ?? p.round_number}`;
      const existing = teamPredAgg.get(key) ?? { goals: 0, disposals: 0, marks: 0 };
      existing.goals += p.predicted_goals ?? 0;
      existing.disposals += p.predicted_disposals ?? 0;
      existing.marks += p.predicted_marks ?? 0;
      teamPredAgg.set(key, existing);
    }

    // Build actual team stats lookup: match_id|team -> { gl, di, mk }
    const teamActualMap = new Map<string, { gl: number; di: number; mk: number }>();
    for (const tm of teamMatches) {
      teamActualMap.set(`${tm.match_id}|${tm.team}`, {
        gl: tm.gl ?? 0,
        di: tm.di ?? 0,
        mk: tm.mk ?? 0,
      });
    }

    // Build results from completed matches
    const results: any[] = matches.map((m: any) => {
      const gp =
        gpByMatchId.get(m.match_id) ??
        gpByTeams.get(`${m.home_team}_${m.away_team}_${m.round_number}`);

      const homeWinProb = gp?.home_win_prob != null ? +Number(gp.home_win_prob).toFixed(4) : null;
      const predictedWinner =
        gp?.predicted_winner ??
        (homeWinProb != null
          ? homeWinProb > 0.5
            ? m.home_team
            : m.away_team
          : null);

      let correct: boolean | null = null;
      if (predictedWinner && m.home_score != null && m.away_score != null) {
        const actualWinner =
          m.home_score > m.away_score
            ? m.home_team
            : m.away_score > m.home_score
              ? m.away_team
              : "Draw";
        correct = predictedWinner === actualWinner;
      }

      const homeKey1 = `${m.home_team}|${m.match_id}`;
      const homeKey2 = `${m.home_team}|${m.round_number}`;
      const awayKey1 = `${m.away_team}|${m.match_id}`;
      const awayKey2 = `${m.away_team}|${m.round_number}`;
      const homePred = teamPredAgg.get(homeKey1) ?? teamPredAgg.get(homeKey2);
      const awayPred = teamPredAgg.get(awayKey1) ?? teamPredAgg.get(awayKey2);

      // Actual team stats
      const homeActual = teamActualMap.get(`${m.match_id}|${m.home_team}`);
      const awayActual = teamActualMap.get(`${m.match_id}|${m.away_team}`);

      return {
        match_id: m.match_id,
        home_team: m.home_team,
        away_team: m.away_team,
        date: m.date ? String(m.date).slice(0, 10) : null,
        round_number: m.round_number,
        venue: m.venue,
        home_score: m.home_score,
        away_score: m.away_score,
        home_win_prob: homeWinProb,
        predicted_winner: predictedWinner,
        predicted_margin: gp?.predicted_margin ?? null,
        correct,
        home_pred: homePred ? { pred_gl: +homePred.goals.toFixed(1), pred_di: +homePred.disposals.toFixed(0), pred_mk: +homePred.marks.toFixed(0) } : null,
        away_pred: awayPred ? { pred_gl: +awayPred.goals.toFixed(1), pred_di: +awayPred.disposals.toFixed(0), pred_mk: +awayPred.marks.toFixed(0) } : null,
        home_actual: homeActual ? { actual_gl: homeActual.gl, actual_di: homeActual.di, actual_mk: homeActual.mk } : null,
        away_actual: awayActual ? { actual_gl: awayActual.gl, actual_di: awayActual.di, actual_mk: awayActual.mk } : null,
      };
    });

    // Add upcoming fixtures that don't have completed matches
    for (const f of fixtures) {
      const key = `${f.team}|${f.opponent}|${f.round_number}`;
      if (completedKeys.has(key)) continue;

      const gp = gpByTeams.get(`${f.team}_${f.opponent}_${f.round_number}`);
      const homeWinProb = gp?.home_win_prob != null ? +Number(gp.home_win_prob).toFixed(4) : null;
      const predictedWinner =
        gp?.predicted_winner ??
        (homeWinProb != null ? (homeWinProb > 0.5 ? f.team : f.opponent) : null);

      const homePred = teamPredAgg.get(`${f.team}|${f.round_number}`);
      const awayPred = teamPredAgg.get(`${f.opponent}|${f.round_number}`);

      results.push({
        match_id: null,
        home_team: f.team,
        away_team: f.opponent,
        date: f.date ?? null,
        round_number: f.round_number,
        venue: f.venue,
        home_score: null,
        away_score: null,
        home_win_prob: homeWinProb,
        predicted_winner: predictedWinner,
        predicted_margin: gp?.predicted_margin ?? null,
        correct: null,
        home_pred: homePred ? { pred_gl: +homePred.goals.toFixed(1), pred_di: +homePred.disposals.toFixed(0), pred_mk: +homePred.marks.toFixed(0) } : null,
        away_pred: awayPred ? { pred_gl: +awayPred.goals.toFixed(1), pred_di: +awayPred.disposals.toFixed(0), pred_mk: +awayPred.marks.toFixed(0) } : null,
        home_actual: null,
        away_actual: null,
      });
    }

    // Sort by round number
    results.sort((a, b) => a.round_number - b.round_number);

    return NextResponse.json(results);
  } catch (err) {
    console.error("Season matches error:", err);
    return NextResponse.json([], { status: 500 });
  }
}
