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

    // Load fixtures (all scheduled games), matches (completed), game predictions, and player predictions
    const [fixRes, matchesRes, gpRes, predsRes] = await Promise.all([
      supabase
        .from("fixtures")
        .select("team, opponent, venue, date, round_number, is_home")
        .eq("year", y)
        .eq("is_home", true)
        .order("round_number", { ascending: true }),
      supabase
        .from("matches")
        .select("match_id, home_team, away_team, venue, date, round_number, home_score, away_score")
        .eq("year", y),
      supabase
        .from("game_predictions")
        .select("match_id, home_team, away_team, round_number, home_win_prob, predicted_winner, predicted_margin")
        .eq("year", y),
      supabase
        .from("predictions")
        .select("team, match_id, round_number, predicted_goals, predicted_disposals, predicted_marks")
        .eq("year", y),
    ]);

    const fixtures = fixRes.data ?? [];
    const matches = matchesRes.data ?? [];
    const gamePreds = gpRes.data ?? [];
    const playerPreds = predsRes.data ?? [];

    // Build match lookup by teams+round (completed games)
    const matchLookup = new Map<string, any>();
    for (const m of matches) {
      matchLookup.set(`${m.home_team}|${m.away_team}|${m.round_number}`, m);
    }

    // Build game prediction lookup
    const gpByTeams = new Map<string, any>();
    for (const gp of gamePreds) {
      gpByTeams.set(`${gp.home_team}_${gp.away_team}_${gp.round_number}`, gp);
    }

    // Aggregate player predictions per team per round
    const teamPredAgg = new Map<string, { goals: number; disposals: number; marks: number }>();
    for (const p of playerPreds) {
      const key = `${p.team}|${p.round_number}`;
      const existing = teamPredAgg.get(key) ?? { goals: 0, disposals: 0, marks: 0 };
      existing.goals += p.predicted_goals ?? 0;
      existing.disposals += p.predicted_disposals ?? 0;
      existing.marks += p.predicted_marks ?? 0;
      teamPredAgg.set(key, existing);
    }

    // Group fixtures by round
    const roundMap = new Map<number, any[]>();
    for (const f of fixtures) {
      const arr = roundMap.get(f.round_number) ?? [];
      arr.push(f);
      roundMap.set(f.round_number, arr);
    }

    // Build round entries
    const rounds = [...roundMap.keys()].sort((a, b) => a - b).map((rn) => {
      const roundFixtures = roundMap.get(rn)!;

      let playedCount = 0;
      const matchEntries = roundFixtures.map((f: any) => {
        const homeTeam = f.team;
        const awayTeam = f.opponent;
        const key = `${homeTeam}|${awayTeam}|${rn}`;
        const m = matchLookup.get(key);
        const isPlayed = m && m.home_score != null && m.away_score != null;
        if (isPlayed) playedCount++;

        const gp = gpByTeams.get(`${homeTeam}_${awayTeam}_${rn}`);
        const homeWinProb = gp?.home_win_prob != null ? +Number(gp.home_win_prob).toFixed(4) : null;
        const predictedWinner =
          gp?.predicted_winner ??
          (homeWinProb != null ? (homeWinProb > 0.5 ? homeTeam : awayTeam) : null);
        const predictedMargin = gp?.predicted_margin != null ? +Number(gp.predicted_margin).toFixed(1) : null;

        const homePred = teamPredAgg.get(`${homeTeam}|${rn}`);
        const awayPred = teamPredAgg.get(`${awayTeam}|${rn}`);

        return {
          home_team: homeTeam,
          away_team: awayTeam,
          venue: m?.venue ?? f.venue ?? null,
          date: m?.date ? String(m.date).slice(0, 10) : f.date ?? null,
          match_id: m?.match_id ?? null,
          home_score: m?.home_score ?? null,
          away_score: m?.away_score ?? null,
          prediction: homeWinProb != null ? {
            home_win_prob: homeWinProb,
            away_win_prob: +(1 - homeWinProb).toFixed(4),
            predicted_winner: predictedWinner,
            predicted_margin: predictedMargin,
          } : null,
          home_pred: homePred ? {
            pred_gl: +homePred.goals.toFixed(1),
            pred_di: +homePred.disposals.toFixed(0),
            pred_mk: +homePred.marks.toFixed(0),
          } : null,
          away_pred: awayPred ? {
            pred_gl: +awayPred.goals.toFixed(1),
            pred_di: +awayPred.disposals.toFixed(0),
            pred_mk: +awayPred.marks.toFixed(0),
          } : null,
        };
      });

      let status: "completed" | "in_progress" | "upcoming" | "future";
      if (playedCount === matchEntries.length) {
        status = "completed";
      } else if (playedCount > 0) {
        status = "in_progress";
      } else {
        const hasPreds = gamePreds.some((gp: any) => gp.round_number === rn);
        status = hasPreds ? "upcoming" : "future";
      }

      return {
        round_number: rn,
        status,
        matches: matchEntries,
      };
    });

    return NextResponse.json({ year: y, rounds });
  } catch (err) {
    console.error("Season schedule error:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
