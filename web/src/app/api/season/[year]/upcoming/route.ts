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

    // Load all matches for the year to find the next unplayed round
    const { data: matches } = await supabase
      .from("matches")
      .select("match_id, home_team, away_team, venue, date, round_number, home_score, away_score")
      .eq("year", y)
      .order("round_number", { ascending: true });

    const allMatches = matches ?? [];

    // Group by round and find the first round with unplayed matches
    const roundMap = new Map<number, any[]>();
    for (const m of allMatches) {
      const arr = roundMap.get(m.round_number) ?? [];
      arr.push(m);
      roundMap.set(m.round_number, arr);
    }

    let upcomingRound: number | null = null;
    for (const rn of [...roundMap.keys()].sort((a, b) => a - b)) {
      const roundMatches = roundMap.get(rn)!;
      const hasUnplayed = roundMatches.some(
        (m: any) => m.home_score == null || m.away_score == null
      );
      if (hasUnplayed) {
        upcomingRound = rn;
        break;
      }
    }

    // If all rounds are played, try to find any predictions for a future round
    if (upcomingRound == null) {
      const { data: predRounds } = await supabase
        .from("predictions")
        .select("round_number")
        .eq("year", y)
        .order("round_number", { ascending: false })
        .limit(1);

      if (predRounds && predRounds.length > 0) {
        upcomingRound = predRounds[0].round_number;
      }
    }

    if (upcomingRound == null) {
      return NextResponse.json({
        year: y,
        round_number: null,
        matches: [],
        predictions: [],
      });
    }

    // Get unplayed matches for the upcoming round
    const upcomingMatches = (roundMap.get(upcomingRound) ?? []).filter(
      (m: any) => m.home_score == null || m.away_score == null
    );

    // If no match rows exist yet (future round), check predictions for fixture info
    let fixtureMatches = upcomingMatches.map((m: any) => ({
      home_team: m.home_team,
      away_team: m.away_team,
      venue: m.venue ?? null,
      date: m.date ? String(m.date).slice(0, 10) : null,
    }));

    // Load predictions for the upcoming round
    const { data: preds } = await supabase
      .from("predictions")
      .select("player, team, opponent, predicted_goals, predicted_disposals, predicted_marks")
      .eq("year", y)
      .eq("round_number", upcomingRound);

    const predictions = (preds ?? []).map((p: any) => ({
      player: p.player,
      team: p.team,
      opponent: p.opponent,
      predicted_goals: p.predicted_goals ?? null,
      predicted_disposals: p.predicted_disposals ?? null,
      predicted_marks: p.predicted_marks ?? null,
    }));

    // If no match rows but we have predictions, derive fixtures from predictions
    if (fixtureMatches.length === 0 && predictions.length > 0) {
      const teamPairs = new Set<string>();
      for (const p of predictions) {
        const pair = [p.team, p.opponent].sort().join("|");
        if (!teamPairs.has(pair)) {
          teamPairs.add(pair);
          fixtureMatches.push({
            home_team: p.team,
            away_team: p.opponent,
            venue: null,
            date: null,
          });
        }
      }
    }

    return NextResponse.json({
      year: y,
      round_number: upcomingRound,
      matches: fixtureMatches,
      predictions,
    });
  } catch (err) {
    console.error("Upcoming error:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
