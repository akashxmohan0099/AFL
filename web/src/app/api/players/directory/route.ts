import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

function round2(v: number): number {
  return Math.round(v * 100) / 100;
}

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const year = parseInt(searchParams.get("year") ?? "2026", 10);

    // Fetch all player_games rows for the given year
    const { data, error } = await supabase
      .from("player_games")
      .select("player_id, player, team, date, gl, bh, di, mk, ki, hb, tk, ho")
      .eq("year", year);

    if (error) {
      console.error("Player directory error:", error);
      return NextResponse.json([]);
    }

    if (!data || data.length === 0) {
      return NextResponse.json([]);
    }

    // Group by (player_id) and aggregate
    const playerMap = new Map<
      string,
      {
        player_id: string;
        name: string;
        team: string;
        latest_date: string;
        games: number;
        gl: number[];
        bh: number[];
        di: number[];
        mk: number[];
        ki: number[];
        hb: number[];
        tk: number[];
        ho: number[];
      }
    >();

    for (const row of data) {
      const pid = row.player_id;
      let entry = playerMap.get(pid);
      if (!entry) {
        entry = {
          player_id: pid,
          name: row.player,
          team: row.team,
          latest_date: row.date,
          games: 0,
          gl: [],
          bh: [],
          di: [],
          mk: [],
          ki: [],
          hb: [],
          tk: [],
          ho: [],
        };
        playerMap.set(pid, entry);
      }

      entry.games++;
      entry.gl.push(Number(row.gl ?? 0));
      entry.bh.push(Number(row.bh ?? 0));
      entry.di.push(Number(row.di ?? 0));
      entry.mk.push(Number(row.mk ?? 0));
      entry.ki.push(Number(row.ki ?? 0));
      entry.hb.push(Number(row.hb ?? 0));
      entry.tk.push(Number(row.tk ?? 0));
      entry.ho.push(Number(row.ho ?? 0));

      // Track latest date for most recent name/team
      if (row.date > entry.latest_date) {
        entry.name = row.player;
        entry.team = row.team;
        entry.latest_date = row.date;
      }
    }

    const directory = Array.from(playerMap.values())
      .map((p) => ({
        player_id: p.player_id,
        name: p.name,
        team: p.team,
        games: p.games,
        avg_goals: round2(mean(p.gl)),
        avg_disposals: round2(mean(p.di)),
        avg_marks: round2(mean(p.mk)),
        avg_tackles: round2(mean(p.tk)),
        avg_kicks: round2(mean(p.ki)),
        avg_handballs: round2(mean(p.hb)),
        avg_hitouts: round2(mean(p.ho)),
      }))
      .sort((a, b) => b.games - a.games || a.name.localeCompare(b.name));

    return NextResponse.json(directory);
  } catch (err) {
    console.error("Player directory error:", err);
    return NextResponse.json([]);
  }
}
