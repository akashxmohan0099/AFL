import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const yearParam = searchParams.get("year");

    let query = supabase
      .from("predictions")
      .select("year, round_number");

    if (yearParam) {
      query = query.eq("year", Number(yearParam));
    }

    const { data, error } = await query;

    if (error) {
      console.error("Rounds query error:", error);
      return NextResponse.json([], { status: 500 });
    }

    if (!data || data.length === 0) {
      return NextResponse.json([]);
    }

    // Aggregate distinct (year, round_number) with counts in JS
    const countMap = new Map<string, { year: number; round_number: number; n_predictions: number }>();

    for (const row of data) {
      const key = `${row.year}_${row.round_number}`;
      const existing = countMap.get(key);
      if (existing) {
        existing.n_predictions++;
      } else {
        countMap.set(key, {
          year: row.year,
          round_number: row.round_number,
          n_predictions: 1,
        });
      }
    }

    const results = Array.from(countMap.values()).sort((a, b) => {
      if (a.year !== b.year) return b.year - a.year;
      return a.round_number - b.round_number;
    });

    return NextResponse.json(results);
  } catch (err) {
    console.error("Rounds error:", err);
    return NextResponse.json([], { status: 500 });
  }
}
