import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ year: string; round: string }> }
) {
  try {
    const { year, round } = await params;
    const y = Number(year);
    const r = Number(round);

    const { data, error } = await supabase
      .from("predictions")
      .select("*")
      .eq("year", y)
      .eq("round_number", r);

    if (error) {
      console.error("Round predictions query error:", error);
      return NextResponse.json([], { status: 500 });
    }

    return NextResponse.json(data ?? []);
  } catch (err) {
    console.error("Round predictions error:", err);
    return NextResponse.json([], { status: 500 });
  }
}
