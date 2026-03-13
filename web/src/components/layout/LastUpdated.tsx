"use client";

import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";

export function LastUpdated() {
  const [label, setLabel] = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then((h) => {
        if (h?.latest_data) {
          // latest_data is "YYYY-MM-DD"
          const d = new Date(h.latest_data + "T00:00:00");
          const now = new Date();
          const diffMs = now.getTime() - d.getTime();
          const diffDays = Math.floor(diffMs / 86_400_000);

          if (diffDays === 0) setLabel("Updated today");
          else if (diffDays === 1) setLabel("Updated yesterday");
          else setLabel(`Updated ${h.latest_data}`);
        }
      })
      .catch(() => {});
  }, []);

  if (!label) return null;

  return (
    <span className="text-[10px] font-mono text-white/50" title="Last data refresh">
      {label}
    </span>
  );
}
