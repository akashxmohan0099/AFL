"use client";

import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";

function formatUpdated(dateStr: string): string {
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return `Updated ${dateStr}`;

  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffMins = Math.floor(diffMs / 60_000);
  const diffHours = Math.floor(diffMs / 3_600_000);
  const diffDays = Math.floor(diffMs / 86_400_000);

  const time = d.toLocaleTimeString("en-AU", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });

  if (diffMins < 60) return `Updated ${diffMins}m ago`;
  if (diffHours < 24) return `Updated ${diffHours}h ago`;
  if (diffDays === 1) return `Updated yesterday, ${time}`;

  const date = d.toLocaleDateString("en-AU", {
    day: "numeric",
    month: "short",
  });
  return `Updated ${date}, ${time}`;
}

export function LastUpdated() {
  const [label, setLabel] = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then((h) => {
        const full = h?.latest_data_full;
        const dateOnly = h?.latest_data;
        if (full) {
          setLabel(formatUpdated(String(full)));
        } else if (dateOnly) {
          setLabel(`Updated ${dateOnly}`);
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
