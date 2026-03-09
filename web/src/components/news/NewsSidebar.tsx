"use client";

import { useEffect, useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { getIntelFeed, getInjuries } from "@/lib/api";
import type { IntelFeed, IntelSignal, InjuryList, InjuryRecord, SignalType } from "@/lib/types";
import { TEAM_COLORS, TEAM_ABBREVS } from "@/lib/constants";

/* ---------- Constants ---------- */

const SIGNAL_TABS: { key: string; label: string; types?: SignalType[] }[] = [
  { key: "all", label: "All" },
  { key: "breaking", label: "Breaking" },
  { key: "injuries", label: "Injuries", types: ["injury"] },
  { key: "form", label: "Form", types: ["form"] },
  { key: "tactical", label: "Tactical", types: ["tactical", "selection"] },
  { key: "tips", label: "Tips", types: ["prediction"] },
  { key: "suspensions", label: "Bans", types: ["suspension"] },
];

const SIGNAL_COLORS: Record<SignalType, string> = {
  injury: "text-red-400 bg-red-500/10",
  suspension: "text-orange-400 bg-orange-500/10",
  form: "text-emerald-400 bg-emerald-500/10",
  tactical: "text-blue-400 bg-blue-500/10",
  selection: "text-cyan-400 bg-cyan-500/10",
  prediction: "text-violet-400 bg-violet-500/10",
  general: "text-zinc-400 bg-zinc-500/10",
};

const SIGNAL_ICONS: Record<SignalType, string> = {
  injury: "+",
  suspension: "!",
  form: "\u2191",
  tactical: "\u2699",
  selection: "\u221A",
  prediction: "\u25B6",
  general: "\u2022",
};

const SENTIMENT_COLORS: Record<string, string> = {
  positive: "bg-emerald-400",
  negative: "bg-red-400",
  neutral: "bg-zinc-400",
  mixed: "bg-amber-400",
};

const REFRESH_INTERVAL = 5 * 60 * 1000; // 5 minutes

/* ---------- Icons ---------- */

function IconRefresh({ className, spinning }: { className?: string; spinning?: boolean }) {
  return (
    <svg
      className={cn("w-3 h-3", spinning && "animate-spin", className)}
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M17 10a7 7 0 1 1-2-5" />
      <path d="M17 3v4h-4" />
    </svg>
  );
}

/* ---------- Signal card ---------- */

function SignalCard({ signal }: { signal: IntelSignal }) {
  const [expanded, setExpanded] = useState(false);
  const typeColor = SIGNAL_COLORS[signal.signal_type] || SIGNAL_COLORS.general;
  const icon = SIGNAL_ICONS[signal.signal_type] || "\u2022";
  const sentimentColor = SENTIMENT_COLORS[signal.sentiment] || SENTIMENT_COLORS.neutral;

  // Format time
  const timeStr = (() => {
    if (!signal.published_at) return "";
    try {
      const d = new Date(signal.published_at);
      const now = new Date();
      const diffMs = now.getTime() - d.getTime();
      const diffH = Math.floor(diffMs / 3600000);
      if (diffH < 1) return `${Math.max(1, Math.floor(diffMs / 60000))}m`;
      if (diffH < 24) return `${diffH}h`;
      return `${Math.floor(diffH / 24)}d`;
    } catch {
      return "";
    }
  })();

  return (
    <div
      className={cn(
        "border-b border-border/20 hover:bg-muted/20 transition-colors cursor-pointer",
        signal.relevance_score >= 0.7 && "border-l-2 border-l-red-400/60",
      )}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="px-3 py-2">
        {/* Top row: type badge + time */}
        <div className="flex items-center gap-1.5 mb-1">
          <span className={cn("text-[8px] font-mono font-bold px-1 py-0.5 rounded", typeColor)}>
            {icon} {signal.signal_type.toUpperCase()}
          </span>
          {signal.relevance_score >= 0.7 && (
            <span className="text-[8px] font-mono font-bold px-1 py-0.5 rounded bg-red-500/20 text-red-400">
              HIGH
            </span>
          )}
          {/* Sentiment dot */}
          <span className={cn("w-1.5 h-1.5 rounded-full", sentimentColor)} />
          <span className="text-[9px] text-muted-foreground/40 font-mono ml-auto tabular-nums">
            {timeStr}
          </span>
        </div>

        {/* Headline */}
        <p className="text-[11px] font-medium leading-snug text-foreground/90 line-clamp-2">
          {signal.headline}
        </p>

        {/* Teams row */}
        {signal.teams.length > 0 && (
          <div className="flex items-center gap-1 mt-1 flex-wrap">
            {signal.teams.slice(0, 4).map((team) => {
              const color = TEAM_COLORS[team]?.primary || "#555";
              const abbr = TEAM_ABBREVS[team] || team.slice(0, 3).toUpperCase();
              return (
                <span
                  key={team}
                  className="text-[8px] font-mono font-semibold px-1 py-0.5 rounded"
                  style={{ backgroundColor: color + "18", color }}
                >
                  {abbr}
                </span>
              );
            })}
            {/* Direction arrows */}
            {Object.entries(signal.direction || {}).slice(0, 3).map(([entity, dir]) => {
              const arrow = dir === "bullish" ? "\u25B2" : dir === "bearish" ? "\u25BC" : "\u2022";
              const c = dir === "bullish" ? "text-emerald-400" : dir === "bearish" ? "text-red-400" : "text-zinc-400";
              const abbr = TEAM_ABBREVS[entity] || entity.split(",")[0]?.slice(0, 6);
              return (
                <span key={entity} className={cn("text-[8px] font-mono", c)}>
                  {abbr} {arrow}
                </span>
              );
            })}
          </div>
        )}

        {/* Expanded content */}
        {expanded && (
          <div className="mt-2 space-y-1.5">
            {signal.summary && (
              <p className="text-[10px] text-muted-foreground/70 leading-relaxed">
                {signal.summary}
              </p>
            )}
            {signal.key_facts?.length > 0 && (
              <div className="space-y-0.5">
                {signal.key_facts.map((fact, i) => (
                  <p key={i} className="text-[9px] text-muted-foreground/60 font-mono">
                    \u2022 {fact}
                  </p>
                ))}
              </div>
            )}
            {signal.prediction_impact && (
              <p className="text-[9px] text-primary/80 font-mono italic">
                {signal.prediction_impact}
              </p>
            )}
            {signal.source_url && (
              <a
                href={signal.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[9px] text-primary/50 hover:text-primary font-mono underline"
                onClick={(e) => e.stopPropagation()}
              >
                Source
              </a>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------- Injury mini-row (for injuries tab) ---------- */

function InjuryMiniRow({ team, injuries }: { team: string; injuries: InjuryRecord[] }) {
  const [open, setOpen] = useState(false);
  const abbr = TEAM_ABBREVS[team] || team.slice(0, 3).toUpperCase();
  const color = TEAM_COLORS[team]?.primary || "#555";
  const serious = injuries.filter((i) => i.severity >= 3).length;

  return (
    <div className="border-b border-border/20">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-1.5 hover:bg-muted/20 transition-colors text-left"
      >
        <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
        <span className="text-[10px] font-semibold font-mono flex-1">{abbr}</span>
        {serious > 0 && (
          <span className="text-[8px] font-mono text-red-400 tabular-nums">{serious}!</span>
        )}
        <span className="text-[9px] font-mono text-muted-foreground/50 tabular-nums">{injuries.length}</span>
        <svg
          className={cn("w-2.5 h-2.5 transition-transform duration-100 text-muted-foreground/30", open && "rotate-90")}
          viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"
        >
          <path d="M7 4l6 6-6 6" />
        </svg>
      </button>
      {open && (
        <div className="px-3 pb-1.5 space-y-1">
          {injuries.map((inj, i) => {
            const sevColors: Record<number, string> = {
              0: "text-emerald-400", 1: "text-yellow-400",
              2: "text-amber-400", 3: "text-orange-400", 4: "text-red-400",
            };
            return (
              <div key={i} className="flex items-center justify-between gap-2">
                <span className="text-[10px] text-foreground/80 truncate">{inj.player}</span>
                <div className="flex items-center gap-1.5 shrink-0">
                  <span className="text-[8px] text-muted-foreground/50 font-mono">{inj.injury}</span>
                  <span className={cn("text-[8px] font-mono font-bold", sevColors[inj.severity] || sevColors[2])}>
                    {inj.severity_label}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ---------- Stats bar ---------- */

function StatBar({ by_type }: { by_type: Record<string, number> }) {
  const total = Object.values(by_type).reduce((a, b) => a + b, 0);
  if (total === 0) return null;

  const barColors: Record<string, string> = {
    injury: "bg-red-400",
    suspension: "bg-orange-400",
    form: "bg-emerald-400",
    tactical: "bg-blue-400",
    selection: "bg-cyan-400",
    prediction: "bg-violet-400",
    general: "bg-zinc-400",
  };

  return (
    <div className="flex h-1 rounded-full overflow-hidden">
      {Object.entries(by_type).map(([type, count]) => (
        <div
          key={type}
          className={cn("h-full", barColors[type] || "bg-zinc-400")}
          style={{ width: `${(count / total) * 100}%` }}
          title={`${type}: ${count}`}
        />
      ))}
    </div>
  );
}

/* ---------- Main sidebar ---------- */

export function NewsSidebar() {
  const [feed, setFeed] = useState<IntelFeed | null>(null);
  const [injuries, setInjuries] = useState<InjuryList | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState("all");
  const [teamFilter, setTeamFilter] = useState("");

  const fetchData = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    else setRefreshing(true);
    try {
      const [feedData, injData] = await Promise.all([
        getIntelFeed({ limit: 200 }),
        getInjuries(),
      ]);
      setFeed(feedData);
      setInjuries(injData);
    } catch {
      // Keep existing data
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(() => fetchData(true), REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Filter signals by tab and team
  const filteredSignals = (() => {
    if (!feed) return [];
    let signals = feed.signals;

    const tab = SIGNAL_TABS.find((t) => t.key === activeTab);

    if (activeTab === "breaking") {
      signals = signals.filter((s) => s.relevance_score >= 0.7);
    } else if (tab?.types) {
      signals = signals.filter((s) => tab.types!.includes(s.signal_type));
    }

    if (teamFilter) {
      const tf = teamFilter.toLowerCase();
      signals = signals.filter((s) =>
        s.teams.some((t) => t.toLowerCase().includes(tf) ||
          (TEAM_ABBREVS[t] || "").toLowerCase().includes(tf)),
      );
    }

    return signals;
  })();

  // Build injury list for the injuries tab
  const injuryTeams = injuries?.teams ?? {};
  const injuryTeamNames = Object.keys(injuryTeams).sort();
  const filteredInjuryTeams = teamFilter
    ? injuryTeamNames.filter((t) =>
        t.toLowerCase().includes(teamFilter.toLowerCase()) ||
        (TEAM_ABBREVS[t] || "").toLowerCase().includes(teamFilter.toLowerCase()))
    : injuryTeamNames;

  const totalSignals = feed?.total ?? 0;
  const breakingCount = feed?.breaking_count ?? 0;

  return (
    <div className="flex flex-col h-full bg-background/50">
      {/* Header */}
      <div className="px-3 py-2.5 border-b border-border/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <h2 className="text-[11px] font-bold tracking-tight uppercase">Intel Feed</h2>
          </div>
          <div className="flex items-center gap-2">
            {breakingCount > 0 && (
              <span className="text-[8px] font-mono font-bold px-1 py-0.5 rounded bg-red-500/15 text-red-400 animate-pulse">
                {breakingCount} ALERT{breakingCount > 1 ? "S" : ""}
              </span>
            )}
            <button
              onClick={() => fetchData(true)}
              className="p-0.5 rounded hover:bg-muted/30 transition-colors text-muted-foreground/50"
              title="Refresh"
            >
              <IconRefresh spinning={refreshing} />
            </button>
          </div>
        </div>
        {feed?.updated && (
          <p className="text-[8px] text-muted-foreground/40 font-mono mt-0.5">
            {new Date(feed.updated).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </p>
        )}
      </div>

      {/* Summary bar */}
      {!loading && feed && (
        <div className="px-3 py-1.5 border-b border-border/30">
          <div className="flex gap-1.5 mb-1.5">
            <div className="flex-1 bg-muted/20 rounded px-1.5 py-1 text-center">
              <p className="text-sm font-bold font-mono tabular-nums leading-none">{totalSignals}</p>
              <p className="text-[7px] text-muted-foreground/40 uppercase tracking-wider">Signals</p>
            </div>
            <div className="flex-1 bg-red-500/5 rounded px-1.5 py-1 text-center">
              <p className="text-sm font-bold font-mono tabular-nums leading-none text-red-400">
                {feed.by_type?.injury ?? 0}
              </p>
              <p className="text-[7px] text-muted-foreground/40 uppercase tracking-wider">Injuries</p>
            </div>
            <div className="flex-1 bg-orange-500/5 rounded px-1.5 py-1 text-center">
              <p className="text-sm font-bold font-mono tabular-nums leading-none text-orange-400">
                {feed.by_type?.suspension ?? 0}
              </p>
              <p className="text-[7px] text-muted-foreground/40 uppercase tracking-wider">Bans</p>
            </div>
            <div className="flex-1 bg-muted/20 rounded px-1.5 py-1 text-center">
              <p className="text-sm font-bold font-mono tabular-nums leading-none">
                {Object.keys(feed.by_team ?? {}).length}
              </p>
              <p className="text-[7px] text-muted-foreground/40 uppercase tracking-wider">Teams</p>
            </div>
          </div>
          <StatBar by_type={feed.by_type ?? {}} />
        </div>
      )}

      {/* Tab bar */}
      <div className="px-2 py-1.5 border-b border-border/30 flex gap-0.5 overflow-x-auto scrollbar-none">
        {SIGNAL_TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={cn(
              "text-[9px] font-mono font-medium px-1.5 py-1 rounded whitespace-nowrap transition-colors",
              activeTab === tab.key
                ? "bg-primary/10 text-primary"
                : "text-muted-foreground/50 hover:text-muted-foreground hover:bg-muted/20",
            )}
          >
            {tab.label}
            {tab.key === "breaking" && breakingCount > 0 && (
              <span className="ml-0.5 text-red-400">{breakingCount}</span>
            )}
          </button>
        ))}
      </div>

      {/* Team filter */}
      <div className="px-3 py-1.5 border-b border-border/20">
        <input
          type="text"
          placeholder="Filter team..."
          value={teamFilter}
          onChange={(e) => setTeamFilter(e.target.value)}
          className="w-full text-[10px] bg-muted/20 border border-border/30 rounded px-2 py-1 placeholder:text-muted-foreground/25 focus:outline-none focus:border-primary/30 font-mono"
        />
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="px-3 py-4 space-y-2">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <div key={i} className="space-y-1">
                <div className="h-3 bg-muted/30 rounded animate-pulse w-16" />
                <div className="h-4 bg-muted/20 rounded animate-pulse" />
                <div className="h-3 bg-muted/15 rounded animate-pulse w-3/4" />
              </div>
            ))}
          </div>
        ) : activeTab === "injuries" && !teamFilter ? (
          /* Dedicated injury view grouped by team */
          filteredInjuryTeams.length > 0 ? (
            filteredInjuryTeams.map((team) => (
              <InjuryMiniRow key={team} team={team} injuries={injuryTeams[team]} />
            ))
          ) : (
            <EmptyState message="No injury data available" />
          )
        ) : filteredSignals.length > 0 ? (
          filteredSignals.map((signal) => (
            <SignalCard key={signal.id} signal={signal} />
          ))
        ) : (
          <EmptyState
            message={
              activeTab === "breaking"
                ? "No breaking alerts"
                : teamFilter
                  ? "No signals for this team"
                  : "No signals in this category"
            }
          />
        )}
      </div>

      {/* Footer legend */}
      <div className="px-3 py-1.5 border-t border-border/30">
        <div className="flex flex-wrap gap-x-2 gap-y-0.5">
          {(["injury", "suspension", "form", "tactical", "prediction"] as SignalType[]).map((type) => {
            const c = SIGNAL_COLORS[type];
            return (
              <button
                key={type}
                onClick={() => {
                  const tab = SIGNAL_TABS.find((t) => t.types?.includes(type));
                  if (tab) setActiveTab(tab.key);
                }}
                className={cn("text-[7px] font-mono px-1 py-0.5 rounded transition-colors hover:opacity-80", c)}
              >
                {SIGNAL_ICONS[type]} {type}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="px-3 py-8 text-center">
      <div className="text-xl text-muted-foreground/15 mb-2">\u25CE</div>
      <p className="text-[10px] text-muted-foreground/35 font-mono">{message}</p>
    </div>
  );
}
