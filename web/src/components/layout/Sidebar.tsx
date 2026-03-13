"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

function IconDashboard({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="2" width="7" height="7" rx="1.5" />
      <rect x="11" y="2" width="7" height="4" rx="1.5" />
      <rect x="2" y="11" width="7" height="4" rx="1.5" />
      <rect x="11" y="8" width="7" height="7" rx="1.5" />
    </svg>
  );
}

function IconMatch({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="6" cy="10" r="3" />
      <circle cx="14" cy="10" r="3" />
      <path d="M9 10h2" />
      <path d="M2 4h16M2 16h16" />
    </svg>
  );
}

function IconPredict({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 17l4-6 3 3 4-7 3 4" />
      <circle cx="17" cy="7" r="1.5" fill="currentColor" stroke="none" />
    </svg>
  );
}

function IconHistory({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="10" cy="10" r="7" />
      <path d="M10 6v4l3 2" />
      <path d="M3 4l1.5 1.5M3 16l1.5-1.5" />
    </svg>
  );
}

function IconPlayers({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="10" cy="6" r="3" />
      <path d="M4 17c0-3.3 2.7-6 6-6s6 2.7 6 6" />
    </svg>
  );
}

function IconVenue({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="10" cy="14" rx="7" ry="3" />
      <path d="M3 14V8c0-1.7 3.1-3 7-3s7 1.3 7 3v6" />
      <path d="M3 8c0 1.7 3.1 3 7 3s7-1.3 7-3" />
    </svg>
  );
}

function IconLadder({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 3v14M16 3v14" />
      <path d="M4 6h12M4 10h12M4 14h12" />
    </svg>
  );
}

function IconSchedule({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="14" height="13" rx="1.5" />
      <path d="M3 8h14" />
      <path d="M7 2v4M13 2v4" />
      <path d="M7 11h2M11 11h2M7 14h2" />
    </svg>
  );
}

function IconRecap({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 3h14v14H3z" />
      <path d="M7 7h6M7 10h4M7 13h5" />
      <circle cx="15" cy="15" r="3" fill="currentColor" stroke="none" opacity="0.3" />
      <path d="M14 15l1 1 2-2" strokeWidth="1.2" />
    </svg>
  );
}

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: IconDashboard },
  { href: "/ladder", label: "Ladder", icon: IconLadder },
  { href: "/matches", label: "Matches", icon: IconMatch },
  {
    href: "/predictions",
    label: "Predictions",
    icon: IconPredict,
    children: [{ href: "/predictions/history", label: "History", icon: IconHistory }],
  },
  { href: "/players", label: "Players", icon: IconPlayers },
  { href: "/venues", label: "Venues", icon: IconVenue },
  { href: "/recap", label: "Round Recap", icon: IconRecap },
  { href: "/schedule", label: "Schedule", icon: IconSchedule },
];

function NavContent({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();

  return (
    <>
      <div className="px-4 py-4 border-b border-sidebar-border">
        <h1 className="text-base font-extrabold tracking-tight text-[oklch(0.90_0.15_95)]">AFL Predict Pro</h1>
        <p className="text-[11px] text-sidebar-foreground/50 font-mono mt-0.5">Player & Match Analytics</p>
      </div>
      <nav className="flex-1 px-2 py-3 space-y-0.5 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);
          const Icon = item.icon;
          return (
            <div key={item.href}>
              <Link
                href={item.href}
                onClick={onNavigate}
                className={cn(
                  "flex items-center gap-2.5 px-3 py-2 rounded text-sm font-semibold transition-colors duration-100",
                  isActive
                    ? "bg-sidebar-accent text-[oklch(0.90_0.15_95)]"
                    : "text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-accent/50"
                )}
              >
                <Icon className={cn("w-[18px] h-[18px] shrink-0", isActive ? "text-[oklch(0.90_0.15_95)]" : "text-sidebar-foreground/50")} />
                {item.label}
                {isActive && (
                  <span className="ml-auto w-1 h-4 rounded-full bg-[oklch(0.90_0.15_95)]" />
                )}
              </Link>
              {item.children && isActive && (
                <div className="ml-5 mt-0.5 space-y-px border-l border-sidebar-border/50 pl-2.5">
                  {item.children.map((child) => {
                    const childActive = pathname === child.href;
                    const ChildIcon = child.icon;
                    return (
                      <Link
                        key={child.href}
                        href={child.href}
                        onClick={onNavigate}
                        className={cn(
                          "flex items-center gap-2 px-2.5 py-1.5 rounded text-[13px] font-semibold transition-colors",
                          childActive
                            ? "text-[oklch(0.90_0.15_95)] bg-sidebar-accent/50"
                            : "text-sidebar-foreground/60 hover:text-sidebar-foreground hover:bg-sidebar-accent/30"
                        )}
                      >
                        <ChildIcon className="w-4 h-4 shrink-0" />
                        {child.label}
                      </Link>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </nav>
      <div className="px-4 py-3 border-t border-sidebar-border">
        <p className="text-[11px] text-sidebar-foreground/40">
          Predictions for goals, disposals, marks & match winners
        </p>
      </div>
    </>
  );
}

export function Sidebar() {
  return (
    <aside className="hidden md:flex w-56 border-r border-sidebar-border bg-sidebar flex-col h-screen sticky top-0 shrink-0">
      <NavContent />
    </aside>
  );
}

export function MobileSidebar() {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="md:hidden p-1.5 -ml-1.5 rounded hover:bg-white/10 transition-colors"
        aria-label="Open menu"
      >
        <svg className="w-5 h-5 text-white/70" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
          <path d="M3 5h14M3 10h14M3 15h14" />
        </svg>
      </button>

      {open && (
        <div className="md:hidden fixed inset-0 z-50 flex">
          <div
            className="fixed inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setOpen(false)}
          />
          <aside className="relative w-56 bg-sidebar flex flex-col h-full border-r border-sidebar-border shadow-2xl">
            <NavContent onNavigate={() => setOpen(false)} />
          </aside>
        </div>
      )}
    </>
  );
}
