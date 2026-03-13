"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  Trophy,
  Swords,
  TrendingUp,
  History,
  Users,
  MapPin,
  Calendar,
} from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/ladder", label: "Ladder", icon: Trophy },
  { href: "/matches", label: "Matches", icon: Swords },
  { href: "/predictions", label: "Predictions", icon: TrendingUp },
  { href: "/predictions/history", label: "Pred vs Actual", icon: History },
  { href: "/players", label: "Players", icon: Users },
  { href: "/venues", label: "Venues", icon: MapPin },
  { href: "/schedule", label: "Schedule", icon: Calendar },
];

function NavContent({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();

  return (
    <>
      <div className="px-4 py-4 border-b border-sidebar-border">
        <h1 className="text-base font-extrabold tracking-tight text-[oklch(0.90_0.15_95)]">
          AFL Predict Pro
        </h1>
        <p className="text-[11px] text-sidebar-foreground/50 font-mono mt-0.5">
          Season 2026
        </p>
      </div>

      <nav className="flex-1 px-2 py-3 space-y-0.5 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : item.href === "/predictions"
                ? pathname === "/predictions"
                : pathname.startsWith(item.href);
          const Icon = item.icon;

          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={onNavigate}
              className={cn(
                "flex items-center gap-2.5 px-3 py-2 rounded text-sm font-medium transition-colors duration-100",
                isActive
                  ? "bg-sidebar-accent text-[oklch(0.90_0.15_95)]"
                  : "text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-accent/50"
              )}
            >
              <Icon
                className={cn(
                  "w-[18px] h-[18px] shrink-0",
                  isActive ? "text-[oklch(0.90_0.15_95)]" : "text-sidebar-foreground/50"
                )}
                strokeWidth={1.5}
              />
              {item.label}
            </Link>
          );
        })}
      </nav>
    </>
  );
}

export function Sidebar() {
  return (
    <aside className="hidden md:flex w-52 border-r border-sidebar-border bg-sidebar flex-col h-screen sticky top-0 shrink-0">
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
        <svg
          className="w-5 h-5 text-white/70"
          viewBox="0 0 20 20"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
        >
          <path d="M3 5h14M3 10h14M3 15h14" />
        </svg>
      </button>

      {open && (
        <div className="md:hidden fixed inset-0 z-50 flex">
          <div
            className="fixed inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setOpen(false)}
          />
          <aside className="relative w-52 bg-sidebar flex flex-col h-full border-r border-sidebar-border shadow-2xl">
            <NavContent onNavigate={() => setOpen(false)} />
          </aside>
        </div>
      )}
    </>
  );
}
