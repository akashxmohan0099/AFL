"use client";

import { useState, useEffect, createContext, useContext } from "react";
import { cn } from "@/lib/utils";
import { NewsSidebar } from "@/components/news/NewsSidebar";

/* ---------- Context for news panel state ---------- */

const NewsPanelContext = createContext<{
  open: boolean;
  toggle: () => void;
}>({ open: false, toggle: () => {} });

export function useNewsPanel() {
  return useContext(NewsPanelContext);
}

/* ---------- Provider wraps the app shell ---------- */

export function NewsPanelProvider({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem("afl-news-panel");
    if (saved === "open") setOpen(true);
  }, []);

  useEffect(() => {
    localStorage.setItem("afl-news-panel", open ? "open" : "closed");
  }, [open]);

  return (
    <NewsPanelContext.Provider value={{ open, toggle: () => setOpen((o) => !o) }}>
      <div className="flex flex-1 min-w-0">
        {/* Main content area */}
        <div className="flex-1 flex flex-col min-w-0">
          {children}
        </div>

        {/* Right sidebar — desktop: inline, mobile: overlay */}
        {open && (
          <>
            {/* Mobile backdrop */}
            <div
              className="xl:hidden fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
              onClick={() => setOpen(false)}
            />
            <aside
              className={cn(
                "border-l border-border/60 bg-sidebar flex flex-col shrink-0",
                // Mobile: fixed overlay
                "fixed right-0 top-0 h-screen w-80 z-50 shadow-2xl",
                // Desktop: inline in flex layout
                "xl:relative xl:w-80 xl:shadow-none xl:z-auto xl:h-screen xl:sticky xl:top-0"
              )}
            >
              {/* Close button (mobile only) */}
              <button
                onClick={() => setOpen(false)}
                className="xl:hidden absolute top-2.5 right-2.5 p-1 rounded hover:bg-muted/50 text-muted-foreground z-10"
                aria-label="Close news panel"
              >
                <svg className="w-4 h-4" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M5 5l10 10M15 5l-10 10" />
                </svg>
              </button>
              <NewsSidebar />
            </aside>
          </>
        )}
      </div>
    </NewsPanelContext.Provider>
  );
}

/* ---------- Toggle button (placed in header) ---------- */

function IconNews({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="3" width="16" height="14" rx="1.5" />
      <path d="M5 7h10M5 10h6M5 13h8" />
    </svg>
  );
}

export function NewsToggleButton() {
  const { open, toggle } = useNewsPanel();

  return (
    <button
      onClick={toggle}
      className={cn(
        "flex items-center gap-1.5 px-1.5 py-1 rounded transition-colors relative",
        open
          ? "bg-primary/10 text-primary"
          : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
      )}
      aria-label="Toggle intel panel"
      title="Intel Feed"
    >
      <IconNews className="w-4 h-4" />
      <span className="text-[10px] font-mono hidden sm:inline">Intel</span>
    </button>
  );
}
