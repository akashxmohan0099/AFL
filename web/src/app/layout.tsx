import type { Metadata } from "next";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import Script from "next/script";
import "./globals.css";
import { Sidebar, MobileSidebar } from "@/components/layout/Sidebar";
import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { NewsPanelProvider, NewsToggleButton } from "@/components/layout/AppShell";

const ibmPlexSans = IBM_Plex_Sans({
  variable: "--font-ibm-sans",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

const ibmPlexMono = IBM_Plex_Mono({
  variable: "--font-ibm-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "AFL Predict Pro",
  description: "Professional AFL prediction and analytics platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${ibmPlexSans.variable} ${ibmPlexMono.variable} antialiased`}
      >
        <Script id="theme-init" strategy="beforeInteractive">{`(function(){try{var t=localStorage.getItem('afl-theme')||(window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');document.documentElement.classList.toggle('dark',t==='dark')}catch(e){}})()`}</Script>
        <div className="flex min-h-screen">
          <Sidebar />
          <NewsPanelProvider>
            {/* Top bar */}
            <header className="h-11 border-b border-border/50 bg-card/50 backdrop-blur-sm flex items-center justify-between px-4 md:px-5 sticky top-0 z-40">
              <div className="flex items-center gap-3">
                <MobileSidebar />
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-widest">
                  AFL Predict Pro
                </span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-mono">
                  LIVE
                </span>
              </div>
              <div className="flex items-center gap-4 text-[11px] text-muted-foreground font-mono">
                <span>Season <span className="text-foreground font-semibold">2026</span></span>
                <span className="w-px h-3 bg-border" />
                <span>v4.2</span>
                <ThemeToggle />
                <span className="w-px h-3 bg-border" />
                <NewsToggleButton />
              </div>
            </header>
            <main className="flex-1 p-5 overflow-auto">
              <div className="max-w-[1400px] mx-auto">{children}</div>
            </main>
          </NewsPanelProvider>
        </div>
      </body>
    </html>
  );
}
