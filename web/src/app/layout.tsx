import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import Script from "next/script";
import "./globals.css";
import { Sidebar, MobileSidebar } from "@/components/layout/Sidebar";
import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { NewsPanelProvider, NewsToggleButton } from "@/components/layout/AppShell";
import { LastUpdated } from "@/components/layout/LastUpdated";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
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
        className={`${inter.variable} ${jetbrainsMono.variable} antialiased`}
      >
        <Script id="theme-init" strategy="beforeInteractive">{`(function(){try{var t=localStorage.getItem('afl-theme')||'dark';document.documentElement.classList.toggle('dark',t==='dark')}catch(e){document.documentElement.classList.add('dark')}})()`}</Script>
        <div className="flex min-h-screen">
          <Sidebar />
          <NewsPanelProvider>
            {/* Top bar — brand green header */}
            <header className="h-11 border-b border-[oklch(0.30_0.06_170)] bg-[oklch(0.35_0.08_170)] flex items-center justify-between px-3 md:px-5 sticky top-0 z-40">
              <div className="flex items-center gap-2 md:gap-3 min-w-0">
                <MobileSidebar />
                <span className="text-xs font-bold text-[oklch(0.90_0.15_95)] uppercase tracking-widest whitespace-nowrap">
                  AFL Predict Pro
                </span>
              </div>
              <div className="flex items-center gap-2 md:gap-4 text-[11px] text-white/70 font-mono shrink-0">
                <span className="hidden sm:inline"><LastUpdated /></span>
                <span className="hidden sm:block w-px h-3 bg-white/20" />
                <span className="hidden md:inline">Season <span className="text-white font-semibold">2026</span></span>
                <ThemeToggle />
                <span className="w-px h-3 bg-white/20" />
                <NewsToggleButton />
              </div>
            </header>
            <main className="flex-1 p-3 md:p-5 overflow-auto">
              <div className="max-w-[1400px] mx-auto">{children}</div>
            </main>
          </NewsPanelProvider>
        </div>
      </body>
    </html>
  );
}
