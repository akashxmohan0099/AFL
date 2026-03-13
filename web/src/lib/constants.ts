export const TEAM_COLORS: Record<string, { primary: string; secondary: string }> = {
  Adelaide: { primary: "#002B5C", secondary: "#FFD200" },
  "Brisbane Lions": { primary: "#A30046", secondary: "#0055A3" },
  Carlton: { primary: "#001C3D", secondary: "#FFFFFF" },
  Collingwood: { primary: "#000000", secondary: "#FFFFFF" },
  Essendon: { primary: "#CC0000", secondary: "#000000" },
  Fremantle: { primary: "#2A0D45", secondary: "#FFFFFF" },
  Geelong: { primary: "#001F3D", secondary: "#FFFFFF" },
  "Gold Coast": { primary: "#D4001E", secondary: "#FFD200" },
  "Greater Western Sydney": { primary: "#F15C22", secondary: "#333333" },
  Hawthorn: { primary: "#4D2004", secondary: "#FFD200" },
  Melbourne: { primary: "#001C3D", secondary: "#CC0000" },
  "North Melbourne": { primary: "#002B5C", secondary: "#FFFFFF" },
  "Port Adelaide": { primary: "#008AAB", secondary: "#000000" },
  Richmond: { primary: "#000000", secondary: "#FFD200" },
  "St Kilda": { primary: "#000000", secondary: "#CC0000" },
  Sydney: { primary: "#CC0000", secondary: "#FFFFFF" },
  "West Coast": { primary: "#002B5C", secondary: "#FFD200" },
  "Western Bulldogs": { primary: "#003F87", secondary: "#CC0000" },
};

export const TEAM_ABBREVS: Record<string, string> = {
  Adelaide: "ADE",
  "Brisbane Lions": "BRL",
  Carlton: "CAR",
  Collingwood: "COL",
  Essendon: "ESS",
  Fremantle: "FRE",
  Geelong: "GEE",
  "Gold Coast": "GCS",
  "Greater Western Sydney": "GWS",
  Hawthorn: "HAW",
  Melbourne: "MEL",
  "North Melbourne": "NTH",
  "Port Adelaide": "PTA",
  Richmond: "RIC",
  "St Kilda": "STK",
  Sydney: "SYD",
  "West Coast": "WCE",
  "Western Bulldogs": "WBD",
};

export const STAT_LABELS: Record<string, string> = {
  GL: "Goals",
  BH: "Behinds",
  DI: "Disposals",
  MK: "Marks",
  KI: "Kicks",
  HB: "Handballs",
  TK: "Tackles",
  HO: "Hitouts",
};

export const CURRENT_YEAR = 2026;
export const AVAILABLE_YEARS = Array.from({ length: CURRENT_YEAR - 2014 }, (_, i) => CURRENT_YEAR - i);

// Canonical data venue name → display name (current sponsor + city)
export const VENUE_DISPLAY: Record<string, string> = {
  "M.C.G.":            "MCG, Melbourne",
  "Docklands":         "Marvel Stadium, Melbourne",
  "S.C.G.":            "SCG, Sydney",
  "SCG":               "SCG, Sydney",
  "Gabba":             "The Gabba, Brisbane",
  "Kardinia Park":     "GMHBA Stadium, Geelong",
  "Adelaide Oval":     "Adelaide Oval, Adelaide",
  "Perth Stadium":     "Optus Stadium, Perth",
  "Carrara":           "People First Stadium, Gold Coast",
  "Sydney Showground": "Engie Stadium, Sydney",
  "Giants Stadium":    "Engie Stadium, Sydney",
  "York Park":         "UTAS Stadium, Launceston",
  "Bellerive Oval":    "Blundstone Arena, Hobart",
  "Manuka Oval":       "Manuka Oval, Canberra",
  "Subiaco":           "Subiaco Oval, Perth",
  "Eureka Stadium":    "Mars Stadium, Ballarat",
  "TIO Stadium":       "TIO Stadium, Darwin",
  "Marrara Oval":      "TIO Stadium, Darwin",
  "Cazaly's Stadium":  "Cazalys Stadium, Cairns",
  "Traeger Park":      "Traeger Park, Alice Springs",
  "Stadium Australia":  "Accor Stadium, Sydney",
  "Norwood Oval":      "Norwood Oval, Adelaide",
  "Riverway Stadium":  "Riverway Stadium, Townsville",
  "Summit Sports Park": "Summit Sports Park, Gold Coast",
  "Jiangwan Stadium":  "Jiangwan Stadium, Shanghai",
  "Wellington":        "Wellington, NZ",
};

export function displayVenue(venue?: string | null): string {
  if (!venue) return "";
  return VENUE_DISPLAY[venue] || venue;
}

export const CHART_COLORS = {
  goals: "#f43f5e",
  disposals: "#FFD700",
  marks: "#34D399",
  behinds: "#f59e0b",
  tackles: "#a78bfa",
  kicks: "#22d3ee",
  handballs: "#f472b6",
  accent: "#FFD700",
  muted: "#5a7066",
  positive: "#34D399",
  negative: "#f43f5e",
  neutral: "#FFD700",
};
