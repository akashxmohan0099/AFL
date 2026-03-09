"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { getVenueDetail } from "@/lib/api";
import type { VenueDetail } from "@/lib/types";
import { TEAM_ABBREVS, TEAM_COLORS, displayVenue } from "@/lib/constants";
import { MatchCard } from "@/components/matches/MatchCard";
import { MapPin, Thermometer, Wind, Droplets, CloudRain } from "lucide-react";
import Link from "next/link";
import { Breadcrumb } from "@/components/ui/breadcrumb";

export default function VenueDetailPage() {
  const params = useParams();
  const venueName = decodeURIComponent(params.venueName as string);
  const [venue, setVenue] = useState<VenueDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    getVenueDetail(venueName)
      .then(setVenue)
      .catch(() => setVenue(null))
      .finally(() => setLoading(false));
  }, [venueName]);

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Skeleton className="h-64" />
          <Skeleton className="h-64" />
        </div>
      </div>
    );
  }

  if (!venue) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-muted-foreground">Venue not found: {venueName}</p>
        </CardContent>
      </Card>
    );
  }

  const weather = venue.weather;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <Breadcrumb items={[
          { label: "Venues", href: "/venues" },
          { label: displayVenue(venue.venue) },
        ]} />
        <div className="flex items-center gap-2">
          <MapPin className="w-5 h-5 text-muted-foreground" />
          <h1 className="text-2xl font-bold">{displayVenue(venue.venue)}</h1>
        </div>
        <div className="flex gap-2 mt-2 flex-wrap">
          <Badge variant="outline">{venue.total_games} games</Badge>
          {venue.is_roofed && (
            <Badge variant="outline" className="text-green-400 border-green-400/30">
              Roofed
            </Badge>
          )}
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-5 pb-4 text-center">
            <p className="text-xs text-muted-foreground uppercase tracking-wider">Avg Total Score</p>
            <p className="text-2xl font-bold mt-1 tabular-nums">{venue.avg_total_score.toFixed(1)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-5 pb-4 text-center">
            <p className="text-xs text-muted-foreground uppercase tracking-wider">Avg Margin</p>
            <p className="text-2xl font-bold mt-1 tabular-nums">{venue.avg_margin.toFixed(1)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-5 pb-4 text-center">
            <p className="text-xs text-muted-foreground uppercase tracking-wider">Total Games</p>
            <p className="text-2xl font-bold mt-1 tabular-nums">{venue.total_games}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-5 pb-4 text-center">
            <p className="text-xs text-muted-foreground uppercase tracking-wider">
              {venue.is_roofed ? "Roofed Venue" : "Open Air"}
            </p>
            <p className="text-2xl font-bold mt-1">{venue.is_roofed ? "Yes" : "No"}</p>
          </CardContent>
        </Card>
      </div>

      {/* Weather Profile */}
      {weather && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Weather Profile</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-orange-500/10 flex items-center justify-center">
                  <Thermometer className="w-5 h-5 text-orange-400" />
                </div>
                <div>
                  <p className="text-sm font-semibold tabular-nums">{weather.avg_temperature.toFixed(1)} C</p>
                  <p className="text-xs text-muted-foreground">Avg Temperature</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-cyan-500/10 flex items-center justify-center">
                  <Wind className="w-5 h-5 text-cyan-400" />
                </div>
                <div>
                  <p className="text-sm font-semibold tabular-nums">{weather.avg_wind_speed.toFixed(1)} km/h</p>
                  <p className="text-xs text-muted-foreground">Avg Wind</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
                  <CloudRain className="w-5 h-5 text-blue-400" />
                </div>
                <div>
                  <p className="text-sm font-semibold tabular-nums">{(weather.pct_wet * 100).toFixed(0)}%</p>
                  <p className="text-xs text-muted-foreground">Wet Games</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-violet-500/10 flex items-center justify-center">
                  <Droplets className="w-5 h-5 text-violet-400" />
                </div>
                <div>
                  <p className="text-sm font-semibold tabular-nums">{weather.avg_humidity.toFixed(0)}%</p>
                  <p className="text-xs text-muted-foreground">Avg Humidity</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Top Performers */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Top Goal Scorers */}
        {venue.top_goal_scorers && venue.top_goal_scorers.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Top Goal Scorers at {displayVenue(venue.venue)}</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Player</TableHead>
                    <TableHead>Team</TableHead>
                    <TableHead className="text-right">Games</TableHead>
                    <TableHead className="text-right">Total GL</TableHead>
                    <TableHead className="text-right">Avg GL</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {venue.top_goal_scorers.map((p, i) => (
                    <TableRow key={i}>
                      <TableCell className="font-medium text-sm">
                        <Link href={`/players/${encodeURIComponent(`${p.player}_${p.team}`)}`} className="hover:text-primary transition-colors">{p.player}</Link>
                      </TableCell>
                      <TableCell>
                        <span className="flex items-center gap-1.5">
                          <span
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: TEAM_COLORS[p.team]?.primary || "#666" }}
                          />
                          {TEAM_ABBREVS[p.team] || p.team}
                        </span>
                      </TableCell>
                      <TableCell className="text-right tabular-nums">{p.games}</TableCell>
                      <TableCell className="text-right tabular-nums">{p.total_goals}</TableCell>
                      <TableCell className="text-right tabular-nums font-semibold">
                        {p.avg_goals.toFixed(2)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        )}

        {/* Top Disposal Getters */}
        {venue.top_disposal_getters && venue.top_disposal_getters.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Top Disposal Getters at {displayVenue(venue.venue)}</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Player</TableHead>
                    <TableHead>Team</TableHead>
                    <TableHead className="text-right">Games</TableHead>
                    <TableHead className="text-right">Total DI</TableHead>
                    <TableHead className="text-right">Avg DI</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {venue.top_disposal_getters.map((p, i) => (
                    <TableRow key={i}>
                      <TableCell className="font-medium text-sm">
                        <Link href={`/players/${encodeURIComponent(`${p.player}_${p.team}`)}`} className="hover:text-primary transition-colors">{p.player}</Link>
                      </TableCell>
                      <TableCell>
                        <span className="flex items-center gap-1.5">
                          <span
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: TEAM_COLORS[p.team]?.primary || "#666" }}
                          />
                          {TEAM_ABBREVS[p.team] || p.team}
                        </span>
                      </TableCell>
                      <TableCell className="text-right tabular-nums">{p.games}</TableCell>
                      <TableCell className="text-right tabular-nums">{p.total_disposals}</TableCell>
                      <TableCell className="text-right tabular-nums font-semibold">
                        {p.avg_disposals.toFixed(1)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Recent Matches */}
      {venue.recent_matches && venue.recent_matches.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-lg font-semibold">Recent Matches</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {venue.recent_matches.map((m) => (
              <MatchCard key={m.match_id} match={m} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
