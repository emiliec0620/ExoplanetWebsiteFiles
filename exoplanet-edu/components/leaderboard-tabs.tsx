"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Trophy, Award, Medal, Crown } from "lucide-react"
import { cn } from "@/lib/utils"

interface LeaderboardEntry {
  rank: number
  username: string
  count: number
  avatar?: string
}

const exoplanetLeaders: LeaderboardEntry[] = [
  { rank: 1, username: "StarGazer_42", count: 127 },
  { rank: 2, username: "CosmicExplorer", count: 98 },
  { rank: 3, username: "PlanetHunter_X", count: 87 },
  { rank: 4, username: "AstroNinja", count: 76 },
  { rank: 5, username: "NebulaSeeker", count: 65 },
  { rank: 6, username: "OrbitTracker", count: 58 },
  { rank: 7, username: "TransitMaster", count: 52 },
  { rank: 8, username: "ExoDetective", count: 47 },
  { rank: 9, username: "StellarScout", count: 41 },
  { rank: 10, username: "GalaxyWatcher", count: 38 },
]

const falsePositiveLeaders: LeaderboardEntry[] = [
  { rank: 1, username: "DataValidator", count: 89 },
  { rank: 2, username: "SignalAnalyst", count: 76 },
  { rank: 3, username: "BinaryBuster", count: 68 },
  { rank: 4, username: "NoiseFilter_Pro", count: 61 },
  { rank: 5, username: "AccuracyFirst", count: 54 },
  { rank: 6, username: "FalseAlarmFinder", count: 49 },
  { rank: 7, username: "QualityControl", count: 43 },
  { rank: 8, username: "PrecisionSeeker", count: 39 },
  { rank: 9, username: "DataCleaner", count: 35 },
  { rank: 10, username: "TruthHunter", count: 31 },
]

function getRankIcon(rank: number) {
  switch (rank) {
    case 1:
      return <Crown className="w-5 h-5 text-yellow-400" />
    case 2:
      return <Medal className="w-5 h-5 text-gray-300" />
    case 3:
      return <Medal className="w-5 h-5 text-amber-600" />
    default:
      return null
  }
}

function getRankColor(rank: number) {
  switch (rank) {
    case 1:
      return "bg-yellow-500/10 border-yellow-500/30"
    case 2:
      return "bg-gray-400/10 border-gray-400/30"
    case 3:
      return "bg-amber-600/10 border-amber-600/30"
    default:
      return "bg-card/50 border-border"
  }
}

function LeaderboardList({ entries, type }: { entries: LeaderboardEntry[]; type: "exoplanets" | "false-positives" }) {
  return (
    <div className="space-y-3">
      {entries.map((entry) => (
        <Card
          key={entry.rank}
          className={cn(
            "p-4 backdrop-blur transition-all hover:scale-[1.02]",
            getRankColor(entry.rank),
            entry.rank <= 3 && "shadow-lg",
          )}
        >
          <div className="flex items-center gap-4">
            {/* Rank */}
            <div className="flex items-center justify-center w-12 h-12 rounded-full bg-secondary/50 flex-shrink-0">
              {getRankIcon(entry.rank) || (
                <span className="text-lg font-bold text-muted-foreground">#{entry.rank}</span>
              )}
            </div>

            {/* Avatar */}
            <div
              className="w-10 h-10 rounded-full flex-shrink-0"
              style={{
                background: `linear-gradient(135deg, ${getColorFromString(entry.username)}, ${getColorFromString(entry.username + "2")})`,
              }}
            />

            {/* Username */}
            <div className="flex-1 min-w-0">
              <p className="font-semibold text-card-foreground truncate">{entry.username}</p>
              <p className="text-sm text-muted-foreground">
                {type === "exoplanets" ? "Exoplanets Discovered" : "False Positives Identified"}
              </p>
            </div>

            {/* Count */}
            <div className="text-right flex-shrink-0">
              <p className="text-2xl font-bold text-primary">{entry.count}</p>
              <p className="text-xs text-muted-foreground">{type === "exoplanets" ? "planets" : "identified"}</p>
            </div>
          </div>
        </Card>
      ))}
    </div>
  )
}

function getColorFromString(str: string): string {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash)
  }
  const hue = hash % 360
  return `hsl(${hue}, 60%, 50%)`
}

export function LeaderboardTabs() {
  const [activeTab, setActiveTab] = useState("exoplanets")

  return (
    <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
      <TabsList className="grid w-full grid-cols-2 mb-8 bg-card/50 backdrop-blur">
        <TabsTrigger value="exoplanets" className="gap-2">
          <Trophy className="w-4 h-4" />
          <span className="hidden sm:inline">Most Exoplanets Discovered</span>
          <span className="sm:hidden">Exoplanets</span>
        </TabsTrigger>
        <TabsTrigger value="false-positives" className="gap-2">
          <Award className="w-4 h-4" />
          <span className="hidden sm:inline">Most False-Positives Discovered</span>
          <span className="sm:hidden">False-Positives</span>
        </TabsTrigger>
      </TabsList>

      <TabsContent value="exoplanets" className="mt-0">
        <div className="mb-6">
          <Card className="p-6 bg-primary/10 backdrop-blur border-primary/20">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-full bg-primary/20 flex items-center justify-center">
                <Trophy className="w-6 h-6 text-primary" />
              </div>
              <div>
                <h3 className="font-semibold text-card-foreground">Top Exoplanet Discoverers</h3>
                <p className="text-sm text-muted-foreground">
                  Explorers who have confirmed the most exoplanets through transit detection
                </p>
              </div>
            </div>
          </Card>
        </div>
        <LeaderboardList entries={exoplanetLeaders} type="exoplanets" />
      </TabsContent>

      <TabsContent value="false-positives" className="mt-0">
        <div className="mb-6">
          <Card className="p-6 bg-accent/10 backdrop-blur border-accent/20">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-full bg-accent/20 flex items-center justify-center">
                <Award className="w-6 h-6 text-accent" />
              </div>
              <div>
                <h3 className="font-semibold text-card-foreground">Top False-Positive Identifiers</h3>
                <p className="text-sm text-muted-foreground">
                  Experts who have identified the most false positives, ensuring data accuracy
                </p>
              </div>
            </div>
          </Card>
        </div>
        <LeaderboardList entries={falsePositiveLeaders} type="false-positives" />
      </TabsContent>
    </Tabs>
  )
}
