import { SpaceBackground } from "@/components/space-background"
import { Navigation } from "@/components/navigation"
import { LeaderboardTabs } from "@/components/leaderboard-tabs"

export default function LeaderboardPage() {
  return (
    <div className="min-h-screen relative">
      <SpaceBackground />
      <Navigation />

      <main className="relative z-10 pt-24 pb-16">
        <div className="container mx-auto px-4 max-w-4xl">
          <div className="mb-12 text-center">
            <h1 className="text-4xl md:text-5xl font-bold mb-4 text-foreground">Leaderboard</h1>
            <p className="text-lg text-muted-foreground text-pretty leading-relaxed">
              Top explorers who have made the most discoveries and identified the most false positives.
            </p>
          </div>

          <LeaderboardTabs />
        </div>
      </main>
    </div>
  )
}
