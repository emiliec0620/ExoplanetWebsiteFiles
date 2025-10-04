import { SpaceBackground } from "@/components/space-background"
import { Navigation } from "@/components/navigation"
import { PlanetExplorer } from "@/components/planet-explorer"

export default function PlanetsPage() {
  return (
    <div className="min-h-screen relative">
      <SpaceBackground />
      <Navigation />

      <main className="relative z-10 pt-24 pb-16">
        <div className="container mx-auto px-4">
          <div className="mb-8 text-center">
            <h1 className="text-4xl md:text-5xl font-bold mb-4 text-foreground">My Planets</h1>
            <p className="text-lg text-muted-foreground text-pretty leading-relaxed">
              Explore discovered exoplanets and analyze their properties. Click on any planet to dive deeper.
            </p>
          </div>

          <PlanetExplorer />
        </div>
      </main>
    </div>
  )
}
