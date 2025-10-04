"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Share2, Thermometer, Ruler, Clock } from "lucide-react"
import { cn } from "@/lib/utils"

type Disposition = "confirmed" | "candidate" | "false-positive"

interface Planet {
  id: string
  name: string
  disposition: Disposition
  starName: string
  radius: number // Earth radii
  orbitalPeriod?: number // days
  temperature?: number // Kelvin
  color: string
  description: string
}

const samplePlanets: Planet[] = [
  {
    id: "1",
    name: "Kepler-442b",
    disposition: "confirmed",
    starName: "Kepler-442",
    radius: 1.34,
    orbitalPeriod: 112.3,
    temperature: 233,
    color: "#4a90e2",
    description: "A potentially habitable super-Earth in the habitable zone of its star.",
  },
  {
    id: "2",
    name: "TOI-1452b",
    disposition: "confirmed",
    starName: "TOI-1452",
    radius: 1.67,
    orbitalPeriod: 11.1,
    temperature: 326,
    color: "#5ba3d0",
    description: "An ocean world candidate with a thick water layer.",
  },
  {
    id: "3",
    name: "KOI-5123",
    disposition: "candidate",
    starName: "KOI-5123",
    radius: 2.1,
    temperature: 415,
    color: "#f4a261",
    description: "A Neptune-sized candidate awaiting confirmation through additional observations.",
  },
  {
    id: "4",
    name: "KOI-8012",
    disposition: "candidate",
    starName: "KOI-8012",
    radius: 0.89,
    temperature: 278,
    color: "#e76f51",
    description: "An Earth-sized candidate in the habitable zone requiring further validation.",
  },
  {
    id: "5",
    name: "KOI-3421",
    disposition: "false-positive",
    starName: "KOI-3421",
    radius: 3.2,
    color: "#c44536",
    description: "Identified as an eclipsing binary star system, not a planetary transit.",
  },
  {
    id: "6",
    name: "TRAPPIST-1e",
    disposition: "confirmed",
    starName: "TRAPPIST-1",
    radius: 0.92,
    orbitalPeriod: 6.1,
    temperature: 251,
    color: "#2a9d8f",
    description: "One of seven Earth-sized planets orbiting an ultra-cool dwarf star.",
  },
  {
    id: "7",
    name: "KOI-7629",
    disposition: "candidate",
    starName: "KOI-7629",
    radius: 1.45,
    temperature: 302,
    color: "#e9c46a",
    description: "A super-Earth candidate with promising habitability indicators.",
  },
  {
    id: "8",
    name: "KOI-9156",
    disposition: "false-positive",
    starName: "KOI-9156",
    radius: 4.1,
    color: "#d62828",
    description: "Background eclipsing binary causing false transit signal.",
  },
]

export function PlanetExplorer() {
  const [selectedPlanet, setSelectedPlanet] = useState<Planet | null>(null)
  const [viewMode, setViewMode] = useState<"overview" | "size" | "orbit" | "temperature">("overview")

  const getDispositionColor = (disposition: Disposition) => {
    switch (disposition) {
      case "confirmed":
        return "bg-success text-success-foreground"
      case "candidate":
        return "bg-warning text-warning-foreground"
      case "false-positive":
        return "bg-destructive text-destructive-foreground"
    }
  }

  const getDispositionLabel = (disposition: Disposition) => {
    switch (disposition) {
      case "confirmed":
        return "Confirmed"
      case "candidate":
        return "Candidate"
      case "false-positive":
        return "False Positive"
    }
  }

  const handleShare = (planet: Planet) => {
    alert(`Shared discovery: ${planet.name} - ${getDispositionLabel(planet.disposition)} exoplanet!`)
  }

  if (selectedPlanet && viewMode !== "overview") {
    return (
      <div className="max-w-6xl mx-auto">
        <Card className="bg-card/50 backdrop-blur border-border overflow-hidden">
          <div className="relative h-[500px] bg-gradient-to-b from-background to-secondary/30 flex items-center justify-center">
            <Button
              onClick={() => setViewMode("overview")}
              variant="outline"
              size="sm"
              className="absolute top-4 left-4 gap-2"
            >
              <ArrowLeft className="w-4 h-4" />
              Back
            </Button>

            {viewMode === "size" && (
              <div className="text-center">
                <div
                  className="mx-auto rounded-full mb-8 relative"
                  style={{
                    width: `${Math.min(selectedPlanet.radius * 100, 300)}px`,
                    height: `${Math.min(selectedPlanet.radius * 100, 300)}px`,
                    backgroundColor: selectedPlanet.color,
                    boxShadow: `0 0 60px ${selectedPlanet.color}80`,
                  }}
                >
                  <div className="absolute -bottom-12 left-1/2 -translate-x-1/2 w-full">
                    <div className="h-1 bg-primary relative">
                      <div className="absolute -left-2 -top-2 w-1 h-5 bg-primary" />
                      <div className="absolute -right-2 -top-2 w-1 h-5 bg-primary" />
                    </div>
                  </div>
                </div>
                <div className="mt-16">
                  <p className="text-2xl font-bold text-foreground mb-2">
                    {selectedPlanet.radius.toFixed(2)} Earth Radii
                  </p>
                  <p className="text-muted-foreground">Diameter: {(selectedPlanet.radius * 2 * 6371).toFixed(0)} km</p>
                </div>
              </div>
            )}

            {viewMode === "orbit" && (
              <div className="text-center">
                <div className="relative w-[400px] h-[400px] mx-auto mb-8">
                  {/* Star */}
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-16 rounded-full bg-yellow-400 animate-pulse" />

                  {/* Orbit path */}
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[300px] h-[300px] border-2 border-dashed border-primary/30 rounded-full" />

                  {/* Planet */}
                  <div
                    className="absolute top-1/2 left-1/2 rounded-full animate-spin"
                    style={{
                      width: `${Math.min(selectedPlanet.radius * 20, 40)}px`,
                      height: `${Math.min(selectedPlanet.radius * 20, 40)}px`,
                      backgroundColor: selectedPlanet.color,
                      marginLeft: "150px",
                      marginTop: `-${Math.min(selectedPlanet.radius * 10, 20)}px`,
                      animationDuration: `${Math.min(selectedPlanet.orbitalPeriod! / 10, 8)}s`,
                    }}
                  />
                </div>
                <p className="text-2xl font-bold text-foreground mb-2">
                  Orbital Period: {selectedPlanet.orbitalPeriod?.toFixed(1)} days
                </p>
                <p className="text-muted-foreground">One complete orbit around {selectedPlanet.starName}</p>
              </div>
            )}

            {viewMode === "temperature" && (
              <div className="text-center">
                <div
                  className="w-64 h-64 rounded-full mx-auto mb-8 relative flex items-center justify-center"
                  style={{
                    backgroundColor: selectedPlanet.color,
                    boxShadow: `0 0 80px ${selectedPlanet.color}80`,
                  }}
                >
                  {/* Temperature arrows animation */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="space-y-4">
                      <div className="flex gap-4 animate-pulse">
                        <div className="w-12 h-1 bg-red-500 rounded" />
                        <div className="w-12 h-1 bg-blue-500 rounded" />
                      </div>
                      <div className="flex gap-4 animate-pulse" style={{ animationDelay: "0.5s" }}>
                        <div className="w-12 h-1 bg-red-500 rounded" />
                        <div className="w-12 h-1 bg-blue-500 rounded" />
                      </div>
                    </div>
                  </div>
                </div>
                <p className="text-2xl font-bold text-foreground mb-2">
                  {selectedPlanet.temperature}K ({(selectedPlanet.temperature! - 273).toFixed(0)}°C)
                </p>
                <p className="text-muted-foreground">Equilibrium temperature at the surface</p>
              </div>
            )}
          </div>
        </Card>
      </div>
    )
  }

  if (selectedPlanet) {
    return (
      <div className="max-w-6xl mx-auto">
        <Button onClick={() => setSelectedPlanet(null)} variant="outline" size="sm" className="mb-4 gap-2">
          <ArrowLeft className="w-4 h-4" />
          Back to All Planets
        </Button>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Planet Visualization */}
          <Card className="bg-card/50 backdrop-blur border-border overflow-hidden">
            <div className="relative h-[400px] bg-gradient-to-b from-background to-secondary/30 flex items-center justify-center">
              <div
                className="w-48 h-48 rounded-full animate-pulse"
                style={{
                  backgroundColor: selectedPlanet.color,
                  boxShadow: `0 0 100px ${selectedPlanet.color}80`,
                }}
              />
            </div>
          </Card>

          {/* Planet Details */}
          <div className="space-y-4">
            <Card className="p-6 bg-card/50 backdrop-blur border-border">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h2 className="text-3xl font-bold text-card-foreground mb-2">{selectedPlanet.name}</h2>
                  <p className="text-muted-foreground">Orbiting {selectedPlanet.starName}</p>
                </div>
                <Badge className={cn("text-sm", getDispositionColor(selectedPlanet.disposition))}>
                  {getDispositionLabel(selectedPlanet.disposition)}
                </Badge>
              </div>

              <p className="text-muted-foreground leading-relaxed mb-6">{selectedPlanet.description}</p>

              <Button onClick={() => handleShare(selectedPlanet)} variant="outline" className="w-full gap-2">
                <Share2 className="w-4 h-4" />
                Share Discovery
              </Button>
            </Card>

            {/* Property Cards */}
            <div className="space-y-3">
              <Card
                className="p-4 bg-card/50 backdrop-blur border-border hover:border-primary/50 transition-colors cursor-pointer"
                onClick={() => setViewMode("size")}
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                    <Ruler className="w-5 h-5 text-primary" />
                  </div>
                  <div className="flex-1">
                    <p className="text-sm text-muted-foreground">Size</p>
                    <p className="font-semibold text-card-foreground">{selectedPlanet.radius.toFixed(2)} Earth Radii</p>
                  </div>
                  <ArrowLeft className="w-4 h-4 text-muted-foreground rotate-180" />
                </div>
              </Card>

              {selectedPlanet.orbitalPeriod && (
                <Card
                  className="p-4 bg-card/50 backdrop-blur border-border hover:border-primary/50 transition-colors cursor-pointer"
                  onClick={() => setViewMode("orbit")}
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center">
                      <Clock className="w-5 h-5 text-accent" />
                    </div>
                    <div className="flex-1">
                      <p className="text-sm text-muted-foreground">Orbital Period</p>
                      <p className="font-semibold text-card-foreground">
                        {selectedPlanet.orbitalPeriod.toFixed(1)} days
                      </p>
                    </div>
                    <ArrowLeft className="w-4 h-4 text-muted-foreground rotate-180" />
                  </div>
                </Card>
              )}

              {selectedPlanet.temperature && (
                <Card
                  className="p-4 bg-card/50 backdrop-blur border-border hover:border-primary/50 transition-colors cursor-pointer"
                  onClick={() => setViewMode("temperature")}
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-warning/10 flex items-center justify-center">
                      <Thermometer className="w-5 h-5 text-warning" />
                    </div>
                    <div className="flex-1">
                      <p className="text-sm text-muted-foreground">Equilibrium Temperature</p>
                      <p className="font-semibold text-card-foreground">
                        {selectedPlanet.temperature}K ({(selectedPlanet.temperature - 273).toFixed(0)}°C)
                      </p>
                    </div>
                    <ArrowLeft className="w-4 h-4 text-muted-foreground rotate-180" />
                  </div>
                </Card>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {samplePlanets.map((planet) => (
          <Card
            key={planet.id}
            className="bg-card/50 backdrop-blur border-border hover:border-primary/50 transition-all cursor-pointer group overflow-hidden"
            onClick={() => setSelectedPlanet(planet)}
          >
            <div className="relative h-48 bg-gradient-to-b from-background to-secondary/30 flex items-center justify-center overflow-hidden">
              <div
                className="w-24 h-24 rounded-full group-hover:scale-110 transition-transform"
                style={{
                  backgroundColor: planet.color,
                  boxShadow: `0 0 40px ${planet.color}60`,
                }}
              />
            </div>
            <div className="p-4">
              <div className="flex items-start justify-between mb-2">
                <h3 className="font-semibold text-card-foreground">{planet.name}</h3>
                <Badge className={cn("text-xs", getDispositionColor(planet.disposition))}>
                  {getDispositionLabel(planet.disposition)}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground mb-3">{planet.starName}</p>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Ruler className="w-3 h-3" />
                <span>{planet.radius.toFixed(2)} R⊕</span>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}
