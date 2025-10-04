"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { ChevronDown, ChevronUp } from "lucide-react"
import { cn } from "@/lib/utils"

const learningTopics = [
  {
    id: 1,
    title: "Transit Method Basics",
    description:
      "The transit method detects exoplanets by measuring the dimming of a star's light when a planet passes in front of it. This is one of the most successful techniques for discovering exoplanets.",
    importance: "Critical",
    usage:
      "Primary method for exoplanet detection. Used by missions like Kepler and TESS to discover thousands of exoplanets.",
    diagram: "/transit-method-diagram-showing-planet-passing-in-f.jpg",
  },
  {
    id: 2,
    title: "Brightness and Light Curves",
    description:
      "A light curve shows how a star's brightness changes over time. During a transit, the brightness dips slightly. The depth and shape of this dip reveal information about the planet.",
    importance: "Critical",
    usage:
      "Essential for confirming exoplanets. The light curve pattern helps distinguish real planets from false positives.",
    diagram: "/exoplanet-light-curve-graph-showing-brightness-dip.jpg",
  },
  {
    id: 3,
    title: "Planet Size and Radius",
    description:
      "The depth of the transit (how much the star dims) tells us the size of the planet. Larger planets block more light, creating deeper dips in the light curve.",
    importance: "Very Important",
    usage:
      "Determines if the planet is Earth-sized, Neptune-sized, or Jupiter-sized. Critical for understanding planet composition.",
    diagram: "/planet-size-comparison-showing-different-transit-d.jpg",
  },
  {
    id: 4,
    title: "Transit Duration",
    description:
      "How long the planet takes to cross in front of the star. This depends on the planet's orbital speed and the star's size.",
    importance: "Important",
    usage:
      "Helps calculate orbital parameters and validate the detection. Unusually short or long transits may indicate false positives.",
    diagram: "/transit-duration-diagram-showing-planet-crossing-s.jpg",
  },
  {
    id: 5,
    title: "Orbital Period",
    description:
      "The time it takes for a planet to complete one orbit around its star. Detected by observing multiple transits and measuring the time between them.",
    importance: "Critical",
    usage:
      "Determines the planet's distance from its star and helps calculate the habitable zone. Essential for confirming periodic behavior.",
    diagram: "/orbital-period-diagram-showing-planet-orbit-around.jpg",
  },
  {
    id: 6,
    title: "Transit Depth Consistency",
    description:
      "Real planets produce consistent transit depths across multiple observations. Variations may indicate eclipsing binary stars or other false positives.",
    importance: "Very Important",
    usage:
      "Key method for distinguishing confirmed exoplanets from false positives. Inconsistent depths are red flags.",
    diagram: "/consistent-vs-inconsistent-transit-depths-comparis.jpg",
  },
  {
    id: 7,
    title: "Secondary Eclipse",
    description:
      "When the planet passes behind the star, we can detect a smaller dip. This confirms the planet's existence and helps measure its temperature.",
    importance: "Important",
    usage:
      "Provides additional confirmation and allows measurement of the planet's thermal emission and atmospheric properties.",
    diagram: "/secondary-eclipse-diagram-showing-planet-behind-st.jpg",
  },
  {
    id: 8,
    title: "Transit Timing Variations",
    description:
      "Small changes in when transits occur can reveal additional planets in the system through gravitational interactions.",
    importance: "Moderate",
    usage:
      "Helps discover additional planets and refine orbital parameters. Useful for detecting multi-planet systems.",
    diagram: "/transit-timing-variations-showing-gravitational-ef.jpg",
  },
  {
    id: 9,
    title: "False Positive Identification",
    description:
      "Not all transit signals are planets. Eclipsing binary stars, background stars, and instrumental artifacts can mimic planet transits.",
    importance: "Critical",
    usage:
      "Essential skill for validating discoveries. Requires checking for V-shaped transits, odd-even transit differences, and centroid shifts.",
    diagram: "/false-positive-examples-showing-eclipsing-binaries.jpg",
  },
  {
    id: 10,
    title: "Signal-to-Noise Ratio",
    description:
      "The strength of the transit signal compared to background noise. Higher signal-to-noise ratios make detections more reliable.",
    importance: "Very Important",
    usage:
      "Determines detection confidence. Low signal-to-noise may require additional observations to confirm the planet.",
    diagram: "/signal-to-noise-ratio-comparison-in-transit-data.jpg",
  },
]

export function LearningContent() {
  const [openTopics, setOpenTopics] = useState<number[]>([])

  const toggleTopic = (id: number) => {
    setOpenTopics((prev) => (prev.includes(id) ? prev.filter((topicId) => topicId !== id) : [...prev, id]))
  }

  return (
    <div className="space-y-4 mb-12">
      <h2 className="text-2xl font-bold mb-6 text-foreground">Key Concepts in Exoplanet Detection</h2>

      {learningTopics.map((topic) => {
        const isOpen = openTopics.includes(topic.id)
        return (
          <Card key={topic.id} className="bg-card/50 backdrop-blur border-border overflow-hidden">
            <button
              onClick={() => toggleTopic(topic.id)}
              className="w-full p-6 flex items-center justify-between text-left hover:bg-secondary/50 transition-colors"
            >
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-1">
                  <h3 className="text-lg font-semibold text-card-foreground">
                    {topic.id}. {topic.title}
                  </h3>
                  <span
                    className={cn(
                      "text-xs px-2 py-1 rounded-full font-medium",
                      topic.importance === "Critical" && "bg-destructive/20 text-destructive",
                      topic.importance === "Very Important" && "bg-warning/20 text-warning-foreground",
                      topic.importance === "Important" && "bg-primary/20 text-primary",
                      topic.importance === "Moderate" && "bg-muted text-muted-foreground",
                    )}
                  >
                    {topic.importance}
                  </span>
                </div>
              </div>
              {isOpen ? (
                <ChevronUp className="w-5 h-5 text-muted-foreground flex-shrink-0" />
              ) : (
                <ChevronDown className="w-5 h-5 text-muted-foreground flex-shrink-0" />
              )}
            </button>

            {isOpen && (
              <div className="px-6 pb-6 space-y-4 border-t border-border pt-4">
                <div>
                  <h4 className="text-sm font-semibold text-primary mb-2">Description</h4>
                  <p className="text-muted-foreground leading-relaxed">{topic.description}</p>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-primary mb-2">How It's Used</h4>
                  <p className="text-muted-foreground leading-relaxed">{topic.usage}</p>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-primary mb-2">Visual Diagram</h4>
                  <div className="rounded-lg overflow-hidden bg-secondary/30">
                    <img
                      src={topic.diagram || "/placeholder.svg"}
                      alt={`${topic.title} diagram`}
                      className="w-full h-auto"
                    />
                  </div>
                </div>
              </div>
            )}
          </Card>
        )
      })}
    </div>
  )
}
