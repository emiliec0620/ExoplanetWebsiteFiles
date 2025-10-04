import { SpaceBackground } from "@/components/space-background"
import { Navigation } from "@/components/navigation"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { ArrowRight, Sparkles, Target, Users } from "lucide-react"
import Link from "next/link"

export default function HomePage() {
  return (
    <div className="min-h-screen relative">
      <SpaceBackground />
      <Navigation />

      <main className="relative z-10 pt-24 pb-16">
        <div className="container mx-auto px-4">
          {/* Hero Section */}
          <div className="max-w-4xl mx-auto text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 mb-6">
              <Sparkles className="w-4 h-4 text-primary" />
              <span className="text-sm text-primary font-medium">Interactive Exoplanet Detection</span>
            </div>

            <h1 className="text-5xl md:text-6xl font-bold mb-6 text-balance">
              Discover the Universe, <span className="text-primary">One Planet at a Time</span>
            </h1>

            <p className="text-xl text-muted-foreground mb-8 text-pretty leading-relaxed">
              Learn the fundamentals of exoplanet detection through transit methods. Master the art of distinguishing
              between false positives and confirmed exoplanets with interactive simulations and real-world data.
            </p>

            <div className="flex flex-wrap items-center justify-center gap-4">
              <Button asChild size="lg" className="gap-2">
                <Link href="/learn">
                  Start Learning
                  <ArrowRight className="w-4 h-4" />
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg">
                <Link href="/planets">Explore Planets</Link>
              </Button>
            </div>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-6 mb-16">
            <Card className="p-6 bg-card/50 backdrop-blur border-border hover:border-primary/50 transition-colors">
              <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                <Target className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2 text-card-foreground">Interactive Learning</h3>
              <p className="text-muted-foreground leading-relaxed">
                Explore 10 key features of exoplanet detection with detailed explanations, diagrams, and AI-generated
                content.
              </p>
            </Card>

            <Card className="p-6 bg-card/50 backdrop-blur border-border hover:border-primary/50 transition-colors">
              <div className="w-12 h-12 rounded-lg bg-accent/10 flex items-center justify-center mb-4">
                <Sparkles className="w-6 h-6 text-accent" />
              </div>
              <h3 className="text-lg font-semibold mb-2 text-card-foreground">Planet Simulations</h3>
              <p className="text-muted-foreground leading-relaxed">
                Discover and explore exoplanets in an immersive 3D environment. Analyze their properties and confirm
                discoveries.
              </p>
            </Card>

            <Card className="p-6 bg-card/50 backdrop-blur border-border hover:border-primary/50 transition-colors">
              <div className="w-12 h-12 rounded-lg bg-success/10 flex items-center justify-center mb-4">
                <Users className="w-6 h-6 text-success" />
              </div>
              <h3 className="text-lg font-semibold mb-2 text-card-foreground">Compete & Share</h3>
              <p className="text-muted-foreground leading-relaxed">
                Track your discoveries on the leaderboard and share your findings with the community.
              </p>
            </Card>
          </div>

          {/* Demo Slideshow Section */}
          <div className="max-w-5xl mx-auto">
            <h2 className="text-3xl font-bold text-center mb-8 text-foreground">See It In Action</h2>
            <Card className="p-8 bg-card/50 backdrop-blur border-border">
              <div className="grid md:grid-cols-3 gap-4">
                <div className="aspect-video rounded-lg bg-secondary/50 flex items-center justify-center overflow-hidden">
                  <img
                    src="/exoplanet-transit-light-curve.png"
                    alt="Transit detection visualization"
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="aspect-video rounded-lg bg-secondary/50 flex items-center justify-center overflow-hidden">
                  <img
                    src="/3d-planet-exploration-interface.jpg"
                    alt="Planet exploration interface"
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="aspect-video rounded-lg bg-secondary/50 flex items-center justify-center overflow-hidden">
                  <img
                    src="/educational-quiz-interface-space-theme.jpg"
                    alt="Interactive quiz system"
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
              <div className="mt-6 text-center">
                <p className="text-muted-foreground">
                  Interactive simulations, real-time data analysis, and engaging quizzes to test your knowledge
                </p>
              </div>
            </Card>
          </div>
        </div>
      </main>
    </div>
  )
}
