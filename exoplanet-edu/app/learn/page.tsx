import { SpaceBackground } from "@/components/space-background"
import { Navigation } from "@/components/navigation"
import { LearningContent } from "@/components/learning-content"
import { QuizGenerator } from "@/components/quiz-generator"

export default function LearnPage() {
  return (
    <div className="min-h-screen relative">
      <SpaceBackground />
      <Navigation />

      <main className="relative z-10 pt-24 pb-16">
        <div className="container mx-auto px-4 max-w-4xl">
          <div className="mb-12 text-center">
            <h1 className="text-4xl md:text-5xl font-bold mb-4 text-foreground">Learn Exoplanet Detection</h1>
            <p className="text-lg text-muted-foreground text-pretty leading-relaxed">
              Master the fundamentals of detecting exoplanets through the transit method. Explore key features and test
              your knowledge with interactive quizzes.
            </p>
          </div>

          <LearningContent />
          <QuizGenerator />
        </div>
      </main>
    </div>
  )
}
