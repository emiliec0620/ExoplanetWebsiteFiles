"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Sparkles, CheckCircle2, XCircle, RotateCw } from "lucide-react"
import { cn } from "@/lib/utils"

interface QuizQuestion {
  question: string
  options: string[]
  correctAnswer: number
  explanation: string
}

const sampleQuizzes: QuizQuestion[][] = [
  [
    {
      question: "What causes the dip in brightness during a planetary transit?",
      options: [
        "The planet emits light",
        "The planet blocks some of the star's light",
        "The star moves away from us",
        "The planet reflects light back to us",
      ],
      correctAnswer: 1,
      explanation:
        "When a planet passes in front of its star from our perspective, it blocks a small portion of the star's light, causing a temporary dip in brightness.",
    },
    {
      question: "Which factor determines the depth of a transit in a light curve?",
      options: [
        "The planet's temperature",
        "The planet's orbital speed",
        "The planet's size relative to the star",
        "The planet's distance from Earth",
      ],
      correctAnswer: 2,
      explanation:
        "The transit depth depends on how much of the star's light the planet blocks, which is determined by the planet's size relative to the star. Larger planets create deeper transits.",
    },
    {
      question: "What is a common false positive in exoplanet detection?",
      options: ["Planetary rings", "Eclipsing binary star systems", "Asteroid belts", "Solar flares"],
      correctAnswer: 1,
      explanation:
        "Eclipsing binary star systems can mimic planet transits. When two stars orbit each other and one passes in front of the other, it creates a dip in brightness similar to a planetary transit.",
    },
  ],
  [
    {
      question: "Why is observing multiple transits important for confirming an exoplanet?",
      options: [
        "To make the discovery more exciting",
        "To verify the periodic nature and consistency of the signal",
        "To measure the planet's color",
        "To count how many moons it has",
      ],
      correctAnswer: 1,
      explanation:
        "Multiple transits allow us to confirm that the signal is periodic and consistent, which is characteristic of a real planet. This helps rule out one-time events or instrumental errors.",
    },
    {
      question: "What does the orbital period of an exoplanet tell us?",
      options: [
        "The planet's color",
        "How long it takes to orbit its star",
        "The planet's composition",
        "The number of moons it has",
      ],
      correctAnswer: 1,
      explanation:
        "The orbital period is the time it takes for a planet to complete one full orbit around its star. This is measured by timing the intervals between successive transits.",
    },
    {
      question: "Which characteristic would suggest a transit signal is NOT a planet?",
      options: [
        "Consistent transit depth",
        "Regular periodic timing",
        "V-shaped transit instead of U-shaped",
        "Multiple transits observed",
      ],
      correctAnswer: 2,
      explanation:
        "A V-shaped transit suggests an eclipsing binary star system rather than a planet. Real planetary transits typically show U-shaped or flat-bottomed dips because the planet is much smaller than the star.",
    },
  ],
  [
    {
      question: "What is the habitable zone?",
      options: [
        "The region where humans can live",
        "The distance from a star where liquid water could exist",
        "The area inside a planet's atmosphere",
        "The space between two planets",
      ],
      correctAnswer: 1,
      explanation:
        "The habitable zone is the range of distances from a star where temperatures allow liquid water to exist on a planet's surface. This is considered crucial for life as we know it.",
    },
    {
      question: "How can we detect a secondary eclipse?",
      options: [
        "When the planet passes behind the star",
        "When the planet reflects sunlight",
        "When the star explodes",
        "When two planets collide",
      ],
      correctAnswer: 0,
      explanation:
        "A secondary eclipse occurs when the planet passes behind the star from our viewpoint. This causes a smaller dip in brightness and helps confirm the planet's existence while providing data about its temperature.",
    },
    {
      question: "What does a high signal-to-noise ratio indicate?",
      options: [
        "The planet is very noisy",
        "The detection is more reliable and clear",
        "The star is unstable",
        "The data is corrupted",
      ],
      correctAnswer: 1,
      explanation:
        "A high signal-to-noise ratio means the transit signal stands out clearly from background noise and random variations. This makes the detection more reliable and easier to confirm.",
    },
  ],
]

export function QuizGenerator() {
  const [currentQuiz, setCurrentQuiz] = useState<QuizQuestion[] | null>(null)
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null)
  const [showExplanation, setShowExplanation] = useState(false)
  const [usedQuizzes, setUsedQuizzes] = useState<number[]>([])

  const generateQuiz = () => {
    // Find unused quizzes
    const availableQuizzes = sampleQuizzes.map((_, index) => index).filter((index) => !usedQuizzes.includes(index))

    // If all quizzes used, reset
    if (availableQuizzes.length === 0) {
      setUsedQuizzes([])
      const randomIndex = Math.floor(Math.random() * sampleQuizzes.length)
      setCurrentQuiz(sampleQuizzes[randomIndex])
      setUsedQuizzes([randomIndex])
    } else {
      const randomIndex = availableQuizzes[Math.floor(Math.random() * availableQuizzes.length)]
      setCurrentQuiz(sampleQuizzes[randomIndex])
      setUsedQuizzes([...usedQuizzes, randomIndex])
    }

    setCurrentQuestionIndex(0)
    setSelectedAnswer(null)
    setShowExplanation(false)
  }

  const handleAnswerSelect = (answerIndex: number) => {
    if (showExplanation) return
    setSelectedAnswer(answerIndex)
    setShowExplanation(true)
  }

  const handleNextQuestion = () => {
    if (currentQuiz && currentQuestionIndex < currentQuiz.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1)
      setSelectedAnswer(null)
      setShowExplanation(false)
    }
  }

  if (!currentQuiz) {
    return (
      <Card className="p-8 bg-card/50 backdrop-blur border-border text-center">
        <div className="max-w-md mx-auto">
          <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-4">
            <Sparkles className="w-8 h-8 text-primary" />
          </div>
          <h2 className="text-2xl font-bold mb-3 text-card-foreground">Test Your Knowledge</h2>
          <p className="text-muted-foreground mb-6 leading-relaxed">
            Ready to put your exoplanet detection skills to the test? Generate a random quiz with multiple-choice
            questions.
          </p>
          <Button onClick={generateQuiz} size="lg" className="gap-2">
            <Sparkles className="w-4 h-4" />
            Generate Quiz
          </Button>
        </div>
      </Card>
    )
  }

  const currentQuestion = currentQuiz[currentQuestionIndex]
  const isCorrect = selectedAnswer === currentQuestion.correctAnswer

  return (
    <Card className="p-8 bg-card/50 backdrop-blur border-border">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold text-card-foreground">Practice Quiz</h2>
          <span className="text-sm text-muted-foreground">
            Question {currentQuestionIndex + 1} of {currentQuiz.length}
          </span>
        </div>
        <div className="w-full bg-secondary rounded-full h-2">
          <div
            className="bg-primary h-2 rounded-full transition-all duration-300"
            style={{
              width: `${((currentQuestionIndex + 1) / currentQuiz.length) * 100}%`,
            }}
          />
        </div>
      </div>

      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-4 text-card-foreground leading-relaxed">{currentQuestion.question}</h3>

        <div className="space-y-3">
          {currentQuestion.options.map((option, index) => {
            const isSelected = selectedAnswer === index
            const isCorrectAnswer = index === currentQuestion.correctAnswer
            const showCorrect = showExplanation && isCorrectAnswer
            const showIncorrect = showExplanation && isSelected && !isCorrect

            return (
              <button
                key={index}
                onClick={() => handleAnswerSelect(index)}
                disabled={showExplanation}
                className={cn(
                  "w-full p-4 rounded-lg border-2 text-left transition-all",
                  "hover:border-primary/50 disabled:cursor-not-allowed",
                  !showExplanation && "hover:bg-secondary/50",
                  isSelected && !showExplanation && "border-primary bg-primary/10",
                  showCorrect && "border-success bg-success/10",
                  showIncorrect && "border-destructive bg-destructive/10",
                  !isSelected && !showExplanation && "border-border",
                )}
              >
                <div className="flex items-center justify-between">
                  <span className="text-card-foreground">{option}</span>
                  {showCorrect && <CheckCircle2 className="w-5 h-5 text-success flex-shrink-0" />}
                  {showIncorrect && <XCircle className="w-5 h-5 text-destructive flex-shrink-0" />}
                </div>
              </button>
            )
          })}
        </div>
      </div>

      {showExplanation && (
        <div
          className={cn(
            "p-4 rounded-lg mb-6",
            isCorrect ? "bg-success/10 border border-success/20" : "bg-destructive/10 border border-destructive/20",
          )}
        >
          <div className="flex items-start gap-3">
            {isCorrect ? (
              <CheckCircle2 className="w-5 h-5 text-success flex-shrink-0 mt-0.5" />
            ) : (
              <XCircle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
            )}
            <div>
              <p className={cn("font-semibold mb-1", isCorrect ? "text-success" : "text-destructive")}>
                {isCorrect ? "Correct!" : "Incorrect"}
              </p>
              <p className="text-muted-foreground leading-relaxed">{currentQuestion.explanation}</p>
            </div>
          </div>
        </div>
      )}

      <div className="flex gap-3">
        {showExplanation && currentQuestionIndex < currentQuiz.length - 1 && (
          <Button onClick={handleNextQuestion} className="flex-1">
            Next Question
          </Button>
        )}
        {showExplanation && currentQuestionIndex === currentQuiz.length - 1 && (
          <Button onClick={generateQuiz} className="flex-1 gap-2">
            <RotateCw className="w-4 h-4" />
            Generate New Quiz
          </Button>
        )}
      </div>
    </Card>
  )
}
