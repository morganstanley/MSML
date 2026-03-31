import { useState } from "react";

interface QuestionCardProps {
  questionId: string;
  question: string;
  onAnswer: (text: string) => void;
}

export default function QuestionCard({ questionId, question, onAnswer }: QuestionCardProps) {
  const [answer, setAnswer] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = () => {
    if (!answer.trim()) return;
    setSubmitted(true);
    onAnswer(answer.trim());
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="question-card">
      <div className="question-text">{question}</div>
      <div className="question-input">
        <input
          type="text"
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your answer..."
          disabled={submitted}
        />
        <button className="btn btn-primary" onClick={handleSubmit} disabled={submitted}>
          {submitted ? "Sent" : "Send"}
        </button>
      </div>
    </div>
  );
}
