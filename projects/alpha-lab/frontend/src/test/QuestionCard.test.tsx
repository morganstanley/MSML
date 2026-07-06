import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import QuestionCard from "../QuestionCard";

describe("QuestionCard", () => {
  let onAnswer: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    onAnswer = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the question text", () => {
    render(
      <QuestionCard
        questionId="q1"
        question="What is the air-speed velocity?"
        onAnswer={onAnswer}
      />,
    );
    expect(
      screen.getByText("What is the air-speed velocity?"),
    ).toBeInTheDocument();
  });

  it("calls onAnswer with trimmed text when Send is clicked", async () => {
    const user = userEvent.setup();
    render(
      <QuestionCard questionId="q1" question="Hello?" onAnswer={onAnswer} />,
    );
    await user.type(screen.getByRole("textbox"), "  my answer  ");
    await user.click(screen.getByRole("button", { name: /send/i }));
    expect(onAnswer).toHaveBeenCalledWith("my answer");
  });

  it("calls onAnswer when Enter is pressed", async () => {
    const user = userEvent.setup();
    render(
      <QuestionCard questionId="q1" question="Hello?" onAnswer={onAnswer} />,
    );
    await user.type(screen.getByRole("textbox"), "keyboard answer{Enter}");
    expect(onAnswer).toHaveBeenCalledWith("keyboard answer");
  });

  it("does not submit when the answer is blank", async () => {
    const user = userEvent.setup();
    render(
      <QuestionCard questionId="q1" question="Hello?" onAnswer={onAnswer} />,
    );
    await user.click(screen.getByRole("button", { name: /send/i }));
    expect(onAnswer).not.toHaveBeenCalled();
  });

  it("disables input and button after submission", async () => {
    const user = userEvent.setup();
    render(
      <QuestionCard questionId="q1" question="Hello?" onAnswer={onAnswer} />,
    );
    await user.type(screen.getByRole("textbox"), "answer");
    await user.click(screen.getByRole("button", { name: /send/i }));
    expect(screen.getByRole("textbox")).toBeDisabled();
    expect(screen.getByRole("button", { name: /sent/i })).toBeDisabled();
  });
});
