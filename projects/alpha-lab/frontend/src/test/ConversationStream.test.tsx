import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, afterEach } from "vitest";
import ConversationStream from "../ConversationStream";
import type { AgentEvent } from "../useWebSocket";

// No fake timers here: ConversationStream has a setInterval for the elapsed
// clock which would loop infinitely with vi.runAllTimers(). Real timers are
// safe because tests complete well within the 1-second tick interval.

afterEach(() => {
  vi.restoreAllMocks();
});

const onAnswer = vi.fn();

function renderStream(events: AgentEvent[]) {
  return render(<ConversationStream events={events} onAnswer={onAnswer} />);
}

describe("ConversationStream", () => {
  it("shows waiting message when events list is empty", () => {
    renderStream([]);
    expect(screen.getByText(/waiting for events/i)).toBeInTheDocument();
  });

  it("displays a tool_result action item", () => {
    const events: AgentEvent[] = [
      {
        type: "tool_result",
        timestamp: 1000,
        name: "shell_exec",
        call_id: "c1",
        output: "hello stdout",
      },
    ];
    renderStream(events);
    expect(screen.getByText("shell_exec")).toBeInTheDocument();
    expect(screen.getByText("hello stdout")).toBeInTheDocument();
  });

  it("renders a QuestionCard for question events", () => {
    const events: AgentEvent[] = [
      {
        type: "question",
        timestamp: 1000,
        question_id: "q1",
        question: "Should I continue?",
      },
    ];
    renderStream(events);
    expect(screen.getByText("Should I continue?")).toBeInTheDocument();
  });

  it("renders error events", () => {
    const events: AgentEvent[] = [
      {
        type: "error",
        timestamp: 1000,
        message: "Something went wrong",
      },
    ];
    renderStream(events);
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
  });

  it("renders phase starting markers", () => {
    const events: AgentEvent[] = [
      {
        type: "phase",
        timestamp: 1000,
        phase: "phase1",
        step: "",
        status: "starting",
        detail: "",
      },
    ];
    renderStream(events);
    expect(screen.getByText("phase1")).toBeInTheDocument();
  });

  it("shows Done status when pipeline completes", () => {
    const events: AgentEvent[] = [
      { type: "status", timestamp: 1000, status: "starting" },
      { type: "status", timestamp: 2000, status: "done" },
    ];
    renderStream(events);
    expect(screen.getByText("Done")).toBeInTheDocument();
  });

  it("truncates long tool output to avoid flooding the UI", () => {
    const longOutput = Array.from({ length: 60 }, (_, i) => `line ${i}`).join(
      "\n",
    );
    const events: AgentEvent[] = [
      {
        type: "tool_result",
        timestamp: 1000,
        name: "shell_exec",
        call_id: "c1",
        output: longOutput,
      },
    ];
    renderStream(events);
    expect(screen.getByText(/lines omitted/)).toBeInTheDocument();
  });
});
