import { render, screen, act, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import App from "../App";
import { MockWebSocket } from "./setup";

let ws: MockWebSocket;

beforeEach(() => {
  // Use fake timers (without shouldAdvanceTime) so setInterval callbacks from
  // child components (ConversationStream tick, useExperiments poll, etc.) are
  // frozen and never fire automatically.
  vi.useFakeTimers();

  const MockWS = vi.fn().mockImplementation((url: string) => {
    ws = new MockWebSocket(url);
    return ws;
  });
  Object.assign(MockWS, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });
  vi.stubGlobal("WebSocket", MockWS);

  // Silence all API calls made by child components (FileTree, Leaderboard, etc.)
  (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
    ok: true,
    json: () =>
      Promise.resolve({
        experiments: [],
        summary: {},
        leaderboard: [],
        children: [],
      }),
  });
});

afterEach(() => {
  // clearAllTimers stops all recurring intervals without infinite looping.
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("App view switching", () => {
  it("renders the Files tab as active by default", async () => {
    render(<App />);
    await act(async () => {
      await Promise.resolve();
    });
    expect(screen.getByRole("button", { name: /^files$/i })).toHaveClass(
      "active",
    );
  });

  it("switches to Board view when Board button is clicked", async () => {
    render(<App />);
    await act(async () => {
      await Promise.resolve();
    });
    fireEvent.click(screen.getByRole("button", { name: /^board$/i }));
    expect(screen.getByRole("button", { name: /^board$/i })).toHaveClass(
      "active",
    );
  });

  it("switches to Status view when Status button is clicked", async () => {
    render(<App />);
    await act(async () => {
      await Promise.resolve();
    });
    fireEvent.click(screen.getByRole("button", { name: /^status$/i }));
    expect(screen.getByRole("button", { name: /^status$/i })).toHaveClass(
      "active",
    );
  });

  it("auto-switches to Board view when Phase 3 events arrive", async () => {
    render(<App />);
    await act(async () => {
      ws.simulateOpen();
      ws.simulateMessage({
        type: "phase",
        phase: "phase3",
        status: "starting",
        timestamp: Date.now() / 1000,
      });
      // Advance past FLUSH_MS (150ms) so the WS event batch is flushed to state.
      vi.advanceTimersByTime(200);
      await Promise.resolve();
    });
    expect(screen.getByRole("button", { name: /^board$/i })).toHaveClass(
      "active",
    );
  });

  it("auto-switches back to Files view when pipeline finishes", async () => {
    render(<App />);
    // First: Phase 3 starts → switch to board
    await act(async () => {
      ws.simulateOpen();
      ws.simulateMessage({
        type: "phase",
        phase: "phase3",
        status: "starting",
        timestamp: Date.now() / 1000,
      });
      vi.advanceTimersByTime(200);
      await Promise.resolve();
    });
    // Then: pipeline done → switch back to files
    await act(async () => {
      ws.simulateMessage({
        type: "status",
        status: "done",
        timestamp: Date.now() / 1000,
      });
      vi.advanceTimersByTime(200);
      await Promise.resolve();
    });
    expect(screen.getByRole("button", { name: /^files$/i })).toHaveClass(
      "active",
    );
  });
});
