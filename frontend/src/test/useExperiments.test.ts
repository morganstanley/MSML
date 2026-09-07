import { renderHook, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { useExperiments } from "../useExperiments";
import type { AgentEvent } from "../useWebSocket";

const EXPERIMENTS_RESPONSE = {
  experiments: [
    {
      id: 1,
      name: "exp-1",
      description: "desc",
      hypothesis: "hyp",
      status: "running",
      config_json: "{}",
      worker_id: null,
      slurm_job_id: null,
      results_json: null,
      error: null,
      debrief_path: null,
      created_at: 1000,
      updated_at: 1001,
      started_at: null,
      finished_at: null,
    },
  ],
  summary: { running: 1 },
};

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  // clearAllTimers stops recurring intervals without looping infinitely.
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.clearAllMocks();
  vi.restoreAllMocks();
});

function mockFetchOnce(response: unknown) {
  (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
    ok: true,
    json: () => Promise.resolve(response),
  });
}

describe("useExperiments", () => {
  it("fetches experiments on mount", async () => {
    mockFetchOnce(EXPERIMENTS_RESPONSE);
    const { result } = renderHook(() => useExperiments([]));

    await act(async () => {
      await Promise.resolve();
    });

    expect(global.fetch).toHaveBeenCalledWith("/api/experiments");
    expect(result.current.experiments).toHaveLength(1);
    expect(result.current.experiments[0].name).toBe("exp-1");
    expect(result.current.summary).toEqual({ running: 1 });
  });

  it("re-fetches on experiment WebSocket event", async () => {
    mockFetchOnce(EXPERIMENTS_RESPONSE);
    const { result, rerender } = renderHook(
      ({ evts }: { evts: AgentEvent[] }) => useExperiments(evts),
      { initialProps: { evts: [] as AgentEvent[] } },
    );

    await act(async () => {
      await Promise.resolve();
    });

    // Advance time past the 1-second debounce window so re-fetch is allowed.
    vi.advanceTimersByTime(1100);

    const updated = { ...EXPERIMENTS_RESPONSE, summary: { running: 2 } };
    mockFetchOnce(updated);

    await act(async () => {
      rerender({ evts: [{ type: "experiment", timestamp: Date.now() }] });
      await Promise.resolve();
    });

    expect(result.current.summary).toEqual({ running: 2 });
  });

  it("polls every 5 seconds", async () => {
    mockFetchOnce(EXPERIMENTS_RESPONSE);
    renderHook(() => useExperiments([]));

    await act(async () => {
      await Promise.resolve();
    });

    mockFetchOnce(EXPERIMENTS_RESPONSE);
    await act(async () => {
      vi.advanceTimersByTime(5000);
      await Promise.resolve();
    });

    expect(global.fetch).toHaveBeenCalledTimes(2);
  });

  it("handles fetch errors gracefully", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      new Error("network error"),
    );
    const { result } = renderHook(() => useExperiments([]));
    await act(async () => {
      await Promise.resolve();
    });
    // Should not throw; experiments stay empty
    expect(result.current.experiments).toHaveLength(0);
  });
});
