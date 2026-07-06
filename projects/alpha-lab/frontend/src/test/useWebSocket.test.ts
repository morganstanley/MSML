import { renderHook, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { useWebSocket } from "../useWebSocket";
import { MockWebSocket } from "./setup";

let ws: MockWebSocket;

beforeEach(() => {
  vi.useFakeTimers();
  // Build a constructor mock that also carries the static constants the hook
  // checks (e.g. `WebSocket.OPEN`) so that `readyState === WebSocket.OPEN` works.
  const MockWS = vi.fn().mockImplementation((url: string) => {
    ws = new MockWebSocket(url);
    return ws;
  });
  Object.assign(MockWS, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });
  vi.stubGlobal("WebSocket", MockWS);
});

afterEach(() => {
  // clearAllTimers stops recurring intervals without looping infinitely.
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("useWebSocket", () => {
  it("starts disconnected and connects to /ws", () => {
    const { result } = renderHook(() => useWebSocket());
    expect(result.current.connected).toBe(false);
    // @ts-expect-error -- mock
    expect(global.WebSocket).toHaveBeenCalledWith(
      expect.stringMatching(/\/ws$/),
    );
  });

  it("becomes connected when WebSocket opens", () => {
    const { result } = renderHook(() => useWebSocket());
    act(() => {
      ws.simulateOpen();
    });
    expect(result.current.connected).toBe(true);
  });

  it("enqueues and flushes incoming events after FLUSH_MS", () => {
    const { result } = renderHook(() => useWebSocket());
    act(() => {
      ws.simulateOpen();
      ws.simulateMessage({ type: "status", status: "thinking", timestamp: 1 });
    });
    // Flush batching timer
    act(() => {
      vi.runAllTimers();
    });
    expect(result.current.events).toHaveLength(1);
    expect(result.current.events[0].type).toBe("status");
  });

  it("ignores malformed JSON messages", () => {
    const { result } = renderHook(() => useWebSocket());
    act(() => {
      ws.simulateOpen();
      // Deliver a raw non-JSON string
      const badEvent = new MessageEvent("message", { data: "not-json{{" });
      if (ws.onmessage) ws.onmessage(badEvent);
    });
    act(() => {
      vi.runAllTimers();
    });
    expect(result.current.events).toHaveLength(0);
  });

  it("reconnects after the connection closes", () => {
    renderHook(() => useWebSocket());
    const firstWs = ws;
    act(() => {
      firstWs.simulateOpen();
      firstWs.close();
    });
    // Reconnect timer fires
    act(() => {
      vi.runAllTimers();
    });
    // A second WebSocket should have been created
    // @ts-expect-error -- mock
    expect(global.WebSocket).toHaveBeenCalledTimes(2);
  });

  it("sendMessage sends JSON when socket is open", () => {
    const { result } = renderHook(() => useWebSocket());
    act(() => {
      ws.simulateOpen();
    });
    act(() => {
      result.current.sendMessage({ type: "answer", text: "yes" });
    });
    expect(ws.send).toHaveBeenCalledWith(
      JSON.stringify({ type: "answer", text: "yes" }),
    );
  });

  it("sendMessage is a no-op when socket is not open", () => {
    const { result } = renderHook(() => useWebSocket());
    // ws is in CONNECTING state, not open
    act(() => {
      result.current.sendMessage({ type: "answer", text: "yes" });
    });
    expect(ws.send).not.toHaveBeenCalled();
  });
});
