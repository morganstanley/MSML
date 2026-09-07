import "@testing-library/jest-dom";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

// scrollIntoView is not implemented in jsdom
Element.prototype.scrollIntoView = vi.fn();

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  // Re-apply the default rejected fetch mock after clearing
  (global.fetch as ReturnType<typeof vi.fn>).mockRejectedValue(
    new Error("fetch not mocked"),
  );
});

// ---------------------------------------------------------------------------
// fetch mock — returns a rejected promise by default; tests override as needed
// ---------------------------------------------------------------------------
global.fetch = vi.fn().mockRejectedValue(new Error("fetch not mocked"));

// ---------------------------------------------------------------------------
// WebSocket mock
// ---------------------------------------------------------------------------
class MockWebSocket extends EventTarget {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState: number = MockWebSocket.CONNECTING;
  url: string;

  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  send = vi.fn();
  close = vi.fn(() => {
    this.readyState = MockWebSocket.CLOSED;
    const event = new Event("close");
    if (this.onclose) this.onclose(event as CloseEvent);
  });

  constructor(url: string) {
    super();
    this.url = url;
  }

  /** Helper: simulate the server opening the connection */
  simulateOpen() {
    this.readyState = MockWebSocket.OPEN;
    const event = new Event("open");
    if (this.onopen) this.onopen(event);
  }

  /** Helper: push a JSON message from the server */
  simulateMessage(data: unknown) {
    const event = new MessageEvent("message", {
      data: JSON.stringify(data),
    });
    if (this.onmessage) this.onmessage(event);
  }
}

// @ts-expect-error -- replacing global with a minimal compatible mock
global.WebSocket = MockWebSocket;

export { MockWebSocket };
