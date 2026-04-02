import { useCallback, useEffect, useRef, useState } from "react";

export interface AgentEvent {
  type: string;
  timestamp: number;
  [key: string]: unknown;
}

interface UseWebSocketResult {
  events: AgentEvent[];
  connected: boolean;
  sendMessage: (msg: Record<string, unknown>) => void;
  clearEvents: () => void;
}

/**
 * Batch incoming WebSocket events to avoid re-rendering on every single
 * message during replay.  Events are accumulated in a buffer and flushed
 * to React state at most every FLUSH_MS milliseconds.
 */
const FLUSH_MS = 150;

export function useWebSocket(): UseWebSocketResult {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Buffer for batching incoming events
  const bufferRef = useRef<AgentEvent[]>([]);
  const flushTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const flush = useCallback(() => {
    flushTimer.current = null;
    const batch = bufferRef.current;
    if (batch.length === 0) return;
    bufferRef.current = [];
    setEvents((prev) => {
      const next = new Array(prev.length + batch.length);
      for (let i = 0; i < prev.length; i++) next[i] = prev[i];
      for (let i = 0; i < batch.length; i++) next[prev.length + i] = batch[i];
      return next;
    });
  }, []);

  const enqueue = useCallback(
    (event: AgentEvent) => {
      bufferRef.current.push(event);
      if (flushTimer.current === null) {
        flushTimer.current = setTimeout(flush, FLUSH_MS);
      }
    },
    [flush],
  );

  const connect = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data) as AgentEvent;
        enqueue(event);
      } catch {
        // ignore malformed messages
      }
    };

    ws.onclose = () => {
      // Flush any remaining buffered events
      if (flushTimer.current) {
        clearTimeout(flushTimer.current);
        flushTimer.current = null;
      }
      flush();

      setConnected(false);
      wsRef.current = null;
      reconnectTimer.current = setTimeout(connect, 2000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [enqueue, flush]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (flushTimer.current) clearTimeout(flushTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const sendMessage = useCallback((msg: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  const clearEvents = useCallback(() => {
    bufferRef.current = [];
    setEvents([]);
  }, []);

  return { events, connected, sendMessage, clearEvents };
}
