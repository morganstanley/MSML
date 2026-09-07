import { useCallback, useEffect, useRef, useState } from "react";
import { AgentEvent } from "./useWebSocket";

export interface ExperimentData {
  id: number;
  name: string;
  description: string;
  hypothesis: string;
  status: string;
  config_json: string;
  worker_id: string | null;
  slurm_job_id: string | null;
  results_json: string | null;
  error: string | null;
  debrief_path: string | null;
  created_at: number;
  updated_at: number;
  started_at: number | null;
  finished_at: number | null;
}

export interface BoardSummary {
  [status: string]: number;
}

export function useExperiments(events: AgentEvent[]) {
  const [experiments, setExperiments] = useState<ExperimentData[]>([]);
  const [summary, setSummary] = useState<BoardSummary>({});
  const lastFetch = useRef(0);

  const fetchExperiments = useCallback(() => {
    fetch("/api/experiments")
      .then((r) => r.json())
      .then((data) => {
        if (data.experiments) setExperiments(data.experiments);
        if (data.summary) setSummary(data.summary);
      })
      .catch(() => {});
    lastFetch.current = Date.now();
  }, []);

  // Poll every 5 seconds
  useEffect(() => {
    fetchExperiments();
    const interval = setInterval(fetchExperiments, 5000);
    return () => clearInterval(interval);
  }, [fetchExperiments]);

  // Also refresh on experiment/board_summary WebSocket events
  useEffect(() => {
    const lastEvent = events[events.length - 1];
    if (
      lastEvent &&
      (lastEvent.type === "experiment" || lastEvent.type === "board_summary")
    ) {
      // Debounce: don't refetch if we just fetched
      if (Date.now() - lastFetch.current > 1000) {
        fetchExperiments();
      }
    }
  }, [events, fetchExperiments]);

  return { experiments, summary, refresh: fetchExperiments };
}
