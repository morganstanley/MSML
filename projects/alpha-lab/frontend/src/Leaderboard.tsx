import { useCallback, useEffect, useState } from "react";
import { AgentEvent } from "./useWebSocket";

interface LeaderEntry {
  id: number;
  name: string;
  status: string;
  metrics: Record<string, number>;
}

interface Props {
  events: AgentEvent[];
}

export default function Leaderboard({ events }: Props) {
  const [entries, setEntries] = useState<LeaderEntry[]>([]);
  const [sortKey, setSortKey] = useState("sharpe");

  const fetchLeaderboard = useCallback(() => {
    fetch("/api/leaderboard")
      .then((r) => r.json())
      .then((data) => {
        if (data.leaderboard) setEntries(data.leaderboard);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetchLeaderboard();
    const interval = setInterval(fetchLeaderboard, 10000);
    return () => clearInterval(interval);
  }, [fetchLeaderboard]);

  // Refresh on experiment events
  useEffect(() => {
    const last = events[events.length - 1];
    if (last && (last.type === "experiment" || last.type === "board_summary")) {
      fetchLeaderboard();
    }
  }, [events, fetchLeaderboard]);

  const sorted = [...entries].sort((a, b) => {
    const av = a.metrics[sortKey] ?? -Infinity;
    const bv = b.metrics[sortKey] ?? -Infinity;
    return bv - av;
  });

  const metricKeys = Array.from(
    new Set(entries.flatMap((e) => Object.keys(e.metrics)))
  ).slice(0, 6);

  if (entries.length === 0) {
    return (
      <div className="leaderboard-empty">
        No experiment results yet
      </div>
    );
  }

  return (
    <div className="leaderboard-container">
      <table className="leaderboard-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Name</th>
            {metricKeys.map((key) => (
              <th
                key={key}
                className={`leaderboard-sortable ${sortKey === key ? "leaderboard-active-sort" : ""}`}
                onClick={() => setSortKey(key)}
              >
                {key}
              </th>
            ))}
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((entry, i) => (
            <tr key={entry.id}>
              <td className="leaderboard-rank">{i + 1}</td>
              <td className="leaderboard-name">{entry.name}</td>
              {metricKeys.map((key) => (
                <td key={key} className="leaderboard-metric">
                  {entry.metrics[key] !== undefined
                    ? typeof entry.metrics[key] === "number"
                      ? entry.metrics[key].toFixed(4)
                      : entry.metrics[key]
                    : "—"}
                </td>
              ))}
              <td className="leaderboard-status">{entry.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
