import { useCallback, useEffect, useState } from "react";
import { AgentEvent } from "./useWebSocket";

interface MetricConfig {
  primary_metric: string;
  direction: string;
  display_name: string;
}

interface StatusData {
  generated_at?: number;
  metric_config?: MetricConfig;
  problem?: {
    summary?: string;
    exploration_available?: boolean;
  };
  baselines?: {
    available?: boolean;
    total_rows?: number;
    horizons?: number[];
    strategies?: string[];
    countries?: string[];
    best_per_horizon?: Record<string, any>[];
    avg_primary_by_strategy?: Record<string, number>;
  };
  experiments?: {
    available?: boolean;
    board?: Record<string, number>;
    total?: number;
    top_models?: Record<string, any>[];
    all_scored_count?: number;
    failures?: { name: string; error: string }[];
    running_slurm?: { name: string; slurm_job_id: string }[];
  };
  comparison?: {
    name: string;
    model_primary: number;
    best_baseline: number;
    avg_baseline: number;
    beats_best_baseline: boolean;
  }[];
}

interface Props {
  events: AgentEvent[];
  sendMessage: (msg: Record<string, unknown>) => void;
}

function fmt(val: number | null | undefined, digits = 4): string {
  if (val === null || val === undefined) return "\u2014";
  return val.toFixed(digits);
}

function timeAgo(ts: number): string {
  const diff = Math.floor(Date.now() / 1000 - ts);
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ${Math.floor((diff % 3600) / 60)}m ago`;
}

export default function StatusReport({ events, sendMessage }: Props) {
  const [data, setData] = useState<StatusData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchReport = useCallback(() => {
    setLoading(true);
    setError(null);
    fetch("/api/status-report")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, []);

  // Fetch on mount
  useEffect(() => {
    fetchReport();
  }, [fetchReport]);

  // Listen for status_report events broadcast via WS (auto-timer or manual)
  useEffect(() => {
    const last = events[events.length - 1];
    if (last && last.type === "status_report") {
      setData(last as unknown as StatusData);
    }
  }, [events]);

  const handleGenerate = () => {
    sendMessage({ type: "generate_status" });
    fetchReport();
  };

  const mc = data?.metric_config;
  const primaryMetric = mc?.primary_metric ?? "sharpe";
  const displayName = mc?.display_name ?? "Sharpe";
  const metricDirection = mc?.direction ?? "maximize";
  const board = data?.experiments?.board;
  const topModels = data?.experiments?.top_models ?? [];
  const comparisons = data?.comparison ?? [];
  const baselines = data?.baselines;

  return (
    <div className="status-report">
      <div className="status-report-header">
        <h2 className="status-report-title">Status Report</h2>
        <div className="status-report-actions">
          {data?.generated_at && (
            <span className="status-report-timestamp">
              {timeAgo(data.generated_at)}
            </span>
          )}
          <button
            className="btn btn-primary"
            onClick={handleGenerate}
            disabled={loading}
          >
            {loading ? "Generating..." : "Refresh"}
          </button>
        </div>
      </div>

      {error && <div className="status-report-error">Error: {error}</div>}

      {data && (
        <div className="status-report-body">

          {/* Pipeline Status */}
          {board && (
            <section className="sr-section">
              <h3 className="sr-section-title">Pipeline Status</h3>
              <div className="sr-board-grid">
                {Object.entries(board).map(([status, count]) => (
                  <div key={status} className="sr-board-cell">
                    <div className="sr-board-count">{count}</div>
                    <div className="sr-board-label">{status.replace(/_/g, " ")}</div>
                  </div>
                ))}
                <div className="sr-board-cell sr-board-total">
                  <div className="sr-board-count">{data.experiments?.total ?? 0}</div>
                  <div className="sr-board-label">total</div>
                </div>
              </div>
              {data.experiments?.running_slurm && data.experiments.running_slurm.length > 0 && (
                <div className="sr-running-jobs">
                  <span className="sr-running-label">SLURM:</span>
                  {data.experiments.running_slurm.map((j) => (
                    <span key={j.slurm_job_id} className="sr-job-chip">
                      {j.name.replace(/^(.*?)_(\d+d).*/, "$1")}
                    </span>
                  ))}
                </div>
              )}
            </section>
          )}

          {/* Baselines */}
          {baselines?.available && (
            <section className="sr-section">
              <h3 className="sr-section-title">Baselines</h3>
              <div className="sr-meta-row">
                {baselines.countries && baselines.countries.length > 0 && (
                  <span>{baselines.countries.length} countries</span>
                )}
                {baselines.strategies && baselines.strategies.length > 0 && (
                  <span>{baselines.strategies.length} strategies</span>
                )}
                {baselines.horizons && baselines.horizons.length > 0 && (
                  <span>{baselines.horizons.map((h) => `h=${h}`).join(", ")}</span>
                )}
                <span>{baselines.total_rows ?? 0} total rows</span>
              </div>

              {baselines.avg_primary_by_strategy && (
                <table className="sr-table">
                  <thead>
                    <tr>
                      <th>Strategy</th>
                      <th>Avg {displayName}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(baselines.avg_primary_by_strategy)
                      .sort((a, b) => metricDirection === "maximize" ? b[1] - a[1] : a[1] - b[1])
                      .map(([strat, val]) => (
                        <tr key={strat}>
                          <td>{strat}</td>
                          <td className="sr-metric">{fmt(val)}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              )}

              {baselines.best_per_horizon && baselines.best_per_horizon.length > 0 && (
                <>
                  <h4 className="sr-subsection">Best Baseline per Horizon</h4>
                  <table className="sr-table">
                    <thead>
                      <tr>
                        {Object.keys(baselines.best_per_horizon[0]).map((k) => (
                          <th key={k}>{k === primaryMetric ? displayName : k}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {baselines.best_per_horizon.map((b, i) => (
                        <tr key={i}>
                          {Object.entries(b).map(([k, v]) => (
                            <td key={k} className={typeof v === "number" ? "sr-metric" : ""}>
                              {typeof v === "number" ? fmt(v) : String(v ?? "\u2014")}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}
            </section>
          )}

          {/* Top Models vs Baselines */}
          {comparisons.length > 0 && (
            <section className="sr-section">
              <h3 className="sr-section-title">Top Models vs Baselines</h3>
              <table className="sr-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Model</th>
                    <th>{displayName}</th>
                    <th>Best Baseline</th>
                    <th>Beats Best?</th>
                  </tr>
                </thead>
                <tbody>
                  {comparisons.map((c, i) => (
                    <tr key={c.name}>
                      <td className="sr-rank">{i + 1}</td>
                      <td className="sr-model-name">{c.name}</td>
                      <td className="sr-metric">{fmt(c.model_primary)}</td>
                      <td className="sr-metric">{fmt(c.best_baseline)}</td>
                      <td>
                        <span className={c.beats_best_baseline ? "sr-beats-yes" : "sr-beats-no"}>
                          {c.beats_best_baseline ? "YES" : "no"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}

          {/* Full Leaderboard */}
          {topModels.length > 0 && (() => {
            const skipKeys = new Set(["name", "status", "primary_metric", "slurm_job_id"]);
            const metricKeys = Object.keys(topModels[0]).filter((k) => !skipKeys.has(k));
            // Put primary metric first
            metricKeys.sort((a, b) => {
              if (a === primaryMetric) return -1;
              if (b === primaryMetric) return 1;
              return 0;
            });

            return (
              <section className="sr-section">
                <h3 className="sr-section-title">
                  Top {topModels.length} Models
                  {data.experiments?.all_scored_count
                    ? ` (of ${data.experiments.all_scored_count} scored)`
                    : ""}
                </h3>
                <table className="sr-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Name</th>
                      {metricKeys.map((k) => (
                        <th key={k}>{k === primaryMetric ? displayName : k}</th>
                      ))}
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topModels.map((m, i) => (
                      <tr key={m.name}>
                        <td className="sr-rank">{i + 1}</td>
                        <td className="sr-model-name">{m.name}</td>
                        {metricKeys.map((k) => (
                          <td key={k} className="sr-metric">{fmt(m[k])}</td>
                        ))}
                        <td className="sr-status-cell">{m.status}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            );
          })()}

          {/* Failures */}
          {data.experiments?.failures && data.experiments.failures.length > 0 && (
            <section className="sr-section">
              <h3 className="sr-section-title">
                Failures ({data.experiments.failures.length})
              </h3>
              <div className="sr-failures">
                {data.experiments.failures.map((f) => (
                  <div key={f.name} className="sr-failure-row">
                    <span className="sr-failure-name">{f.name}</span>
                    {f.error && (
                      <span className="sr-failure-error">{f.error}</span>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      )}

      {!data && !loading && !error && (
        <div className="status-report-empty">
          No status data available yet. Click Refresh to generate.
        </div>
      )}
    </div>
  );
}
