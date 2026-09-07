import { useCallback, useEffect, useState } from "react";
import { AgentEvent } from "./useWebSocket";

interface StatusData {
  generated_at?: number;
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
    best_per_horizon?: {
      horizon: number | null;
      strategy: string;
      country: string;
      sharpe: number;
      mae: number | null;
    }[];
    avg_sharpe_by_strategy?: Record<string, number>;
  };
  experiments?: {
    available?: boolean;
    board?: Record<string, number>;
    total?: number;
    top_models?: {
      name: string;
      status: string;
      sharpe: number | null;
      mae: number | null;
      max_dd: number | null;
      total_return: number | null;
    }[];
    all_scored_count?: number;
    failures?: { name: string; error: string }[];
    running_slurm?: { name: string; slurm_job_id: string }[];
  };
  comparison?: {
    name: string;
    model_sharpe: number;
    model_mae: number | null;
    best_baseline_sharpe: number;
    avg_baseline_sharpe: number;
    beats_best_baseline: boolean;
    beats_avg_baseline: boolean;
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
                <span>{baselines.countries?.length ?? 0} countries</span>
                <span>{baselines.strategies?.length ?? 0} strategies</span>
                <span>{baselines.horizons?.map((h) => `h=${h}`).join(", ")}</span>
              </div>

              {baselines.avg_sharpe_by_strategy && (
                <table className="sr-table">
                  <thead>
                    <tr>
                      <th>Strategy</th>
                      <th>Avg Sharpe</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(baselines.avg_sharpe_by_strategy)
                      .sort((a, b) => b[1] - a[1])
                      .map(([strat, sharpe]) => (
                        <tr key={strat}>
                          <td>{strat}</td>
                          <td className="sr-metric">{fmt(sharpe)}</td>
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
                        <th>Horizon</th>
                        <th>Strategy</th>
                        <th>Country</th>
                        <th>Sharpe</th>
                        <th>MAE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {baselines.best_per_horizon.map((b) => (
                        <tr key={b.horizon}>
                          <td>h={b.horizon}</td>
                          <td>{b.strategy}</td>
                          <td>{b.country}</td>
                          <td className="sr-metric">{fmt(b.sharpe)}</td>
                          <td className="sr-metric">{fmt(b.mae)}</td>
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
                    <th>Sharpe</th>
                    <th>MAE</th>
                    <th>Best Baseline</th>
                    <th>Beats Best?</th>
                  </tr>
                </thead>
                <tbody>
                  {comparisons.map((c, i) => (
                    <tr key={c.name}>
                      <td className="sr-rank">{i + 1}</td>
                      <td className="sr-model-name">{c.name}</td>
                      <td className="sr-metric">{fmt(c.model_sharpe)}</td>
                      <td className="sr-metric">{fmt(c.model_mae)}</td>
                      <td className="sr-metric">{fmt(c.best_baseline_sharpe)}</td>
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
          {topModels.length > 0 && (
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
                    <th>Sharpe</th>
                    <th>MAE</th>
                    <th>Max DD</th>
                    <th>Return</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {topModels.map((m, i) => (
                    <tr key={m.name}>
                      <td className="sr-rank">{i + 1}</td>
                      <td className="sr-model-name">{m.name}</td>
                      <td className="sr-metric">{fmt(m.sharpe)}</td>
                      <td className="sr-metric">{fmt(m.mae)}</td>
                      <td className="sr-metric">{fmt(m.max_dd)}</td>
                      <td className="sr-metric">{fmt(m.total_return)}</td>
                      <td className="sr-status-cell">{m.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}

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
