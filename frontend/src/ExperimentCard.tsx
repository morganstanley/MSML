import { useState } from "react";
import { ExperimentData } from "./useExperiments";

interface Props {
  experiment: ExperimentData;
  onCancel?: (id: number) => void;
}

const STATUS_COLORS: Record<string, string> = {
  to_implement: "#d29922",
  implemented: "#d18616",
  checked: "#58a6ff",
  queued: "#8b949e",
  running: "#58a6ff",
  finished: "#3fb950",
  analyzed: "#a371f7",
  done: "#3fb950",
  cancelled: "#f85149",
};

function parseMetrics(json: string | null): Record<string, number> {
  if (!json) return {};
  try {
    return JSON.parse(json);
  } catch {
    return {};
  }
}

const CANCELLABLE = new Set([
  "to_implement", "implemented", "checked", "queued", "running", "finished",
]);

export default function ExperimentCard({ experiment, onCancel }: Props) {
  const [expanded, setExpanded] = useState(false);
  const metrics = parseMetrics(experiment.results_json);
  const color = STATUS_COLORS[experiment.status] || "#8b949e";
  const canCancel = CANCELLABLE.has(experiment.status) && onCancel;

  return (
    <div
      className="experiment-card"
      onClick={() => setExpanded(!expanded)}
    >
      <div className="exp-card-header">
        <span className="exp-card-name">{experiment.name}</span>
        <div className="exp-card-header-right">
          <span
            className="exp-card-badge"
            style={{ background: `${color}22`, color }}
          >
            {experiment.status}
          </span>
          {canCancel && (
            <button
              className="exp-card-cancel"
              title="Cancel experiment"
              onClick={(e) => {
                e.stopPropagation();
                onCancel(experiment.id);
              }}
            >
              ×
            </button>
          )}
        </div>
      </div>

      <div className="exp-card-hypothesis">
        {experiment.hypothesis || experiment.description}
      </div>

      {metrics.sharpe !== undefined && (
        <div className="exp-card-metric">
          Sharpe: {typeof metrics.sharpe === "number" ? metrics.sharpe.toFixed(3) : metrics.sharpe}
        </div>
      )}

      <div className="exp-card-footer">
        {experiment.worker_id && (
          <span className="exp-card-worker">{experiment.worker_id}</span>
        )}
        {experiment.slurm_job_id && (
          <span className="exp-card-slurm">#{experiment.slurm_job_id}</span>
        )}
        {experiment.error && (
          <span className="exp-card-error" title={experiment.error}>ERR</span>
        )}
      </div>

      {expanded && (
        <div className="exp-card-detail">
          <div className="exp-card-section">
            <strong>Description:</strong> {experiment.description}
          </div>
          {experiment.hypothesis && (
            <div className="exp-card-section">
              <strong>Hypothesis:</strong> {experiment.hypothesis}
            </div>
          )}
          {Object.keys(metrics).length > 0 && (
            <div className="exp-card-section">
              <strong>Metrics:</strong>
              <div className="exp-card-metrics">
                {Object.entries(metrics).map(([k, v]) => (
                  <span key={k} className="exp-card-metric-item">
                    {k}: {typeof v === "number" ? v.toFixed(4) : v}
                  </span>
                ))}
              </div>
            </div>
          )}
          {experiment.error && (
            <div className="exp-card-section exp-card-error-detail">
              <strong>Error:</strong> {experiment.error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
