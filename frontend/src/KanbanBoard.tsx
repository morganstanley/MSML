import { useCallback, useState } from "react";
import { ExperimentData, BoardSummary } from "./useExperiments";
import ExperimentCard from "./ExperimentCard";

interface Props {
  experiments: ExperimentData[];
  summary: BoardSummary;
  onRefresh?: () => void;
}

const COLUMNS = [
  { key: "to_implement", label: "To Implement" },
  { key: "implemented", label: "Implemented" },
  { key: "checked", label: "Checked" },
  { key: "queued", label: "Queued" },
  { key: "running", label: "Running" },
  { key: "finished", label: "Finished" },
  { key: "analyzed", label: "Analyzed" },
  { key: "done", label: "Done" },
  { key: "cancelled", label: "Cancelled" },
];

// Default number of items to show per column
const DEFAULT_VISIBLE = 10;
const LOAD_MORE_INCREMENT = 20;

export default function KanbanBoard({ experiments, summary, onRefresh }: Props) {
  // Track how many items to show per column
  const [visibleCounts, setVisibleCounts] = useState<Record<string, number>>({});
  // Track collapsed columns
  const [collapsedCols, setCollapsedCols] = useState<Record<string, boolean>>({});

  const handleCancel = useCallback(
    (id: number) => {
      fetch(`/api/experiments/${id}/cancel`, { method: "POST" })
        .then((r) => {
          if (r.ok && onRefresh) onRefresh();
        })
        .catch(() => {});
    },
    [onRefresh],
  );

  const byStatus: Record<string, ExperimentData[]> = {};
  for (const col of COLUMNS) {
    byStatus[col.key] = [];
  }
  for (const exp of experiments) {
    if (byStatus[exp.status]) {
      byStatus[exp.status].push(exp);
    }
  }

  const toggleCollapsed = (key: string) => {
    setCollapsedCols((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const showMore = (key: string) => {
    setVisibleCounts((prev) => ({
      ...prev,
      [key]: (prev[key] || DEFAULT_VISIBLE) + LOAD_MORE_INCREMENT,
    }));
  };

  const showAll = (key: string, total: number) => {
    setVisibleCounts((prev) => ({ ...prev, [key]: total }));
  };

  return (
    <div className="kanban-board">
      {COLUMNS.map((col) => {
        const items = byStatus[col.key];
        const total = items.length;
        const totalFromSummary = summary[col.key] || total;
        const visibleCount = visibleCounts[col.key] || DEFAULT_VISIBLE;
        const isCollapsed = collapsedCols[col.key] || false;
        const visibleItems = items.slice(0, visibleCount);
        const hasMore = total > visibleCount;

        return (
          <div key={col.key} className={`kanban-column ${isCollapsed ? "collapsed" : ""}`}>
            <div
              className="kanban-column-header"
              onClick={() => toggleCollapsed(col.key)}
              style={{ cursor: "pointer" }}
            >
              <span className="kanban-collapse-icon">
                {isCollapsed ? "▶" : "▼"}
              </span>
              <span className="kanban-column-label">{col.label}</span>
              <span className="kanban-column-count">{totalFromSummary}</span>
            </div>
            {!isCollapsed && (
              <div className="kanban-column-body">
                {visibleItems.map((exp) => (
                  <ExperimentCard key={exp.id} experiment={exp} onCancel={handleCancel} />
                ))}
                {hasMore && (
                  <div className="kanban-load-more">
                    <button onClick={() => showMore(col.key)}>
                      Show more ({visibleCount}/{total})
                    </button>
                    <button onClick={() => showAll(col.key, total)}>
                      Show all
                    </button>
                  </div>
                )}
                {total === 0 && (
                  <div className="kanban-empty">No experiments</div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
