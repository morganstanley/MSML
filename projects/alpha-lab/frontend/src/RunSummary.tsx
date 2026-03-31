import { AgentEvent } from "./useWebSocket";

interface RunSummaryProps {
  events: AgentEvent[];
}

function formatElapsed(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

export default function RunSummary({ events }: RunSummaryProps) {
  // Elapsed time
  const startEvent = events.find(
    (e) => e.type === "status" && e.status === "starting",
  );
  const doneEvent = [...events]
    .reverse()
    .find((e) => e.type === "status" && (e.status === "done" || e.status === "error"));
  const elapsed =
    startEvent && doneEvent
      ? Math.floor(
          (doneEvent.timestamp as number) - (startEvent.timestamp as number),
        )
      : 0;

  // Turns
  const turnCount = events.filter((e) => e.type === "api_request").length;

  // Tool call breakdown
  const toolResults = events.filter((e) => e.type === "tool_result");
  const toolCounts: Record<string, number> = {};
  for (const e of toolResults) {
    const name = e.name as string;
    toolCounts[name] = (toolCounts[name] || 0) + 1;
  }

  // Also count web searches from api_response output (works on old logs too)
  for (const e of events) {
    if (e.type === "api_response") {
      const output = e.output as Record<string, unknown>[];
      if (output) {
        for (const item of output) {
          if (item.type === "web_search_call") {
            toolCounts["web_search"] = (toolCounts["web_search"] || 0) + 1;
          }
        }
      }
    }
  }

  // Deduplicate: if we have both tool_result web_search events AND api_response
  // web_search_call items (new logs have both), count only from api_response
  const hasWebSearchResults = toolResults.some((e) => e.name === "web_search");
  const webSearchFromApi = events.some(
    (e) =>
      e.type === "api_response" &&
      (e.output as Record<string, unknown>[])?.some(
        (item) => item.type === "web_search_call",
      ),
  );
  if (hasWebSearchResults && webSearchFromApi) {
    // api_response count includes all; tool_result count is a subset
    // Remove the tool_result count, keep only api_response count
    const fromResults = toolResults.filter((e) => e.name === "web_search").length;
    toolCounts["web_search"] -= fromResults;
    if (toolCounts["web_search"] <= 0) delete toolCounts["web_search"];
  }

  const totalActions = Object.values(toolCounts).reduce((a, b) => a + b, 0);
  const maxToolCount = Math.max(...Object.values(toolCounts), 1);
  const sortedTools = Object.entries(toolCounts).sort((a, b) => b[1] - a[1]);

  // Files created/modified
  const fileEvents = events.filter((e) => e.type === "file_changed");
  const filesAdded = new Set<string>();
  const filesModified = new Set<string>();
  for (const e of fileEvents) {
    const path = e.path as string;
    const change = e.change as string;
    if (change === "added") filesAdded.add(path);
    else if (change === "modified") filesModified.add(path);
  }

  // Token usage
  let inputTokens = 0;
  let outputTokens = 0;
  for (const e of events) {
    if (e.type === "api_response") {
      const usage = e.usage as Record<string, number> | undefined;
      if (usage) {
        inputTokens += usage.input_tokens || 0;
        outputTokens += usage.output_tokens || 0;
      }
    }
  }

  // Determine which phase completed
  const hasPhase2 = events.some(
    (e) => e.type === "phase" && e.phase === "phase2" && e.step === "complete",
  );
  const summaryTitle = hasPhase2 ? "Phase 2 Complete" : "Phase 1 Complete";

  return (
    <div className="run-summary">
      <div className="summary-title">{summaryTitle}</div>

      {/* Key stats row */}
      <div className="summary-stats">
        <div className="stat">
          <div className="stat-value">{formatElapsed(elapsed)}</div>
          <div className="stat-label">Duration</div>
        </div>
        <div className="stat">
          <div className="stat-value">{turnCount}</div>
          <div className="stat-label">Turns</div>
        </div>
        <div className="stat">
          <div className="stat-value">{totalActions}</div>
          <div className="stat-label">Actions</div>
        </div>
        <div className="stat">
          <div className="stat-value">{filesAdded.size}</div>
          <div className="stat-label">Files Created</div>
        </div>
      </div>

      {/* Tool breakdown bar chart */}
      <div className="summary-section">
        <div className="summary-section-title">Tool Calls</div>
        {sortedTools.map(([name, count]) => (
          <div key={name} className="bar-row">
            <div className="bar-label">{name}</div>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{ width: `${(count / maxToolCount) * 100}%` }}
              />
            </div>
            <div className="bar-value">{count}</div>
          </div>
        ))}
      </div>

      {/* Token usage */}
      {(inputTokens > 0 || outputTokens > 0) && (
        <div className="summary-section">
          <div className="summary-section-title">Token Usage</div>
          <div className="token-row">
            <span className="token-label">Input</span>
            <span className="token-value">{formatTokens(inputTokens)}</span>
          </div>
          <div className="token-row">
            <span className="token-label">Output</span>
            <span className="token-value">{formatTokens(outputTokens)}</span>
          </div>
          <div className="token-row token-total">
            <span className="token-label">Total</span>
            <span className="token-value">
              {formatTokens(inputTokens + outputTokens)}
            </span>
          </div>
        </div>
      )}

      {/* Files created */}
      {filesAdded.size > 0 && (
        <div className="summary-section">
          <div className="summary-section-title">
            Files Created ({filesAdded.size})
          </div>
          <div className="file-list">
            {[...filesAdded].sort().map((f) => (
              <div key={f} className="file-entry">
                {f}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
