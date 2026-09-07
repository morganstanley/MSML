import { AgentEvent } from "./useWebSocket";

interface TopBarProps {
  connected: boolean;
  events: AgentEvent[];
}

export default function TopBar({ connected, events }: TopBarProps) {
  // Detect Phase 3 from experiment/board_summary events OR phase events
  const hasPhase3 = events.some(
    (e) => e.type === "experiment" || e.type === "board_summary" ||
      (e.type === "phase" && e.phase === "phase3")
  );

  // Count done experiments for Phase 3 display
  const doneCount = hasPhase3
    ? events.filter(
        (e) => e.type === "experiment" &&
          (e.status === "done" || e.status === "analyzed")
      ).length
    : 0;

  // Derive current phase from most recent phase event
  const lastPhase = [...events]
    .reverse()
    .find((e) => e.type === "phase");

  // Conductor turns now emit phase events with phase="conductor" (from
  // run.py's _run_conductor_steer wrapper) and phase="phase0"/"phase1"/
  // "phase2" (from the Conductor's internal _run_steer). Surface both
  // as "Conductor" so the dashboard doesn't default to "Phase 1" mid-
  // boundary-turn. phase="phase3" + step="conductor" remains in the
  // hasPhase3 branch since those are milestone/timer turns during
  // experimentation.
  const _phase = String(lastPhase?.phase ?? "");
  const _step = String(lastPhase?.step ?? "");
  const isConductorBoundary =
    _phase === "conductor" ||
    (_step === "conductor" && (_phase === "phase0" || _phase === "phase1" || _phase === "phase2"));
  const phaseLabel = hasPhase3
    ? `Phase 3: Experimenting${doneCount > 0 ? ` (${doneCount} done)` : ""}`
    : isConductorBoundary
    ? `Conductor · ${_step || "steer"}`
    : _phase === "phase2"
    ? `Phase 2 · ${_step}`
    : _phase === "phase0"
    ? `Phase 0 · ${_step}`
    : lastPhase
    ? "Phase 1"
    : "Idle";

  // Derive status from most recent status event
  const lastStatus = [...events]
    .reverse()
    .find((e) => e.type === "status");

  const status = (lastStatus?.status as string) || "idle";
  const running = status === "starting" || status === "thinking" || status === "tool_executing";

  const badgeClass =
    status === "done" ? "done" :
    status === "error" ? "error" :
    running ? "running" :
    "";

  const badgeText =
    status === "idle" ? "Idle" :
    status === "starting" ? "Starting..." :
    status === "thinking" ? "Thinking..." :
    status === "tool_executing" ? (lastStatus?.detail as string || "Executing...") :
    status === "done" ? "Done" :
    status === "error" ? "Error" :
    status;

  return (
    <div className="topbar">
      <div className="topbar-left">
        <span className="topbar-title">Alpha Lab</span>
        <span className="phase-tag">{phaseLabel}</span>
        <span className={`status-badge ${badgeClass}`}>{badgeText}</span>
      </div>
      <div className="topbar-right">
        <div className={`status-dot ${connected ? "connected" : ""}`} title={connected ? "Connected" : "Disconnected"} />
      </div>
    </div>
  );
}
