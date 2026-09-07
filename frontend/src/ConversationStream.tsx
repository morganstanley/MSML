import { useEffect, useRef, useState } from "react";
import { AgentEvent } from "./useWebSocket";
import QuestionCard from "./QuestionCard";
import RunSummary from "./RunSummary";

interface Props {
  events: AgentEvent[];
  onAnswer: (text: string) => void;
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

/** Find the matching tool_call event for a tool_result by call_id */
function getToolCommand(event: AgentEvent, events: AgentEvent[]): string {
  const callId = event.call_id as string;
  const toolCall = events.find(
    (e) => e.type === "tool_call" && e.call_id === callId,
  );
  if (!toolCall) return "";

  const name = toolCall.name as string;
  let args: Record<string, unknown>;
  try {
    args = JSON.parse(toolCall.arguments as string);
  } catch {
    return toolCall.arguments as string;
  }

  if (name === "shell_exec") return `$ ${args.command}`;
  if (name === "view_image") return String(args.path);
  if (name === "ask_user") return String(args.question);
  if (name === "report_to_user") return String(args.summary || "");
  return JSON.stringify(args, null, 2);
}

function truncateOutput(output: string, maxLines = 30): string {
  const lines = output.split("\n");
  if (lines.length <= maxLines) return output;
  const half = Math.floor(maxLines / 2);
  return (
    lines.slice(0, half).join("\n") +
    `\n··· ${lines.length - maxLines} lines omitted ···\n` +
    lines.slice(-half).join("\n")
  );
}

export default function ConversationStream({
  events,
  onAnswer,
}: Props) {
  const bodyRef = useRef<HTMLDivElement>(null);
  const [now, setNow] = useState(Date.now());

  // Auto-scroll
  useEffect(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [events]);

  // Tick elapsed time
  useEffect(() => {
    const interval = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(interval);
  }, []);

  // Derive current status
  const lastStatus = [...events]
    .reverse()
    .find((e) => e.type === "status");
  const status = (lastStatus?.status as string) || "idle";
  const statusDetail = (lastStatus?.detail as string) || "";
  const isActive =
    status === "starting" ||
    status === "thinking" ||
    status === "tool_executing";

  // Counts
  const turnCount = events.filter((e) => e.type === "api_request").length;
  const actionCount = events.filter((e) => e.type === "tool_result").length;

  // Elapsed
  const firstEvent = events.find(
    (e) => e.type === "status" && e.status === "starting",
  );
  const elapsed = firstEvent
    ? Math.floor(now / 1000 - (firstEvent.timestamp as number))
    : 0;

  // Status display
  let statusText: string;
  let statusClass: string;
  if (status === "idle" || events.length === 0) {
    statusText = "Idle";
    statusClass = "";
  } else if (status === "starting") {
    statusText = "Starting...";
    statusClass = "active";
  } else if (status === "thinking") {
    statusText = "Thinking...";
    statusClass = "active";
  } else if (status === "tool_executing") {
    statusText = statusDetail || "Executing...";
    statusClass = "active";
  } else if (status === "done") {
    statusText = "Done";
    statusClass = "done";
  } else if (status === "error") {
    statusText = statusDetail ? `Error: ${statusDetail}` : "Error";
    statusClass = "error";
  } else {
    statusText = status;
    statusClass = "";
  }

  let actionNum = 0;

  return (
    <div className="conversation">
      {/* Status header */}
      <div className={`conv-status ${statusClass}`}>
        <div className="conv-status-left">
          <span className={`conv-status-dot ${statusClass}`} />
          <span className="conv-status-text">{statusText}</span>
        </div>
        {turnCount > 0 && (
          <div className="conv-status-right">
            {actionCount} actions · {formatElapsed(elapsed)}
          </div>
        )}
      </div>

      {/* Activity log */}
      <div className="conversation-body" ref={bodyRef}>
        {events.map((event, i) => {
          // Tool results: the main content
          if (event.type === "tool_result") {
            actionNum++;
            const name = event.name as string;
            const output = event.output as string;
            const command = getToolCommand(event, events);
            const imageB64 = event.image_base64 as string | undefined;
            const mediaType = event.media_type as string | undefined;

            return (
              <div
                key={i}
                className={`action-item ${name === "report_to_user" ? "action-report" : ""}`}
              >
                <div className="action-header">
                  <span className="action-num">{actionNum}</span>
                  <span className="action-name">{name}</span>
                </div>
                {command && (
                  <pre className="action-command">{command}</pre>
                )}
                <pre className="action-output">
                  {truncateOutput(output)}
                </pre>
                {imageB64 && mediaType ? (
                  <img
                    src={`data:${mediaType};base64,${imageB64}`}
                    alt={`${name} output`}
                    className="action-image"
                  />
                ) : null}
              </div>
            );
          }

          // Questions
          if (event.type === "question") {
            return (
              <div key={i} className="action-item">
                <QuestionCard
                  questionId={event.question_id as string}
                  question={event.question as string}
                  onAnswer={onAnswer}
                />
              </div>
            );
          }

          // Experiment status changes (Phase 3)
          if (event.type === "experiment") {
            const expName = event.name as string;
            const expStatus = event.status as string;
            const detail = event.detail as string;
            return (
              <div key={i} className="action-item experiment-event">
                <div className="action-header">
                  <span className="action-name">experiment</span>
                </div>
                <div className="experiment-event-line">
                  <span className="experiment-event-name">{expName}</span>
                  <span className="experiment-event-arrow">&rarr;</span>
                  <span className="experiment-event-status">{expStatus}</span>
                </div>
                {detail && (
                  <div className="experiment-event-detail">{detail}</div>
                )}
              </div>
            );
          }

          // Phase transitions
          if (event.type === "phase") {
            const phase = event.phase as string;
            const step = event.step as string;
            const phaseStatus = event.status as string;
            const detail = event.detail as string;
            if (phaseStatus === "starting") {
              return (
                <div key={i} className="phase-marker">
                  <span className="phase-marker-label">
                    {phase === "phase2" ? `Phase 2: ${step}` : phase}
                  </span>
                  {detail && <span className="phase-marker-detail">{detail}</span>}
                </div>
              );
            }
            return null;
          }

          // Errors
          if (event.type === "error") {
            return (
              <div key={i} className="action-error">
                {event.message as string}
              </div>
            );
          }

          // Status: only show context-switching status events
          if (event.type === "status") {
            const s = event.status as string;
            if (s === "done") {
              return (
                <div key={i}>
                  <RunSummary events={events} />
                </div>
              );
            }
            if (s === "error" && event.detail) {
              return (
                <div key={i} className="action-error">
                  {event.detail as string}
                </div>
              );
            }
          }

          return null;
        })}

        {/* Thinking indicator */}
        {isActive && status === "thinking" && (
          <div className="thinking-indicator">
            <span className="thinking-dot" />
            <span className="thinking-dot" />
            <span className="thinking-dot" />
          </div>
        )}

        {events.length === 0 && (
          <div className="conv-empty">
            {"Waiting for events..."}
          </div>
        )}
      </div>
    </div>
  );
}
