import { useEffect, useState } from "react";
import { useWebSocket } from "./useWebSocket";
import TopBar from "./TopBar";
import FileTree from "./FileTree";
import FileViewer from "./FileViewer";
import ConversationStream from "./ConversationStream";
import KanbanBoard from "./KanbanBoard";
import Leaderboard from "./Leaderboard";
import StatusReport from "./StatusReport";
import ChatPanel from "./ChatPanel";
import { useExperiments } from "./useExperiments";

function ResizeHandle({ onResize }: { onResize: (delta: number) => void }) {
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    let lastX = e.clientX;

    const onMouseMove = (ev: MouseEvent) => {
      const delta = ev.clientX - lastX;
      lastX = ev.clientX;
      onResize(delta);
    };

    const onMouseUp = () => {
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  };

  return <div className="resize-handle" onMouseDown={handleMouseDown} />;
}

export default function App() {
  const { events, connected, sendMessage } = useWebSocket();
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [sidebarWidth, setSidebarWidth] = useState(260);
  const [convWidth, setConvWidth] = useState(420);
  const [centerView, setCenterView] = useState<"files" | "board" | "status">("files");
  const { experiments, summary, refresh: refreshExperiments } = useExperiments(events);

  // Auto-switch to board view when Phase 3 events appear
  useEffect(() => {
    const hasPhase3 = events.some(
      (e) => e.type === "experiment" || e.type === "board_summary" ||
        (e.type === "phase" && e.phase === "phase3")
    );
    if (hasPhase3 && centerView === "files") {
      setCenterView("board");
    }
  }, [events, centerView]);

  const handleAnswer = (text: string) => {
    sendMessage({ type: "answer", text });
  };

  return (
    <div className="app">
      <TopBar
        connected={connected}
        events={events}
      />
      <div className="app-body">
        <div className="pane" style={{ width: sidebarWidth, minWidth: 150 }}>
          <FileTree
            events={events}
            selectedFile={selectedFile}
            onSelectFile={setSelectedFile}
          />
        </div>
        <ResizeHandle
          onResize={(delta) =>
            setSidebarWidth((w) => Math.max(150, Math.min(500, w + delta)))
          }
        />
        <div className="pane" style={{ flex: 1, minWidth: 200 }}>
          <div className="view-toggle">
            <button
              className={`view-toggle-btn ${centerView === "files" ? "active" : ""}`}
              onClick={() => setCenterView("files")}
            >
              Files
            </button>
            <button
              className={`view-toggle-btn ${centerView === "board" ? "active" : ""}`}
              onClick={() => setCenterView("board")}
            >
              Board
            </button>
            <button
              className={`view-toggle-btn ${centerView === "status" ? "active" : ""}`}
              onClick={() => setCenterView("status")}
            >
              Status
            </button>
          </div>
          {centerView === "files" ? (
            <FileViewer selectedFile={selectedFile} />
          ) : centerView === "board" ? (
            <div className="board-view">
              <div className="board-main">
                <KanbanBoard experiments={experiments} summary={summary} onRefresh={refreshExperiments} />
                <Leaderboard events={events} />
              </div>
              <ChatPanel />
            </div>
          ) : (
            <StatusReport events={events} sendMessage={sendMessage} />
          )}
        </div>
        <ResizeHandle
          onResize={(delta) =>
            setConvWidth((w) => Math.max(250, Math.min(800, w - delta)))
          }
        />
        <div className="pane" style={{ width: convWidth, minWidth: 250 }}>
          <ConversationStream
            events={events}
            onAnswer={handleAnswer}
          />
        </div>
      </div>
    </div>
  );
}
