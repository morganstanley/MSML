import { useEffect, useState } from "react";
import Markdown from "react-markdown";

interface FileViewerProps {
  selectedFile: string | null;
}

interface FileData {
  type: "text" | "image";
  content?: string;
  data?: string;
  media_type?: string;
  name: string;
}

function CsvTable({ content }: { content: string }) {
  const lines = content.split("\n").filter((l) => l.trim());
  if (lines.length === 0) return <pre>{content}</pre>;

  // Simple CSV parse (handles basic cases)
  const parseRow = (line: string) => {
    const cells: string[] = [];
    let current = "";
    let inQuotes = false;
    for (const ch of line) {
      if (ch === '"') {
        inQuotes = !inQuotes;
      } else if (ch === "," && !inQuotes) {
        cells.push(current.trim());
        current = "";
      } else {
        current += ch;
      }
    }
    cells.push(current.trim());
    return cells;
  };

  const headers = parseRow(lines[0]);
  const rows = lines.slice(1, 101).map(parseRow); // First 100 rows

  return (
    <div style={{ overflowX: "auto" }}>
      <table>
        <thead>
          <tr>
            {headers.map((h, i) => (
              <th key={i}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri}>
              {row.map((cell, ci) => (
                <td key={ci}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {lines.length > 101 && (
        <div style={{ padding: "8px", color: "var(--text-muted)", fontSize: "12px" }}>
          Showing first 100 of {lines.length - 1} rows
        </div>
      )}
    </div>
  );
}

export default function FileViewer({ selectedFile }: FileViewerProps) {
  const [fileData, setFileData] = useState<FileData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedFile) {
      setFileData(null);
      return;
    }

    setLoading(true);
    setError(null);

    const encodedFile = encodeURIComponent(selectedFile);

    fetch(`/api/file/${encodedFile}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        setFileData(data);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, [selectedFile]);

  if (!selectedFile) {
    return (
      <div className="viewer">
        <div className="viewer-empty">Select a file to view</div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="viewer">
        <div className="viewer-header">{selectedFile}</div>
        <div className="viewer-empty">Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="viewer">
        <div className="viewer-header">{selectedFile}</div>
        <div className="viewer-empty" style={{ color: "var(--red)" }}>Error: {error}</div>
      </div>
    );
  }

  if (!fileData) return null;

  const ext = selectedFile.split(".").pop()?.toLowerCase();

  return (
    <div className="viewer">
      <div className="viewer-header">{selectedFile}</div>
      <div className="viewer-content">
        {fileData.type === "image" ? (
          <img src={`data:${fileData.media_type};base64,${fileData.data}`} alt={fileData.name} />
        ) : ext === "md" ? (
          <div className="event-text">
            <Markdown>{fileData.content || ""}</Markdown>
          </div>
        ) : ext === "csv" ? (
          <CsvTable content={fileData.content || ""} />
        ) : (
          <pre>{fileData.content}</pre>
        )}
      </div>
    </div>
  );
}
