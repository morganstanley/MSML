import { useCallback, useEffect, useState } from "react";
import { AgentEvent } from "./useWebSocket";

interface FileNode {
  name: string;
  path: string;
  type: "file" | "directory";
  children?: FileNode[];
}

interface FileTreeProps {
  events: AgentEvent[];
  selectedFile: string | null;
  onSelectFile: (path: string) => void;
}

const FILE_ICONS: Record<string, string> = {
  ".py": "py",
  ".md": "md",
  ".csv": "csv",
  ".json": "{}",
  ".yaml": "ym",
  ".yml": "ym",
  ".png": "img",
  ".jpg": "img",
  ".jpeg": "img",
  ".toml": "tm",
  ".txt": "tx",
};

function getIcon(node: FileNode): string {
  if (node.type === "directory") return "\u25B8";
  const ext = "." + node.name.split(".").pop()?.toLowerCase();
  return FILE_ICONS[ext] || "  ";
}

function TreeNode({
  node,
  selectedFile,
  onSelectFile,
  depth = 0,
}: {
  node: FileNode;
  selectedFile: string | null;
  onSelectFile: (path: string) => void;
  depth?: number;
}) {
  const [expanded, setExpanded] = useState(depth < 2);

  const handleClick = () => {
    if (node.type === "directory") {
      setExpanded(!expanded);
    } else {
      onSelectFile(node.path);
    }
  };

  const isSelected = node.path === selectedFile;
  const icon = node.type === "directory" ? (expanded ? "\u25BE" : "\u25B8") : getIcon(node);

  return (
    <>
      <div
        className={`tree-item ${isSelected ? "selected" : ""}`}
        style={{ paddingLeft: `${16 + depth * 16}px` }}
        onClick={handleClick}
      >
        <span className="tree-icon">{icon}</span>
        <span className="tree-name">{node.name}</span>
      </div>
      {node.type === "directory" && expanded && node.children && (
        <div className="tree-children" style={{ paddingLeft: 0 }}>
          {node.children.map((child) => (
            <TreeNode
              key={child.path}
              node={child}
              selectedFile={selectedFile}
              onSelectFile={onSelectFile}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
    </>
  );
}

export default function FileTree({ events, selectedFile, onSelectFile }: FileTreeProps) {
  const [tree, setTree] = useState<FileNode[]>([]);

  const fetchTree = useCallback(() => {
    fetch("/api/files")
      .then((r) => r.json())
      .then((data) => setTree(data.children || []))
      .catch(() => {});
  }, []);

  // Fetch on mount
  useEffect(() => {
    fetchTree();
  }, [fetchTree]);

  // Re-fetch on file_changed events
  useEffect(() => {
    const fileEvents = events.filter((e) => e.type === "file_changed");
    if (fileEvents.length > 0) {
      fetchTree();
    }
  }, [events, fetchTree]);

  return (
    <div className="sidebar">
      <div className="sidebar-header">Files</div>
      {tree.length === 0 ? (
        <div style={{ padding: "16px", color: "var(--text-muted)", fontSize: "13px" }}>
          No files yet
        </div>
      ) : (
        tree.map((node) => (
          <TreeNode
            key={node.path}
            node={node}
            selectedFile={selectedFile}
            onSelectFile={onSelectFile}
          />
        ))
      )}
    </div>
  );
}
