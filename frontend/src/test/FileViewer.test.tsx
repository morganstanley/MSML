import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import FileViewer from "../FileViewer";

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
});

afterEach(() => {
  vi.runAllTimers();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

function mockFetchText(content: string, name = "notes.md") {
  (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
    ok: true,
    json: () => Promise.resolve({ type: "text", content, name }),
  });
}

function mockFetchError(status = 404) {
  (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
    ok: false,
    status,
    json: () => Promise.resolve({}),
  });
}

describe("FileViewer", () => {
  it("renders placeholder when no file is selected", () => {
    render(<FileViewer selectedFile={null} />);
    expect(screen.getByText(/select a file/i)).toBeInTheDocument();
  });

  it("shows loading state while fetching", () => {
    // fetch stays pending
    (global.fetch as ReturnType<typeof vi.fn>).mockReturnValueOnce(
      new Promise(() => {}),
    );
    render(<FileViewer selectedFile="path/to/notes.md" />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it("renders text file content after fetch", async () => {
    mockFetchText("# Hello World", "notes.md");
    render(<FileViewer selectedFile="path/to/notes.md" />);
    await waitFor(() =>
      expect(screen.queryByText(/loading/i)).not.toBeInTheDocument(),
    );
    // Markdown renders "Hello World" as a heading
    expect(screen.getByText("Hello World")).toBeInTheDocument();
  });

  it("renders plain text for non-markdown files", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({ type: "text", content: "plain text body", name: "script.py" }),
    });
    render(<FileViewer selectedFile="path/to/script.py" />);
    await waitFor(() =>
      expect(screen.queryByText(/loading/i)).not.toBeInTheDocument(),
    );
    expect(screen.getByText("plain text body")).toBeInTheDocument();
  });

  it("displays error message on failed fetch", async () => {
    mockFetchError(404);
    render(<FileViewer selectedFile="missing.txt" />);
    await waitFor(() =>
      expect(screen.queryByText(/loading/i)).not.toBeInTheDocument(),
    );
    expect(screen.getByText(/error/i)).toBeInTheDocument();
  });

  it("shows the file path in the header", async () => {
    mockFetchText("content", "notes.md");
    render(<FileViewer selectedFile="workspace/notes.md" />);
    await waitFor(() =>
      expect(screen.queryByText(/loading/i)).not.toBeInTheDocument(),
    );
    expect(screen.getByText("workspace/notes.md")).toBeInTheDocument();
  });
});
