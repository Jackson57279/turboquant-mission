/**
 * TUI Module - Terminal User Interface for LLM Compress
 *
 * This module provides an interactive terminal interface using OpenTUI concepts.
 * It includes:
 * - Model browser with keyboard navigation
 * - Download screen with progress bar and ETA
 * - Error modal dialogs
 * - Help system
 */

import * as readline from "readline";
import { spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

// Model metadata interface
interface ModelInfo {
  id: string;
  size: string;
  quantized: boolean;
  downloadedAt?: string;
}

// Download progress state
interface DownloadProgress {
  modelId: string;
  percent: number;
  downloadedBytes: number;
  totalBytes: number;
  speed: number; // bytes per second
  eta: number; // seconds
  status: "downloading" | "complete" | "error" | "idle";
  error?: string;
}

// TUI State
interface TUIState {
  models: ModelInfo[];
  selectedIndex: number;
  screen: "browser" | "download" | "help" | "error" | "confirm_download";
  downloadProgress: DownloadProgress;
  errorMessage: string;
  downloadInput: string;
  isDownloading: boolean;
}

// Default cache directory
function getCacheDir(): string {
  return path.join(os.homedir(), ".cache", "llm-compress");
}

// Load models from cache directory
function loadCachedModels(): ModelInfo[] {
  const cacheDir = getCacheDir();
  const models: ModelInfo[] = [];

  if (!fs.existsSync(cacheDir)) {
    return models;
  }

  try {
    const entries = fs.readdirSync(cacheDir);
    for (const entry of entries) {
      const metadataPath = path.join(cacheDir, entry, "metadata.json");
      if (fs.existsSync(metadataPath)) {
        try {
          const metadata = JSON.parse(fs.readFileSync(metadataPath, "utf8"));
          models.push({
            id: metadata.id || entry,
            size: metadata.size || "Unknown",
            quantized: metadata.quantized || false,
            downloadedAt: metadata.downloadedAt,
          });
        } catch {
          // Invalid metadata, skip
        }
      }
    }
  } catch {
    // Error reading cache, return empty
  }

  return models;
}

// Clear screen
function clearScreen(): void {
  process.stdout.write("\x1b[2J\x1b[H");
}

// Format bytes to human readable
function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

// Format ETA
function formatETA(seconds: number): string {
  if (seconds < 0 || !isFinite(seconds)) return "calculating...";
  if (seconds < 60) return `${Math.ceil(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.ceil(seconds % 60);
  return `${mins}m ${secs}s`;
}

// Render progress bar
function renderProgressBar(percent: number, width: number = 40): string {
  const filled = Math.floor((percent / 100) * width);
  const empty = width - filled;
  return "█".repeat(filled) + "░".repeat(empty);
}

// Draw the model browser screen
function drawBrowser(state: TUIState): void {
  clearScreen();

  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log("║           LLM Compress - Model Browser                   ║");
  console.log("╠══════════════════════════════════════════════════════════╣");

  if (state.models.length === 0) {
    console.log("║                                                          ║");
    console.log("║  No models in cache                                      ║");
    console.log("║                                                          ║");
    console.log("║  Press 'd' to download a model                          ║");
  } else {
    console.log("║  Use ↑/↓ to navigate, Enter to select                    ║");
    console.log("║                                                          ║");

    // Show models (limit to 10 for screen space)
    const displayModels = state.models.slice(0, 10);
    displayModels.forEach((model, index) => {
      const isSelected = index === state.selectedIndex;
      const prefix = isSelected ? "▶ " : "  ";
      const quantized = model.quantized ? " [Q]" : "";
      const line = `${prefix}${model.id} (${model.size})${quantized}`.substring(0, 52);
      console.log(`║${isSelected ? "\x1b[7m" : ""}${line.padEnd(58)}\x1b[0m║`);
    });

    // Show scroll indicator if more models
    if (state.models.length > 10) {
      console.log(`║  ... and ${state.models.length - 10} more models`.padEnd(58) + "║");
    }

    console.log("║                                                          ║");
    console.log("║  Press 'd' to download a new model                     ║");
  }

  console.log("║  Press '?' for help                                     ║");
  console.log("║  Press 'q' to quit                                      ║");
  console.log("║                                                          ║");
  console.log("╚══════════════════════════════════════════════════════════╝");
}

// Draw the download screen with progress
function drawDownload(state: TUIState): void {
  clearScreen();

  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log("║           Download Model                                 ║");
  console.log("╠══════════════════════════════════════════════════════════╣");
  console.log("║                                                          ║");

  if (state.downloadProgress.status === "idle") {
    console.log("║  Enter model ID (e.g., microsoft/DialoGPT-medium):       ║");
    console.log("║                                                          ║");
    console.log(`║  > ${state.downloadInput.padEnd(53)}║`);
    console.log("║                                                          ║");
    console.log("║  Press Enter to start download                          ║");
    console.log("║  Press Esc to cancel                                    ║");
  } else if (state.downloadProgress.status === "downloading") {
    const progress = state.downloadProgress;
    const percent = progress.percent.toFixed(1);
    const bar = renderProgressBar(progress.percent);

    console.log(`║  Model: ${progress.modelId.padEnd(48)}║`);
    console.log("║                                                          ║");
    console.log(`║  ${bar} ${percent.padStart(6)}% ║`);
    console.log("║                                                          ║");
    console.log(`║  Downloaded: ${formatBytes(progress.downloadedBytes).padEnd(15)} Total: ${formatBytes(progress.totalBytes).padEnd(15)}║`);
    console.log(`║  Speed: ${(formatBytes(progress.speed) + "/s").padEnd(20)} ETA: ${formatETA(progress.eta).padEnd(18)}║`);
    console.log("║                                                          ║");
    console.log("║  Press 'q' to cancel download                           ║");
  } else if (state.downloadProgress.status === "complete") {
    console.log("║  ✓ Download Complete!                                   ║");
    console.log("║                                                          ║");
    console.log(`║  Model: ${state.downloadProgress.modelId.padEnd(48)}║`);
    console.log("║                                                          ║");
    console.log("║  Press any key to return to browser                     ║");
  }

  console.log("║                                                          ║");
  console.log("╚══════════════════════════════════════════════════════════╝");
}

// Draw the error modal
function drawErrorModal(state: TUIState): void {
  // First draw the background screen
  if (state.screen === "browser") {
    drawBrowser(state);
  } else {
    drawDownload(state);
  }

  // Draw modal overlay
  const lines = state.errorMessage.split("\n");
  const maxWidth = Math.max(...lines.map(l => l.length), 20);
  const modalWidth = Math.min(maxWidth + 4, 50);
  const modalHeight = lines.length + 4;

  // Center the modal
  const startRow = 5;
  const startCol = Math.floor((60 - modalWidth) / 2);

  // Move cursor and draw modal box
  process.stdout.write(`\x1b[${startRow};${startCol}H`);
  console.log("╔" + "═".repeat(modalWidth - 2) + "╗");

  for (let i = 0; i < modalHeight - 2; i++) {
    process.stdout.write(`\x1b[${startRow + i + 1};${startCol}H`);
    if (i === 0) {
      console.log("║" + " ERROR ".padStart(Math.floor((modalWidth - 2) / 2)).padEnd(modalWidth - 2) + "║");
    } else if (i <= lines.length) {
      const line = lines[i - 1] || "";
      console.log("║ " + line.substring(0, modalWidth - 4).padEnd(modalWidth - 4) + " ║");
    } else {
      console.log("║" + " ".repeat(modalWidth - 2) + "║");
    }
  }

  process.stdout.write(`\x1b[${startRow + modalHeight - 1};${startCol}H`);
  console.log("╠" + "═".repeat(modalWidth - 2) + "╣");
  process.stdout.write(`\x1b[${startRow + modalHeight};${startCol}H`);
  console.log("║" + " Press any key to dismiss ".padStart(Math.floor((modalWidth - 2 + 26) / 2)).padEnd(modalWidth - 2) + "║");
  process.stdout.write(`\x1b[${startRow + modalHeight + 1};${startCol}H`);
  console.log("╚" + "═".repeat(modalWidth - 2) + "╝");
}

// Draw the help screen
function drawHelp(): void {
  clearScreen();

  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log("║           Keyboard Shortcuts                             ║");
  console.log("╠══════════════════════════════════════════════════════════╣");
  console.log("║                                                          ║");
  console.log("║  Model Browser:                                          ║");
  console.log("║    ↑ / ↓     Navigate model list                        ║");
  console.log("║    Enter     Select model                               ║");
  console.log("║    d         Download new model                        ║");
  console.log("║                                                          ║");
  console.log("║  Download Screen:                                        ║");
  console.log("║    Enter     Start download                             ║");
  console.log("║    Esc       Cancel / Go back                          ║");
  console.log("║    q         Quit during download                      ║");
  console.log("║                                                          ║");
  console.log("║  General:                                                ║");
  console.log("║    ?         Show this help                            ║");
  console.log("║    q         Quit (with confirmation if active)        ║");
  console.log("║    Ctrl+C    Force quit                                 ║");
  console.log("║                                                          ║");
  console.log("╚══════════════════════════════════════════════════════════╝");
}

// Parse download progress from Python CLI output
function parseProgress(line: string): Partial<DownloadProgress> | null {
  // Try to match patterns like:
  // "Downloading: 45.2% (12.3MB / 27.2MB) [23.4KB/s] ETA: 1m 23s"
  // "Downloaded 12.3MB of 27.2MB at 23.4KB/s"

  const percentMatch = line.match(/(\d+\.?\d*)%/);
  const sizeMatch = line.match(/(\d+\.?\d*)\s*(B|KB|MB|GB|TB).*?(\d+\.?\d*)\s*(B|KB|MB|GB|TB)/i);
  const speedMatch = line.match(/(\d+\.?\d*)\s*(B|KB|MB|GB|TB)/i);
  const etaMatch = line.match(/ETA:\s*(.+)/i);

  const progress: Partial<DownloadProgress> = {};

  if (percentMatch) {
    progress.percent = parseFloat(percentMatch[1]);
  }

  return progress;
}

// Start a model download
function startDownload(
  modelId: string,
  onProgress: (progress: DownloadProgress) => void,
  onComplete: () => void,
  onError: (error: string) => void
): () => void {
  let totalBytes = 0;
  let startTime = Date.now();

  const progress: DownloadProgress = {
    modelId,
    percent: 0,
    downloadedBytes: 0,
    totalBytes: 0,
    speed: 0,
    eta: 0,
    status: "downloading",
  };

  // Spawn Python CLI download command
  const proc = spawn(
    "python",
    ["-m", "llm_compress", "download", modelId, "--cache-dir", getCacheDir()],
    { cwd: "/home/dih/turboquant-mission/python" }
  );

  let stderr = "";

  proc.stdout.on("data", (data: Buffer) => {
    const lines = data.toString().split("\n");
    for (const line of lines) {
      // Parse progress from output
      const parsed = parseProgress(line);
      if (parsed) {
        Object.assign(progress, parsed);
      }

      // Update speed and ETA calculation
      const now = Date.now();
      const elapsed = (now - startTime) / 1000;
      if (elapsed > 0 && progress.downloadedBytes > 0) {
        progress.speed = progress.downloadedBytes / elapsed;
        if (progress.totalBytes > 0 && progress.speed > 0) {
          const remaining = progress.totalBytes - progress.downloadedBytes;
          progress.eta = remaining / progress.speed;
        }
      }

      onProgress({ ...progress });
    }
  });

  proc.stderr.on("data", (data: Buffer) => {
    stderr += data.toString();
  });

  proc.on("close", (code: number) => {
    if (code === 0) {
      progress.status = "complete";
      progress.percent = 100;
      onProgress({ ...progress });
      onComplete();
    } else {
      progress.status = "error";
      onError(stderr || `Download failed with exit code ${code}`);
    }
  });

  // Return cancel function
  return () => {
    proc.kill("SIGTERM");
  };
}

// Main TUI launch function
export function launchTUI(): Promise<void> {
  return new Promise((resolve) => {
    const state: TUIState = {
      models: loadCachedModels(),
      selectedIndex: 0,
      screen: "browser",
      downloadProgress: {
        modelId: "",
        percent: 0,
        downloadedBytes: 0,
        totalBytes: 0,
        speed: 0,
        eta: 0,
        status: "idle",
      },
      errorMessage: "",
      downloadInput: "",
      isDownloading: false,
    };

    let cancelDownload: (() => void) | null = null;

    // Set up stdin for key press
    const stdin = process.stdin;
    stdin.setRawMode(true);
    stdin.resume();
    stdin.setEncoding("utf8");

    // Initial draw
    drawBrowser(state);

    const handleInput = (key: string) => {
      // Ctrl+C always exits
      if (key === "\u0003") {
        if (cancelDownload) {
          cancelDownload();
        }
        stdin.setRawMode(false);
        stdin.pause();
        resolve();
        return;
      }

      // Handle error modal first
      if (state.screen === "error") {
        state.screen = state.models.length > 0 ? "browser" : "browser";
        state.errorMessage = "";
        drawBrowser(state);
        return;
      }

      // Handle help screen
      if (state.screen === "help") {
        if (key === "q" || key === "\u001b" || key === "?") {
          state.screen = "browser";
          drawBrowser(state);
        }
        return;
      }

      // Handle download complete screen
      if (state.screen === "download" && state.downloadProgress.status === "complete") {
        state.models = loadCachedModels();
        state.screen = "browser";
        state.downloadProgress = {
          modelId: "",
          percent: 0,
          downloadedBytes: 0,
          totalBytes: 0,
          speed: 0,
          eta: 0,
          status: "idle",
        };
        state.downloadInput = "";
        state.isDownloading = false;
        cancelDownload = null;
        drawBrowser(state);
        return;
      }

      // Handle download screen
      if (state.screen === "download") {
        if (state.downloadProgress.status === "idle") {
          if (key === "\r" || key === "\n") {
            // Start download
            if (state.downloadInput.trim()) {
              state.isDownloading = true;
              state.downloadProgress.modelId = state.downloadInput.trim();
              state.downloadProgress.status = "downloading";
              drawDownload(state);

              cancelDownload = startDownload(
                state.downloadInput.trim(),
                (progress) => {
                  state.downloadProgress = progress;
                  if (state.screen === "download") {
                    drawDownload(state);
                  }
                },
                () => {
                  state.isDownloading = false;
                  drawDownload(state);
                },
                (error) => {
                  state.isDownloading = false;
                  state.errorMessage = error;
                  state.screen = "error";
                  drawErrorModal(state);
                }
              );
            }
          } else if (key === "\u001b" || key === "\u001b\u001b") {
            // Escape - go back
            state.screen = "browser";
            state.downloadInput = "";
            drawBrowser(state);
          } else if (key === "\u007f") {
            // Backspace
            state.downloadInput = state.downloadInput.slice(0, -1);
            drawDownload(state);
          } else if (key >= " " && key <= "~") {
            // Printable characters
            if (state.downloadInput.length < 50) {
              state.downloadInput += key;
              drawDownload(state);
            }
          }
        } else if (state.downloadProgress.status === "downloading") {
          if (key === "q") {
            // Cancel download
            if (cancelDownload) {
              cancelDownload();
            }
            state.isDownloading = false;
            state.downloadProgress.status = "idle";
            state.screen = "browser";
            drawBrowser(state);
          }
        }
        return;
      }

      // Handle browser screen
      if (state.screen === "browser") {
        switch (key) {
          case "q":
            if (state.isDownloading) {
              state.errorMessage = "Download in progress. Press 'q' again to force quit.";
              state.screen = "error";
              drawErrorModal(state);
            } else {
              stdin.setRawMode(false);
              stdin.pause();
              resolve();
            }
            break;

          case "d":
            state.screen = "download";
            drawDownload(state);
            break;

          case "?":
            state.screen = "help";
            drawHelp();
            break;

          case "\u001b[A": // Up arrow
            if (state.models.length > 0) {
              state.selectedIndex = Math.max(0, state.selectedIndex - 1);
              drawBrowser(state);
            }
            break;

          case "\u001b[B": // Down arrow
            if (state.models.length > 0) {
              state.selectedIndex = Math.min(state.models.length - 1, state.selectedIndex + 1);
              drawBrowser(state);
            }
            break;

          case "\r":
          case "\n":
            // Select model - could navigate to model details
            // For now, just show a simple message
            if (state.models.length > 0 && state.selectedIndex < state.models.length) {
              const model = state.models[state.selectedIndex];
              state.errorMessage = `Selected model: ${model.id}\nSize: ${model.size}\nQuantized: ${model.quantized ? "Yes" : "No"}`;
              state.screen = "error"; // Using error modal as info dialog
              drawErrorModal(state);
            }
            break;
        }
      }
    };

    stdin.on("data", handleInput);
  });
}
