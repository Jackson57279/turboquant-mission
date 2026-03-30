/**
 * TUI Module - Terminal User Interface for LLM Compress
 *
 * This module provides an interactive terminal interface using OpenTUI concepts.
 * It includes:
 * - Model browser with keyboard navigation
 * - Download screen with progress bar and ETA
 * - Error modal dialogs
 * - Help system
 * - Server control panel for start/stop operations
 */

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

// Server control state
interface ServerState {
  isRunning: boolean;
  port: number;
  host: string;
  backend: "vllm" | "llama-cpp";
  modelId: string | null;
  status: "stopped" | "starting" | "running" | "error";
  error?: string;
  pid?: number;
}

// TUI State
interface TUIState {
  models: ModelInfo[];
  selectedIndex: number;
  screen: "browser" | "download" | "help" | "error" | "confirm_download" | "quantize" | "confirm_quantize" | "server_control" | "chat";
  downloadProgress: DownloadProgress;
  quantizeProgress: QuantizeProgress;
  serverState: ServerState;
  errorMessage: string;
  downloadInput: string;
  isDownloading: boolean;
  selectedModelForQuantize: ModelInfo | null;
  quantizeOptions: QuantizeOptions;
}

// Quantization options
interface QuantizeOptions {
  bits: 4 | 8;
  kvCache: boolean;
}

// Quantization progress state
interface QuantizeProgress {
  modelId: string;
  percent: number;
  layer: number;
  totalLayers: number;
  layerName: string;
  eta: number; // seconds
  status: "quantizing" | "complete" | "error" | "idle";
  error?: string;
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

// Check if server is running
async function checkServerStatus(port: number): Promise<boolean> {
  try {
    const response = await fetch(`http://127.0.0.1:${port}/health`, {
      method: "GET",
      signal: AbortSignal.timeout(1000),
    });
    return response.ok;
  } catch {
    return false;
  }
}

// Start server process
function startServer(
  modelId: string,
  port: number,
  host: string,
  backend: "vllm" | "llama-cpp",
  onStatusChange: (state: Partial<ServerState>) => void,
  onError: (error: string) => void
): () => void {
  const args = [
    "-m", "llm_compress",
    "serve",
    modelId,
    "--port", port.toString(),
    "--host", host,
    "--backend", backend,
  ];

  const proc = spawn("python", args, {
    cwd: "/home/dih/turboquant-mission/python",
    detached: true,
    stdio: ["ignore", "pipe", "pipe"],
  });

  let stderr = "";

  proc.stdout?.on("data", (data: Buffer) => {
    const output = data.toString();
    // Check for server ready message
    if (output.includes("Starting API server") || output.includes("Server ready")) {
      onStatusChange({ isRunning: true, status: "running" });
    }
  });

  proc.stderr?.on("data", (data: Buffer) => {
    stderr += data.toString();
  });

  proc.on("error", (err: Error) => {
    onError(`Failed to start server: ${err.message}`);
  });

  proc.on("close", (code: number) => {
    if (code !== 0 && code !== null) {
      onStatusChange({ isRunning: false, status: "error", error: stderr || `Server exited with code ${code}` });
    } else {
      onStatusChange({ isRunning: false, status: "stopped" });
    }
  });

  // Initial status
  onStatusChange({ isRunning: true, status: "starting", pid: proc.pid });

  // Poll for server readiness
  const pollInterval = setInterval(async () => {
    const isRunning = await checkServerStatus(port);
    if (isRunning) {
      clearInterval(pollInterval);
      onStatusChange({ isRunning: true, status: "running" });
    }
  }, 500);

  // Timeout after 30 seconds
  setTimeout(() => {
    clearInterval(pollInterval);
  }, 30000);

  // Return stop function
  return () => {
    clearInterval(pollInterval);
    if (proc.pid) {
      try {
        process.kill(-proc.pid, "SIGTERM"); // Kill process group
      } catch {
        // Process may have already exited
      }
    }
  };
}

// Stop server process
function stopServer(serverState: ServerState, onStopped: () => void): void {
  if (serverState.pid) {
    try {
      process.kill(-serverState.pid, "SIGTERM");
    } catch {
      // Process may have already exited
    }
  }
  onStopped();
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
    console.log("║  Press 't' to quantize selected model                  ║");
    console.log("║  Press 's' for server control                          ║");
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

// Draw the quantization configuration screen
function drawQuantizeConfig(state: TUIState): void {
  clearScreen();

  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log("║           Quantization Configuration                     ║");
  console.log("╠══════════════════════════════════════════════════════════╣");
  console.log("║                                                          ║");

  if (state.selectedModelForQuantize) {
    console.log(`║  Model: ${state.selectedModelForQuantize.id.substring(0, 50).padEnd(48)}║`);
    console.log(`║  Size: ${state.selectedModelForQuantize.size.padEnd(49)}║`);
    console.log("║                                                          ║");
  }

  console.log("║  Bit Width:                                              ║");
  const bits4Selected = state.quantizeOptions.bits === 4;
  const bits8Selected = state.quantizeOptions.bits === 8;
  console.log(`║    ${bits4Selected ? "▶" : " "} 4-bit (4x compression, ~99% accuracy)           ║`);
  console.log(`║    ${bits8Selected ? "▶" : " "} 8-bit (2x compression, ~99.5% accuracy)         ║`);
  console.log("║                                                          ║");

  console.log("║  KV Cache Quantization:                                ║");
  const kvEnabled = state.quantizeOptions.kvCache;
  console.log(`║    ${kvEnabled ? "▶" : " "} Enable KV cache compression (3-bit keys, 2-bit values) ║`);
  console.log("║                                                          ║");

  console.log("║                                                          ║");
  console.log("║  Press ←/→ to change bit width                        ║");
  console.log("║  Press 'k' to toggle KV cache                          ║");
  console.log("║  Press Enter to start quantization                    ║");
  console.log("║  Press Esc to cancel                                   ║");
  console.log("║                                                          ║");
  console.log("╚══════════════════════════════════════════════════════════╝");
}

// Draw the server control screen
function drawServerControl(state: TUIState): void {
  clearScreen();

  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log("║           Server Control Panel                           ║");
  console.log("╠══════════════════════════════════════════════════════════╣");
  console.log("║                                                          ║");

  const serverState = state.serverState;

  if (serverState.status === "stopped" || serverState.status === "error") {
    console.log("║  Server Status: Stopped                                 ║");
    console.log("║                                                          ║");
    console.log("║  Configuration:                                          ║");
    console.log("║                                                          ║");

    // Port selection
    console.log(`║    Port: ${serverState.port.toString().padEnd(49)}║`);
    console.log("║    Use ↑/↓ to adjust port                               ║");
    console.log("║                                                          ║");

    // Host selection
    console.log(`║    Host: ${serverState.host.padEnd(49)}║`);
    console.log("║    Press 'h' to toggle host (127.0.0.1 / 0.0.0.0)       ║");
    console.log("║                                                          ║");

    // Backend selection
    const vllmSelected = serverState.backend === "vllm";
    const llamaSelected = serverState.backend === "llama-cpp";
    console.log("║    Backend:                                              ║");
    console.log(`║      ${vllmSelected ? "▶" : " "} vLLM (high-throughput GPU)                  ║`);
    console.log(`║      ${llamaSelected ? "▶" : " "} llama.cpp (broad hardware support)           ║`);
    console.log("║      Press ←/→ to change backend                       ║");
    console.log("║                                                          ║");

    if (serverState.error) {
      console.log("║                                                          ║");
      console.log(`║  Error: ${serverState.error.substring(0, 50).padEnd(49)}║`);
    }

    console.log("║                                                          ║");
    console.log("║  Press Enter to START server                            ║");
    console.log("║  Press Esc to return to browser                         ║");
  } else if (serverState.status === "starting") {
    console.log("║  Server Status: Starting...                             ║");
    console.log("║                                                          ║");
    console.log(`║  Port: ${serverState.port.toString().padEnd(50)}║`);
    console.log(`║  Backend: ${serverState.backend.padEnd(47)}║`);
    console.log("║                                                          ║");
    console.log("║  Initializing...                                        ║");
    console.log("║                                                          ║");
    console.log("║  Press 'q' to cancel                                    ║");
  } else if (serverState.status === "running") {
    console.log("║  Server Status: ✓ RUNNING                               ║");
    console.log("║                                                          ║");
    console.log(`║  Port: ${serverState.port.toString().padEnd(50)}║`);
    console.log(`║  Backend: ${serverState.backend.padEnd(47)}║`);
    console.log("║                                                          ║");
    console.log(`║  Health: http://127.0.0.1:${serverState.port}/health`.padEnd(58) + "║");
    console.log(`║  API: http://127.0.0.1:${serverState.port}/v1/chat/completions`.padEnd(58) + "║");
    console.log("║                                                          ║");
    console.log("║  Press 'c' to open chat interface                       ║");
    console.log("║  Press Enter to STOP server                             ║");
  }

  console.log("║                                                          ║");
  console.log("╚══════════════════════════════════════════════════════════╝");
}

// Draw the chat screen (placeholder)
function drawChat(_state: TUIState): void {
  clearScreen();

  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log(

  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log("║           Chat Interface                                 ║");
  console.log("╠══════════════════════════════════════════════════════════╣");
  console.log("║                                                          ║");
  console.log("║  Chat interface coming soon!                             ║");
  console.log("║                                                          ║");
  console.log("║  Press Esc to return to server control                   ║");
  console.log("║                                                          ║");
  console.log("╚══════════════════════════════════════════════════════════╝");
}

// Draw the quantization progress screen
function drawQuantizeProgress(state: TUIState): void {
  clearScreen();

  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log("║           Quantizing Model                               ║");
  console.log("╠══════════════════════════════════════════════════════════╣");
  console.log("║                                                          ║");

  const progress = state.quantizeProgress;

  if (progress.status === "idle") {
    console.log("║  Preparing quantization...                               ║");
  } else if (progress.status === "quantizing") {
    const percent = progress.percent.toFixed(1);
    const bar = renderProgressBar(progress.percent, 40);

    console.log(`║  Model: ${progress.modelId.substring(0, 50).padEnd(48)}║`);
    console.log("║                                                          ║");
    console.log(`║  ${bar} ${percent.padStart(6)}% ║`);
    console.log("║                                                          ║");
    console.log(`║  Layer ${progress.layer} of ${progress.totalLayers}`.padEnd(58) + "║");
    if (progress.layerName) {
      console.log(`║  ${progress.layerName.substring(0, 56).padEnd(56)}║`);
    }
    console.log("║                                                          ║");
    console.log(`║  ETA: ${formatETA(progress.eta).padEnd(52)}║`);
    console.log("║                                                          ║");
    console.log("║  Press 'q' to cancel quantization                       ║");
  } else if (progress.status === "complete") {
    console.log("║  ✓ Quantization Complete!                                 ║");
    console.log("║                                                          ║");
    console.log(`║  Model: ${progress.modelId.substring(0, 50).padEnd(48)}║`);
    console.log("║                                                          ║");
    console.log(`║  Quantized to ${state.quantizeOptions.bits}-bit`.padEnd(56) + "║");
    if (state.quantizeOptions.kvCache) {
      console.log("║  KV cache compression enabled                            ║");
    }
    console.log("║                                                          ║");
    console.log("║  Press any key to return to browser                     ║");
  }

  console.log("║                                                          ║");
  console.log("╚══════════════════════════════════════════════════════════╝");
}

// Parse quantization progress from Python CLI output
function parseQuantizeProgress(line: string): Partial<QuantizeProgress> | null {
  // Try to match patterns like:
  // "Quantizing layer 5/32: model.layers.4.self_attn"
  // "Progress: 45.2%"
  // "ETA: 1m 23s"

  const layerMatch = line.match(/layer\s+(\d+)\/(\d+)[\s:]*(.*)/i);
  const percentMatch = line.match(/(\d+\.?\d*)%/);

  const progress: Partial<QuantizeProgress> = {};
  let found = false;

  if (layerMatch) {
    progress.layer = parseInt(layerMatch[1], 10);
    progress.totalLayers = parseInt(layerMatch[2], 10);
    progress.layerName = layerMatch[3]?.trim() || "";
    if (progress.totalLayers && progress.totalLayers > 0) {
      progress.percent = ((progress.layer || 0) / progress.totalLayers) * 100;
    }
    found = true;
  }

  if (percentMatch) {
    progress.percent = parseFloat(percentMatch[1]);
    found = true;
  }

  return found ? progress : null;
}

// Start model quantization
function startQuantization(
  modelId: string,
  options: QuantizeOptions,
  onProgress: (progress: QuantizeProgress) => void,
  onComplete: () => void,
  onError: (error: string) => void
): () => void {
  let startTime = Date.now();

  const progress: QuantizeProgress = {
    modelId,
    percent: 0,
    layer: 0,
    totalLayers: 32, // Approximate, will be updated from CLI output
    layerName: "",
    eta: 0,
    status: "quantizing",
  };

  // Build CLI arguments
  const args = [
    "-m", "llm_compress",
    "quantize",
    modelId,
    "--bits", options.bits.toString(),
    "--cache-dir", getCacheDir(),
  ];

  if (options.kvCache) {
    args.push("--kv-cache");
  }

  // Spawn Python CLI quantization command
  const proc = spawn("python", args, { cwd: "/home/dih/turboquant-mission/python" });

  let stderr = "";

  proc.stdout.on("data", (data: Buffer) => {
    const lines = data.toString().split("\n");
    for (const line of lines) {
      // Parse progress from output
      const parsed = parseQuantizeProgress(line);
      if (parsed) {
        Object.assign(progress, parsed);
      }

      // Update ETA calculation
      const now = Date.now();
      const elapsed = (now - startTime) / 1000;
      if (elapsed > 0 && progress.percent > 0) {
        const totalEstimated = elapsed / (progress.percent / 100);
        const remaining = totalEstimated - elapsed;
        progress.eta = Math.max(0, remaining);
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
      progress.layer = progress.totalLayers;
      onProgress({ ...progress });
      onComplete();
    } else {
      progress.status = "error";
      onError(stderr || `Quantization failed with exit code ${code}`);
    }
  });

  // Return cancel function
  return () => {
    proc.kill("SIGTERM");
  };
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
  console.log("║    t         Quantize selected model                   ║");
  console.log("║    s         Server control panel                      ║");
  console.log("║                                                          ║");
  console.log("║  Download Screen:                                        ║");
  console.log("║    Enter     Start download                             ║");
  console.log("║    Esc       Cancel / Go back                          ║");
  console.log("║    q         Quit during download                      ║");
  console.log("║                                                          ║");
  console.log("║  Quantization Screen:                                    ║");
  console.log("║    ← / →     Change bit width (4-bit / 8-bit)         ║");
  console.log("║    k         Toggle KV cache quantization              ║");
  console.log("║    Enter     Start quantization                        ║");
  console.log("║    Esc       Cancel / Go back                          ║");
  console.log("║    q         Cancel during quantization                ║");
  console.log("║                                                          ║");
  console.log("║  Server Control:                                         ║");
  console.log("║    ↑ / ↓     Adjust port number                        ║");
  console.log("║    h         Toggle host (127.0.0.1 / 0.0.0.0)         ║");
  console.log("║    ← / →     Change backend (vLLM / llama.cpp)        ║");
  console.log("║    Enter     Start / Stop server                       ║");
  console.log("║    c         Open chat (when running)                  ║");
  console.log("║    Esc       Go back to browser                        ║");
  console.log("║    q         Cancel while starting                     ║");
  console.log("║                                                          ║");
  console.log("║  General:                                                ║");
  console.log("║    ?         Show this help                            ║");
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
      quantizeProgress: {
        modelId: "",
        percent: 0,
        layer: 0,
        totalLayers: 32,
        layerName: "",
        eta: 0,
        status: "idle",
      },
      serverState: {
        isRunning: false,
        port: 3200,
        host: "127.0.0.1",
        backend: "vllm",
        modelId: null,
        status: "stopped",
      },
      errorMessage: "",
      downloadInput: "",
      isDownloading: false,
      selectedModelForQuantize: null,
      quantizeOptions: {
        bits: 4,
        kvCache: false,
      },
    };

    let cancelDownload: (() => void) | null = null;
    let cancelQuantize: (() => void) | null = null;
    let cancelServer: (() => void) | null = null;

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
        if (cancelServer) {
          cancelServer();
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

      // Handle chat screen
      if (state.screen === "chat") {
        if (key === "\u001b" || key === "\u001b\u001b") {
          state.screen = "server_control";
          drawServerControl(state);
        }
        return;
      }

      // Handle server control screen
      if (state.screen === "server_control") {
        const serverState = state.serverState;

        if (serverState.status === "stopped" || serverState.status === "error") {
          // Config mode
          if (key === "\r" || key === "\n") {
            // Start server with selected model
            if (state.models.length > 0 && state.selectedIndex < state.models.length) {
              const model = state.models[state.selectedIndex];
              state.serverState.modelId = model.id;
              state.serverState.status = "starting";
              drawServerControl(state);

              cancelServer = startServer(
                model.id,
                state.serverState.port,
                state.serverState.host,
                state.serverState.backend,
                (newState) => {
                  Object.assign(state.serverState, newState);
                  if (state.screen === "server_control") {
                    drawServerControl(state);
                  }
                },
                (error) => {
                  state.serverState.status = "error";
                  state.serverState.error = error;
                  state.serverState.isRunning = false;
                  if (state.screen === "server_control") {
                    drawServerControl(state);
                  }
                }
              );
            }
          } else if (key === "\u001b" || key === "\u001b\u001b") {
            // Escape - go back
            state.screen = "browser";
            drawBrowser(state);
          } else if (key === "\u001b[A") { // Up arrow - increase port
            state.serverState.port = Math.min(65535, state.serverState.port + 1);
            drawServerControl(state);
          } else if (key === "\u001b[B") { // Down arrow - decrease port
            state.serverState.port = Math.max(1024, state.serverState.port - 1);
            drawServerControl(state);
          } else if (key === "\u001b[D") { // Left arrow - vllm
            state.serverState.backend = "vllm";
            drawServerControl(state);
          } else if (key === "\u001b[C") { // Right arrow - llama-cpp
            state.serverState.backend = "llama-cpp";
            drawServerControl(state);
          } else if (key === "h" || key === "H") {
            // Toggle host
            state.serverState.host = state.serverState.host === "127.0.0.1" ? "0.0.0.0" : "127.0.0.1";
            drawServerControl(state);
          }
        } else if (serverState.status === "starting") {
          if (key === "q") {
            // Cancel starting
            if (cancelServer) {
              cancelServer();
            }
            state.serverState.status = "stopped";
            state.serverState.isRunning = false;
            drawServerControl(state);
          }
        } else if (serverState.status === "running") {
          if (key === "\r" || key === "\n") {
            // Stop server
            if (cancelServer) {
              cancelServer();
            }
            stopServer(state.serverState, () => {
              state.serverState.status = "stopped";
              state.serverState.isRunning = false;
              state.serverState.pid = undefined;
              drawServerControl(state);
            });
          } else if (key === "c" || key === "C") {
            // Open chat interface (placeholder for now)
            state.screen = "chat";
            drawChat(state);
          } else if (key === "\u001b" || key === "\u001b\u001b") {
            // Escape - go back to browser
            state.screen = "browser";
            drawBrowser(state);
          }
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

      // Handle quantization complete screen
      if (state.screen === "quantize" && state.quantizeProgress.status === "complete") {
        state.models = loadCachedModels();
        state.screen = "browser";
        state.quantizeProgress = {
          modelId: "",
          percent: 0,
          layer: 0,
          totalLayers: 32,
          layerName: "",
          eta: 0,
          status: "idle",
        };
        state.selectedModelForQuantize = null;
        cancelQuantize = null;
        drawBrowser(state);
        return;
      }

      // Handle quantization screen
      if (state.screen === "quantize") {
        if (state.quantizeProgress.status === "idle") {
          // Config mode - selecting options
          if (key === "\r" || key === "\n") {
            // Start quantization
            if (state.selectedModelForQuantize) {
              const model = state.selectedModelForQuantize;
              state.quantizeProgress.modelId = model.id;
              state.quantizeProgress.status = "quantizing";
              drawQuantizeProgress(state);

              cancelQuantize = startQuantization(
                model.id,
                state.quantizeOptions,
                (progress) => {
                  state.quantizeProgress = progress;
                  if (state.screen === "quantize") {
                    drawQuantizeProgress(state);
                  }
                },
                () => {
                  drawQuantizeProgress(state);
                },
                (error) => {
                  state.errorMessage = error;
                  state.screen = "error";
                  drawErrorModal(state);
                }
              );
            }
          } else if (key === "\u001b" || key === "\u001b\u001b") {
            // Escape - go back
            state.screen = "browser";
            state.selectedModelForQuantize = null;
            drawBrowser(state);
          } else if (key === "\u001b[D") { // Left arrow
            state.quantizeOptions.bits = 4;
            drawQuantizeConfig(state);
          } else if (key === "\u001b[C") { // Right arrow
            state.quantizeOptions.bits = 8;
            drawQuantizeConfig(state);
          } else if (key === "k" || key === "K") {
            state.quantizeOptions.kvCache = !state.quantizeOptions.kvCache;
            drawQuantizeConfig(state);
          }
        } else if (state.quantizeProgress.status === "quantizing") {
          if (key === "q") {
            // Cancel quantization
            if (cancelQuantize) {
              cancelQuantize();
            }
            state.quantizeProgress.status = "idle";
            state.screen = "browser";
            drawBrowser(state);
          }
        }
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

          case "t":
            // Open quantization config for selected model
            if (state.models.length > 0 && state.selectedIndex < state.models.length) {
              const model = state.models[state.selectedIndex];
              state.selectedModelForQuantize = model;
              state.quantizeProgress.status = "idle";
              state.screen = "quantize";
              drawQuantizeConfig(state);
            }
            break;

          case "s":
            // Open server control
            state.screen = "server_control";
            state.serverState.status = state.serverState.isRunning ? "running" : "stopped";
            drawServerControl(state);
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
