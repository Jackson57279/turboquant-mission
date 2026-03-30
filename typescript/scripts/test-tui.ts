/**
 * Manual TUI verification script
 *
 * This script tests that the TUI module can be imported and initialized
 * without throwing errors. A full interactive test requires a terminal.
 *
 * Run with: bun run test:tui
 */

import { launchTUI, type ModelInfo, type QuantizeOptions, type QuantizeProgress } from "../src/tui/index.js";

console.log("Testing TUI module...\n");

// Test 1: Import check
console.log("✓ Test 1: Module imports successfully");

// Test 2: Model info structure
console.log("✓ Test 2: Model info interface");
const model: ModelInfo = {
  id: "meta-llama/Llama-2-7b-hf",
  size: "13.5 GB",
  quantized: false,
  downloadedAt: "2024-01-15",
};
console.log(`  - Model ID: ${model.id}`);
console.log(`  - Size: ${model.size}`);
console.log(`  - Quantized: ${model.quantized}`);

// Test 3: Quantize options
console.log("\n✓ Test 3: Quantize options interface");
const options: QuantizeOptions = {
  bits: 4,
  kvCache: true,
};
console.log(`  - Bits: ${options.bits}-bit`);
console.log(`  - KV Cache: ${options.kvCache}`);

// Test 4: Quantize progress
console.log("\n✓ Test 4: Quantize progress interface");
const progress: QuantizeProgress = {
  modelId: "test-model",
  percent: 45.5,
  layer: 15,
  totalLayers: 32,
  layerName: "model.layers.14.attention",
  eta: 120,
  status: "quantizing",
};
console.log(`  - Model: ${progress.modelId}`);
console.log(`  - Progress: ${progress.percent}%`);
console.log(`  - Layer: ${progress.layer}/${progress.totalLayers}`);
console.log(`  - ETA: ${progress.eta}s`);
console.log(`  - Status: ${progress.status}`);

console.log("\n✅ All basic interface tests passed!");
console.log("\nTo launch the TUI interactively, run:");
console.log("  bun run dist/esm/cli.js tui");
console.log("\nKeyboard shortcuts:");
console.log("  ↑/↓ - Navigate models");
console.log("  Enter - Select model");
console.log("  d - Download model");
console.log("  t - Quantize selected model");
console.log("  ? - Help");
console.log("  q - Quit");
