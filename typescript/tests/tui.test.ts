/**
 * TUI module tests
 *
 * Tests for the OpenTUI-based terminal interface.
 *
 * @module tests/tui
 */

import { describe, expect, it } from "bun:test";
import { launchTUI, type ModelInfo } from "../src/tui/index.js";

describe("TUI Module", () => {
  it("should export launchTUI function", () => {
    expect(typeof launchTUI).toBe("function");
  });
});

describe("Model Info Interface", () => {
  it("should accept valid model info", () => {
    const model: ModelInfo = {
      id: "test-model",
      size: "10 GB",
      quantized: true,
      downloadedAt: "2024-01-15",
    };

    expect(model.id).toBe("test-model");
    expect(model.size).toBe("10 GB");
    expect(model.quantized).toBe(true);
    expect(model.downloadedAt).toBe("2024-01-15");
  });
});
