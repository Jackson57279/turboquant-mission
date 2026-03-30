/**
 * TUI module tests
 *
 * Tests for the OpenTUI-based terminal interface.
 *
 * @module tests/tui
 */

import { describe, expect, it } from "bun:test";
import { createInitialState, type TUIState, type ModelInfo } from "../src/tui/index.js";

describe("TUI State Management", () => {
  it("should create initial state with correct defaults", () => {
    const state = createInitialState();
    
    expect(state.screen).toBe("main");
    expect(state.selectedIndex).toBe(0);
    expect(state.selectedModel).toBeNull();
    expect(state.serverRunning).toBe(false);
    expect(state.shouldExit).toBe(false);
    expect(state.lastKey).toBe("");
  });
});

describe("Model Info Interface", () => {
  it("should accept valid model info", () => {
    const model: ModelInfo = {
      id: "test-model",
      name: "Test Model",
      size: "10 GB",
      files: 3,
      quantized: true,
      downloaded: "2024-01-15",
    };
    
    expect(model.id).toBe("test-model");
    expect(model.name).toBe("Test Model");
    expect(model.size).toBe("10 GB");
    expect(model.files).toBe(3);
    expect(model.quantized).toBe(true);
    expect(model.downloaded).toBe("2024-01-15");
  });
});

describe("TUI Screen States", () => {
  const validScreens: TUIState["screen"][] = [
    "main",
    "models",
    "model_detail",
    "help",
    "quit_confirm",
  ];
  
  it("should accept all valid screen states", () => {
    for (const screen of validScreens) {
      const state: TUIState = {
        ...createInitialState(),
        screen,
      };
      expect(state.screen).toBe(screen);
    }
  });
});
