/**
 * TUI (Terminal User Interface) module.
 *
 * This module provides OpenTUI-based components for the
 * interactive terminal interface.
 *
 * @module tui
 */

/**
 * TUI application state.
 */
export interface TUIState {
  /** Current screen */
  screen: 'main' | 'models' | 'quantize' | 'serve' | 'chat';
  /** Selected model */
  selectedModel: string | null;
  /** Server status */
  serverRunning: boolean;
}

/**
 * Create initial TUI state.
 *
 * @returns Initial state
 */
export function createInitialState(): TUIState {
  return {
    screen: 'main',
    selectedModel: null,
    serverRunning: false,
  };
}

/**
 * Launch the TUI.
 *
 * This is a placeholder for the full TUI implementation.
 */
export function launchTUI(): void {
  console.log('TUI would launch here with OpenTUI components.');
  console.log('Full implementation coming in tui-main-interface feature.');
}
