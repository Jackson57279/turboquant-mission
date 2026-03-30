/**
 * TUI (Terminal User Interface) module.
 *
 * This module provides OpenTUI-based components for the
 * interactive terminal interface.
 *
 * @module tui
 */

import { createCliRenderer, TextRenderable, BoxRenderable } from "@opentui/core";
import type { KeyEvent } from "@opentui/core";

/**
 * Model metadata interface.
 */
export interface ModelInfo {
  id: string;
  name: string;
  size: string;
  files: number;
  quantized: boolean;
  downloaded: string;
}

/**
 * TUI application state.
 */
export interface TUIState {
  /** Current screen */
  screen: 'main' | 'models' | 'model_detail' | 'help' | 'quit_confirm';
  /** Selected model index */
  selectedIndex: number;
  /** Selected model */
  selectedModel: ModelInfo | null;
  /** Server status */
  serverRunning: boolean;
  /** Whether TUI should exit */
  shouldExit: boolean;
  /** Last key pressed */
  lastKey: string;
}

/**
 * Create initial TUI state.
 *
 * @returns Initial state
 */
export function createInitialState(): TUIState {
  return {
    screen: 'main',
    selectedIndex: 0,
    selectedModel: null,
    serverRunning: false,
    shouldExit: false,
    lastKey: '',
  };
}

/**
 * Sample models for demo purposes.
 */
const SAMPLE_MODELS: ModelInfo[] = [
  {
    id: 'meta-llama/Llama-2-7b-hf',
    name: 'Llama 2 7B',
    size: '13.5 GB',
    files: 3,
    quantized: false,
    downloaded: '2024-01-15',
  },
  {
    id: 'meta-llama/Llama-2-7b-chat-hf',
    name: 'Llama 2 7B Chat',
    size: '3.8 GB',
    files: 2,
    quantized: true,
    downloaded: '2024-01-20',
  },
  {
    id: 'microsoft/Phi-3-mini-4k-instruct',
    name: 'Phi-3 Mini',
    size: '7.6 GB',
    files: 4,
    quantized: false,
    downloaded: '2024-02-01',
  },
];

/**
 * Launch the TUI.
 *
 * Creates and runs the OpenTUI-based terminal interface.
 */
export async function launchTUI(): Promise<void> {
  const renderer = await createCliRenderer();
  const state = createInitialState();

  // Create main container box
  const mainBox = new BoxRenderable(renderer, {
    id: 'main-box',
    border: true,
    borderStyle: 'single',
    title: ' LLM Compress ',
    shouldFill: true,
    backgroundColor: '#1a1a2e',
    borderColor: '#4a4a6a',
    focusedBorderColor: '#6a6a8a',
  });

  renderer.root.add(mainBox);

  // Create navigation state
  let currentScreen: TUIState['screen'] = 'main';
  let selectedIndex = 0;
  let selectedModel: ModelInfo | null = null;
  let shouldExit = false;

  // Create menu items
  const menuItems = [
    { label: 'Browse Models', action: 'models', shortcut: 'b' },
    { label: 'Quantize Model', action: 'quantize', shortcut: 'q' },
    { label: 'Serve Model', action: 'serve', shortcut: 's' },
    { label: 'Settings', action: 'settings', shortcut: 't' },
    { label: 'Quit', action: 'quit', shortcut: 'q' },
  ];

  // Clear and render helper
  function clearMainBox(): void {
    const children = mainBox.getChildren();
    for (const child of children) {
      mainBox.remove(child.id);
    }
  }

  // Render main menu screen
  function renderMainMenu(): void {
    clearMainBox();
    
    // Title
    const title = new TextRenderable(renderer, {
      id: 'menu-title',
      content: 'Main Menu',
    });
    title.fg = '#ffffff';
    title.attributes = 1; // bold
    title.paddingTop = 1;
    title.paddingLeft = 2;
    mainBox.add(title);

    // Menu items
    menuItems.forEach((item, index) => {
      const isSelected = index === selectedIndex;
      const prefix = isSelected ? '> ' : '  ';
      const suffix = isSelected ? ' <' : '';
      
      const menuItem = new TextRenderable(renderer, {
        id: `menu-item-${index}`,
        content: `${prefix}${item.label} (${item.shortcut})${suffix}`,
      });
      menuItem.fg = isSelected ? '#00ff88' : '#a0a0a0';
      menuItem.attributes = isSelected ? 1 : 0;
      menuItem.paddingLeft = 4;
      menuItem.paddingTop = index === 0 ? 2 : 0;
      mainBox.add(menuItem);
    });

    // Footer with hints
    const footer = new TextRenderable(renderer, {
      id: 'menu-footer',
      content: '↑↓ Navigate | Enter Select | ? Help | q Quit',
    });
    footer.fg = '#606060';
    footer.paddingLeft = 2;
    footer.paddingTop = menuItems.length + 2;
    mainBox.add(footer);
  }

  // Render model browser screen
  function renderModelBrowser(): void {
    clearMainBox();
    
    // Title
    const title = new TextRenderable(renderer, {
      id: 'browser-title',
      content: 'Model Browser',
    });
    title.fg = '#ffffff';
    title.attributes = 1;
    title.paddingTop = 1;
    title.paddingLeft = 2;
    mainBox.add(title);

    // Model count
    const count = new TextRenderable(renderer, {
      id: 'model-count',
      content: `${SAMPLE_MODELS.length} models in cache`,
    });
    count.fg = '#808080';
    count.paddingLeft = 2;
    count.paddingTop = 1;
    mainBox.add(count);

    // Model list header
    const header = new TextRenderable(renderer, {
      id: 'list-header',
      content: 'NAME                    SIZE      QUANTIZED  DOWNLOADED',
    });
    header.fg = '#606060';
    header.paddingLeft = 2;
    header.paddingTop = 1;
    mainBox.add(header);

    // Model items
    SAMPLE_MODELS.forEach((model, index) => {
      const isSelected = index === selectedIndex;
      const prefix = isSelected ? '> ' : '  ';
      const suffix = isSelected ? ' <' : '';
      const quantizedStr = model.quantized ? 'Yes' : 'No';
      
      // Pad model name to 22 chars
      const paddedName = model.name.padEnd(22);
      
      const modelItem = new TextRenderable(renderer, {
        id: `model-${index}`,
        content: `${prefix}${paddedName}${model.size.padEnd(10)}${quantizedStr.padEnd(11)}${model.downloaded}${suffix}`,
      });
      modelItem.fg = isSelected ? '#00ff88' : '#a0a0a0';
      modelItem.attributes = isSelected ? 1 : 0;
      modelItem.paddingLeft = 2;
      modelItem.paddingTop = 0;
      mainBox.add(modelItem);
    });

    // Footer
    const footer = new TextRenderable(renderer, {
      id: 'browser-footer',
      content: '↑↓ Navigate | Enter Select | m Menu | ? Help | q Quit',
    });
    footer.fg = '#606060';
    footer.paddingLeft = 2;
    footer.paddingTop = SAMPLE_MODELS.length + 2;
    mainBox.add(footer);
  }

  // Render model detail screen
  function renderModelDetail(): void {
    clearMainBox();
    
    if (!selectedModel) {
      const error = new TextRenderable(renderer, {
        id: 'error',
        content: 'No model selected',
      });
      error.fg = '#ff4444';
      mainBox.add(error);
      return;
    }

    // Title
    const title = new TextRenderable(renderer, {
      id: 'detail-title',
      content: 'Model Details',
    });
    title.fg = '#ffffff';
    title.attributes = 1;
    title.paddingTop = 1;
    title.paddingLeft = 2;
    mainBox.add(title);

    // Model info
    const details = [
      { label: 'ID:', value: selectedModel.id },
      { label: 'Name:', value: selectedModel.name },
      { label: 'Size:', value: selectedModel.size },
      { label: 'Files:', value: String(selectedModel.files) },
      { label: 'Quantized:', value: selectedModel.quantized ? 'Yes' : 'No' },
      { label: 'Downloaded:', value: selectedModel.downloaded },
    ];

    let currentY = 2;
    details.forEach((detail, index) => {
      const labelText = new TextRenderable(renderer, {
        id: `detail-label-${index}`,
        content: detail.label,
      });
      labelText.fg = '#808080';
      labelText.attributes = 1;
      labelText.paddingLeft = 4;
      labelText.paddingTop = currentY;
      currentY = 0; // Only first has padding
      mainBox.add(labelText);

      const valueText = new TextRenderable(renderer, {
        id: `detail-value-${index}`,
        content: '  ' + detail.value,
      });
      valueText.fg = '#ffffff';
      valueText.paddingLeft = 4;
      mainBox.add(valueText);
    });

    // Actions
    const actionsText = new TextRenderable(renderer, {
      id: 'detail-actions',
      content: '\nActions:',
    });
    actionsText.fg = '#808080';
    actionsText.attributes = 1;
    actionsText.paddingLeft = 4;
    mainBox.add(actionsText);

    const actions = [
      'q - Quantize this model',
      's - Serve this model',
      'd - Delete from cache',
    ];

    actions.forEach((action, index) => {
      const actionText = new TextRenderable(renderer, {
        id: `action-${index}`,
        content: '  ' + action,
      });
      actionText.fg = '#a0a0a0';
      actionText.paddingLeft = 4;
      mainBox.add(actionText);
    });

    // Footer
    const footer = new TextRenderable(renderer, {
      id: 'detail-footer',
      content: 'Esc Back | ? Help',
    });
    footer.fg = '#606060';
    footer.paddingLeft = 2;
    footer.paddingTop = 4;
    mainBox.add(footer);
  }

  // Render help screen
  function renderHelp(): void {
    clearMainBox();
    
    const title = new TextRenderable(renderer, {
      id: 'help-title',
      content: 'Keyboard Shortcuts',
    });
    title.fg = '#ffffff';
    title.attributes = 1;
    title.paddingTop = 1;
    title.paddingLeft = 2;
    mainBox.add(title);

    const shortcuts = [
      { key: '↑ / ↓', description: 'Navigate up/down in lists' },
      { key: 'Enter', description: 'Select/confirm' },
      { key: 'm', description: 'Go to main menu' },
      { key: 'b', description: 'Browse models' },
      { key: '?', description: 'Show this help' },
      { key: 'q', description: 'Quit application' },
      { key: 'Esc', description: 'Go back' },
    ];

    let currentY = 2;
    shortcuts.forEach((shortcut, index) => {
      const keyText = new TextRenderable(renderer, {
        id: `help-key-${index}`,
        content: shortcut.key.padEnd(12),
      });
      keyText.fg = '#00ff88';
      keyText.attributes = 1;
      keyText.paddingLeft = 4;
      keyText.paddingTop = currentY;
      currentY = 0;
      mainBox.add(keyText);

      const descText = new TextRenderable(renderer, {
        id: `help-desc-${index}`,
        content: shortcut.description,
      });
      descText.fg = '#a0a0a0';
      descText.paddingLeft = 16;
      mainBox.add(descText);
    });

    const footer = new TextRenderable(renderer, {
      id: 'help-footer',
      content: '\nPress any key to close help',
    });
    footer.fg = '#606060';
    footer.paddingLeft = 2;
    mainBox.add(footer);
  }

  // Render quit confirmation
  function renderQuitConfirm(): void {
    clearMainBox();
    
    const title = new TextRenderable(renderer, {
      id: 'quit-title',
      content: 'Quit?',
    });
    title.fg = '#ffffff';
    title.attributes = 1;
    title.paddingTop = 1;
    title.paddingLeft = 2;
    mainBox.add(title);

    const message = new TextRenderable(renderer, {
      id: 'quit-message',
      content: 'Are you sure you want to quit?',
    });
    message.fg = '#a0a0a0';
    message.paddingLeft = 4;
    message.paddingTop = 2;
    mainBox.add(message);

    const options = new TextRenderable(renderer, {
      id: 'quit-options',
      content: '[Y]es  [N]o',
    });
    options.fg = '#00ff88';
    options.paddingLeft = 4;
    options.paddingTop = 2;
    mainBox.add(options);
  }

  // Render current screen
  function render(): void {
    switch (currentScreen) {
      case 'main':
        renderMainMenu();
        break;
      case 'models':
        renderModelBrowser();
        break;
      case 'model_detail':
        renderModelDetail();
        break;
      case 'help':
        renderHelp();
        break;
      case 'quit_confirm':
        renderQuitConfirm();
        break;
    }
    renderer.requestRender();
  }

  // Handle key presses using keyHandler
  function handleKey(key: string, name: string): boolean {
    // Track last key for debugging
    state.lastKey = name || key;

    // Help screen - any key closes
    if (currentScreen === 'help') {
      // Go back to previous screen
      if (selectedModel) {
        currentScreen = 'models';
      } else {
        currentScreen = 'main';
      }
      render();
      return true;
    }

    // Quit confirmation
    if (currentScreen === 'quit_confirm') {
      if (key === 'y' || key === 'Y' || name === 'return') {
        shouldExit = true;
        return true;
      }
      if (key === 'n' || key === 'N' || name === 'escape') {
        currentScreen = 'main';
        selectedIndex = 0;
        render();
        return true;
      }
      return true;
    }

    // Global shortcuts (not in quit confirm)
    if (key === '?') {
      currentScreen = 'help';
      render();
      return true;
    }

    if (key === 'q') {
      currentScreen = 'quit_confirm';
      render();
      return true;
    }

    if (key === 'm' || name === 'escape') {
      if (currentScreen === 'model_detail') {
        currentScreen = 'models';
      } else if (currentScreen === 'models') {
        currentScreen = 'main';
        selectedIndex = 0;
      } else {
        currentScreen = 'main';
        selectedIndex = 0;
      }
      render();
      return true;
    }

    if (key === 'b') {
      currentScreen = 'models';
      selectedIndex = 0;
      render();
      return true;
    }

    // Main menu navigation
    if (currentScreen === 'main') {
      if (name === 'up') {
        selectedIndex = Math.max(0, selectedIndex - 1);
        render();
        return true;
      }
      if (name === 'down') {
        selectedIndex = Math.min(menuItems.length - 1, selectedIndex + 1);
        render();
        return true;
      }
      if (name === 'return') {
        const selected = menuItems[selectedIndex];
        if (selected.action === 'models') {
          currentScreen = 'models';
          selectedIndex = 0;
          render();
          return true;
        }
        if (selected.action === 'quit') {
          currentScreen = 'quit_confirm';
          render();
          return true;
        }
        // Other actions show placeholder
        const actionText = new TextRenderable(renderer, {
          id: 'action-message',
          content: `Selected: ${selected.label}`,
        });
        actionText.fg = '#00ff88';
        actionText.paddingLeft = 4;
        actionText.paddingTop = menuItems.length + 3;
        mainBox.add(actionText);
        renderer.requestRender();
        return true;
      }
    }

    // Model browser navigation
    if (currentScreen === 'models') {
      if (name === 'up') {
        selectedIndex = Math.max(0, selectedIndex - 1);
        render();
        return true;
      }
      if (name === 'down') {
        selectedIndex = Math.min(SAMPLE_MODELS.length - 1, selectedIndex + 1);
        render();
        return true;
      }
      if (name === 'return') {
        selectedModel = SAMPLE_MODELS[selectedIndex];
        currentScreen = 'model_detail';
        render();
        return true;
      }
    }

    // Model detail screen
    if (currentScreen === 'model_detail') {
      if (name === 'escape') {
        currentScreen = 'models';
        render();
        return true;
      }
    }

    return true;
  }

  // Set up keyboard handling using keyHandler
  renderer.keyInput.on('keypress', (key: KeyEvent) => {
    const keyChar = key.raw || key.name || '';
    handleKey(keyChar, key.name || '');
  });

  // Start renderer and initial render
  renderer.start();
  render();

  // Keep process alive until exit
  await new Promise<void>((resolve) => {
    const checkInterval = setInterval(() => {
      if (shouldExit) {
        clearInterval(checkInterval);
        renderer.stop();
        resolve();
      }
    }, 100);
  });
}

export default launchTUI;
