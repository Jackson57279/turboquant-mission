---
name: tui-worker
description: OpenTUI-based terminal user interface components and interactions
---

# tui-worker

## When to Use This Skill

Use this skill for:
- Creating OpenTUI-based terminal interfaces
- Implementing screens (model browser, chat, settings)
- Adding keyboard navigation and shortcuts
- Creating progress displays and modals
- End-to-end TUI testing with tuistory

## Required Skills

- **tuistory** - For automating TUI interactions and capturing screenshots

## Work Procedure

1. **Read mission context**
   - Read /home/dih/.factory/missions/07fa55b8-a7aa-40b1-89fb-dfb79186e53a/mission.md
   - Read /home/dih/.factory/missions/07fa55b8-a7aa-40b1-89fb-dfb79186e53a/AGENTS.md
   - Review validation contract for TUI assertions

2. **Check OpenTUI setup**
   - Verify Zig is installed
   - Verify Bun is available
   - Run `bun install` to get @opentui packages

3. **Design the screen**
   - Plan the component hierarchy
   - Define state management
   - Plan keyboard shortcuts

4. **Implement the screen**
   - Use @opentui/react for React-style components
   - Use @opentui/core for imperative API
   - Add proper error boundaries
   - Implement keyboard handlers

5. **Test with tuistory**
   - Create test script that launches TUI
   - Automate key sequences
   - Capture screenshots at each step
   - Verify expected state

6. **Manual verification**
   - Run TUI interactively
   - Test all keyboard shortcuts
   - Verify visual appearance

## Example Handoff

```json
{
  "salientSummary": "Implemented model browser screen in OpenTUI. Shows downloaded models with sizes and quantization status. Arrow keys navigate, Enter selects, ? shows help. Tested with tuistory automation.",
  "whatWasImplemented": "Created src/tui/screens/ModelBrowser.tsx using @opentui/react. Displays scrollable list of models with metadata. Uses useState for selection and useEffect for keyboard handling. Integrates with backend API for model list.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "bun run tui", "exitCode": 0, "observation": "TUI launches, model browser visible"},
      {"command": "tuistory --test tests/tui/model_browser.test.ts", "exitCode": 0, "observation": "All 5 test scenarios passed, screenshots captured"}
    ],
    "interactiveChecks": [
      {"action": "Launch llm-compress tui", "observed": "Model browser shows with 3 test models"},
      {"action": "Press Down arrow 2 times", "observed": "Highlight moves to third model"},
      {"action": "Press Enter", "observed": "Model detail screen opens"},
      {"action": "Press ?", "observed": "Help modal appears with shortcuts"}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- OpenTUI fails to build (Zig/Bun issues)
- Terminal compatibility issues
- Screen design conflicts with existing screens
- Need to coordinate with backend API changes
