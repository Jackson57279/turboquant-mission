---
name: infrastructure-worker
description: Packaging, CI/CD, documentation, and release management
---

# infrastructure-worker

## When to Use This Skill

Use this skill for:
- Creating package configuration (pyproject.toml, package.json)
- Setting up CI/CD pipelines
- Writing documentation (README, API docs)
- Creating Docker images
- Publishing to PyPI and NPM
- Repository structure and tooling

## Required Skills

- **None** - Standard tools only

## Work Procedure

1. **Read mission context**
   - Read /home/dih/.factory/missions/07fa55b8-a7aa-40b1-89fb-dfb79186e53a/mission.md
   - Read /home/dih/.factory/missions/07fa55b8-a7aa-40b1-89fb-dfb79186e53a/AGENTS.md
   - Check existing package files

2. **Create package configuration**
   - Python: pyproject.toml with dependencies, entry points
   - TypeScript: package.json with scripts, dependencies
   - Version numbers must match across files

3. **Write documentation**
   - README with quickstart, installation, usage
   - API documentation (OpenAPI for FastAPI)
   - Examples directory

4. **Set up CI/CD**
   - GitHub Actions workflow
   - Test on multiple Python/Node versions
   - Automated publishing on tag

5. **Create Docker images**
   - Multi-stage build
   - Runtime image with only production deps
   - Document usage

6. **Test packaging**
   - Build package locally
   - Install in clean environment
   - Verify all commands work

7. **Publish**
   - PyPI: twine upload
   - NPM: npm publish
   - Create GitHub release

## Example Handoff

```json
{
  "salientSummary": "Created comprehensive README with quickstart, API docs, and examples. Set up GitHub Actions CI/CD. Published llm-compress v0.1.0 to PyPI and NPM. Docker image builds and runs correctly.",
  "whatWasImplemented": "Created README.md with installation (pip/npm), quickstart guide, CLI reference, API documentation. Set up .github/workflows/ci.yml for testing on Python 3.10-3.12. Created Dockerfile with multi-stage build. Published packages - pip install llm-compress and npm install llm-compress both work.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "python -m build", "exitCode": 0, "observation": "Wheel and sdist created in dist/"},
      {"command": "pip install dist/llm_compress-0.1.0-py3-none-any.whl", "exitCode": 0, "observation": "Package installs successfully"},
      {"command": "llm-compress --version", "exitCode": 0, "observation": "0.1.0"},
      {"command": "docker build -t llm-compress:latest .", "exitCode": 0, "observation": "Image builds without errors"}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Publishing credentials not available
- Package name conflict on PyPI/NPM
- Version numbering conflicts
- CI/CD infrastructure not accessible
