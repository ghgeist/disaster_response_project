# Agent Prompts Directory

This directory contains specialized AI agent prompts for different coding tasks. All agents follow consistent structural coherence principles and support multiple AI coding platforms.

## Platform Support

All agents support **multi-platform** usage:

- ✅ **Cursor IDE** - Full support with semantic code search
- ✅ **Claude Code** (claude.ai/code) - Full support with tool adaptation
- ✅ **Gemini CLI** - Full support with ReAct loop
- ✅ **Codex** (OpenAI) - Full support with IDE integration

**See**: `_platform-detection-guide.md` for platform detection and tool mapping details.

## Active Agents

### Core Development Agents
- **`code-improvement-agent.md`** - Analyze codebases and implement incremental enhancements
- **`performance-agent.md`** - Optimize code to meet production performance requirements
- **`security-agent.md`** - Implement essential security measures for production
- **`test-agent.md`** - Validate code works correctly and ships safely
- **`debug-agent.md`** - Systematically identify root causes and implement robust fixes

### Planning & Management Agents
- **`planning-agent.md`** - Create actionable plans that lead to working code
- **`dev-note-agent.md`** - Synthesize daily engineering activities from GitHub history
- **`release-orchestrator-agent.md`** - Evaluate repository readiness for production deployment
- **`coding-session-manager.md`** - Manage structured coding sessions and prevent chaos

### Specialized Agents
- **`machine-learning-engineer-agent.md`** - Move ML models from experimentation to production
- **`flask-ui-ux-agent.md`** - Improve Flask web applications through UX optimizations
- **`readme-agent.md`** - Review and improve README.md for completeness and accuracy

## Infrastructure Files

### Integration Standards
- **`_cursor-integration-standard.md`** - Cursor IDE tool usage patterns
- **`_claude-code-integration-standard.md`** - Claude Code tool usage patterns
- **`_gemini-cli-integration-standard.md`** - Gemini CLI tool usage patterns
- **`_codex-integration-standard.md`** - Codex tool usage patterns
- **`_platform-detection-guide.md`** - Platform detection and tool mapping guide

### Session Management
- **`_session-management-core.md`** - Mandatory session management rules
- **`session-management-best-practices.md`** - Comprehensive session management guide

### Templates
- **`agent-type-objective-template.md`** - Template for creating new agents

## Archive

The `archive/` directory contains:
- Historical implementation guides
- Consolidated agents (functionality merged into active agents)
- Completed standardization reports

## Usage

### For Cursor IDE
Agents automatically detect Cursor and use full tool support including `codebase_search`.

### For Claude Code
Agents automatically adapt to Claude Code tools (bash tool, text editor tool, memory tool).

### For Gemini CLI
Agents automatically adapt to Gemini CLI tools (file system tools, `run_shell_command`).

### For Codex
Agents automatically adapt to Codex tools (IDE integration, terminal/CLI, skills system).

## Structural Coherence

All agents incorporate structural coherence principles:
- **Connectedness**: Address coherent problem spaces
- **Explicit Transformations**: Document what's preserved, transformed, and added
- **Compositional Integrity**: Ensure improvements compose correctly
- **Valid No-Op State**: System works when improvements are disabled
- **Intent Preservation**: Improvements preserve original intent

## Version History

- **v2.0** (2026-02-03): Added multi-platform support, structural coherence principles, YAML frontmatter
- **v1.0** (2025-09-12): Initial standardization with Cursor integration

---

**See individual agent files for detailed usage instructions and examples.**
