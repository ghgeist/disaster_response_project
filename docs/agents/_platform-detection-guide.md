# Platform Detection Guide for Agent Prompts

**Status**: Reference Guide  
**Version**: 1.0  
**Last Updated**: 2026-02-03

## Purpose

This guide helps agents detect which platform they're running on and use the appropriate integration standard.

## Platform Detection

### How to Detect Platform

**Cursor IDE:**
- Has `codebase_search` tool available
- Has `read_lints` tool available
- Has `update_memory` tool available
- Has `MultiEdit` tool available
- **Integration Standard**: `docs/agents/_cursor-integration-standard.md`

**Claude Code (claude.ai/code):**
- Has bash tool, text editor tool, memory tool
- Does NOT have semantic code search (`codebase_search`)
- Uses natural language + file reading for codebase exploration
- **Integration Standard**: `docs/agents/_claude-code-integration-standard.md`

**Gemini CLI:**
- Has `run_shell_command` tool (with safety measures)
- Has file system tools (read, write, list, search)
- Uses ReAct (reason and act) loop
- Does NOT have semantic code search
- **Integration Standard**: `docs/agents/_gemini-cli-integration-standard.md`

**Codex (OpenAI):**
- Has IDE integration capabilities
- Has terminal/CLI tool
- Has memory/skills system
- May have IDE-native search (varies by IDE)
- **Integration Standard**: `docs/agents/_codex-integration-standard.md`

## Universal Principles (All Platforms)

Regardless of platform, all agents should follow:

1. **Structural Coherence Requirements** - Connectedness, explicit transformations, compositional integrity
2. **Phase-Based Analysis** - Discovery → Assessment → Implementation → Validation
3. **Session Management** - Follow `docs/agents/_session-management-core.md`
4. **Shipping Philosophy** - Working > Perfect, Incremental > Big Bang

## Tool Availability Matrix

| Tool | Cursor | Claude Code | Gemini CLI | Codex |
|------|--------|-------------|------------|-------|
| Semantic Code Search | ✅ `codebase_search` | ❌ (use file reading + queries) | ❌ (use file search) | ⚠️ (varies by IDE) |
| Pattern Search | ✅ `grep` | ✅ (shell commands) | ✅ `run_shell_command` | ✅ (IDE/terminal) |
| File Reading | ✅ `read_file` | ✅ (text editor tool) | ✅ (file system read) | ✅ (IDE) |
| File Writing | ✅ `write` | ✅ (text editor tool) | ✅ (file system write) | ✅ (IDE) |
| File Editing | ✅ `search_replace` | ✅ (text editor tool) | ✅ (file system modify) | ✅ (IDE) |
| Multi-File Edit | ✅ `MultiEdit` | ✅ (text editor tool) | ⚠️ (sequential) | ✅ (IDE) |
| Terminal Commands | ✅ `run_terminal_cmd` | ✅ (bash tool) | ✅ `run_shell_command` | ✅ (terminal/CLI) |
| Linting | ✅ `read_lints` | ✅ (bash tool) | ✅ `run_shell_command` | ✅ (IDE/terminal) |
| Memory | ✅ `update_memory` | ✅ (memory tool) | ⚠️ (session files) | ✅ (skills system) |
| Directory Listing | ✅ `list_dir` | ✅ (shell commands) | ✅ (file system list) | ✅ (IDE/terminal) |

## Recommended Approach

### For Agent Prompt Writers

Include platform detection logic in agent prompts:

```markdown
## PLATFORM INTEGRATION

**DETECT PLATFORM**: Check available tools to determine platform:
- If `codebase_search` available → Use `docs/agents/_cursor-integration-standard.md`
- If bash tool + text editor tool available → Use `docs/agents/_claude-code-integration-standard.md`
- If `run_shell_command` + file system tools available → Use `docs/agents/_gemini-cli-integration-standard.md`
- If IDE integration + terminal available → Use `docs/agents/_codex-integration-standard.md`

**FALLBACK**: If platform unclear, use generic tool descriptions and adapt based on available capabilities.
```

### For Users

When using agents in different platforms:
1. **Cursor**: Agents work as-is with full tool support
2. **Claude Code**: Agents adapt automatically, semantic search replaced with file reading + queries
3. **Gemini CLI**: Agents adapt automatically, uses ReAct loop with file system tools
4. **Codex**: Agents adapt automatically, uses IDE-native features

## Migration Notes

- **Universal principles** (structural coherence, phase-based analysis) work across all platforms
- **Tool-specific features** (like `codebase_search`) are platform-dependent
- **Session management** works across all platforms (file-based)
- **Output formats** remain consistent across platforms

---

**See individual integration standards for platform-specific details:**
- `_cursor-integration-standard.md` - Cursor IDE
- `_claude-code-integration-standard.md` - Claude Code
- `_gemini-cli-integration-standard.md` - Gemini CLI
- `_codex-integration-standard.md` - Codex
