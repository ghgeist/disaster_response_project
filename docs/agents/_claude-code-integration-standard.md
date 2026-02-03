# Standard Claude Code Integration for All Agents

**Status**: Canonical Reference - Include in all agent prompts for Claude Code (claude.ai/code)  
**Version**: 1.0  
**Last Updated**: 2026-02-03

## CLAUDE CODE INTEGRATION (STANDARD)

You operate within Claude Code (claude.ai/code) with access to integrated tools. **Follow the session management rules in `docs/agents/_session-management-core.md`**.

### Tool Mapping (Claude Code equivalents)

**Cursor → Claude Code mappings:**
- `codebase_search` → Use natural language queries in conversation + file reading to understand codebase
- `grep` → Use text editor search or shell commands (`grep`, `rg`, `find`)
- `read_file` → Use text editor tool to read files
- `list_dir` → Use shell commands (`ls`, `dir`) or file system navigation
- `search_replace` → Use text editor tool to modify files
- `write` → Use text editor tool to create new files
- `run_terminal_cmd` → Use bash tool to execute shell commands
- `read_lints` → Use bash tool to run linting commands (`pylint`, `ruff`, `mypy`)
- `update_memory` → Use memory tool to store important learnings
- `MultiEdit` → Use text editor tool with multiple file edits

### Discovery Phase (ALWAYS FIRST)
- Use shell commands (`ls`, `find`) to check `docs/sessions/active/` for existing relevant sessions
- Use natural language queries + file reading to understand overall system architecture and patterns
- Use shell commands (`grep`, `rg`) to find specific patterns, implementations, or related work
- Use text editor tool to examine relevant files, configurations, and existing sessions

### Analysis Phase
- Use natural language queries + file reading with semantic questions like "How is [X] implemented?" or "Where does [Y] happen?"
- Use shell commands (`grep`, `rg`) to search for specific code patterns, errors, or configurations
- Use text editor tool to analyze code structure, documentation, and session context

### Implementation Phase
- Use text editor tool for code changes (single or multiple files)
- Use text editor tool to create new files, documentation, or session updates
- Use bash tool to execute tests, validations, and deployments
- Use bash tool to run linting commands (`pylint`, `ruff`, `mypy`) for code quality checks

### Session Management (MANDATORY)
- **ALWAYS** check `docs/sessions/active/` before starting work using shell commands
- **CREATE** session using template from `_session-management-core.md` if none exists (use text editor tool)
- **UPDATE** session progress throughout work using text editor tool
- **COMPLETE** session by moving to `docs/sessions/completed/` when done (use shell commands)

### Validation Phase
- Use bash tool to run linting commands and check for introduced errors
- Use bash tool to validate changes work correctly
- Use memory tool to preserve important learnings for future sessions

### Claude Code Specific Notes
- **No semantic code search**: Use natural language queries + file reading instead of `codebase_search`
- **File operations**: Use text editor tool for all file read/write operations
- **Command execution**: Use bash tool for all terminal commands
- **Memory**: Use memory tool for persistent information storage

---

**REFERENCE**: See `docs/agents/_session-management-core.md` for complete session management rules.
