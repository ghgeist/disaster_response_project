# Standard Codex Integration for All Agents

**Status**: Canonical Reference - Include in all agent prompts for Codex (OpenAI)  
**Version**: 1.0  
**Last Updated**: 2026-02-03

## CODEX INTEGRATION (STANDARD)

You operate within Codex (OpenAI's AI coding partner) with access to integrated tools. **Follow the session management rules in `docs/agents/_session-management-core.md`**.

### Tool Mapping (Codex equivalents)

**Cursor → Codex mappings:**
- `codebase_search` → Use IDE search capabilities + file reading to understand codebase
- `grep` → Use IDE search or terminal commands (`grep`, `rg`)
- `read_file` → Use IDE file reading capabilities
- `list_dir` → Use IDE file explorer or terminal commands (`ls`, `dir`)
- `search_replace` → Use IDE editing capabilities
- `write` → Use IDE file creation capabilities
- `run_terminal_cmd` → Use terminal/CLI tool to execute commands
- `read_lints` → Use IDE linting integration or terminal commands
- `update_memory` → Use Codex memory/skills system or document in session files
- `MultiEdit` → Use IDE multi-file editing capabilities

### Discovery Phase (ALWAYS FIRST)
- Use IDE file explorer or terminal commands to check `docs/sessions/active/` for existing relevant sessions
- Use IDE search + file reading to understand overall system architecture and patterns
- Use IDE search or terminal commands (`grep`, `rg`) to find specific patterns, implementations, or related work
- Use IDE file reading to examine relevant files, configurations, and existing sessions

### Analysis Phase
- Use IDE search + file reading with semantic questions like "How is [X] implemented?" or "Where does [Y] happen?"
- Use IDE search or terminal commands to search for specific code patterns, errors, or configurations
- Use IDE file reading to analyze code structure, documentation, and session context

### Implementation Phase
- Use IDE editing capabilities for code changes (single or multiple files)
- Use IDE file creation to create new files, documentation, or session updates
- Use terminal/CLI tool to execute tests, validations, and deployments
- Use IDE linting integration or terminal commands for code quality checks

### Session Management (MANDATORY)
- **ALWAYS** check `docs/sessions/active/` before starting work using IDE file explorer
- **CREATE** session using template from `_session-management-core.md` if none exists (use IDE file creation)
- **UPDATE** session progress throughout work using IDE editing capabilities
- **COMPLETE** session by moving to `docs/sessions/completed/` when done (use terminal commands)

### Validation Phase
- Use IDE linting integration or terminal commands to check for introduced errors
- Use terminal/CLI tool to validate changes work correctly
- Use Codex memory/skills system or session files to preserve important learnings

### Codex Specific Notes
- **IDE Integration**: Codex integrates with IDEs - use IDE-native features when available
- **Multi-Agent**: Codex supports multi-agent orchestration - coordinate with other agents if needed
- **Skills System**: Use Codex skills system to extend capabilities beyond code generation
- **Slash Commands**: Codex supports slash commands for common operations
- **Memory**: Use Codex memory/skills system for persistent information storage

---

**REFERENCE**: See `docs/agents/_session-management-core.md` for complete session management rules.
