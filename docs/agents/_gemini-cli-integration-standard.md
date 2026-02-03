# Standard Gemini CLI Integration for All Agents

**Status**: Canonical Reference - Include in all agent prompts for Gemini CLI  
**Version**: 1.0  
**Last Updated**: 2026-02-03

## GEMINI CLI INTEGRATION (STANDARD)

You operate within Gemini CLI with access to integrated tools through the ReAct (reason and act) loop. **Follow the session management rules in `docs/agents/_session-management-core.md`**.

### Tool Mapping (Gemini CLI equivalents)

**Cursor → Gemini CLI mappings:**
- `codebase_search` → Use file system search tools + natural language queries to understand codebase
- `grep` → Use `run_shell_command` with `grep` or `rg` commands
- `read_file` → Use file system read tool
- `list_dir` → Use file system list tool or `run_shell_command` with `ls`/`dir`
- `search_replace` → Use file system write/modify tools
- `write` → Use file system write tool to create new files
- `run_terminal_cmd` → Use `run_shell_command` tool (with safety measures and user confirmation)
- `read_lints` → Use `run_shell_command` to run linting commands
- `update_memory` → Use memory/storage capabilities if available, or document in session files
- `MultiEdit` → Use file system tools to modify multiple files sequentially

### Discovery Phase (ALWAYS FIRST)
- Use file system list tool or `run_shell_command` to check `docs/sessions/active/` for existing relevant sessions
- Use file system search + file reading to understand overall system architecture and patterns
- Use `run_shell_command` with `grep`/`rg` to find specific patterns, implementations, or related work
- Use file system read tool to examine relevant files, configurations, and existing sessions

### Analysis Phase
- Use file system search + file reading with semantic questions like "How is [X] implemented?" or "Where does [Y] happen?"
- Use `run_shell_command` with `grep`/`rg` to search for specific code patterns, errors, or configurations
- Use file system read tool to analyze code structure, documentation, and session context

### Implementation Phase
- Use file system write/modify tools for code changes
- Use file system write tool to create new files, documentation, or session updates
- Use `run_shell_command` to execute tests, validations, and deployments (with user confirmation for safety)
- Use `run_shell_command` to run linting commands for code quality checks

### Session Management (MANDATORY)
- **ALWAYS** check `docs/sessions/active/` before starting work using file system tools
- **CREATE** session using template from `_session-management-core.md` if none exists (use file system write tool)
- **UPDATE** session progress throughout work using file system modify tools
- **COMPLETE** session by moving to `docs/sessions/completed/` when done (use `run_shell_command`)

### Validation Phase
- Use `run_shell_command` to run linting commands and check for introduced errors
- Use `run_shell_command` to validate changes work correctly
- Document important learnings in session files for future reference

### Gemini CLI Specific Notes
- **ReAct loop**: Gemini CLI uses reasoning and action loops - explain your reasoning before taking actions
- **Safety measures**: `run_shell_command` includes safety measures and may require user confirmation
- **File operations**: Use file system tools for all file operations
- **No semantic code search**: Use file system search + file reading instead of `codebase_search`
- **Memory**: Document important information in session files rather than separate memory system

---

**REFERENCE**: See `docs/agents/_session-management-core.md` for complete session management rules.
