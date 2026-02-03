# Standard Cursor Integration for All Agents

**Status**: Canonical Reference - Include in all agent prompts for Cursor IDE  
**Version**: 2.0  
**Last Updated**: 2026-02-03

## PLATFORM DETECTION

**If you're running in Cursor IDE**: Use this integration standard (`_cursor-integration-standard.md`)  
**If you're running in Claude Code**: Use `docs/agents/_claude-code-integration-standard.md`  
**If you're running in Gemini CLI**: Use `docs/agents/_gemini-cli-integration-standard.md`  
**If you're running in Codex**: Use `docs/agents/_codex-integration-standard.md`

**See**: `docs/agents/_platform-detection-guide.md` for platform detection and tool mapping details.

## CURSOR INTEGRATION (STANDARD)

You operate within Cursor IDE with access to powerful integrated tools. **Follow the session management rules in `docs/agents/_session-management-core.md`**.

### Discovery Phase (ALWAYS FIRST)
- Use `list_dir` to check `docs/sessions/active/` for existing relevant sessions
- Use `codebase_search` to understand overall system architecture and patterns
- Use `grep` to find specific patterns, implementations, or related work
- Use `read_file` to examine relevant files, configurations, and existing sessions

### Analysis Phase
- Use `codebase_search` with semantic queries like "How is [X] implemented?" or "Where does [Y] happen?"
- Use `grep` to search for specific code patterns, errors, or configurations
- Use `read_file` to analyze code structure, documentation, and session context

### Implementation Phase
- Use `search_replace` or `MultiEdit` for code changes
- Use `write` to create new files, documentation, or session updates
- Use `run_terminal_cmd` to execute tests, validations, and deployments
- Use `read_lints` to check for code quality issues and best practices

### Session Management (MANDATORY)
- **ALWAYS** check `docs/sessions/active/` before starting work
- **CREATE** session using template from `_session-management-core.md` if none exists
- **UPDATE** session progress throughout work using `search_replace`
- **COMPLETE** session by moving to `docs/sessions/completed/` when done

### Validation Phase
- Use `read_lints` to check for introduced errors
- Use `run_terminal_cmd` to validate changes work correctly
- Use `update_memory` to preserve important learnings for future sessions

---

**REFERENCE**: See `docs/agents/_session-management-core.md` for complete session management rules.
