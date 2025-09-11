# Core Session Management Rules for All Agents

**Status**: Canonical Reference - All agents must follow these rules  
**Version**: 1.0  
**Last Updated**: 2025-09-12

## 🎯 MANDATORY SESSION WORKFLOW

### Phase 1: Session Discovery (ALWAYS FIRST)
```markdown
1. Use `list_dir` to check `docs/sessions/active/` for existing sessions
2. Use `grep` to search for related work in `backlog/` and `completed/`
3. Use `read_file` to understand context of any relevant sessions found
```

### Phase 2: Session Management (REQUIRED)
```markdown
IF no relevant session exists:
- Use `write` to create new session in `docs/sessions/active/`
- Follow naming: `YYYY-MM-DD-[agent-type]-[description].md`

IF relevant session exists:
- Use `search_replace` to update existing session with progress
- Continue work within established session context
```

### Phase 3: Session Updates (THROUGHOUT WORK)
```markdown
- Update session file with progress using `search_replace`
- Document decisions, blockers, and solutions
- Link to any files created or modified
- Track success criteria and outcomes
```

### Phase 4: Session Completion (WHEN DONE)
```markdown
- Mark session as completed with final outcomes
- Document next steps or follow-up work needed
- Move session from `active/` to `completed/` (use file operations)
- Create follow-up sessions in `backlog/` if needed
```

## 📋 SESSION FILE TEMPLATE

```yaml
---
title: "[Agent Type]: [Clear Description]"
date: "YYYY-MM-DD"
status: "active|completed|blocked"
session_type: "[agent-type]"
priority: "high|medium|low"
tags: ["relevant", "tags"]
author: "[agent-name]"
related: ["paths/to/related/sessions"]
---

# [Agent Type]: [Clear Description]

**Session Type**: [RESEARCH|PLAN|EXECUTE|DEBUG|DEPLOY|REFACTOR|TEST]
**Priority**: [High|Medium|Low]
**Estimated Duration**: [Time estimate]
**Status**: [Active|Completed|Blocked]

## 🎯 Objective
[Clear, specific description of what this session should accomplish]

## 📋 Success Criteria
- [ ] Specific measurable outcome 1
- [ ] Specific measurable outcome 2
- [ ] Specific measurable outcome 3

## 🔍 Context
[Background information and current state]

## 📝 Progress Log
[Real-time updates throughout the session]

## 🎉 Outcomes
[Final results when session is complete]

## 🔗 Related Work
[Links to files modified, related sessions, etc.]

## 📈 Next Steps
[Follow-up actions or sessions needed]
```

## 🚨 ENFORCEMENT RULES

### MUST DO:
✅ Check for existing sessions before creating new ones  
✅ Use standardized naming convention: `YYYY-MM-DD-[type]-[description].md`  
✅ Keep maximum 2 sessions in `active/` directory  
✅ Update session progress throughout work  
✅ Move completed sessions to `completed/` directory  
✅ Link session to any files created or modified  

### MUST NOT:
❌ Create sessions without checking for existing ones  
❌ Skip session creation for significant work  
❌ Leave sessions in `active/` after completion  
❌ Use inconsistent naming conventions  
❌ Work without documenting progress in session  

## 🔧 AGENT TYPE MAPPINGS

| Agent Type | Session Prefix | Primary Output Location |
|------------|---------------|------------------------|
| `code-improvement` | `refactor` | `docs/code_improvement_log/` + session |
| `flask-ui-ux` | `ui-ux` | Code changes + session |
| `performance` | `performance` | Performance reports + session |
| `security` | `security` | Security fixes + session |
| `debug` | `debug` | Bug fixes + session |
| `plan` | `plan` | Session only |
| `integrate` | `deploy` | Deployment artifacts + session |
| `test` | `test` | Test results + session |
| `refactor` | `refactor` | Code changes + session |
| `ml-engineer` | `ml` | Model artifacts + session |
| `research` | `research` | Research findings + session |

## 📊 SESSION STATE LIMITS

- **Active Sessions**: Maximum 2 at any time
- **Session Duration**: Close sessions within 24-48 hours
- **File Size**: Keep sessions under 1000 lines (split if needed)
- **Updates**: Update session at least every 30 minutes during active work

## 🎯 QUALITY GATES

Before completing any session, verify:
- [ ] Objective clearly achieved or reason for incompletion documented
- [ ] All success criteria addressed
- [ ] Related files and sessions linked
- [ ] Next steps clearly defined
- [ ] Session moved to appropriate directory

---

**IMPORTANT**: This document is the canonical reference for session management. All agents must implement these rules without exception.
