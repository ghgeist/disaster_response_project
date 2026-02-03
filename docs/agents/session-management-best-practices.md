# Session Management Best Practices for Cursor Agents

**Date**: 2025-09-12  
**Status**: Active  
**Purpose**: Standardize how agents manage session artifacts and outputs

## 🎯 Current Session Structure

```
docs/sessions/
├── active/          # Currently working sessions (1-2 max)
├── backlog/         # Planned future sessions  
└── completed/       # Finished sessions with outcomes
```

## 📋 Session File Naming Convention

Based on user preferences [[memory:8730623]] [[memory:8730527]]:

**Format**: `YYYY-MM-DD-[session-type]-[description].md`

**Examples**:
- `2025-09-12-deploy-production-model.md`
- `2025-09-12-debug-authentication-issue.md` 
- `2025-09-12-refactor-database-queries.md`

## 🔧 Agent Output Guidelines

### 1. Session Artifact Creation

**All agents should**:
```markdown
## Session Artifacts Management

When starting work:
1. Use `list_dir` to check `docs/sessions/active/` for existing sessions
2. If no relevant active session exists, create one using `write`
3. If relevant session exists, use `read_file` to understand context
4. Update session file throughout work using `search_replace`
```

### 2. Session Types & Outputs

**CRITICAL**: ALL session files MUST be stored in `docs/sessions/active/` during active work.

| Session Type | Session File Location | Secondary Outputs |
|-------------|---------------------|-------------------|
| **RESEARCH** | `docs/sessions/active/` | `docs/research/` if needed |
| **PLAN** | `docs/sessions/active/` | None |
| **EXECUTE** | `docs/sessions/active/` | Code changes in place |
| **DEBUG** | `docs/sessions/active/` | Bug fix documentation |
| **DEPLOY** | `docs/sessions/active/` | Deployment logs/scripts |
| **REFACTOR** | `docs/sessions/active/` | `docs/code_improvement_log/` |
| **TEST** | `docs/sessions/active/` | Test results in session |

### 3. Session State Management

**Active Sessions (1-2 maximum)**:
- Current work in progress
- Updated in real-time during agent interactions
- Moved to `completed/` when finished

**Session Lifecycle**:
```markdown
1. **Create**: Agent creates new session in `active/`
2. **Update**: Agent updates session throughout work
3. **Complete**: Agent moves session to `completed/` when done
4. **Archive**: Old sessions stay in `completed/` for reference
```

## 🚀 Agent Implementation Patterns

### Pattern 1: Session-Aware Agent Start
```markdown
Every agent should begin with:

1. **Check Active Sessions**:
   ```
   Use `list_dir` to check docs/sessions/active/
   Use `read_file` to understand current context
   ```

2. **Create or Update Session**:
   ```
   If no relevant session: Use `write` to create new session
   If relevant session exists: Use `search_replace` to update
   ```

3. **Set Session Context**:
   ```
   Document what agent is doing
   Set expected outcomes
   Track progress throughout work
   ```
```

### Pattern 2: Work Documentation
```markdown
Throughout agent work:

1. **Progress Updates**:
   - Update session file with current progress
   - Document decisions and rationale
   - Track blockers and solutions

2. **Artifact Links**:
   - Link to created/modified files
   - Reference related sessions
   - Document dependencies

3. **Outcome Recording**:
   - Final results and deliverables
   - Success criteria met/unmet
   - Next steps or follow-up needed
```

### Pattern 3: Session Completion
```markdown
When agent completes work:

1. **Final Update**:
   - Mark session as completed
   - Document final outcomes
   - Add success/failure status

2. **Move to Completed**:
   - Use file operations to move from active/ to completed/
   - Preserve all session history
   - Update any references

3. **Clean Up**:
   - Remove from active/ directory
   - Update any related sessions
   - Create follow-up sessions if needed
```

## 📝 Session Template Integration

### Enhanced Agent Prompts
All agents should include this session management pattern:

```markdown
## SESSION MANAGEMENT INTEGRATION

### Session Discovery
- Use `list_dir` to check `docs/sessions/active/` for relevant work
- Use `read_file` to understand existing session context
- Use `grep` to find related sessions in `backlog/` and `completed/`

### Session Creation/Updates
- Use `write` to create new session if none exists
- Use `search_replace` to update existing sessions
- Follow YYYY-MM-DD-[type]-[description].md naming convention

### Session Lifecycle
- Keep sessions in `active/` during work
- Update progress throughout agent interaction
- Move to `completed/` when work is finished
- Create follow-up sessions in `backlog/` if needed
```

## 🎯 Specific Agent Behaviors

### Code Improvement Agent
- **Always** check `docs/code_improvement_log/` for previous work
- **Create session** in `docs/sessions/active/` for current improvement work
- **Link** to improvement log entries from session
- **Move** session to `completed/` when improvement is done

### Deployment Agents
- **Create detailed deployment session** with rollback plans
- **Track deployment progress** in real-time
- **Document** all deployment steps and outcomes
- **Keep** deployment sessions for audit trail

### Debug Agents  
- **Create debug session** with problem description
- **Document** debugging process and findings
- **Track** reproduction steps and solutions
- **Link** to any code fixes made

### Planning Agents
- **Create planning session** with objectives and approach
- **Break down** work into implementable tasks
- **Create follow-up sessions** in `backlog/` for execution
- **Link** planning sessions to execution sessions

## 🔍 Session Quality Checklist

### Every Session Should Have:
- [ ] **Clear title** following naming convention
- [ ] **Session type** clearly identified
- [ ] **Objectives** or goals stated
- [ ] **Progress tracking** throughout work
- [ ] **Outcomes documented** when complete
- [ ] **Links** to related files/sessions
- [ ] **Next steps** or follow-up actions

### Session Metadata:
```yaml
---
title: "Session Title"
date: "YYYY-MM-DD"
status: "active|completed|blocked"
session_type: "research|plan|execute|debug|deploy|refactor|test"
priority: "high|medium|low"
tags: ["relevant", "tags"]
author: "agent-name"
related: ["path/to/related/sessions"]
---
```

## 🚨 Common Pitfalls to Avoid

### DON'T:
❌ Create multiple active sessions for the same type of work  
❌ Skip session creation for significant agent work  
❌ Leave sessions in `active/` after work is complete  
❌ Create sessions without clear objectives  
❌ Forget to link related work and artifacts  

### DO:
✅ Keep active sessions to 1-2 maximum  
✅ Update sessions throughout agent work  
✅ Move completed sessions promptly  
✅ Use consistent naming conventions  
✅ Document decisions and rationale  

## 🎉 Expected Benefits

### For Users:
- **Clear work tracking** across all agent interactions
- **Session continuity** between different agent types
- **Audit trail** of all development work
- **Context preservation** for future work

### For Agents:
- **Better context awareness** of ongoing work
- **Reduced duplicate effort** through session discovery
- **Consistent output patterns** across agent types
- **Improved collaboration** between different agents

---

**Implementation Status**: Ready for immediate adoption by all agents  
**Next Step**: Update agent prompts to include session management patterns
