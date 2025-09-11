# Agent Standardization Implementation Script

**Purpose**: Update all agents to use centralized Cursor integration and session management rules  
**Status**: Implementation Guide  
**Date**: 2025-09-12

## 🎯 Standardization Approach

### 1. Centralized Rules Created
- ✅ `_cursor-integration-standard.md` - Standard Cursor tool usage patterns
- ✅ `_session-management-core.md` - Mandatory session management rules
- ✅ Applied to: `flask-ui-ux-agent.md`, `code-improvement-agent.md`

### 2. Remaining Agents to Update

#### Replace Custom Cursor Integration Sections With:
```markdown
## CURSOR INTEGRATION

**STANDARD INTEGRATION**: Follow the standard Cursor integration patterns defined in `docs/agents/_cursor-integration-standard.md`.

**MANDATORY SESSION MANAGEMENT**: Follow session management rules in `docs/agents/_session-management-core.md`.

### [Agent-Type] Specific Tool Usage
[Only agent-specific tool usage patterns that differ from standard]
```

#### Agents Needing Updates:
- [ ] `performance-agent.md`
- [ ] `security-agent.md` 
- [ ] `debug-agent.md`
- [ ] `plan-agent.md`
- [ ] `integrate-agent.md`
- [ ] `test-agent.md`
- [ ] `refactor-agent.md`
- [ ] `machine-learning-engineer-agent.md`
- [ ] `coding-session-manager.md`
- [ ] `release-orchestrator-agent.md` (already has good Cursor integration)

### 3. Standard Replacement Pattern

**Find sections like:**
```markdown
## CURSOR INTEGRATION

You operate within Cursor IDE with access to [various tools]:

### Discovery Phase
- Use `codebase_search` to...
- Use `grep` to...
[etc.]
```

**Replace with:**
```markdown
## CURSOR INTEGRATION

**STANDARD INTEGRATION**: Follow the standard Cursor integration patterns defined in `docs/agents/_cursor-integration-standard.md`.

**MANDATORY SESSION MANAGEMENT**: Follow session management rules in `docs/agents/_session-management-core.md`.

### [Agent-Type] Specific Tool Usage
[Only unique patterns for this agent type]
```

## 🔧 Benefits of Centralization

### Consistency
- All agents follow identical session management rules
- Standardized Cursor tool usage patterns
- Uniform session file naming and structure

### Maintainability  
- Single source of truth for session rules
- Easy to update all agents by changing central files
- Reduced duplication across agent prompts

### Enforcement
- Clear reference documents for session management
- Mandatory compliance with centralized rules
- Quality gates defined in central location

## 📋 Implementation Checklist

### Phase 1: Core Infrastructure ✅
- [x] Create `_cursor-integration-standard.md`
- [x] Create `_session-management-core.md` 
- [x] Update `flask-ui-ux-agent.md`
- [x] Update `code-improvement-agent.md`

### Phase 2: Remaining Agents
- [ ] Update all remaining agents with standardized references
- [ ] Remove duplicate Cursor integration sections
- [ ] Preserve agent-specific tool usage patterns
- [ ] Validate all agents reference central rules

### Phase 3: Validation
- [ ] Test agents follow session management rules
- [ ] Verify consistent session file creation
- [ ] Confirm proper session lifecycle management
- [ ] Validate centralized rule compliance

## 🎉 Expected Outcome

After standardization:
- **15 agents** all reference centralized rules
- **Single source of truth** for session management
- **Consistent behavior** across all agent types
- **Easy maintenance** through central rule updates
- **Enforced compliance** with session standards

---

**Next Step**: Apply standardization pattern to all remaining agents
