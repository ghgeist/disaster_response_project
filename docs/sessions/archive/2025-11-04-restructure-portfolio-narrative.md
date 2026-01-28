---
title: "Restructure Portfolio Narrative for Storm Signal"
date: "2025-11-04"
status: "active"
tags: ["portfolio", "narrative", "documentation"]
author: "grant"
related: []
---

# Restructure Portfolio Narrative for Storm Signal

## Goal
Make it immediately clear that this project solves a real-world problem, accessible to anyone, while providing technical depth for technical audiences.

## Current State
- README is technical-first, missing clear problem statement
- Web app hero is generic ("Emergency Message Intelligence")
- Technical details are mixed with narrative
- Real-world impact is buried in technical documentation

## Strategy: Layered Narrative

### Layer 1: Problem → Solution → Impact (Everyone)
- Clear problem statement: What happens during disasters?
- Solution: How does this tool help?
- Impact: Why does this matter?

### Layer 2: Technical Details (Technical Readers)
- Architecture, implementation details, metrics
- Accessible but clearly marked as technical

## Implementation Plan

### Phase 1: Hero/Narrative (Web App) - PRIORITY
**File**: `app/templates/home.html`
- **Hero Section**: Rewrite with clear problem → solution → impact
  - Problem: During disasters, thousands of messages flood emergency channels
  - Solution: Storm Signal automatically categorizes messages so responders can prioritize
  - Impact: Faster response times, better resource allocation
- **Make "How It Works" more accessible**: Plain language, avoid jargon
- **Add real-world context**: "In a disaster with 10,000 messages, this tool..."

### Phase 2: README Summary Section
**File**: `README.md`
- Add "At a Glance" summary section at very top (before current content)
- Keep all existing technical content below (maintains structure for AI agents)
- Summary: 3-4 paragraphs covering problem, solution, key results

### Phase 3: Technical Details Layer (Web App) - DEFERRED
- Add collapsible technical sections to web app (Option A)
- Can be implemented after shipping initial narrative improvements

## Files to Modify

1. `app/templates/home.html` - Update hero section with problem-first narrative
2. `README.md` - Add summary section at top, keep all existing content

## Success Criteria

- ✅ Average person understands the problem and solution in 30 seconds
- ✅ Real-world impact is front and center
- ✅ No technical jargon in main narrative
- ✅ README maintains comprehensive structure for AI agents
- ✅ Web app hero clearly communicates value proposition

## Decisions Made

1. **Technical Details Placement**: Option A - Collapsible sections in web app (implement later)
2. **README**: Add summary section at top, keep comprehensive version (needed for AI agents)
3. **Web App Priority**: Focus on hero/narrative first (technical layer deferred)
