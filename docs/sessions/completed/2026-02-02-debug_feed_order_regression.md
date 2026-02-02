---
title: "Debug: Feed Order Regression After Manual Dispatch"
date: "2026-02-02"
status: "completed"
session_type: "debug"
priority: "high"
tags: ["dashboard", "feed", "regression", "react"]
author: "gpt-5.2-codex"
related: []
---

# Debug: Feed Order Regression After Manual Dispatch

**Session Type**: DEBUG  
**Priority**: High  
**Estimated Duration**: 2-3 hours  
**Status**: Completed

## 🎯 Objective
Stabilize the Live Feed ordering so manual dispatches appear at the top after classification.

## 📋 Success Criteria
- [x] Manual dispatch item renders at the top of the feed consistently.
- [x] Feed ordering remains stable across refreshes and re-renders.
- [x] Regression is documented for future prevention.

## 🔍 Context
Manual dispatches appeared mid-feed even when the state array was prepended. Debugging revealed a reconciliation/order issue in the rendered list.

## 📝 Progress Log
- Inspected feed ordering, filters, and timestamps.
- Verified manual dispatch items were index 0 in state.
- Identified React reconciliation behavior as likely cause.
- Applied composite key including index to force DOM order to match array order.
- Added a smoke test to enforce render order.

## 🎉 Outcomes
- Manual dispatch renders at top reliably.
- Ordering regression addressed via composite key.
- Added a render-order test for `FeedPanel`.

## 🔗 Related Work
- `_vendor/figma_make/src/app/components/dashboard/FeedPanel.tsx`
- `_vendor/figma_make/src/app/components/dashboard/FeedPanel.test.tsx`
- `_vendor/figma_make/src/app/App.tsx`

## 📈 Next Steps
- [ ] Consider a dedicated `sortKey` if list ordering becomes more complex.
