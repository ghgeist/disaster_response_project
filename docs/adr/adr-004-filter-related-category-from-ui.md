---
title: "Filter 'related' category from UI display"
date: "2025-09-15"
status: "accepted"
tags: ["ui", "ux", "classification", "bug-fix"]
author: "AI Assistant"
related: ["adr-008-class-weighting-over-sampling.md", "adr-009-algorithm-selection-logistic-regression-over-random-forest.md"]
---

# Filter 'related' category from UI display

**Date**: 2025-09-15  
**Status**: Accepted  
**Deciders**: AI Assistant, User  
**Tags**: ui, ux, classification, bug-fix

## Context

The disaster response classification system was displaying the 'related' category as a predicted category in the UI alongside specific disaster types (e.g., "Child Alone", "Medical Help"). This created user confusion because:

1. The 'related' category is a meta-category indicating disaster relevance (0=not related, 1=related, 2=ambiguous)
2. It's not a specific disaster type like "water", "food", or "medical_help"
3. Users were seeing "Related" as a category tag, which was semantically confusing
4. The mock service was hardcoded to always return `related=1` for any input, making testing unreliable
5. With class weighting (ADR-008) and LogisticRegression (ADR-009), out-of-distribution or non-message input (e.g. Lorem Ipsum) can score high on "related" (~66%) because the "related" head defaults toward the training prior (~76% positive). Filtering "related" from API/UI avoids showing that misleading signal.

## Decision

Filter the 'related' category from UI display while maintaining its internal use for classification logic:

1. **Backend filtering**: Exclude 'related' from `sorted_predictions` in both GET and POST routes
2. **Frontend filtering**: Remove 'related' from confidence chart display
3. **Mock service fix**: Implement intelligent keyword-based detection for 'related' category
4. **Enhanced UX**: Add contextual messaging to distinguish between disaster-related messages with/without specific categories

## Consequences

### Positive
- **Clearer UI**: Users only see specific disaster categories, eliminating confusion
- **Better UX**: Contextual messaging explains disaster relevance status appropriately
- **Accurate testing**: Mock service now provides realistic 'related' category predictions
- **Maintained functionality**: 'related' category still used internally for proper classification logic
- **Consistent behavior**: Both production and testing environments handle 'related' category correctly

### Negative  
- **Additional complexity**: More filtering logic in routes and templates
- **Potential confusion**: Users might not understand why some disaster-related messages show no specific categories

### Neutral
- **No performance impact**: Filtering is lightweight and doesn't affect model inference
- **Backward compatibility**: Existing model predictions continue to work unchanged

## Alternatives Considered

1. **Keep 'related' in UI with different styling**: Would still confuse users about what 'related' means
2. **Rename 'related' to 'disaster-relevant'**: Doesn't solve the fundamental issue of it being a meta-category
3. **Remove 'related' category entirely**: Would break the model's internal classification logic
4. **Show 'related' only in admin/debug views**: Adds complexity without user benefit

## References

- [Column definitions showing 'related' as meta-category](src/disasterproject/data/column_definitions.py)
- [Model labels including 'related' as first category](model/label_order.json)
- [UI filtering implementation](app/routes.py)
- [Mock service fix](app/utils.py)
- [Template filtering for confidence chart](app/templates/results.html)
