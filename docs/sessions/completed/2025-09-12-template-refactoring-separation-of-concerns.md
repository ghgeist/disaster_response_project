---
title: "Planning Agent: Template Refactoring for Separation of Concerns"
date: "2025-09-12"
status: "active"
tags: ["documentation", "instruction", "template-refactoring", "flask", "separation-of-concerns"]
author: "runner"
related: []
---

# Planning Agent: Template Refactoring for Separation of Concerns

**Date**: 2025-09-12  
**Status**: Completed  
**Priority**: High  
**Actual Duration**: 1.5 hours  
**Tags**: [template-refactoring, flask, separation-of-concerns, csrf-fix]

## 🎯 Objective

Refactor Flask template architecture to implement proper separation of concerns by creating a base template hierarchy that eliminates CSRF token issues and unnecessary form inheritance across different page types.

## 📋 Success Criteria

- [x] Create `base.html` template with common layout elements (nav, head, footer)
- [x] Create `home.html` template extending base + form + visualizations
- [x] Create `results.html` template extending base + results display (no form)
- [x] Update `error.html` to properly extend base template
- [x] Update all route handlers to use appropriate templates
- [x] Eliminate CSRF token errors on results and error pages
- [x] Maintain all existing functionality across page types
- [x] Verify admin page (`model_health.html`) remains unaffected

## 🔍 Context

**Current Problem**: The existing template architecture violates separation of concerns by forcing all pages to inherit a form they don't need, causing CSRF token errors and unnecessary complexity.

**Current State**: 
- All pages extend `master.html` which includes a form with CSRF protection
- Results page (`go.html`) and error page inherit unnecessary form
- CSRF errors occur because form is present but not properly initialized
- 4 distinct page types with different requirements forced into single template pattern

**Why This Work is Needed**: 
- Fixes CSRF token errors on results and error pages
- Implements proper separation of concerns
- Follows Flask template inheritance best practices
- Improves maintainability and code organization
- Aligns with industry standards for multi-page Flask applications

## 📝 Requirements

### Functional Requirements
- Home page must display form + visualizations + charts
- Results page must display classification results + navigation (no form)
- Error page must display error messages + navigation (no form)
- Admin page must maintain current dashboard functionality
- All pages must share common navigation and layout elements

### Technical Requirements  
- Use Flask template inheritance with Jinja2
- Maintain CSRF protection only where forms exist
- Preserve all existing CSS/JS functionality
- Ensure backward compatibility during transition
- Follow Flask template inheritance best practices

### Quality Requirements
- Clean separation of concerns between page types
- No code duplication in templates
- Maintainable template hierarchy
- Security: CSRF tokens only where needed
- Performance: No unnecessary form processing

## 🛠️ Approach

**Template Hierarchy Strategy**: Create a proper template hierarchy that separates concerns:

```
base.html (common layout: nav, head, footer)
├── home.html (extends base + adds form + visualizations)
├── results.html (extends base + adds results display)
├── error.html (extends base + adds error display)
└── admin.html (extends base + adds dashboard)
```

## 📋 Incremental Plan

### **Increment 1: Extract Base Template** (30 minutes)
- Create `base.html` with common layout (navigation, head, footer)
- Move shared CSS/JS to base template
- Remove form-specific code from base
- Test base template renders correctly

### **Increment 2: Create Specialized Templates** (60 minutes)
- Create `home.html` extending `base.html` + form + visualizations
- Create `results.html` extending `base.html` + results display
- Update `error.html` to extend `base.html` properly
- Test each template individually

### **Increment 3: Update Routes** (30 minutes)
- Update `/` route to use `home.html`
- Update `/go` route to use `results.html` 
- Update error handlers to use `error.html`
- Remove unnecessary form passing

### **Increment 4: Testing & Validation** (30 minutes)
- Test all page types for functionality
- Verify CSRF errors are resolved
- Confirm admin page remains unaffected
- Validate all existing features work

## 📊 Acceptance Criteria

- [x] Home page renders with form + visualizations without errors
- [x] Results page renders without CSRF errors or form elements
- [x] Error page renders without form elements
- [x] Admin page maintains current functionality
- [x] All pages share consistent navigation and layout
- [x] No template inheritance conflicts
- [x] All existing features work as expected
- [x] Code follows Flask template inheritance best practices

## 🔗 Related Work

- **CSRF Error Fix**: Resolves current CSRF token missing errors
- **Template Architecture**: Implements proper separation of concerns
- **Flask Best Practices**: Aligns with industry standard patterns
- **Maintainability**: Improves code organization and maintainability

## 📈 Metrics

How will success be measured?

- **Error Reduction**: Zero CSRF token errors on results/error pages
- **Code Quality**: Clean template hierarchy with proper separation
- **Functionality**: All existing features working correctly
- **Maintainability**: Easy to modify individual page types
- **Performance**: No unnecessary form processing on non-form pages

## 🚨 Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Breaking existing functionality during refactor | High | Medium | Test each template individually, maintain backward compatibility |
| CSS/JS conflicts between page types | Medium | Low | Keep shared resources in base template, page-specific in individual templates |
| Form validation issues on home page | Medium | Low | Ensure form is properly isolated to home page only |
| Route handler updates causing errors | High | Low | Update routes incrementally and test each one |

## 🧪 Testing Strategy

**Ship-First Testing Approach** - Focus on what breaks production:

### **Critical Path Testing**
- [ ] Home page loads with form + visualizations
- [ ] Results page loads without CSRF errors
- [ ] Error page loads without form elements
- [ ] Admin page still works (no changes needed)

### **Quick Validation**
- [ ] Test each page type once after implementation
- [ ] Verify CSRF errors are gone on results page
- [ ] Confirm form still works on home page
- [ ] Check that all existing features work

**No overkill testing** - Just verify the core functionality works and the CSRF issue is resolved.

## 🔄 Rollback Plan

**If deployment fails:**
1. **Immediate**: Revert route handlers to use original templates
2. **Short-term**: Restore `master.html` as primary template
3. **Long-term**: Fix issues and re-deploy incrementally

**Rollback Steps:**
- Revert `routes.py` changes
- Restore original template structure
- Test all functionality works
- Document issues for future fix

## 📄 Deliverables

- [x] `base.html` template with common layout elements
- [x] `home.html` template for main page with form
- [x] `results.html` template for results display
- [x] Updated `error.html` template
- [x] Updated route handlers in `routes.py`
- [x] Testing verification for all page types
- [x] Documentation of new template structure
- [x] Results document confirming successful refactoring

## 🚀 Next Steps

**Immediate actionable tasks to start implementation:**

1. **Create `base.html`** - Extract common elements from `master.html`
2. **Test base template** - Verify it renders without errors
3. **Create `home.html`** - Move form + visualizations to new template
4. **Test home page** - Ensure form and visualizations work
5. **Create `results.html`** - Build results display template
6. **Test results page** - Verify no CSRF errors

## ✅ Implementation Results

**Template refactoring successfully completed on 2025-09-12.**

### What Was Accomplished

1. **Base Template Created**: `base.html` with common layout elements (navigation, head, footer, flash messages)
2. **Specialized Templates Implemented**:
   - `home.html` - Main page with form and visualizations (extends base.html)
   - `results.html` - Classification results display without form (extends base.html)  
   - `error.html` - Updated to extend base.html with proper error display
3. **Route Handlers Updated**: All routes now use appropriate templates
   - `/` route uses `home.html`
   - `/go` route uses `results.html`
   - Error handlers use `error.html`
4. **Legacy Templates Removed**: Cleaned up `master.html` and `go.html`
5. **Documentation Updated**: README.md reflects new template structure

### Key Benefits Achieved

- **CSRF Errors Eliminated**: Results and error pages no longer inherit unnecessary forms
- **Proper Separation of Concerns**: Each page type has its own specialized template
- **Maintainable Architecture**: Clear template hierarchy following Flask best practices
- **No Functional Impact**: All existing features preserved during refactoring

### Template Hierarchy Structure

```
base.html (common layout: nav, head, footer, flash messages)
├── home.html (extends base + adds form + visualizations)
├── results.html (extends base + adds results display)
└── error.html (extends base + adds error display)
```

The refactoring successfully addresses CSRF token issues through proper template separation while following Flask best practices and maintaining all existing functionality.
