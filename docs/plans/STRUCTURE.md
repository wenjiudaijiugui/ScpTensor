# Implementation Plan Structure

## Document Overview

```
docs/plans/
├── README.md (Quick Reference - 5 KB)
├── 2026-02-25-core-refactoring-design.md (Design - 58 KB)
└── 2026-02-25-core-refactoring-implementation-plan.md (Implementation - 73 KB)

Total: 3 documents, ~136 KB
```

## Implementation Plan Structure

### 1. Header (Required)
```markdown
# ScpTensor Core Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans

**Goal:** Refactor ScpTensor codebase to reduce code by 30%
**Architecture:** Module-by-module refactoring with 8 parallel team members
**Tech Stack:** Python 3.12+, NumPy, SciPy, Polars, Numba, pytest, mypy, ruff
```

### 2. Table of Contents
- 13 main sections
- Hierarchical structure
- Module-by-module breakdown

### 3. Project Overview
- Current state table
- Timeline (6 weeks)
- Success criteria (3 categories)

### 4. Phase 0: Setup and Preparation
- Task 0.1: Create refactoring branch
- Task 0.2: Establish code quality tools

### 5. Module Sections (1-8)

Each module contains:
- Owner name
- Current LOC → Target LOC
- Duration
- **Detailed tasks with 5 steps each**

Example Task Structure:
```markdown
### Task X.Y: [Task Name]

**Goal:** [Brief description]

**Files:**
- Modify: `path/to/file.py:line-range`
- Test: `path/to/test.py:line-range`

**Step 1: Write the failing test**
[code]

**Step 2: Run test to verify it fails**
[command]

**Step 3: Write minimal implementation**
[code]

**Step 4: Run test to verify it passes**
[command]

**Step 5: Commit**
[command + message]
```

### 6. Testing Strategy
- Test coverage requirements
- Test organization
- Running tests commands

### 7. Validation Checklist
- Phase 1: Code Quality (5 items)
- Phase 2: Functional (4 items)
- Phase 3: Performance (4 items)
- Phase 4: Mathematical (4 items)
- Phase 5: API (5 items)

### 8. Risk Mitigation
- 5 major risks
- Mitigation strategies

### 9. Success Metrics
- Quantitative metrics table
- Qualitative metrics checklist

### 10. Timeline Summary
- Week-by-week breakdown

### 11. Appendix: Task Template
- Reusable 5-step template

## Module Breakdown

### Module 1: Core (15 tasks, ~75 steps)
- Task 1.1: Simplify ScpContainer initialization
- Task 1.2: Simplify Assay structure
- Task 1.3: Optimize ScpMatrix with slots
- Task 1.4: Simplify Provenance tracking
- Task 1.5: Extract mask operations
- Task 1.6: Add fluent interface
- Task 1.7: Simplify JIT operations
- Task 1.8-1.15: Continue pattern...

### Module 2: Normalization (12 tasks, ~60 steps)
- Task 2.1: Design unified interface
- Task 2.2: Refactor log normalization
- Task 2.3-2.10: Refactor remaining methods
- Task 2.11: Add main normalize() function
- Task 2.12: Update module exports

### Module 3: Imputation (10 tasks, ~50 steps)
- Task 3.1: Design unified interface
- Task 3.2: Refactor KNN imputation
- Task 3.3-3.10: Continue pattern...

### Module 4: Integration (10 tasks, ~50 steps)
- Task 4.1: Design unified interface
- Task 4.2: Refactor ComBat
- Task 4.3-4.10: Continue pattern...

### Module 5: QC (9 tasks, ~45 steps)
- Task 5.1: Reorganize module
- Task 5.2: Simplify pipeline
- Task 5.3-5.9: Continue pattern...

### Module 6: DR + Clustering (12 tasks, ~60 steps)
- Task 6.1: Refactor PCA
- Task 6.2: Refactor UMAP
- Task 6.3: Refactor KMeans
- Task 6.4-6.12: Continue pattern...

### Module 7: Feature Selection + DE (10 tasks, ~50 steps)
- Task 7.1: Refactor HVG
- Task 7.2: Refactor DE
- Task 7.3-7.10: Continue pattern...

### Module 8: Utils + Viz + I/O (12 tasks, ~60 steps)
- Task 8.1: Reorganize utils
- Task 8.2: Simplify viz base
- Task 8.3: Simplify viz recipes
- Task 8.4-8.12: Continue pattern...

## Total Scope

- **8 modules**
- **~90 tasks**
- **~450 steps**
- **6 weeks**
- **8 team members**

## Key Features

### Bite-Sized Tasks
- Each task is self-contained
- Clear file locations
- Complete code examples
- Accurate commands

### 5-Step Pattern
1. Write failing test (TDD)
2. Verify it fails
3. Write implementation
4. Verify it passes
5. Commit with message

### Comprehensive Examples
- Real code snippets
- Actual commands
- Expected outputs
- Commit message templates

### Parallel Work
- Each member owns complete module
- No file conflicts
- Independent validation
- Clear deliverables

## Usage

### For Team Lead
1. Read DESIGN document (strategy)
2. Assign modules to members
3. Track progress with checklist
4. Coordinate validation

### For Team Member
1. Read module section in IMPLEMENTATION
2. Follow 5-step pattern
3. Run tests after each step
4. Commit with provided template

### For QA
1. Use validation checklist
2. Run test suite
3. Verify metrics
4. Check mathematical correctness

## Success Indicators

### Document Quality
- [x] Clear structure
- [x] Complete examples
- [x] Accurate commands
- [x] Reusable templates

### Implementation Readiness
- [x] All modules covered
- [x] All tasks bite-sized
- [x] All steps documented
- [x] All risks addressed

### Team Coordination
- [x] Clear ownership
- [x] No conflicts
- [x] Parallel workflow
- [x] Shared timeline

---

**Status:** Complete and ready for implementation
**Total Documents:** 3 (Design, Implementation, README)
**Total Size:** ~136 KB
**Total Tasks:** ~90
**Total Steps:** ~450
