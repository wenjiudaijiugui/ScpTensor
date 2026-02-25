# ScpTensor Refactoring Plans - Quick Reference

## Available Documents

### 1. Design Document (58 KB)
**File:** `2026-02-25-core-refactoring-design.md`

**Purpose:** Strategic overview and architectural specifications

**Contents:**
- Executive Summary
- Quantitative Objectives
- Refactoring Principles (YAGNI, SRP, etc.)
- Current Codebase Analysis
- Team Division (8 members)
- Module-by-Module Specifications
- API Simplification Examples
- Mathematical Verification Requirements

**When to Read:**
- Understanding overall strategy
- Learning refactoring principles
- Reviewing architectural decisions
- Understanding team structure

### 2. Implementation Plan (73 KB)
**File:** `2026-02-25-core-refactoring-implementation-plan.md`

**Purpose:** Detailed, bite-sized tasks for implementation

**Contents:**
- Phase 0: Setup and Preparation
- 8 Modules with detailed tasks
- Each task has 5 steps (Test → Run → Implement → Run → Commit)
- Testing Strategy
- Validation Checklist
- Risk Mitigation

**When to Read:**
- Starting implementation work
- Writing specific tasks
- Running tests
- Committing changes

---

## Quick Navigation

### For Team Leads

1. **Start here:** Design Document - Sections 1-3 (Executive Summary, Objectives, Principles)
2. **Team setup:** Design Document - Section 5 (Team Division)
3. **Track progress:** Implementation Plan - Validation Checklist

### For Module Members

1. **Understand your module:** Design Document - Section 7 (Module Specifications)
2. **Start implementation:** Implementation Plan - Your module section
3. **Follow task template:** Implementation Plan - Appendix

---

## Module Ownership

| Member | Module(s) | Current LOC | Target LOC | Tasks |
|--------|-----------|-------------|------------|-------|
| 1 | core/ | 5,475 | ~3,800 | 15 tasks |
| 2 | normalization/ | 658 | ~450 | 12 tasks |
| 3 | impute/ | 2,381 | ~1,600 | 10 tasks |
| 4 | integration/ | 1,708 | ~1,150 | 10 tasks |
| 5 | qc/ | 3,589 | ~2,400 | 9 tasks |
| 6 | dim_reduction/ + cluster/ | 1,723 | ~1,170 | 12 tasks |
| 7 | feature_selection/ + diff_expr/ | 4,185 | ~2,800 | 10 tasks |
| 8 | utils/ + viz/ + io/ | 12,992 | ~8,350 | 12 tasks |

---

## Task Template (5 Steps)

Every task in the Implementation Plan follows this pattern:

### Step 1: Write the failing test
```python
def test_feature():
    """Test description."""
    # Arrange, Act, Assert
```

### Step 2: Run test to verify it fails
```bash
uv run pytest path/to/test.py::test_name -v
# Expected: FAIL
```

### Step 3: Write minimal implementation
```python
def function_name(data):
    """Docstring with Parameters, Returns, Examples."""
    return result
```

### Step 4: Run test to verify it passes
```bash
uv run pytest path/to/test.py::test_name -v
# Expected: PASS
```

### Step 5: Commit
```bash
git add [files]
git commit -m "refactor(module): description

- Change 1
- Change 2

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Common Commands

### Setup
```bash
# Create refactoring branch
git checkout -b refactor/2026-02-core-refactoring

# Install dependencies
uv sync
uv pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
uv run pytest tests/ -v

# Run specific module
uv run pytest tests/core/ -v

# Run with coverage
uv run pytest --cov=scptensor --cov-report=html
```

### Code Quality
```bash
# Lint
uv run ruff check scptensor/

# Format
uv run ruff format scptensor/

# Type check
uv run mypy scptensor/
```

### Metrics
```bash
# Cyclomatic complexity
radon cc scptensor/ -a

# Code duplication
lizard scptensor/

# Line counts
find scptensor/ -name "*.py" -type f | xargs wc -l
```

---

## Success Criteria

### Code Metrics
- [ ] Total LOC: 34,223 → ~24,000 (-30%)
- [ ] Functions < 50 lines
- [ ] Complexity < 10
- [ ] Type coverage > 95%
- [ ] Test coverage > 85%

### Quality Metrics
- [ ] All tests pass
- [ ] Zero regressions
- [ ] Performance maintained
- [ ] Math verified

### Developer Experience
- [ ] API calls 40% shorter
- [ ] Parameters 40% fewer
- [ ] Full chaining support
- [ ] Auto type inference

---

## Timeline

| Week | Phase | Focus |
|------|-------|-------|
| 1 | Setup & Analysis | Branch, baseline metrics |
| 2-3 | Implementation (1-4) | Core, Norm, Impute, Integration |
| 4-5 | Implementation (5-8) | QC, DR/Cluster, Feat/DE, Utils |
| 5 | Validation | Tests, benchmarks, math |
| 6 | Release | Docs, migration guide, v0.2.0 |

---

## Risk Mitigation

### Breaking Changes
- Comprehensive test suite
- Incremental changes
- Migration guide
- Version bump to v0.2.0

### Performance Regression
- Benchmark suite
- Performance tests in CI
- JIT compilation preserved

### Coordination
- Clear module ownership
- Daily syncs
- Shared task tracking

---

## Questions?

### Design Questions
See: `2026-02-25-core-refactoring-design.md`

### Implementation Questions
See: `2026-02-25-core-refactoring-implementation-plan.md`

### Quick Start
1. Read Design Document (Sections 1-3)
2. Find your module in Implementation Plan
3. Start with Task 0.1 (Setup)
4. Follow 5-step pattern for each task

---

**Last Updated:** 2026-02-25
**Status:** Ready for Implementation
