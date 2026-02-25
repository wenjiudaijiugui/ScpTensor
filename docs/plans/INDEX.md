# ScpTensor Refactoring Plans Index

**Last Updated:** 2026-02-25
**Status:** Ready for Implementation
**Version:** v0.2.0-beta

---

## Quick Navigation

### 🚀 Start Here
**[README.md](README.md)** - Quick reference guide (5 min read)

- Module ownership table
- Task template (5 steps)
- Common commands
- Success criteria

### 📖 Strategy & Design
**[2026-02-25-core-refactoring-design.md](2026-02-25-core-refactoring-design.md)** - Full design document (30 min read)

- Executive summary
- Refactoring principles (YAGNI, SRP, etc.)
- Team division and responsibilities
- Module-by-module specifications
- API simplification examples
- Mathematical verification requirements

### 📝 Implementation Plan
**[2026-02-25-core-refactoring-implementation-plan.md](2026-02-25-core-refactoring-implementation-plan.md)** - Detailed tasks (reference during implementation)

- Phase 0: Setup and preparation
- 8 modules with ~90 bite-sized tasks
- Each task has 5 steps with code examples
- Testing strategy
- Validation checklist
- Risk mitigation

### 📐 Document Structure
**[STRUCTURE.md](STRUCTURE.md)** - Implementation plan structure (5 min read)

- Document overview
- Module breakdown
- Usage guidelines
- Success indicators

---

## Document Summary

| Document | Size | Lines | Purpose | Audience |
|----------|------|-------|---------|----------|
| DESIGN.md | 58 KB | 2,436 | Strategy & architecture | All |
| IMPLEMENTATION.md | 73 KB | 3,134 | Detailed tasks | Implementers |
| README.md | 5 KB | 243 | Quick reference | All |
| STRUCTURE.md | 5 KB | 233 | Document structure | All |

**Total:** 136 KB, 6,046 lines

---

## How to Use These Documents

### For Team Lead
1. **Start:** Read [README.md](README.md) (5 min)
2. **Strategy:** Read [DESIGN.md](2026-02-25-core-refactoring-design.md) Sections 1-5 (20 min)
3. **Assign:** Review team division (Section 5 of DESIGN.md)
4. **Track:** Use validation checklist (Section 12 of IMPLEMENTATION.md)

### For Module Member
1. **Start:** Read [README.md](README.md) (5 min)
2. **Understand:** Read your module section in [DESIGN.md](2026-02-25-core-refactoring-design.md) (10 min)
3. **Implement:** Follow tasks in [IMPLEMENTATION.md](2026-02-25-core-refactoring-implementation-plan.md) (reference)
4. **Validate:** Use validation checklist

### For QA/Reviewer
1. **Start:** Read [README.md](README.md) (5 min)
2. **Strategy:** Skim [DESIGN.md](2026-02-25-core-refactoring-design.md) Sections 2-3 (10 min)
3. **Validate:** Use validation checklist (Section 12 of IMPLEMENTATION.md)
4. **Verify:** Run tests and check metrics

---

## Key Sections

### Design Document Highlights

**Section 1: Executive Summary** (5 min)
- Overview and strategic priorities
- Scope and goals

**Section 2: Quantitative Objectives** (5 min)
- Code metrics targets
- Module-specific targets
- User experience metrics

**Section 3: Refactoring Principles** (10 min)
- YAGNI (You Aren't Gonna Need It)
- Single Responsibility Principle
- Avoid Premature Abstraction
- Functional Style

**Section 5: Team Division** (10 min)
- 8 members and their responsibilities
- Module ownership
- Success criteria per module

**Section 7: Module-by-Module Specifications** (30 min)
- Detailed specifications for each module
- Proposed APIs
- Key improvements

### Implementation Plan Highlights

**Phase 0: Setup and Preparation** (1 day)
- Task 0.1: Create refactoring branch
- Task 0.2: Establish code quality tools

**Module Sections 1-8** (25 days)
- Each module has 9-15 tasks
- Each task has 5 steps with complete code examples
- Parallel work structure

**Testing Strategy** (reference)
- Test coverage requirements
- Test organization
- Running tests commands

**Validation Checklist** (reference)
- 5 phases of validation
- 22 total checklist items

**Risk Mitigation** (reference)
- 5 major risks
- Mitigation strategies

---

## Task Template

Every task in the implementation plan follows this pattern:

```markdown
### Task X.Y: [Task Name]

**Goal:** [Brief description]

**Files:**
- Modify: `path/to/file.py:line-range`
- Test: `path/to/test.py:line-range`

**Step 1: Write the failing test**
[code example]

**Step 2: Run test to verify it fails**
[command]

**Step 3: Write minimal implementation**
[code example]

**Step 4: Run test to verify it passes**
[command]

**Step 5: Commit**
[command + message template]
```

---

## Common Commands

### Setup
```bash
git checkout -b refactor/2026-02-core-refactoring
uv sync
uv pip install -e ".[dev]"
```

### Testing
```bash
uv run pytest tests/ -v
uv run pytest --cov=scptensor --cov-report=html
```

### Code Quality
```bash
uv run ruff check scptensor/
uv run ruff format scptensor/
uv run mypy scptensor/
```

### Metrics
```bash
radon cc scptensor/ -a
lizard scptensor/
find scptensor/ -name "*.py" | xargs wc -l
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

## Module Ownership

| Member | Module(s) | Tasks | Steps |
|--------|-----------|-------|-------|
| 1 | core/ | 15 | ~75 |
| 2 | normalization/ | 12 | ~60 |
| 3 | impute/ | 10 | ~50 |
| 4 | integration/ | 10 | ~50 |
| 5 | qc/ | 9 | ~45 |
| 6 | dim_reduction/ + cluster/ | 12 | ~60 |
| 7 | feature_selection/ + diff_expr/ | 10 | ~50 |
| 8 | utils/ + viz/ + io/ | 12 | ~60 |

**Total:** 8 members, ~90 tasks, ~450 steps

---

## Questions?

### "Where do I start?"
→ Read [README.md](README.md)

### "What's the strategy?"
→ Read [DESIGN.md](2026-02-25-core-refactoring-design.md) Sections 1-3

### "What are my tasks?"
→ Find your module in [IMPLEMENTATION.md](2026-02-25-core-refactoring-implementation-plan.md)

### "How do I validate?"
→ Use validation checklist in [IMPLEMENTATION.md](2026-02-25-core-refactoring-implementation-plan.md) Section 12

### "What's the structure?"
→ Read [STRUCTURE.md](STRUCTURE.md)

---

**Status:** ✅ Complete and ready for implementation
**Created:** 2026-02-25
**Version:** 1.0
**Team:** ScpTensor
