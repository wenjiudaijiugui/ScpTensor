# ScpTensor Design Documents Index

**Purpose:** Quick reference for locating specific information in design documents
**Last Updated:** 2025-01-14
**Usage:** Use `python scripts/doc_loader.py <doc> <lines>` to load specific sections

---

## Quick Reference

| Document | Lines | Purpose | When to Use |
|----------|-------|---------|-------------|
| MASTER.md | 1-653 | Strategic overview + navigation | Understanding project status, priorities |
| ARCHITECTURE.md | 1-1099 | Module specifications, data structures | Designing new features, understanding API |
| ROADMAP.md | 1-287 | Execution plan, milestones, sprints | Planning work, tracking progress |
| MIGRATION.md | 1-791 | v0.1.0-alpha → v0.1.0-beta guide | Upgrading from alpha to beta |
| API_REFERENCE.md | 1-1184 | Complete API documentation | Using ScpTensor APIs |

---

## Document Navigation

### MASTER.md (26 KB, ~650 lines)

**Document Path:** `docs/design/MASTER.md`

**Status:** ✅ Updated for v0.1.0-beta

**Sections:**
```
Lines 1-50:     Executive Summary (current state - v0.1.0-beta)
Lines 51-150:   Architecture Overview (module structure, data flow)
Lines 151-250:  Priority Matrix (P0/P1/P2 tasks with completion status)
Lines 251-300:  Document Ecosystem (navigation hub)
Lines 301-350:  Maintenance Protocol
Lines 351-400:  Success Metrics
Lines 600-653:  Document History and Appendix
```

**Quick Commands:**
```bash
# Load executive summary
python scripts/doc_loader.py MASTER 1-50

# Load priority matrix
python scripts/doc_loader.py MASTER 151-250

# Load full document
python scripts/doc_loader.py MASTER all
```

**When to Reference:**
- Project status check
- Understanding critical path
- Resource allocation decisions
- Document navigation

---

### ARCHITECTURE.md (30 KB, ~1100 lines)

**Document Path:** `docs/design/ARCHITECTURE.md`

**Status:** ✅ Complete

**Sections:**
```
Lines 1-100:    Module Responsibility Matrix
Lines 101-200:  Data Structure Specifications
  - Lines 120-150: ScpContainer
  - Lines 151-180: Assay
  - Lines 181-220: ScpMatrix
Lines 201-300:  Design Patterns
  - Lines 210-240: Immutable Layer Creation
  - Lines 241-270: Mask-Based Provenance
Lines 301-450:  Module APIs
  - Lines 310-350: normalization module
  - Lines 351-400: impute module
  - Lines 401-450: integration module
Lines 451-550:  Integration Patterns
Lines 551-650:  Error Handling Strategy
Lines 651-750:  Performance Considerations
Lines 751-850:  Extension Points
Lines 851-950:  Testing Strategy
Lines 951-1050: Deprecation Policy
Lines 1051-1100: Appendix (standards, glossary)
```

**Quick Commands:**
```bash
# Load module responsibility matrix
python scripts/doc_loader.py ARCHITECTURE 1-100

# Load ScpContainer specification
python scripts/doc_loader.py ARCHITECTURE 120-150

# Load normalization module API
python scripts/doc_loader.py ARCHITECTURE 310-350

# Load design patterns
python scripts/doc_loader.py ARCHITECTURE 201-300
```

**When to Reference:**
- Designing new modules
- Understanding data structures
- Implementing new features
- Code review guidance
- API changes impact analysis

---

### ROADMAP.md (10 KB, ~287 lines)

**Document Path:** `docs/design/ROADMAP.md`

**Status:** ✅ Updated for v0.1.0-beta

**Sections:**
```
Lines 1-50:     Executive Summary (v0.1.0-beta complete)
Lines 51-90:    Priority Matrix Detail (P0/P1/P2 status)
Lines 91-197:   Milestones (M1-M5 all complete)
Lines 200-240:  v0.2.0 Planning
Lines 218-240:  Sprint Retrospective
Lines 241-287:  Effort Tracking and Change Log
```

**Quick Commands:**
```bash
# Load executive summary
python scripts/doc_loader.py ROADMAP 1-50

# Load completed milestones
python scripts/doc_loader.py ROADMAP 91-197

# Load v0.2.0 planning
python scripts/doc_loader.py ROADMAP 200-240
```

**When to Reference:**
- Sprint planning
- Task prioritization
- Progress tracking
- Risk assessment
- Timeline estimation

---

### MIGRATION.md (18 KB, ~791 lines)

**Document Path:** `docs/design/MIGRATION.md`

**Status:** ✅ Complete

**Sections:**
```
Lines 1-50:     Quick Start & Checklist
Lines 51-150:   Breaking Changes
  - Lines 60-80: Import path changes
  - Lines 81-110: Function signatures
  - Lines 111-140: Error handling
  - Lines 141-150: Dependency versions
Lines 151-250:  Data Compatibility
Lines 251-350:  Migration Strategies
  - Lines 260-280: Incremental migration
  - Lines 281-310: Parallel validation
  - Lines 311-330: Complete rewrite
Lines 351-500:  Step-by-Step Migration
  - Lines 360-380: Environment setup
  - Lines 381-400: Code audit
  - Lines 401-420: Update imports
  - Lines 421-500: Validation & testing
Lines 501-600:  FAQ & Troubleshooting
Lines 601-791:  Reference and Appendices
```

**Quick Commands:**
```bash
# Load breaking changes
python scripts/doc_loader.py MIGRATION 51-150

# Load migration strategies
python scripts/doc_loader.py MIGRATION 251-350

# Load step-by-step guide
python scripts/doc_loader.py MIGRATION 351-500

# Load FAQ
python scripts/doc_loader.py MIGRATION 501-600
```

**When to Reference:**
- Upgrading from alpha to beta
- Debugging migration issues
- Planning upgrade timeline
- Understanding breaking changes

---

### API_REFERENCE.md (26 KB, ~1184 lines)

**Document Path:** `docs/design/API_REFERENCE.md`

**Status:** ✅ Complete

**Sections:**
```
Lines 1-80:     Table of Contents
Lines 81-200:   Core Data Structures
  - Lines 100-130: ScpContainer
  - Lines 131-160: Assay
  - Lines 161-190: ScpMatrix
Lines 201-350:   Normalization Module
  - Lines 220-250: log_normalize
  - Lines 251-280: Other normalization methods
Lines 351-500:   Imputation Module
  - Lines 380-420: knn
  - Lines 421-460: ppca, svd
Lines 501-600:   Integration Module
  - Lines 520-550: combat
  - Lines 551-580: harmony
Lines 601-700:   QC Module
Lines 701-800:   Dimensionality Reduction
  - Lines 720-750: pca
  - Lines 751-780: umap
Lines 801-1184: Clustering, Visualization, Utils, Appendices
```

**Quick Commands:**
```bash
# Load core structures
python scripts/doc_loader.py API_REFERENCE 81-200

# Load specific module
python scripts/doc_loader.py API_REFERENCE 351-500  # impute

# Load specific function
python scripts/doc_loader.py API_REFERENCE 380-420  # knn function
```

**When to Reference:**
- Using ScpTensor APIs
- Writing integration code
- Understanding function signatures
- Checking parameter types
- Finding examples

---

## Additional Documentation (New in v0.1.0-beta)

| Document | Location | Purpose |
|----------|----------|---------|
| COMPETITOR_BENCHMARK.md | docs/ | Benchmark vs Scanpy, Seurat etc. |
| DEVELOPER_GUIDE.md | docs/ | Developer onboarding and workflow |
| CONTRIBUTING.md | docs/ | Contribution guidelines |
| COMPLETION_STATUS.md | docs/ | Detailed task completion tracking |
| PROJECT_STATUS.md | docs/ | Current project overview |
| Tutorials | docs/tutorials/ | 4 Jupyter notebooks for learning |

---

## Common Use Cases

### Use Case 1: Starting Development on a New Feature

**Load Order:**
1. `MASTER.md` lines 1-50 (Executive Summary) - Understand project status
2. `ARCHITECTURE.md` lines 1-100 (Module Responsibility) - Find your module
3. `ARCHITECTURE.md` lines 751-850 (Extension Points) - Learn how to extend
4. `API_REFERENCE.md` relevant section - See similar APIs

**Commands:**
```bash
python scripts/doc_loader.py MASTER 1-50
python scripts/doc_loader.py ARCHITECTURE 1-100
python scripts/doc_loader.py ARCHITECTURE 751-850
```

---

### Use Case 2: Fixing a Critical Bug

**Load Order:**
1. `ISSUES_AND_LIMITATIONS.md` - Check if known issue
2. `ARCHITECTURE.md` lines 551-650 (Error Handling) - Understand error strategy
3. `API_REFERENCE.md` relevant function - Check contract
4. `ROADMAP.md` lines 218-240 (Risks) - Check risk register

---

### Use Case 3: Planning a Sprint

**Load Order:**
1. `ROADMAP.md` lines 1-50 (Executive Summary) - See current status
2. `ROADMAP.md` lines 200-240 (v0.2.0 Planning) - See proposed tasks
3. `MASTER.md` lines 224-283 (Priority Matrix) - Cross-reference

---

### Use Case 4: Upgrading to v0.1.0-beta

**Load Order:**
1. `MIGRATION.md` lines 1-50 (Quick Start) - Overview
2. `MIGRATION.md` lines 51-150 (Breaking Changes) - What changed
3. `MIGRATION.md` lines 351-500 (Step-by-Step) - Follow guide
4. `MIGRATION.md` lines 501-600 (FAQ) - Troubleshooting

---

## Search Patterns

### By Keyword

**"architecture"** → ARCHITECTURE.md
**"priority"** → MASTER.md lines 151-250, ROADMAP.md lines 51-90
**"api"** → API_REFERENCE.md
**"migration"** → MIGRATION.md
**"sprint"** → ROADMAP.md (v0.2.0 planning section)
**"module"** → ARCHITECTURE.md lines 1-100
**"data structure"** → ARCHITECTURE.md lines 101-200
**"error"** → ARCHITECTURE.md lines 551-650, MIGRATION.md FAQ

---

### By Task Type

**"I want to add a new normalization method"**
1. ARCHITECTURE.md lines 310-350 (normalization module)
2. ARCHITECTURE.md lines 751-850 (Extension Points)
3. API_REFERENCE.md lines 201-350 (normalization APIs)

**"I need to fix a test failure"**
1. ISSUES_AND_LIMITATIONS.md (check if known issue)
2. ARCHITECTURE.md lines 851-950 (Testing Strategy)
3. ROADMAP.md retrospective (check if blocking)

**"I want to understand the data flow"**
1. MASTER.md lines 78-145 (Architecture Overview)
2. ARCHITECTURE.md lines 451-550 (Integration Patterns)

**"I need to implement imputation"**
1. ARCHITECTURE.md lines 351-400 (impute module)
2. API_REFERENCE.md lines 351-500 (impute APIs)
3. ROADMAP.md v0.2.0 planning (check priorities)

---

## Script Usage

### Basic Usage

```bash
# Load specific lines from a document
python scripts/doc_loader.py <DOC_NAME> <LINE_RANGE>

# Examples:
python scripts/doc_loader.py MASTER 1-50
python scripts/doc_loader.py ARCHITECTURE 120-150
python scripts/doc_loader.py ROADMAP 91-197
python scripts/doc_loader.py API_REFERENCE 380-420

# Load full document
python scripts/doc_loader.py <DOC_NAME> all
```

### Advanced Usage

```bash
# Search for keyword in document
python scripts/doc_loader.py <DOC_NAME> search "<keyword>"

# Get document outline
python scripts/doc_loader.py <DOC_NAME> outline

# Count lines in document
python scripts/doc_loader.py <DOC_NAME> count
```

---

## Document Metadata

| Document | File Size | Line Count | Last Modified |
|----------|-----------|------------|---------------|
| MASTER.md | 26.5 KB | 653 | 2025-01-14 |
| ARCHITECTURE.md | 30.0 KB | 1099 | 2025-01-05 |
| ROADMAP.md | 9.9 KB | 287 | 2025-01-14 |
| MIGRATION.md | 17.6 KB | 791 | 2025-01-05 |
| API_REFERENCE.md | 25.9 KB | 1184 | 2025-01-05 |
| **Total** | **109.9 KB** | **4014** | - |

---

## Best Practices

1. **Always start with INDEX** - Load this file first to find what you need
2. **Load only what you need** - Use specific line ranges to minimize context
3. **Follow use case guides** - Use the "Common Use Cases" section above
4. **Update index when docs change** - Keep line numbers accurate
5. **Use script for consistency** - Always use `doc_loader.py` to load docs

---

## Maintenance

**When to Update INDEX:**
- After modifying any design document
- After adding new sections
- After reorganizing content
- After project releases

**How to Update:**
1. Recalculate line numbers: `python scripts/doc_loader.py <DOC> count`
2. Update section ranges in this document
3. Test with script: `python scripts/doc_loader.py <DOC> <range>`

---

**Index Size:** ~418 lines
**Last Updated:** 2025-01-14
**Maintainer:** ScpTensor Team

**End of INDEX**
