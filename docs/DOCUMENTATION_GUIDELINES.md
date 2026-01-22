# ScpTensor Documentation Classification Guidelines

**Version:** 1.0.0
**Last Updated:** 2026-01-22
**Maintainer:** ScpTensor Documentation Team

---

## Table of Contents

1. [Documentation Classification Overview](#1-documentation-classification-overview)
2. [Directory Structure](#2-directory-structure)
3. [Document Classification Rules](#3-document-classification-rules)
4. [Document Naming Conventions](#4-document-naming-conventions)
5. [Content Guidelines by Category](#5-content-guidelines-by-category)
6. [Maintenance Guidelines](#6-maintenance-guidelines)
7. [Quick Reference](#7-quick-reference)

---

## 1. Documentation Classification Overview

### 1.1 Purpose

A well-organized documentation structure is critical for:
- **Findability:** Quickly locating relevant information
- **Maintainability:** Keeping documentation up-to-date and consistent
- **Scalability:** Accommodating project growth without chaos
- **Collaboration:** Clear ownership and responsibility for documentation areas

### 1.2 Classification Goals

The documentation classification system aims to:
1. **Separate concerns:** Different types of documents have different lifecycles
2. **Establish clear ownership:** Know who is responsible for what
3. **Support different audiences:** Developers, users, contributors, and project managers
4. **Enable automated workflows:** CI/CD, documentation generation, and reporting

### 1.3 Core Principles

- **Location-based classification:** Document type is determined by its directory location
- **Consistent naming:** Files follow predictable naming patterns
- **Minimal duplication:** Each document has a single, clear purpose
- **Lifecycle awareness:** Documents are archived or removed when obsolete

---

## 2. Directory Structure

### 2.1 Overview

```
docs/
├── PROJECT_STRUCTURE.md          # Main project documentation (root-level)
├── DEVELOPER_GUIDE.md            # Developer onboarding and workflows
├── API_QUICK_REFERENCE.md        # Quick API reference
├── COMPLETION_STATUS.md          # Overall project completion status
├── PROJECT_STATUS.md             # Current project status
├── MIGRATION_GUIDE.md            # Version migration guides
├── SCPTENSOR_CODING_STANDARDS.md # Coding standards
├── SCPTENSOR_API_AUDIT.md        # API audit reports
├── PERFORMANCE_REPORT.md         # Performance analysis reports
├── ISSUES_AND_LIMITATIONS.md     # Known issues and limitations
├── DATASETS.md                   # Dataset documentation
├── README.md                     # Documentation overview and navigation
│
├── api/                          # Auto-generated API documentation
│   └── referee.rst               # API reference files
│
├── design/                       # Design documents (architectural decisions)
│   ├── INDEX.md                  # Design document index
│   ├── MASTER.md                 # Strategic overview
│   ├── ARCHITECTURE.md           # Technical architecture
│   ├── ROADMAP.md                # Project roadmap
│   ├── API_REFERENCE.md          # Complete API reference
│   ├── MIGRATION.md              # Migration design (alpha→beta)
│   ├── YYYY-MM-DD-*.md           # Date-stamped design documents
│   └── *.md                      # Other design documents
│
├── benchmarks/                   # Benchmark testing reports
│   ├── COMPREHENSIVE_BENCHMARK_REPORT.md # Comprehensive reports
│   ├── BENCHMARK_PHASE2_COMPLETION_REPORT.md # Completion reports
│   ├── BENCHMARK_REFACTOR_VERIFICATION.md    # Verification reports
│   ├── COMPETITOR_BENCHMARK.md              # Competitor comparisons
│   ├── BENCHMARK_SUMMARY.txt                # Summary outputs
│   ├── accuracy_report.md                   # Specific metric reports
│   └── scanpy_comparison_report.md          # Method comparisons
│
├── guides/                       # Technical guides and implementation details
│   ├── FILTER_API_MIGRATION.md              # Migration implementation reports
│   ├── lazy_validation_implementation.md    # Feature implementation reports
│   ├── psm_qc_guide.md                      # Feature guides
│   └── *.md                      # Technical guides for specific features
│
├── reference/                    # Reference materials (plans, management, reports)
│   ├── plans/                    # Detailed implementation plans
│   │   └── YYYY-MM-DD-*.md       # Date-stamped implementation plans
│   ├── management/               # Project management documents
│   │   ├── BENCHMARK_PHASE2_PLAN.md          # Phase-specific plans
│   │   ├── BENCHMARK_REFACTOR_PLAN.md        # Refactoring plans
│   │   └── *_PLAN.md                          # Other project plans
│   └── reports/                  # Implementation reports
│       ├── core_implementation.md            # Implementation reports
│       ├── docstring_standardization.md      # Standardization reports
│       ├── pipeline_api_fixes_complete.md    # Fix completion reports
│       ├── sparse_optimization.md            # Optimization reports
│       ├── visualization_fixes.md            # Visualization reports
│       └── *.md                  # Other reports
│
├── comparison_study/             # Comparative analysis documents
│   ├── README.md                 # Study overview
│   ├── START_HERE.md             # Getting started guide
│   ├── QUICK_REFERENCE.md        # Quick reference
│   ├── DELIVERY_SUMMARY.md       # Delivery summaries
│   ├── IMPLEMENTATION_COMPLETE.md # Implementation completion
│   ├── RUNNER_IMPLEMENTATION_REPORT.md # Component reports
│   ├── configs/                  # Configuration files
│   ├── data/                     # Study data
│   ├── evaluation/               # Evaluation outputs
│   ├── examples/                 # Example usage
│   ├── outputs/                  # Generated outputs
│   ├── pipelines/                # Pipeline definitions
│   └── visualization/            # Visualization outputs
│
├── tutorials/                    # Tutorial notebooks
│   └── README.md                 # Tutorial index
│
└── notebooks/                    # Example Jupyter notebooks
```

### 2.2 Directory Purposes

#### Root Level (`docs/`)
**Purpose:** High-level project documentation accessible to all audiences
**Audience:** Developers, users, contributors, project managers
**Lifecycle:** Long-term, actively maintained
**Examples:**
- `/home/shenshang/projects/ScpTensor/docs/PROJECT_STRUCTURE.md`
- `/home/shenshang/projects/ScpTensor/docs/DEVELOPER_GUIDE.md`

#### `docs/design/`
**Purpose:** Architectural decisions, system design, and technical specifications
**Audience:** Architects, senior developers, technical leads
**Lifecycle:** Long-term reference, updated when architecture changes
**Examples:**
- `/home/shenshang/projects/ScpTensor/docs/design/ARCHITECTURE.md`
- `/home/shenshang/projects/ScpTensor/docs/design/2026-01-15-io-export-design.md`

#### `docs/benchmarks/`
**Purpose:** Benchmark testing results, performance analysis, and competitor comparisons
**Audience:** Performance engineers, project managers, stakeholders
**Lifecycle:** Short to medium-term (superseded by new benchmarks)
**Examples:**
- `/home/shenshang/projects/ScpTensor/docs/benchmarks/COMPREHENSIVE_BENCHMARK_REPORT.md`
- `/home/shenshang/projects/ScpTensor/docs/benchmarks/BENCHMARK_REFACTOR_VERIFICATION.md`

#### `docs/guides/`
**Purpose:** Technical guides, implementation details, and how-to instructions
**Audience:** Developers, power users
**Lifecycle:** Long-term, updated as features evolve
**Examples:**
- `/home/shenshang/projects/ScpTensor/docs/guides/psm_qc_guide.md`
- `/home/shenshang/projects/ScpTensor/docs/guides/FILTER_API_MIGRATION.md`

#### `docs/reference/`
**Purpose:** Reference materials including plans, management documents, and reports
**Audience:** Project managers, developers, stakeholders
**Lifecycle:** Varied (plans: medium-term, management: short-term, reports: long-term)
**Subdirectories:**
- `plans/` - Detailed implementation plans
- `management/` - Project planning and coordination
- `reports/` - Implementation completion reports
**Examples:**
- `/home/shenshang/projects/ScpTensor/docs/reference/plans/2026-01-14-v0.2.0-visualization-implementation.md`
- `/home/shenshang/projects/ScpTensor/docs/reference/management/BENCHMARK_PHASE2_PLAN.md`
- `/home/shenshang/projects/ScpTensor/docs/reference/reports/core_implementation.md`

#### `docs/comparison_study/`
**Purpose:** Comparative analysis studies with supporting data and outputs
**Audience:** Researchers, analysts, stakeholders
**Lifecycle:** Long-term (reference studies)
**Examples:**
- `/home/shenshang/projects/ScpTensor/docs/comparison_study/README.md`
- `/home/shenshang/projects/ScpTensor/docs/comparison_study/DELIVERY_SUMMARY.md`

#### `docs/tutorials/`
**Purpose:** Educational notebooks and examples for users
**Audience:** Users, developers learning the system
**Lifecycle:** Long-term, updated with API changes
**Examples:**
- `/home/shenshang/projects/ScpTensor/docs/tutorials/README.md`

---

## 3. Document Classification Rules

### 3.1 Decision Tree

Use this decision tree to determine where to place a document:

```
Is the document about...
├─ System architecture, design decisions, or technical specifications?
│  └─ YES → docs/design/
│
├─ Benchmark results, performance analysis, or competitor comparison?
│  └─ YES → docs/benchmarks/
│
├─ Technical guide or how-to instructions?
│  └─ YES → docs/guides/
│
├─ Project planning, milestones, or coordination?
│  └─ YES → docs/reference/management/
│
├─ Detailed implementation steps for a feature?
│  └─ YES → docs/reference/plans/
│
├─ Completed implementation, migration, or feature delivery?
│  └─ YES → docs/reference/reports/
│
├─ Comparative analysis with supporting data/outputs?
│  └─ YES → docs/comparison_study/
│
├─ Tutorial or educational content?
│  └─ YES → docs/tutorials/
│
└─ High-level project information (overview, status, guides)?
   └─ YES → docs/ (root level)
```

### 3.2 Document Type Mapping

| Document Type | Suffix/Keyword | Target Directory | Examples |
|--------------|----------------|------------------|----------|
| **Plan** | `_PLAN.md` | `docs/reference/management/` | `BENCHMARK_PHASE2_PLAN.md` |
| **Implementation Plan** | `-implementation.md` | `docs/reference/plans/` | `2026-01-15-io-export-implementation.md` |
| **Design** | `-design.md` | `docs/design/` | `2026-01-15-io-export-design.md` |
| **Completion Report** | `_COMPLETION_REPORT.md` | `docs/benchmarks/` | `BENCHMARK_PHASE2_COMPLETION_REPORT.md` |
| **Verification** | `_VERIFICATION.md` | `docs/benchmarks/` | `BENCHMARK_REFACTOR_VERIFICATION.md` |
| **Implementation Report** | `_MIGRATION.md`, `_implementation.md` | `docs/reference/reports/` | `FILTER_API_MIGRATION.md` |
| **Guide** | `_guide.md` | `docs/guides/` | `psm_qc_guide.md` |
| **Benchmark Report** | `_BENCHMARK_REPORT.md`, `_REPORT.md` | `docs/benchmarks/` | `COMPREHENSIVE_BENCHMARK_REPORT.md` |
| **Refactor Plan** | `_REFACTOR_PLAN.md` | `docs/reference/management/` | `BENCHMARK_REFACTOR_PLAN.md` |
| **Summary** | `_SUMMARY.md`, `.txt` | Varies by context | `BENCHMARK_SUMMARY.txt` |

### 3.3 Special Cases

#### Date-Stamped Documents
**Rule:** Documents created as part of a planned initiative should use date stamps
**Format:** `YYYY-MM-DD-description-type.md`
**Location:** Depends on document purpose (design, plans, etc.)
**Examples:**
- `/home/shenshang/projects/ScpTensor/docs/design/2026-01-15-io-export-design.md`
- `/home/shenshang/projects/ScpTensor/docs/reference/plans/2026-01-14-v0.2.0-visualization-implementation.md`

#### Auto-Generated Documents
**Rule:** Auto-generated documentation goes in designated subdirectories
**Location:** `docs/api/` for API docs, `docs/reference/reports/` for generated reports
**Examples:**
- `/home/shenshang/projects/ScpTensor/docs/api/referee.rst`

#### Study Artifacts
**Rule:** Studies with multiple artifacts (data, configs, outputs) use subdirectories
**Location:** `docs/comparison_study/` or dedicated study directory
**Examples:**
- `/home/shenshang/projects/ScpTensor/docs/comparison_study/configs/`
- `/home/shenshang/projects/ScpTensor/docs/comparison_study/data/`
- `/home/shenshang/projects/ScpTensor/docs/comparison_study/outputs/`

---

## 4. Document Naming Conventions

### 4.1 General Rules

1. **Use lowercase letters** for regular words
2. **Use uppercase letters** for acronyms (API, QC, PSM, etc.)
3. **Use underscores** to separate words within a name
4. **Use hyphens** to separate name components (date, type, etc.)
5. **Be descriptive but concise** (aim for 50-60 characters max)
6. **Avoid special characters** except `_`, `-`, and `.`

### 4.2 Naming Patterns by Category

#### Project Management Documents
**Pattern:** `[PHASE/FEATURE]_[ACTION]_PLAN.md`
**Examples:**
- `BENCHMARK_PHASE2_PLAN.md`
- `BENCHMARK_REFACTOR_PLAN.md`
- `FEATURE_X_IMPLEMENTATION_PLAN.md`

#### Design Documents
**Pattern:** `YYYY-MM-DD-[feature]-design.md` (for dated designs)
**Pattern:** `[MODULE].md` (for core design docs)
**Examples:**
- `2026-01-15-io-export-design.md`
- `2026-01-16-qc-enhancement-design.md`
- `ARCHITECTURE.md`
- `API_REFERENCE.md`

#### Implementation Plans
**Pattern:** `YYYY-MM-DD-[version/feature]-implementation.md`
**Examples:**
- `2026-01-14-v0.2.0-visualization-implementation.md`
- `2026-01-15-io-export-implementation.md`
- `2026-01-16-qc-enhancement-implementation.md`

#### Benchmark Reports
**Pattern:** `[SCOPE]_BENCHMARK_[REPORT_TYPE].md`
**Examples:**
- `COMPREHENSIVE_BENCHMARK_REPORT.md`
- `BENCHMARK_PHASE2_COMPLETION_REPORT.md`
- `BENCHMARK_REFACTOR_VERIFICATION.md`
- `COMPETITOR_BENCHMARK.md`
- `accuracy_report.md`

#### Implementation Reports
**Pattern:** `[FEATURE]_[ACTION].md` or `[FEATURE]_implementation.md`
**Examples:**
- `FILTER_API_MIGRATION.md`
- `lazy_validation_implementation.md`
- `psm_qc_guide.md`

#### Technical Guides
**Pattern:** `[feature]_guide.md` or `[topic]_guide.md`
**Examples:**
- `psm_qc_guide.md`
- `batch_correction_guide.md`
- `advanced_filtering_guide.md`

### 4.3 File Extension Guidelines

| Extension | Use Case | Example |
|-----------|----------|---------|
| `.md` | All documentation (preferred) | `ARCHITECTURE.md` |
| `.rst` | Sphinx/reStructuredText files | `referee.rst` |
| `.txt` | Raw output, logs, summaries | `BENCHMARK_SUMMARY.txt` |
| `.ipynb` | Jupyter notebooks | `tutorial_01.ipynb` |

### 4.4 Version Control in Names

**Avoid version numbers in filenames** (e.g., `v1.0.0_guide.md`). Instead:
1. Use git for version control
2. Update content in place
3. Use date stamps for major revisions
4. Create new files only when documenting a different feature

**Exception:** Version-specific implementation plans
- `2026-01-14-v0.2.0-visualization-implementation.md` (acceptable)

---

## 5. Content Guidelines by Category

### 5.1 Design Documents (`docs/design/`)

**Purpose:** Document architectural decisions and technical specifications

**Required Sections:**
1. Overview
2. Motivation/Problem Statement
3. Proposed Solution
4. Technical Details
5. Impact Analysis
6. Alternatives Considered
7. References

**Example:**
```markdown
# IO Export System Design

## Overview
Describe the high-level design...

## Motivation
Why is this design needed?

## Proposed Solution
Detailed technical design...

## Impact Analysis
How does this affect existing code?
```

### 5.2 Implementation Plans (`docs/reference/plans/`)

**Purpose:** Detailed step-by-step implementation guide

**Required Sections:**
1. Objective
2. Scope
3. Implementation Steps (ordered list)
4. Testing Strategy
5. Success Criteria
6. Timeline/Effort Estimate

**Example:**
```markdown
# IO Export Implementation Plan

## Objective
Implement export functionality for multiple formats...

## Scope
- CSV export
- HDF5 export
- NPZ export

## Implementation Steps
1. Create base exporter interface
2. Implement CSV exporter
3. ...

## Testing Strategy
- Unit tests for each format
- Integration tests with real data
```

### 5.3 Project Management (`docs/reference/management/`)

**Purpose:** Track project progress, milestones, and coordination

**Required Sections:**
1. Overview
2. Objectives
3. Deliverables
4. Timeline
5. Dependencies
6. Risk Assessment
7. Status Tracking

**Example:**
```markdown
# Benchmark Phase 2 Plan

## Overview
Phase 2 focuses on accuracy evaluation...

## Objectives
- Implement accuracy evaluator
- Compare with competitors
- Generate comprehensive report

## Deliverables
1. Accuracy evaluator module
2. Comparison report
3. Visualization updates
```

### 5.4 Benchmark Reports (`docs/benchmarks/`)

**Purpose:** Document benchmark results and performance analysis

**Required Sections:**
1. Executive Summary
2. Methodology
3. Results (with figures/tables)
4. Analysis
5. Conclusions
6. Recommendations

**Example:**
```markdown
# Comprehensive Benchmark Report

## Executive Summary
Benchmark results show...

## Methodology
Compared ScpTensor against Scanpy and Scrublet...

## Results
- Runtime: 2.3x faster than Scanpy
- Memory: 40% reduction
```

### 5.5 Implementation Reports (`docs/reference/reports/`)

**Purpose:** Document completed implementations and migrations

**Required Sections:**
1. Summary
2. Changes Made
3. Migration Path (if applicable)
4. Breaking Changes
5. Testing Results
6. Known Issues

**Example:**
```markdown
# Filter API Migration

## Summary
Migrated filtering API from function-based to method-based...

## Changes Made
- Added filter methods to ScpMatrix
- Deprecated standalone functions
- Updated all call sites
```

### 5.6 Technical Guides (`docs/guides/`)

**Purpose:** Provide how-to instructions for specific tasks

**Required Sections:**
1. Introduction
2. Prerequisites
3. Step-by-Step Guide
4. Examples
5. Troubleshooting
6. References

**Example:**
```markdown
# PSM QC Guide

## Introduction
This guide explains how to use PSM-level QC...

## Prerequisites
- ScpTensor v0.1.0 or higher
- Raw PSM data files

## Step-by-Step Guide
1. Load PSM data
2. Configure QC metrics
3. Run QC pipeline
```

---

## 6. Maintenance Guidelines

### 6.1 Regular Maintenance Tasks

#### Weekly
- [ ] Review root-level documentation for accuracy
- [ ] Update project status and completion tracking
- [ ] Archive completed project plans

#### Monthly
- [ ] Review design documents for relevance
- [ ] Archive outdated benchmark reports
- [ ] Update technical guides with latest patterns
- [ ] Clean up comparison study artifacts

#### Per Release
- [ ] Update API documentation
- [ ] Update migration guides
- [ ] Archive implementation reports for the release
- [ ] Update coding standards if needed

### 6.2 Document Lifecycle

1. **Creation:** Document follows naming conventions and is placed in correct directory
2. **Active Maintenance:** Updated as project evolves
3. **Obsolescence:** When document is no longer relevant:
   - Move to `docs/archive/` (if historically valuable)
   - Delete if no longer needed (with PR review)
4. **Supersession:** New versions should reference old versions in appendix

### 6.3 Archival Process

**When to Archive:**
- Implementation plans after feature is complete
- Old benchmark reports when newer ones available
- Deprecated design documents

**How to Archive:**
```bash
# Create archive directory if needed
mkdir -p docs/archive/2026-Q1

# Move document with timestamp
mv docs/reference/management/BENCHMARK_PHASE2_PLAN.md \
   docs/archive/2026-Q1/2026-01-22_BENCHMARK_PHASE2_PLAN.md
```

**Archive Structure:**
```
docs/archive/
├── 2025-Q4/
│   └── YYYY-MM-DD-original-name.md
└── 2026-Q1/
    └── YYYY-MM-DD-original-name.md
```

### 6.4 Review Process

Before creating new documentation:
1. **Search existing docs** to avoid duplication
2. **Check if similar doc exists** that could be updated instead
3. **Verify directory placement** using decision tree
4. **Follow naming conventions**
5. **Include required sections** for document type

When updating documentation:
1. **Preserve historical context** in appendices if needed
2. **Update modification date** in document header
3. **Review related documents** for consistency
4. **Update cross-references** in other docs

### 6.5 Quality Standards

All documentation should:
- **Use clear, concise language** (English-only for code docs)
- **Include examples** where applicable
- **Have consistent formatting** (markdown standards)
- **Be spell-checked** before committing
- **Have descriptive titles** (not "Documentation", "Notes", etc.)
- **Include dates** for time-sensitive content
- **Link to related documents** for context

### 6.6 Cleanup Checklist

Use this checklist during quarterly cleanup:

```bash
# Find documents older than 6 months in reference/management
find docs/reference/management -name "*.md" -mtime +180

# Find duplicate or similar documents
# (manual review required)

# Check for broken links
# (use markdown linter)

# Archive completed plans
# (move completed implementation plans to archive)

# Update outdated references
# (search and replace)
```

---

## 7. Quick Reference

### 7.1 Common Tasks

| Task | Command/Action |
|------|----------------|
| **Find a design document** | `ls docs/design/` or `grep -r "keyword" docs/design/` |
| **Find a project plan** | `ls docs/reference/management/*_PLAN.md` |
| **Find benchmark results** | `ls docs/benchmarks/*_REPORT.md` |
| **Find implementation guide** | `ls docs/guides/*_guide.md` |
| **Create new design** | Use `docs/design/YYYY-MM-DD-feature-design.md` |
| **Create new plan** | Use `docs/reference/management/FEATURE_PLAN.md` |
| **Archive old document** | Move to `docs/archive/YYYY-QQ/` with date prefix |

### 7.2 Document Templates

#### Design Document Template
```markdown
# [Feature Name] Design

**Date:** YYYY-MM-DD
**Author:** [Name]
**Status:** [Draft/Review/Approved]

## Overview
[High-level description]

## Motivation
[Why this design is needed]

## Proposed Solution
[Technical details]

## Implementation Notes
[Key implementation considerations]

## Impact Analysis
[Effects on existing systems]

## Alternatives Considered
[Other approaches evaluated]

## References
[Related documents, issues, etc.]
```

#### Implementation Plan Template
```markdown
# [Feature Name] Implementation Plan

**Date:** YYYY-MM-DD
**Author:** [Name]
**Target Release:** [Version]

## Objective
[What will be implemented]

## Scope
[In-scope and out-of-scope items]

## Implementation Steps
1. [Step 1]
2. [Step 2]
3. ...

## Testing Strategy
[Unit, integration, end-to-end tests]

## Success Criteria
[Definition of done]

## Timeline
[Estimated effort and duration]

## Dependencies
[Prerequisites and blocking items]
```

#### Benchmark Report Template
```markdown
# [Benchmark Scope] Report

**Date:** YYYY-MM-DD
**Authors:** [Names]
**Benchmark Version:** [Version]

## Executive Summary
[Key findings and conclusions]

## Methodology
[Experimental setup, datasets, metrics]

## Results
[Findings with figures/tables]

## Analysis
[Interpretation of results]

## Conclusions
[What was learned]

## Recommendations
[Actionable next steps]

## Appendix
[Raw data, detailed metrics]
```

### 7.3 Troubleshooting

**Problem:** Can't find where to put a document
**Solution:** Use the decision tree in Section 3.1

**Problem:** Found duplicate documents
**Solution:**
1. Compare content
2. Merge if complementary
3. Delete outdated version if superseded
4. Archive both if historically valuable

**Problem:** Documentation is outdated
**Solution:**
1. Check document date
2. Search for newer versions
3. Check git history for updates
4. Create issue to update if needed

**Problem:** Broken links between documents
**Solution:**
1. Use relative paths
2. Update when moving documents
3. Use markdown linter to detect
4. Test links in documentation build

---

## Appendix: Real Examples

### Example 1: Creating a New Feature Design

**Scenario:** Designing a new normalization method

**Steps:**
1. Create document: `/home/shenshang/projects/ScpTensor/docs/design/2026-01-22-new-normalization-design.md`
2. Follow design document template
3. Include: overview, motivation, algorithm, implementation notes
4. Link from `/home/shenshang/projects/ScpTensor/docs/design/INDEX.md`

### Example 2: Documenting Benchmark Results

**Scenario:** Completing accuracy benchmark

**Steps:**
1. Create report: `/home/shenshang/projects/ScpTensor/docs/benchmarks/ACCURACY_BENCHMARK_REPORT.md`
2. Include: methodology, results, figures, analysis
3. Update summary: `/home/shenshang/projects/ScpTensor/docs/benchmarks/COMPREHENSIVE_BENCHMARK_REPORT.md`
4. Archive old accuracy reports if needed

### Example 3: Writing a Technical Guide

**Scenario:** Creating guide for new feature

**Steps:**
1. Create guide: `/home/shenshang/projects/ScpTensor/docs/guides/new_feature_guide.md`
2. Include: introduction, prerequisites, step-by-step, examples
3. Add examples in `/home/shenshang/projects/ScpTensor/docs/tutorials/`
4. Update `/home/shenshang/projects/ScpTensor/docs/DEVELOPER_GUIDE.md` to reference

---

## Summary

This documentation classification system provides:

- **Clear structure:** Every document has a logical place
- **Consistent naming:** Predictable file names across categories
- **Scalable organization:** Grows with the project
- **Maintainable workflow:** Clear lifecycle and archival processes

**Key Takeaways:**
1. Use the decision tree to determine document location
2. Follow naming conventions for consistency
3. Include required sections for document type
4. Archive obsolete documents regularly
5. Review and update documentation periodically

For questions or suggestions about these guidelines, please open an issue or discussion.

---

**Document History:**
- 2026-01-22: Initial version (v1.0.0)
