# ScpTensor Design Documentation - Progressive Loading System

**Purpose:** Load design documentation on-demand without overwhelming context
**Last Updated:** 2025-01-06

---

## Quick Start

### Step 1: Load INDEX First

```bash
python3 scripts/doc_loader.py INDEX 1-100
```

INDEX contains all document summaries and line ranges.

### Step 2: Load Specific Sections

```bash
# Load by section name (recommended)
python3 scripts/doc_loader.py MASTER section executive

# Load by line range
python3 scripts/doc_loader.py ARCHITECTURE 310-350

# Search for keyword
python3 scripts/doc_loader.py ROADMAP search "P0"
```

---

## Command Reference

### Section Loading (Recommended)

```bash
python3 scripts/doc_loader.py <DOC> section <section_name>
```

**Available Sections:**
- **MASTER**: executive, architecture, priority, ecosystem, maintenance, metrics
- **ARCHITECTURE**: modules, structures, patterns, apis, integration, errors, performance, extensions, testing
- **ROADMAP**: summary, priorities, milestones, dependencies, sprints, risks
- **MIGRATION**: quickstart, breaking, compatibility, strategies, steps, faq
- **API_REFERENCE**: toc, core, normalization, impute, integration, qc, dim_reduction

**Examples:**
```bash
python3 scripts/doc_loader.py MASTER section executive
python3 scripts/doc_loader.py ARCHITECTURE section modules
python3 scripts/doc_loader.py ROADMAP section priorities
```

---

### Line Range Loading

```bash
python3 scripts/doc_loader.py <DOC> <START>-<END>
```

**Examples:**
```bash
python3 scripts/doc_loader.py MASTER 1-50        # Executive summary
python3 scripts/doc_loader.py ARCHITECTURE 310-350  # Normalization module
python3 scripts/doc_loader.py ROADMAP 151-250     # Milestones
```

---

### Search Commands

**Keyword Search:**
```bash
python3 scripts/doc_loader.py <DOC> search "<keyword>"
```

**Regex Search:**
```bash
python3 scripts/doc_loader.py <DOC> regex "<pattern>"
```

**Examples:**
```bash
# Keyword search
python3 scripts/doc_loader.py ARCHITECTURE search "normalization"

# Regex search (find all headings)
python3 scripts/doc_loader.py MASTER regex "^##"

# Regex search (find function definitions)
python3 scripts/doc_loader.py API_REFERENCE regex "def.*normalize"
```

---

### Other Commands

```bash
# Load entire document (use sparingly!)
python3 scripts/doc_loader.py <DOC> all

# Get document outline
python3 scripts/doc_loader.py <DOC> outline

# Count lines and file size
python3 scripts/doc_loader.py <DOC> count

# Show help
python3 scripts/doc_loader.py help
```

---

### Save to File

```bash
python3 scripts/doc_loader.py <DOC> <ACTION> --output <filename>
```

**Examples:**
```bash
python3 scripts/doc_loader.py MASTER section executive --output exec.md
python3 scripts/doc_loader.py ROADMAP section priorities --output priority.md
```

---

## Quick Reference Tables

### By Task

| Task | Command |
|------|---------|
| **Project status** | `doc_loader.py MASTER section executive` |
| **What to work on** | `doc_loader.py ROADMAP section priorities` |
| **Module architecture** | `doc_loader.py ARCHITECTURE section modules` |
| **Data structures** | `doc_loader.py ARCHITECTURE section structures` |
| **Add normalization** | `doc_loader.py ARCHITECTURE section apis` |
| **Add imputation** | `doc_loader.py API_REFERENCE section impute` |
| **API documentation** | `doc_loader.py API_REFERENCE section core` |
| **Upgrade guide** | `doc_loader.py MIGRATION section quickstart` |
| **Testing strategy** | `doc_loader.py ARCHITECTURE section testing` |
| **Risk assessment** | `doc_loader.py ROADMAP section risks` |

---

### Available Documents

| Document | Lines | Purpose |
|----------|-------|---------|
| **INDEX** | 418 | Navigation hub - start here! |
| **MASTER** | 639 | Strategic overview, priorities |
| **ARCHITECTURE** | 1100 | Module specs, data structures |
| **ROADMAP** | 700 | Execution plan, milestones |
| **MIGRATION** | 600 | Upgrade guide alpha→beta |
| **API_REFERENCE** | 900 | Complete API documentation |

---

## Best Practices

### 1. Always Start with INDEX

```bash
python3 scripts/doc_loader.py INDEX 1-100
```

### 2. Use Section Names (Not Line Numbers)

**Good:**
```bash
python3 scripts/doc_loader.py MASTER section executive
```

**Avoid:**
```bash
python3 scripts/doc_loader.py MASTER 1-50
```

### 3. Load Only What You Need

**Good:**
```bash
python3 scripts/doc_loader.py ARCHITECTURE section modules
```

**Bad:**
```bash
python3 scripts/doc_loader.py ARCHITECTURE all  # 1100 lines!
```

### 4. Use Search Before Loading

```bash
# Step 1: Search first
python3 scripts/doc_loader.py INDEX search "imputation"

# Step 2: Load specific section
python3 scripts/doc_loader.py ARCHITECTURE section apis
```

---

## Troubleshooting

### Unknown Document

```bash
$ python3 scripts/doc_loader.py INVALID 1-50
Error: Unknown document 'INVALID'
Available documents: MASTER, ARCHITECTURE, ROADMAP, MIGRATION, API_REFERENCE, INDEX
```

**Solution:** Use one of the available document names (case-insensitive).

---

### Line Range Out of Bounds

```bash
$ python3 scripts/doc_loader.py MASTER 1000-1050
Error: Line range 1000-1050 out of bounds (document has 639 lines)
```

**Solution:** Check document size first with `count` command.

---

### Unknown Section

```bash
$ python3 scripts/doc_loader.py MASTER section fake
Error: Unknown section 'fake' for MASTER
Available sections: executive, architecture, priority, ecosystem, maintenance, metrics
```

**Solution:** Use one of the available section names.

---

## What's New (v1.1)

**Recent Improvements:**
- 🎨 **Colored Output** - Color-coded terminal messages (green=success, red=error, cyan=info)
- 📦 **Section Names** - Load sections by name instead of line numbers
- 🔍 **Regex Search** - Advanced pattern matching with `regex` command
- 💾 **Save Parameter** - Built-in `--output` parameter for saving sections
- ✅ **Better Errors** - Helpful error messages with available options
- 🧪 **Unit Tests** - Built-in test coverage for core functionality

---

## System Advantages

✅ **Reduced Context** - Load only what you need (50-200 lines vs 4000+)
✅ **Faster Loading** - Smaller sections load quicker
✅ **Better Focus** - See only relevant information
✅ **Easy Navigation** - Section names, no line numbers needed
✅ **Enhanced UX** - Color-coded output for better readability
✅ **Searchable** - Find keywords and regex patterns quickly
✅ **Save to File** - Built-in `--output` parameter
✅ **Version Control Friendly** - Easy to track changes

---

## Common Workflows

### Starting Development on a New Feature

```bash
# Step 1: Check current status
python3 scripts/doc_loader.py MASTER section executive

# Step 2: Find your module
python3 scripts/doc_loader.py ARCHITECTURE section modules

# Step 3: Learn extension points
python3 scripts/doc_loader.py ARCHITECTURE section extensions

# Step 4: Check similar APIs
python3 scripts/doc_loader.py API_REFERENCE section normalization
```

### Fixing a Bug

```bash
# Step 1: Check if known issue
python3 scripts/doc_loader.py INDEX search "bug"

# Step 2: Understand error handling
python3 scripts/doc_loader.py ARCHITECTURE section errors

# Step 3: Check API contract
python3 scripts/doc_loader.py API_REFERENCE search "<function_name>"
```

### Planning a Sprint

```bash
# Step 1: Review priority matrix
python3 scripts/doc_loader.py ROADMAP section priorities

# Step 2: Check milestones
python3 scripts/doc_loader.py ROADMAP section milestones

# Step 3: Review risks
python3 scripts/doc_loader.py ROADMAP section risks
```

---

**Last Updated:** 2025-01-06
**Maintainer:** ScpTensor Team

For questions or issues, refer to the project README or create an issue.
