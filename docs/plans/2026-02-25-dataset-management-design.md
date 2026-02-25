# ScpTensor High-Quality Dataset Management System Design

**Date:** 2026-02-25
**Status:** Approved
**Priority:** High

---

## Overview

Establish a high-quality single-cell proteomics dataset management system focusing on:
- Orbitrap Astral / Astral Zoom instruments
- DIA-NN processed result files
- Data from public databases (PRIDE, MassIVE)

---

## Directory Structure

```
scptensor/datasets/
├── registry.json            # Dataset registry
├── pride/                   # PRIDE database source
│   ├── PXD0xxxx/
│   │   ├── report.tsv       # DIA-NN result file
│   │   └── metadata.json    # Metadata
│   └── ...
├── massive/                 # MassIVE database source
│   └── ...
└── scripts/                 # Management scripts
    ├── search_datasets.py   # Search candidate datasets
    ├── download_dataset.py  # Download dataset
    └── validate_dataset.py  # Validate dataset
```

---

## Metadata Format (metadata.json)

```json
{
  "id": "PXD0xxxx",
  "source": "PRIDE",
  "title": "Single-cell proteomics of X using Orbitrap Astral",
  "instrument": "Orbitrap Astral Zoom",
  "software": "DIA-NN",
  "software_version": "1.9",
  "n_samples": 500,
  "n_proteins": 3500,
  "organism": "Homo sapiens",
  "sample_type": "cell_line",
  "citation_doi": "10.1016/j.xxxx.2024.xx",
  "pride_url": "https://www.ebi.ac.uk/pride/archive/projects/PXD0xxxx",
  "download_date": "2026-02-25",
  "checksum": "sha256:abc123...",
  "status": "validated"
}
```

---

## Registry Format (registry.json)

```json
{
  "version": "1.0",
  "last_updated": "2026-02-25",
  "datasets": [
    {
      "id": "PXD0xxxx",
      "path": "pride/PXD0xxxx",
      "instrument": "Orbitrap Astral",
      "status": "validated"
    }
  ],
  "statistics": {
    "total_datasets": 3,
    "by_instrument": {
      "Orbitrap Astral": 2,
      "Orbitrap Astral Zoom": 1
    }
  }
}
```

---

## Workflow

Five-step process:

| Step | Operation | Output |
|------|-----------|--------|
| 1 | Automatic search of PRIDE/MassIVE | candidates.csv |
| 2 | Manual review of candidate list | approved.csv |
| 3 | One-click download of approved datasets | Data files |
| 4 | Validate dataset integrity | Validation report |
| 5 | Update registry | registry.json |

---

## Selection Criteria

**Must satisfy:**
- Instrument: Orbitrap Astral / Astral Zoom
- Software: DIA-NN processed
- Data type: Single-cell proteomics
- Has published literature citation

**Priority to:**
- Sample count > 100
- Protein count > 2000
- Has complete sample metadata
- Data from 2023 onwards

---

## Data Sources

- PRIDE: https://www.ebi.ac.uk/pride/
- MassIVE: https://massive.ucsd.edu/
- ProteomeXchange: https://proteomexchange.org/
