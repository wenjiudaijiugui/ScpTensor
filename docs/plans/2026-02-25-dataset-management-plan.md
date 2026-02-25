# Dataset Management System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a high-quality single-cell proteomics dataset management system for ScpTensor

**Architecture:** Five-step workflow (search -> review -> download -> validate -> register) with JSON metadata and registry, supporting PRIDE and MassIVE data sources

**Tech Stack:** Python 3.12+, requests, pandas, polars, JSON

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Task 1: Create Directory Structure](#task-1-create-directory-structure)
3. [Task 2: Create Registry File](#task-2-create-registry-file)
4. [Task 3: Create Search Script](#task-3-create-search-script)
5. [Task 4: Create Download Script](#task-4-create-download-script)
6. [Task 5: Create Validation Script](#task-5-create-validation-script)
7. [Task 6: Execute Search and Download First Dataset](#task-6-execute-search-and-download-first-dataset)
8. [Validation Checklist](#validation-checklist)
9. [Success Metrics](#success-metrics)

---

## Project Overview

### Current State

| Component | Status | Action Required |
|-----------|--------|-----------------|
| Directory structure | Missing | Create `scptensor/datasets/` hierarchy |
| Registry file | Missing | Create `registry.json` |
| Search script | Missing | Create `search_datasets.py` |
| Download script | Missing | Create `download_dataset.py` |
| Validation script | Missing | Create `validate_dataset.py` |
| Datasets | 0 | Download and validate first dataset |

### Timeline

| Task | Duration | Dependencies |
|------|----------|--------------|
| Task 1: Directory structure | 5 minutes | None |
| Task 2: Registry file | 5 minutes | Task 1 |
| Task 3: Search script | 30 minutes | Task 1 |
| Task 4: Download script | 30 minutes | Task 1 |
| Task 5: Validation script | 30 minutes | Task 1 |
| Task 6: Execute workflow | 1 hour | Tasks 2-5 |

**Total Duration:** ~2.5 hours

### Success Criteria

- [ ] Directory structure created with proper organization
- [ ] Registry file initialized with valid JSON
- [ ] Search script can query PRIDE API
- [ ] Download script can fetch DIA-NN files
- [ ] Validation script checks integrity
- [ ] First dataset downloaded and validated
- [ ] Registry updated with first entry

---

## Task 1: Create Directory Structure

**Goal:** Establish the dataset management directory hierarchy

**Files:**
- Create: `scptensor/datasets/.gitkeep`
- Create: `scptensor/datasets/pride/.gitkeep`
- Create: `scptensor/datasets/massive/.gitkeep`
- Create: `scptensor/datasets/scripts/.gitkeep`

### Step 1: Create Directories and .gitkeep Files

**Action:** Execute the following commands

```bash
# Create directory structure
mkdir -p scptensor/datasets/pride
mkdir -p scptensor/datasets/massive
mkdir -p scptensor/datasets/scripts

# Create .gitkeep files to preserve empty directories in git
touch scptensor/datasets/.gitkeep
touch scptensor/datasets/pride/.gitkeep
touch scptensor/datasets/massive/.gitkeep
touch scptensor/datasets/scripts/.gitkeep
```

**Expected Output:** No errors, directories created successfully

### Step 2: Verify Directory Structure

**Action:** Run ls command to verify structure

```bash
ls -la scptensor/datasets/
```

**Expected Output:**

```
total 20
drwxr-xr-x 5 user user 4096 Feb 25 10:00 .
drwxr-xr-x 3 user user 4096 Feb 25 10:00 ..
-rw-r--r-- 1 user user    0 Feb 25 10:00 .gitkeep
drwxr-xr-x 2 user user 4096 Feb 25 10:00 massive/
drwxr-xr-x 2 user user 4096 Feb 25 10:00 pride/
drwxr-xr-x 2 user user 4096 Feb 25 10:00 scripts/
```

**Verify subdirectories:**

```bash
ls -la scptensor/datasets/pride/
ls -la scptensor/datasets/massive/
ls -la scptensor/datasets/scripts/
```

**Expected Output:** Each directory should contain a `.gitkeep` file

### Step 3: Commit Changes

**Action:** Stage and commit the directory structure

```bash
git add scptensor/datasets/
git commit -m "$(cat <<'EOF'
feat(datasets): create dataset directory structure

- Add pride/ and massive/ subdirectories for data sources
- Add scripts/ for management tools
- Add .gitkeep files to preserve empty directories

This establishes the foundation for the dataset management system.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

**Expected Output:** Git commit successful with commit hash

---

## Task 2: Create Registry File

**Goal:** Initialize the dataset registry with valid JSON structure

**Files:**
- Create: `scptensor/datasets/registry.json`

### Step 1: Create Initial Registry

**Action:** Create the registry.json file

```bash
cat > scptensor/datasets/registry.json << 'EOF'
{
  "version": "1.0",
  "last_updated": "2026-02-25",
  "datasets": [],
  "statistics": {
    "total_datasets": 0,
    "by_instrument": {},
    "by_source": {}
  }
}
EOF
```

**Expected Output:** File created successfully

### Step 2: Validate JSON Format

**Action:** Verify JSON is valid and parseable

```bash
python3 -c "import json; data = json.load(open('scptensor/datasets/registry.json')); print('✓ JSON is valid'); print(f'Version: {data[\"version\"]}'); print(f'Total datasets: {data[\"statistics\"][\"total_datasets\"]}')"
```

**Expected Output:**

```
✓ JSON is valid
Version: 1.0
Total datasets: 0
```

**Alternative validation using jq (if available):**

```bash
jq . scptensor/datasets/registry.json
```

### Step 3: Commit Registry File

**Action:** Stage and commit the registry

```bash
git add scptensor/datasets/registry.json
git commit -m "$(cat <<'EOF'
feat(datasets): add initial registry.json

- Initialize dataset registry with version 1.0
- Add statistics tracking structure
- Prepare for dataset registration

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

**Expected Output:** Git commit successful

---

## Task 3: Create Search Script

**Goal:** Create script to search PRIDE API for single-cell proteomics datasets

**Files:**
- Create: `scptensor/datasets/scripts/search_datasets.py`

### Step 1: Create Search Script

**Action:** Create the search script with complete implementation

```bash
cat > scptensor/datasets/scripts/search_datasets.py << 'PYEOF'
#!/usr/bin/env python3
"""Search PRIDE database for single-cell proteomics datasets.

This script queries the PRIDE API to find datasets matching specific criteria:
- Instrument: Orbitrap Astral or Orbitrap Astral Zoom
- Software: DIA-NN
- Data type: Single-cell proteomics

Output:
    candidates.csv: List of candidate datasets with metadata
"""

from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any

import requests


# PRIDE API endpoints
PRIDE_API_BASE = "https://www.ebi.ac.uk/pride/ws/archive/v2"
PROJECTS_ENDPOINT = f"{PRIDE_API_BASE}/projects"


def search_pride_projects(
    keywords: List[str],
    instrument_filter: str | None = None,
    software_filter: str | None = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Search PRIDE database for projects matching criteria.

    Args:
        keywords: List of keywords to search for
        instrument_filter: Filter by instrument name (e.g., "Orbitrap Astral")
        software_filter: Filter by software name (e.g., "DIA-NN")
        limit: Maximum number of results to return

    Returns:
        List of project dictionaries with metadata
    """
    projects = []
    skip = 0
    page_size = 100

    print(f"Searching PRIDE for: {' + '.join(keywords)}")

    while True:
        params = {
            "keywords": " ".join(keywords),
            "pageSize": page_size,
            "skip": skip,
        }

        try:
            response = requests.get(PROJECTS_ENDPOINT, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data.get("_embedded", {}).get("projects"):
                break

            page_projects = data["_embedded"]["projects"]
            projects.extend(page_projects)

            print(f"Fetched {len(page_projects)} projects (total: {len(projects)})")

            # Check if we've reached the limit or end of results
            if len(projects) >= limit or len(page_projects) < page_size:
                break

            skip += page_size

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break

    # Apply filters
    filtered_projects = []
    for project in projects:
        acc = project.get("accession", "")
        title = project.get("title", "")
        desc = project.get("description", "")

        # Instrument filter
        if instrument_filter:
            instrument_lower = instrument_filter.lower()
            if not (
                instrument_lower in title.lower()
                or instrument_lower in desc.lower()
            ):
                continue

        # Software filter
        if software_filter:
            software_lower = software_filter.lower()
            if not (
                software_lower in title.lower()
                or software_lower in desc.lower()
            ):
                continue

        filtered_projects.append(project)

    print(f"Found {len(filtered_projects)} projects after filtering")

    return filtered_projects


def extract_project_metadata(project: Dict[str, Any]) -> Dict[str, str]:
    """Extract relevant metadata from a PRIDE project.

    Args:
        project: Project dictionary from PRIDE API

    Returns:
        Dictionary with extracted metadata fields
    """
    accession = project.get("accession", "")
    title = project.get("title", "")
    description = project.get("description", "")

    # Extract instrument information
    instrument = "Unknown"
    if "instrumentNames" in project and project["instrumentNames"]:
        instrument = ", ".join(project["instrumentNames"])

    # Extract software information
    software = "Unknown"
    if "softwares" in project and project["softwares"]:
        software_list = [s.get("name", "Unknown") for s in project["softwares"]]
        software = ", ".join(software_list)

    # Extract organism
    organism = "Unknown"
    if "organismScientificNames" in project and project["organismScientificNames"]:
        organism = ", ".join(project["organismScientificNames"])

    # Extract publication info
    publication_doi = ""
    if "references" in project and project["references"]:
        for ref in project["references"]:
            if ref.get("referenceType") == "Publication":
                publication_doi = ref.get("doi", "")
                break

    # Count files
    n_files = 0
    if "fileCount" in project:
        n_files = project["fileCount"]

    # Project URL
    pride_url = f"https://www.ebi.ac.uk/pride/archive/projects/{accession}"

    return {
        "accession": accession,
        "title": title[:200],  # Truncate long titles
        "description": description[:500],  # Truncate long descriptions
        "instrument": instrument,
        "software": software,
        "organism": organism,
        "publication_doi": publication_doi,
        "n_files": n_files,
        "pride_url": pride_url,
    }


def save_candidates_to_csv(
    candidates: List[Dict[str, str]], output_path: Path
) -> None:
    """Save candidate datasets to CSV file.

    Args:
        candidates: List of candidate metadata dictionaries
        output_path: Path to output CSV file
    """
    if not candidates:
        print("No candidates to save")
        return

    fieldnames = [
        "accession",
        "title",
        "instrument",
        "software",
        "organism",
        "publication_doi",
        "n_files",
        "pride_url",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for candidate in candidates:
            # Write only selected fields
            row = {k: candidate.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"Saved {len(candidates)} candidates to {output_path}")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Search PRIDE for single-cell proteomics datasets"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="candidates.csv",
        help="Output CSV file path (default: candidates.csv)",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        default=["single-cell", "proteomics"],
        help="Search keywords (default: 'single-cell proteomics')",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="Orbitrap Astral",
        help="Filter by instrument name (default: 'Orbitrap Astral')",
    )
    parser.add_argument(
        "--software",
        type=str,
        default="DIA-NN",
        help="Filter by software name (default: 'DIA-NN')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of results (default: 100)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PRIDE Dataset Search")
    print("=" * 60)
    print(f"Keywords: {args.keywords}")
    print(f"Instrument filter: {args.instrument}")
    print(f"Software filter: {args.software}")
    print(f"Limit: {args.limit}")
    print("=" * 60)

    # Search PRIDE
    projects = search_pride_projects(
        keywords=args.keywords,
        instrument_filter=args.instrument,
        software_filter=args.software,
        limit=args.limit,
    )

    if not projects:
        print("No projects found matching criteria")
        return

    # Extract metadata
    candidates = [extract_project_metadata(p) for p in projects]

    # Save to CSV
    output_path = Path(args.output)
    save_candidates_to_csv(candidates, output_path)

    print("=" * 60)
    print(f"✓ Search complete! Found {len(candidates)} candidates")
    print(f"✓ Output saved to: {output_path.absolute()}")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the candidates.csv file")
    print("2. Create an approved.csv with datasets to download")
    print("3. Run: python download_datasets.py --input approved.csv")


if __name__ == "__main__":
    main()
PYEOF
```

**Expected Output:** Script created successfully at `scptensor/datasets/scripts/search_datasets.py`

### Step 2: Make Script Executable and Test Syntax

**Action:** Set executable permission and check syntax

```bash
# Make script executable
chmod +x scptensor/datasets/scripts/search_datasets.py

# Test Python syntax
python3 -m py_compile scptensor/datasets/scripts/search_datasets.py
echo "✓ Syntax check passed"
```

**Expected Output:**

```
✓ Syntax check passed
```

**Verify script help:**

```bash
python3 scptensor/datasets/scripts/search_datasets.py --help
```

**Expected Output:** Help text displaying command-line options

### Step 3: Commit Search Script

**Action:** Stage and commit the script

```bash
git add scptensor/datasets/scripts/search_datasets.py
git commit -m "$(cat <<'EOF'
feat(datasets): add search_datasets.py script

- Search PRIDE for Orbitrap Astral datasets
- Filter by DIA-NN software and single-cell keywords
- Output candidates.csv with project metadata
- Support customizable search parameters

Usage:
    python scptensor/datasets/scripts/search_datasets.py \
        --keywords single-cell proteomics \
        --instrument "Orbitrap Astral" \
        --software "DIA-NN" \
        --output candidates.csv

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

**Expected Output:** Git commit successful

---

## Task 4: Create Download Script

**Goal:** Create script to download approved datasets from PRIDE/MassIVE

**Files:**
- Create: `scptensor/datasets/scripts/download_dataset.py`

### Step 1: Create Download Script

**Action:** Create the download script with complete implementation

```bash
cat > scptensor/datasets/scripts/download_dataset.py << 'PYEOF'
#!/usr/bin/env python3
"""Download datasets from PRIDE or MassIVE repositories.

This script reads a list of approved datasets and downloads the DIA-NN
report files along with metadata.

Usage:
    python download_dataset.py --accession PXD000123 --source pride
    python download_dataset.py --input approved.csv

Input:
    CSV file with columns: accession, source
    OR command-line arguments for single dataset

Output:
    - DIA-NN report.tsv file
    - metadata.json with dataset information
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List
import re

import requests


# Repository endpoints
PRIDE_API_BASE = "https://www.ebi.ac.uk/pride/ws/archive/v2"
PRIDE_FILES_ENDPOINT = f"{PRIDE_API_BASE}/projects"

MASSIVE_API_BASE = "https://massive.ucsd.edu/ProteoSAFe/proteomes"
MASSIVE_DOWNLOAD_BASE = "https://massive.ucsd.edu/ProteoSAFe/downloads"


def get_pride_files(accession: str) -> List[Dict[str, Any]]:
    """Get list of files from PRIDE project.

    Args:
        accession: PRIDE accession (e.g., PXD000123)

    Returns:
        List of file dictionaries with metadata
    """
    url = f"{PRIDE_FILES_ENDPOINT}/{accession}/files"
    params = {"pageSize": 1000}

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        files = data.get("_embedded", {}).get("files", [])
        return files

    except requests.exceptions.RequestException as e:
        print(f"Error fetching files from PRIDE: {e}")
        return []


def find_diann_file_pride(files: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Find DIA-NN report file in PRIDE file list.

    Args:
        files: List of file dictionaries from PRIDE

    Returns:
        Dictionary of the DIA-NN report file or None
    """
    # Common DIA-NN report file patterns
    patterns = [
        r".*dia.*nn.*\.tsv$",
        r".*diann.*\.tsv$",
        r".*report.*\.tsv$",
        r".*protein.*\.tsv$",
        r".*ms.*\.tsv$",
        r".*output.*\.tsv$",
    ]

    for file_info in files:
        filename = file_info.get("fileName", "").lower()

        for pattern in patterns:
            if re.match(pattern, filename, re.IGNORECASE):
                return file_info

    return None


def download_file_pride(
    file_info: Dict[str, Any],
    output_dir: Path,
    accession: str,
) -> Path:
    """Download a single file from PRIDE.

    Args:
        file_info: File information dictionary from PRIDE API
        output_dir: Output directory path
        accession: Project accession

    Returns:
        Path to downloaded file
    """
    filename = file_info.get("fileName", "")
    download_url = file_info.get("downloadUrl", "")

    if not filename or not download_url:
        raise ValueError("Invalid file information")

    output_path = output_dir / filename

    print(f"Downloading: {filename}")

    try:
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"✓ Downloaded: {output_path.name}")
        return output_path

    except requests.exceptions.RequestException as e:
        print(f"✗ Download failed: {e}")
        raise


def get_pride_project_metadata(accession: str) -> Dict[str, Any]:
    """Get project metadata from PRIDE.

    Args:
        accession: PRIDE accession

    Returns:
        Dictionary with project metadata
    """
    url = f"{PRIDE_API_BASE}/projects/{accession}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching project metadata: {e}")
        return {}


def create_metadata(
    accession: str,
    source: str,
    project_info: Dict[str, Any],
    report_file: Path,
) -> Dict[str, Any]:
    """Create metadata dictionary for dataset.

    Args:
        accession: Dataset accession
        source: Data source (PRIDE or MassIVE)
        project_info: Project metadata from repository
        report_file: Path to DIA-NN report file

    Returns:
        Metadata dictionary
    """
    # Extract instrument
    instrument = "Unknown"
    if "instrumentNames" in project_info and project_info["instrumentNames"]:
        instrument = ", ".join(project_info["instrumentNames"])

    # Extract software
    software = "Unknown"
    if "softwares" in project_info and project_info["softwares"]:
        software_list = [s.get("name", "Unknown") for s in project_info["softwares"]]
        software = ", ".join(software_list)

    # Extract organism
    organism = "Unknown"
    if "organismScientificNames" in project_info:
        organisms = project_info["organismScientificNames"]
        if organisms:
            organism = ", ".join(organisms)

    # Extract publication DOI
    citation_doi = ""
    if "references" in project_info and project_info["references"]:
        for ref in project_info["references"]:
            if ref.get("referenceType") == "Publication":
                citation_doi = ref.get("doi", "")
                break

    # Calculate file checksum
    checksum = calculate_file_checksum(report_file)

    # Parse DIA-NN report to get dimensions (if file exists)
    n_samples = 0
    n_proteins = 0

    if report_file.exists():
        try:
            import polars as pl

            df = pl.read_csv(report_file, separator="\\t", n_rows=1)
            # Count columns (first column is usually protein/precursor)
            n_samples = len(df.columns) - 1
            n_proteins = sum(1 for _ in report_file.open("r")) - 1

        except Exception as e:
            print(f"Warning: Could not parse report file for dimensions: {e}")

    metadata = {
        "id": accession,
        "source": source.upper(),
        "title": project_info.get("title", "")[:200],
        "instrument": instrument,
        "software": software,
        "software_version": "Unknown",
        "n_samples": n_samples,
        "n_proteins": n_proteins,
        "organism": organism,
        "sample_type": "Unknown",
        "citation_doi": citation_doi,
        "pride_url": f"https://www.ebi.ac.uk/pride/archive/projects/{accession}",
        "download_date": Path.cwd().strftime("%Y-%m-%d"),
        "checksum": checksum,
        "status": "downloaded",
    }

    return metadata


def calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file.

    Args:
        file_path: Path to file

    Returns:
        SHA256 checksum as hex string with "sha256:" prefix
    """
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return f"sha256:{sha256_hash.hexdigest()}"


def save_metadata(metadata: Dict[str, Any], output_path: Path) -> None:
    """Save metadata to JSON file.

    Args:
        metadata: Metadata dictionary
        output_path: Path to output JSON file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✓ Metadata saved to: {output_path}")


def download_dataset(
    accession: str,
    source: str,
    output_base_dir: Path,
) -> bool:
    """Download a single dataset.

    Args:
        accession: Dataset accession
        source: Data source (pride or massive)
        output_base_dir: Base output directory

    Returns:
        True if successful, False otherwise
    """
    print("=" * 60)
    print(f"Downloading: {accession} from {source.upper()}")
    print("=" * 60)

    # Create output directory
    if source.lower() == "pride":
        output_dir = output_base_dir / "pride" / accession
    else:
        output_dir = output_base_dir / "massive" / accession

    output_dir.mkdir(parents=True, exist_ok=True)

    # For now, only PRIDE is implemented
    if source.lower() != "pride":
        print(f"Error: Only PRIDE source is currently implemented")
        return False

    # Get project metadata
    project_info = get_pride_project_metadata(accession)
    if not project_info:
        print(f"Error: Could not fetch project info for {accession}")
        return False

    # Get file list
    files = get_pride_files(accession)
    if not files:
        print(f"Error: No files found for {accession}")
        return False

    # Find DIA-NN report file
    diann_file = find_diann_file_pride(files)
    if not diann_file:
        print(f"Error: No DIA-NN report file found for {accession}")
        print(f"Available files: {[f.get('fileName', '') for f in files[:10]]}")
        return False

    # Download report file
    try:
        report_path = download_file_pride(diann_file, output_dir, accession)
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

    # Create metadata
    metadata = create_metadata(accession, source, project_info, report_path)

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    save_metadata(metadata, metadata_path)

    print("=" * 60)
    print(f"✓ Successfully downloaded {accession}")
    print(f"  Files: {report_path.name}")
    print(f"  Location: {output_dir.absolute()}")
    print("=" * 60)

    return True


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Download datasets from PRIDE or MassIVE"
    )
    parser.add_argument(
        "--accession",
        type=str,
        help="Dataset accession (e.g., PXD000123)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["pride", "massive"],
        default="pride",
        help="Data source (default: pride)",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input CSV file with accessions to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scptensor/datasets",
        help="Base output directory (default: scptensor/datasets)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.accession and not args.input:
        parser.error("Either --accession or --input must be provided")

    output_dir = Path(args.output_dir)

    # Download single dataset
    if args.accession:
        success = download_dataset(args.accession, args.source, output_dir)
        if not success:
            exit(1)

    # Download multiple datasets from CSV
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            exit(1)

        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            total = 0
            successful = 0
            failed = 0

            for row in reader:
                accession = row.get("accession", "").strip()
                source = row.get("source", "pride").strip()

                if not accession:
                    continue

                total += 1
                if download_dataset(accession, source, output_dir):
                    successful += 1
                else:
                    failed += 1

        print("=" * 60)
        print(f"Download Summary:")
        print(f"  Total: {total}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print("=" * 60)


if __name__ == "__main__":
    main()
PYEOF
```

**Expected Output:** Script created successfully

### Step 2: Make Script Executable and Test Syntax

**Action:** Set executable permission and check syntax

```bash
# Make script executable
chmod +x scptensor/datasets/scripts/download_dataset.py

# Test Python syntax
python3 -m py_compile scptensor/datasets/scripts/download_dataset.py
echo "✓ Syntax check passed"
```

**Expected Output:**

```
✓ Syntax check passed
```

**Verify script help:**

```bash
python3 scptensor/datasets/scripts/download_dataset.py --help
```

### Step 3: Commit Download Script

**Action:** Stage and commit the script

```bash
git add scptensor/datasets/scripts/download_dataset.py
git commit -m "$(cat <<'EOF'
feat(datasets): add download_dataset.py script

- Download DIA-NN report files from PRIDE
- Generate metadata.json with dataset information
- Support single dataset or batch download from CSV
- Calculate SHA256 checksums for integrity

Usage (single):
    python scptensor/datasets/scripts/download_dataset.py \
        --accession PXD000123 \
        --source pride

Usage (batch):
    python scptensor/datasets/scripts/download_dataset.py \
        --input approved.csv

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

**Expected Output:** Git commit successful

---

## Task 5: Create Validation Script

**Goal:** Create script to validate downloaded datasets

**Files:**
- Create: `scptensor/datasets/scripts/validate_dataset.py`

### Step 1: Create Validation Script

**Action:** Create the validation script with complete implementation

```bash
cat > scptensor/datasets/scripts/validate_dataset.py << 'PYEOF'
#!/usr/bin/env python3
"""Validate downloaded datasets for completeness and integrity.

This script checks:
- File existence and readability
- DIA-NN report.tsv format
- metadata.json required fields
- File checksums
- Data dimensions

Usage:
    python validate_dataset.py --accession PXD000123
    python validate_dataset.py --all
    python validate_dataset.py --input approved.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


# Required metadata fields
REQUIRED_METADATA_FIELDS = [
    "id",
    "source",
    "title",
    "instrument",
    "software",
    "n_samples",
    "n_proteins",
    "organism",
    "download_date",
    "checksum",
    "status",
]

# Optional metadata fields
OPTIONAL_METADATA_FIELDS = [
    "software_version",
    "sample_type",
    "citation_doi",
    "pride_url",
]


def calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file.

    Args:
        file_path: Path to file

    Returns:
        SHA256 checksum as hex string with "sha256:" prefix
    """
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return f"sha256:{sha256_hash.hexdigest()}"


def validate_report_file(report_path: Path) -> Tuple[bool, List[str]]:
    """Validate DIA-NN report file format.

    Args:
        report_path: Path to DIA-NN report.tsv file

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    # Check file exists
    if not report_path.exists():
        issues.append(f"Report file not found: {report_path}")
        return False, issues

    # Check file is readable
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        if not first_line:
            issues.append("Report file is empty")
            return False, issues

    except Exception as e:
        issues.append(f"Cannot read report file: {e}")
        return False, issues

    # Check TSV format
    if POLARS_AVAILABLE:
        try:
            df = pl.read_csv(report_path, separator="\t", n_rows=1)

            # Check minimum columns
            if len(df.columns) < 2:
                issues.append(
                    f"Report file has insufficient columns: {len(df.columns)} "
                    "(minimum 2 expected)"
                )
                return False, issues

            # Typical DIA-NN columns (at least one should be present)
            expected_patterns = [
                "protein",
                "precursor",
                "q-value",
                "spectral",
                "intensity",
            ]

            columns_lower = [c.lower() for c in df.columns]
            has_expected = any(
                any(pattern in col for col in columns_lower)
                for pattern in expected_patterns
            )

            if not has_expected:
                issues.append(
                    "Report file does not contain expected DIA-NN columns. "
                    f"Found columns: {df.columns[:5]}"
                )

        except Exception as e:
            issues.append(f"Failed to parse report file as TSV: {e}")
            return False, issues

    return len(issues) == 0, issues


def validate_metadata(metadata_path: Path) -> Tuple[bool, List[str]]:
    """Validate metadata.json file.

    Args:
        metadata_path: Path to metadata.json file

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    # Check file exists
    if not metadata_path.exists():
        issues.append(f"Metadata file not found: {metadata_path}")
        return False, issues

    # Check file is valid JSON
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    except json.JSONDecodeError as e:
        issues.append(f"Invalid JSON in metadata file: {e}")
        return False, issues

    except Exception as e:
        issues.append(f"Cannot read metadata file: {e}")
        return False, issues

    # Check required fields
    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            issues.append(f"Missing required field: {field}")
        elif not metadata[field]:
            issues.append(f"Empty required field: {field}")

    # Check data types
    if "n_samples" in metadata:
        if not isinstance(metadata["n_samples"], int) or metadata["n_samples"] < 0:
            issues.append("n_samples must be a non-negative integer")

    if "n_proteins" in metadata:
        if not isinstance(metadata["n_proteins"], int) or metadata["n_proteins"] < 0:
            issues.append("n_proteins must be a non-negative integer")

    return len(issues) == 0, issues


def validate_checksum(report_path: Path, metadata: Dict[str, Any]) -> bool:
    """Validate file checksum against metadata.

    Args:
        report_path: Path to report file
        metadata: Metadata dictionary

    Returns:
        True if checksum matches, False otherwise
    """
    expected_checksum = metadata.get("checksum", "")

    if not expected_checksum:
        print("  ⚠ No checksum in metadata")
        return False

    actual_checksum = calculate_file_checksum(report_path)

    if actual_checksum != expected_checksum:
        print(f"  ✗ Checksum mismatch!")
        print(f"    Expected: {expected_checksum}")
        print(f"    Actual:   {actual_checksum}")
        return False

    print(f"  ✓ Checksum verified: {actual_checksum[:20]}...")
    return True


def validate_dataset(
    accession: str,
    base_dir: Path,
    verbose: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate a single dataset.

    Args:
        accession: Dataset accession
        base_dir: Base datasets directory
        verbose: Print detailed validation results

    Returns:
        Tuple of (is_valid, validation_report)
    """
    if verbose:
        print("=" * 60)
        print(f"Validating: {accession}")
        print("=" * 60)

    # Find dataset directory
    pride_dir = base_dir / "pride" / accession
    massive_dir = base_dir / "massive" / accession

    dataset_dir = None
    if pride_dir.exists():
        dataset_dir = pride_dir
    elif massive_dir.exists():
        dataset_dir = massive_dir
    else:
        if verbose:
            print(f"✗ Dataset directory not found")
        return False, {"status": "failed", "error": "Directory not found"}

    # Find files
    report_files = list(dataset_dir.glob("*.tsv")) + list(dataset_dir.glob("*.tsv.gz"))
    metadata_path = dataset_dir / "metadata.json"

    if not report_files:
        if verbose:
            print(f"✗ No report files found")
        return False, {"status": "failed", "error": "No report files"}

    report_path = report_files[0]

    # Validate report file
    if verbose:
        print(f"Validating report file: {report_path.name}")

    report_valid, report_issues = validate_report_file(report_path)

    if not report_valid:
        if verbose:
            print("  ✗ Report file validation failed:")
            for issue in report_issues:
                print(f"    - {issue}")
        return False, {"status": "failed", "report_issues": report_issues}

    if verbose:
        print(f"  ✓ Report file is valid")

    # Validate metadata
    if verbose:
        print(f"Validating metadata file")

    metadata_valid, metadata_issues = validate_metadata(metadata_path)

    if not metadata_valid:
        if verbose:
            print("  ✗ Metadata validation failed:")
            for issue in metadata_issues:
                print(f"    - {issue}")
        return False, {"status": "failed", "metadata_issues": metadata_issues}

    # Load metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if verbose:
        print(f"  ✓ Metadata is valid")

    # Validate checksum
    if verbose:
        print(f"Validating checksum")

    checksum_valid = validate_checksum(report_path, metadata)

    # Check data dimensions
    if POLARS_AVAILABLE and report_valid:
        try:
            df = pl.read_csv(report_path, separator="\t", n_rows=1)
            n_samples = len(df.columns) - 1

            # Count rows (excluding header)
            n_proteins = sum(1 for _ in open(report_path)) - 1

            if verbose:
                print(f"  Data dimensions:")
                print(f"    Proteins:  {n_proteins}")
                print(f"    Samples:   {n_samples}")

            # Update metadata if dimensions changed
            if metadata.get("n_samples") != n_samples or metadata.get(
                "n_proteins"
            ) != n_proteins:
                if verbose:
                    print("  ⚠ Data dimensions differ from metadata")

        except Exception as e:
            if verbose:
                print(f"  ⚠ Could not verify data dimensions: {e}")

    # Overall result
    all_valid = report_valid and metadata_valid and checksum_valid

    if verbose:
        print("=" * 60)
        if all_valid:
            print(f"✓ {accession} is VALIDATED")
        else:
            print(f"✗ {accession} validation FAILED")
        print("=" * 60)

    report = {
        "accession": accession,
        "status": "validated" if all_valid else "failed",
        "report_valid": report_valid,
        "metadata_valid": metadata_valid,
        "checksum_valid": checksum_valid,
        "issues": report_issues + metadata_issues,
    }

    return all_valid, report


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Validate downloaded datasets"
    )
    parser.add_argument(
        "--accession",
        type=str,
        help="Dataset accession to validate (e.g., PXD000123)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all datasets in the registry",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input CSV file with accessions to validate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scptensor/datasets",
        help="Base datasets directory (default: scptensor/datasets)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Validate single dataset
    if args.accession:
        is_valid, report = validate_dataset(
            args.accession, output_dir, verbose=not args.quiet
        )
        sys.exit(0 if is_valid else 1)

    # Validate all datasets
    elif args.all:
        # Find all dataset directories
        pride_dirs = list((output_dir / "pride").glob("PXD*"))
        massive_dirs = list((output_dir / "massive").glob("MSV*"))

        all_dirs = pride_dirs + massive_dirs

        if not all_dirs:
            print("No datasets found to validate")
            sys.exit(1)

        print(f"Found {len(all_dirs)} datasets to validate")
        print()

        results = {
            "total": len(all_dirs),
            "validated": 0,
            "failed": 0,
            "details": [],
        }

        for dataset_dir in sorted(all_dirs):
            accession = dataset_dir.name
            is_valid, report = validate_dataset(
                accession, output_dir, verbose=not args.quiet
            )

            if is_valid:
                results["validated"] += 1
            else:
                results["failed"] += 1

            results["details"].append(report)

        # Print summary
        print()
        print("=" * 60)
        print("Validation Summary")
        print("=" * 60)
        print(f"Total:       {results['total']}")
        print(f"Validated:   {results['validated']}")
        print(f"Failed:      {results['failed']}")
        print("=" * 60)

        sys.exit(0 if results["failed"] == 0 else 1)

    else:
        parser.error("Either --accession or --all must be provided")


if __name__ == "__main__":
    main()
PYEOF
```

**Expected Output:** Script created successfully

### Step 2: Make Script Executable and Test Syntax

**Action:** Set executable permission and check syntax

```bash
# Make script executable
chmod +x scptensor/datasets/scripts/validate_dataset.py

# Test Python syntax
python3 -m py_compile scptensor/datasets/scripts/validate_dataset.py
echo "✓ Syntax check passed"
```

**Expected Output:**

```
✓ Syntax check passed
```

**Verify script help:**

```bash
python3 scptensor/datasets/scripts/validate_dataset.py --help
```

### Step 3: Commit Validation Script

**Action:** Stage and commit the script

```bash
git add scptensor/datasets/scripts/validate_dataset.py
git commit -m "$(cat <<'EOF'
feat(datasets): add validate_dataset.py script

- Validate dataset integrity and completeness
- Check DIA-NN report.tsv format
- Verify metadata.json required fields
- Calculate and verify SHA256 checksums
- Support single dataset or batch validation

Usage:
    python scptensor/datasets/scripts/validate_dataset.py \
        --accession PXD000123

    python scptensor/datasets/scripts/validate_dataset.py \
        --all

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

**Expected Output:** Git commit successful

---

## Task 6: Execute Search and Download First Dataset

**Goal:** Complete the full workflow: search -> review -> download -> validate -> register

**Files:**
- Modify: `scptensor/datasets/registry.json`
- Create: `scptensor/datasets/pride/PXD0XXXX/`
- Create: `scptensor/datasets/pride/PXD0XXXX/report.tsv`
- Create: `scptensor/datasets/pride/PXD0XXXX/metadata.json`

### Step 1: Execute Search

**Action:** Run the search script to find candidate datasets

```bash
# Navigate to project root
cd /home/shenshang/projects/ScpTensor

# Run search script
python3 scptensor/datasets/scripts/search_datasets.py \
    --keywords single-cell proteomics "Orbitrap Astral" \
    --instrument "Orbitrap Astral" \
    --software "DIA-NN" \
    --output candidates.csv \
    --limit 50
```

**Expected Output:**

```
============================================================
PRIDE Dataset Search
============================================================
Keywords: single-cell proteomics Orbitrap Astral
Instrument filter: Orbitrap Astral
Software filter: DIA-NN
Limit: 50
============================================================
Searching PRIDE for: single-cell proteomics + Orbitrap Astral
Fetched X projects (total: X)
Found X projects after filtering
============================================================
✓ Search complete! Found X candidates
✓ Output saved to: /home/shenshang/projects/ScpTensor/candidates.csv
```

**Alternative: Manual Search**

If automated search doesn't yield results, manually search:
1. Visit PRIDE website: https://www.ebi.ac.uk/pride/
2. Search for: "single-cell proteomics Orbitrap Astral"
3. Identify candidate datasets
4. Create candidates.csv manually:

```bash
cat > candidates.csv << 'EOF'
accession,title,instrument,software,organism,publication_doi,n_files,pride_url
PXD000000,Example Dataset,Orbitrap Astral,DIA-NN,Homo sapiens,10.1016/j.cell.2024.01.001,5,https://www.ebi.ac.uk/pride/archive/projects/PXD000000
EOF
```

### Step 2: Review Candidates

**Action:** Examine the candidates.csv file

```bash
# View candidates
cat candidates.csv

# Or use polars for better formatting
python3 -c "import polars as pl; df = pl.read_csv('candidates.csv'); print(df)"
```

**Expected Output:** Table of candidate datasets with metadata

**Create approved.csv:**

```bash
# After reviewing candidates, create approved.csv
# Select one dataset for initial download

# Example (replace with actual accession):
head -n 2 candidates.csv | tail -n 1 > approved.csv

# Or manually create:
cat > approved.csv << 'EOF'
accession,source
PXD0XXXX,pride
EOF
```

**Selection Criteria:**
- Instrument: Orbitrap Astral or Astral Zoom
- Software: DIA-NN
- Has DIA-NN report file (report.tsv, output.tsv, etc.)
- Publication available
- Recent (2023+)

### Step 3: Download Dataset

**Action:** Download the approved dataset

```bash
# Download single dataset
python3 scptensor/datasets/scripts/download_dataset.py \
    --accession PXD0XXXX \
    --source pride \
    --output-dir scptensor/datasets
```

**Expected Output:**

```
============================================================
Downloading: PXD0XXXX from PRIDE
============================================================
Downloading: report.tsv
✓ Downloaded: report.tsv
✓ Metadata saved to: .../metadata.json
============================================================
✓ Successfully downloaded PXD0XXXX
  Files: report.tsv
  Location: /home/shenshang/projects/ScpTensor/scptensor/datasets/pride/PXD0XXXX
============================================================
```

**Verify downloaded files:**

```bash
# Check directory structure
ls -lh scptensor/datasets/pride/PXD0XXXX/

# Expected output:
# -rw-r--r-- 1 user user X.XM Feb 25 10:00 metadata.json
# -rw-r--r-- 1 user user XX.M Feb 25 10:00 report.tsv

# View metadata
cat scptensor/datasets/pride/PXD0XXXX/metadata.json | python3 -m json.tool
```

### Step 4: Validate Dataset

**Action:** Run validation script

```bash
# Validate the downloaded dataset
python3 scptensor/datasets/scripts/validate_dataset.py \
    --accession PXD0XXXX \
    --output-dir scptensor/datasets
```

**Expected Output:**

```
============================================================
Validating: PXD0XXXX
============================================================
Validating report file: report.tsv
  ✓ Report file is valid
Validating metadata file
  ✓ Metadata is valid
Validating checksum
  ✓ Checksum verified: sha256:abc123...
  Data dimensions:
    Proteins:  XXXX
    Samples:   XX
============================================================
✓ PXD0XXXX is VALIDATED
============================================================
```

**If validation fails:**
- Check report file format (should be TSV)
- Verify metadata.json structure
- Recalculate checksum if needed
- See validation error messages for details

### Step 5: Update Registry

**Action:** Add validated dataset to registry.json

```bash
# Read current registry
python3 << 'PYEOF'
import json
from pathlib import Path

# Load registry
registry_path = Path("scptensor/datasets/registry.json")
with open(registry_path, "r") as f:
    registry = json.load(f)

# Add new dataset (replace with actual values)
new_dataset = {
    "id": "PXD0XXXX",
    "path": "pride/PXD0XXXX",
    "instrument": "Orbitrap Astral",  # Update from metadata
    "status": "validated"
}

# Load metadata for details
metadata_path = Path("scptensor/datasets/pride/PXD0XXXX/metadata.json")
with open(metadata_path, "r") as f:
    metadata = json.load(f)

new_dataset["instrument"] = metadata.get("instrument", "Unknown")
new_dataset["n_samples"] = metadata.get("n_samples", 0)
new_dataset["n_proteins"] = metadata.get("n_proteins", 0)

# Add to registry
registry["datasets"].append(new_dataset)

# Update statistics
registry["statistics"]["total_datasets"] = len(registry["datasets"])

# Update by_instrument
instrument = new_dataset["instrument"]
if instrument not in registry["statistics"]["by_instrument"]:
    registry["statistics"]["by_instrument"][instrument] = 0
registry["statistics"]["by_instrument"][instrument] += 1

# Update by_source
source = metadata.get("source", "Unknown")
if source not in registry["statistics"]["by_source"]:
    registry["statistics"]["by_source"][source] = 0
registry["statistics"]["by_source"][source] += 1

# Update date
registry["last_updated"] = "2026-02-25"

# Save registry
with open(registry_path, "w") as f:
    json.dump(registry, f, indent=2)

print("✓ Registry updated")
print(f"Total datasets: {registry['statistics']['total_datasets']}")
print(f"By instrument: {registry['statistics']['by_instrument']}")
print(f"By source: {registry['statistics']['by_source']}")
PYEOF
```

**Expected Output:**

```
✓ Registry updated
Total datasets: 1
By instrument: {'Orbitrap Astral': 1}
By source: {'PRIDE': 1}
```

**Verify registry:**

```bash
cat scptensor/datasets/registry.json | python3 -m json.tool
```

**Expected Output:** Formatted JSON with one dataset entry

### Step 6: Commit First Dataset

**Action:** Stage and commit the downloaded dataset

```bash
# Add all new files
git add scptensor/datasets/

# Verify what will be committed
git status

# Commit
git commit -m "$(cat <<'EOF'
feat(datasets): add first validated dataset

- Download PXD0XXXX from PRIDE
- Validate DIA-NN report format
- Add metadata.json with dataset information
- Update registry.json

Dataset details:
- Instrument: Orbitrap Astral
- Samples: XX
- Proteins: XXXX
- Status: validated

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

**Expected Output:** Git commit successful

---

## Validation Checklist

### Phase 1: Directory Structure

- [ ] `scptensor/datasets/` directory created
- [ ] `scptensor/datasets/pride/` subdirectory created
- [ ] `scptensor/datasets/massive/` subdirectory created
- [ ] `scptensor/datasets/scripts/` subdirectory created
- [ ] All directories have `.gitkeep` files
- [ ] Git commit created for directory structure

### Phase 2: Registry File

- [ ] `scptensor/datasets/registry.json` created
- [ ] JSON is valid (parseable)
- [ ] Contains `version`, `last_updated`, `datasets`, `statistics` fields
- [ ] Initial `datasets` array is empty
- [ ] `statistics` has `total_datasets`, `by_instrument`, `by_source`
- [ ] Git commit created for registry file

### Phase 3: Search Script

- [ ] `scptensor/datasets/scripts/search_datasets.py` created
- [ ] Script is executable (`chmod +x`)
- [ ] Python syntax is valid (`py_compile` passes)
- [ ] `--help` flag works
- [ ] Accepts `--keywords`, `--instrument`, `--software`, `--output` parameters
- [ ] Returns candidate datasets in CSV format
- [ ] Git commit created for search script

### Phase 4: Download Script

- [ ] `scptensor/datasets/scripts/download_dataset.py` created
- [ ] Script is executable
- [ ] Python syntax is valid
- [ ] `--help` flag works
- [ ] Accepts `--accession`, `--source`, `--input`, `--output-dir` parameters
- [ ] Downloads DIA-NN report files
- [ ] Creates metadata.json with required fields
- [ ] Calculates SHA256 checksums
- [ ] Git commit created for download script

### Phase 5: Validation Script

- [ ] `scptensor/datasets/scripts/validate_dataset.py` created
- [ ] Script is executable
- [ ] Python syntax is valid
- [ ] `--help` flag works
- [ ] Accepts `--accession`, `--all`, `--input`, `--output-dir` parameters
- [ ] Validates report.tsv format
- [ ] Validates metadata.json fields
- [ ] Verifies SHA256 checksums
- [ ] Git commit created for validation script

### Phase 6: End-to-End Workflow

- [ ] Search script executes successfully
- [ ] candidates.csv file generated
- [ ] At least one suitable dataset identified
- [ ] approved.csv file created
- [ ] Download script executes successfully
- [ ] Dataset files downloaded (report.tsv, metadata.json)
- [ ] Validation script executes successfully
- [ ] Dataset passes all validation checks
- [ ] Registry updated with dataset entry
- [ ] Final commit created with dataset

---

## Success Metrics

### Quantitative Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Scripts created | 3 | Count `.py` files in `scripts/` |
| Lines of code | ~600 | `wc -l scptensor/datasets/scripts/*.py` |
| Test coverage | 0% (optional) | Manual testing only |
| Datasets downloaded | ≥1 | Count entries in `registry.json` |
| Validation pass rate | 100% | All validated datasets pass checks |
| Documentation completeness | 100% | All tasks have examples |

### Qualitative Metrics

- [ ] Code follows ScpTensor style guide
- [ ] All scripts have complete docstrings
- [ ] Error handling is robust
- [ ] User-facing messages are clear
- [ ] Scripts are idempotent (can re-run safely)
- [ ] Git commits follow project conventions
- [ ] Implementation matches design document
- [ ] README or usage guide available

---

## Appendix: Quick Reference

### Common Commands

```bash
# Search for datasets
python3 scptensor/datasets/scripts/search_datasets.py \
    --keywords "single-cell proteomics" \
    --output candidates.csv

# Download single dataset
python3 scptensor/datasets/scripts/download_dataset.py \
    --accession PXD000123 \
    --source pride

# Validate single dataset
python3 scptensor/datasets/scripts/validate_dataset.py \
    --accession PXD000123

# Validate all datasets
python3 scptensor/datasets/scripts/validate_dataset.py --all

# View registry
cat scptensor/datasets/registry.json | python3 -m json.tool

# Check dataset location
ls -lh scptensor/datasets/pride/
```

### File Locations

| Component | Path |
|-----------|------|
| Registry | `scptensor/datasets/registry.json` |
| Search script | `scptensor/datasets/scripts/search_datasets.py` |
| Download script | `scptensor/datasets/scripts/download_dataset.py` |
| Validation script | `scptensor/datasets/scripts/validate_dataset.py` |
| PRIDE datasets | `scptensor/datasets/pride/PXD0XXXX/` |
| MassIVE datasets | `scptensor/datasets/massive/MSV0XXXX/` |

### Troubleshooting

**Problem:** Search returns no results
**Solution:** Try broader keywords, check internet connection, verify PRIDE API is accessible

**Problem:** Download fails
**Solution:** Check accession number, verify file exists in repository, check network connection

**Problem:** Validation fails on checksum
**Solution:** Re-download file, verify download completed, check for corruption

**Problem:** Permission denied
**Solution:** Ensure scripts are executable (`chmod +x`), check directory permissions

**Problem:** JSON parse error
**Solution:** Validate JSON syntax, check for trailing commas, verify proper escaping

---

**Status:** Ready for implementation
**Total Tasks:** 6
**Estimated Duration:** 2.5 hours
**Last Updated:** 2026-02-25
