#!/usr/bin/env python3
"""Search PRIDE database for single-cell proteomics datasets.

This script searches the PRIDE database for datasets that meet the following criteria:
- Instrument: Orbitrap Astral / Astral Zoom
- Software: DIA-NN
- Data type: Single-cell proteomics

Output: candidates.csv with dataset metadata

Usage:
    python -m scptensor.datasets.scripts.search_datasets --output candidates.csv --verbose

Example:
    python -m scptensor.datasets.scripts.search_datasets -o my_candidates.csv -v
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# PRIDE API endpoints
PRIDE_API_BASE = "https://www.ebi.ac.uk/pride/ws/archive/v2"
PRIDE_PROJECT_SEARCH = f"{PRIDE_API_BASE}/search/projects"
PRIDE_PROJECT_BY_ACCESSION = f"{PRIDE_API_BASE}/projects"

# Search parameters
SEARCH_KEYWORDS = [
    "single-cell proteomics",
    "Orbitrap Astral",
    "Astral Zoom",
    "DIA-NN",
    "single cell mass spectrometry",
    "SCoPE-MS",
    "SCoPE2",
]

# Instruments to filter for
TARGET_INSTRUMENTS = [
    "Orbitrap Astral",
    "Astral Zoom",
    "Orbitrap Exploris",
]

# Software to filter for
TARGET_SOFTWARE = [
    "DIA-NN",
    "MaxQuant",
    "Spectronaut",
]

# Request settings
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def make_request_with_retry(
    url: str, params: dict | None = None, retry_count: int = 0
) -> dict | None:
    """Make HTTP request with retry logic.

    Parameters
    ----------
    url : str
        URL to request
    params : dict, optional
        Query parameters
    retry_count : int
        Current retry attempt number

    Returns
    -------
    dict or None
        JSON response or None if failed
    """
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if retry_count < MAX_RETRIES:
            logger.warning(f"Request failed (attempt {retry_count + 1}/{MAX_RETRIES}): {e}")
            logger.info(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            return make_request_with_retry(url, params, retry_count + 1)
        else:
            logger.error(f"Request failed after {MAX_RETRIES} attempts: {e}")
            return None


def search_pride(keyword: str, page: int = 0, size: int = 100) -> list[dict] | None:
    """Search PRIDE database with keyword.

    Parameters
    ----------
    keyword : str
        Search keyword
    page : int
        Page number (0-indexed)
    size : int
        Number of results per page

    Returns
    -------
    list[dict] or None
        List of project results or None if failed
    """
    params = {
        "keyword": keyword,
        "page": page,
        "size": size,
        "sortDirection": "desc",
        "sortFields": "submissionDate",
    }

    logger.info(f"Searching PRIDE for: '{keyword}' (page {page})")
    result = make_request_with_retry(PRIDE_PROJECT_SEARCH, params)

    # PRIDE API returns a list of projects directly
    if result and isinstance(result, list):
        logger.info(f"Found {len(result)} results for '{keyword}' on page {page}")
        return result
    elif result:
        logger.warning(f"Unexpected result type: {type(result)}")
        return None

    return None


def get_project_details(accession: str) -> dict | None:
    """Get detailed information for a specific project.

    Parameters
    ----------
    accession : str
        Project accession number (e.g., PXD012345)

    Returns
    -------
    dict or None
        Project details or None if failed
    """
    url = f"{PRIDE_PROJECT_BY_ACCESSION}/{accession}"
    return make_request_with_retry(url)


def extract_dataset_info(project: dict) -> dict:
    """Extract relevant information from PRIDE project.

    Parameters
    ----------
    project : dict
        PRIDE project metadata

    Returns
    -------
    dict
        Extracted dataset information
    """
    # Extract basic info
    accession = project.get("accession", "")
    title = project.get("title", "")

    # Extract instrument information
    instruments = project.get("instrumentNames", [])
    instrument_str = "; ".join(instruments) if instruments else "Unknown"

    # Extract software/analysis information
    # PRIDE stores this in different fields
    software_list = []

    # Check projectTags for software
    project_tags = project.get("projectTags", [])
    software_list.extend([tag for tag in project_tags if tag])

    # Check references for software mentions
    references = project.get("references", [])
    for ref in references:
        # Handle both dict and string references
        if isinstance(ref, dict):
            ref_title = ref.get("title", "")
            if ref_title:
                software_list.append(ref_title)
        elif isinstance(ref, str) and ref:
            software_list.append(ref)

    software_str = "; ".join(software_list) if software_list else "Unknown"

    # Extract sample count (if available)
    # This might be in different fields depending on the project
    n_samples = "Unknown"
    if "numAssays" in project:
        n_samples = str(project["numAssays"])
    elif "sampleCount" in project:
        n_samples = str(project["sampleCount"])

    # Extract submission date
    submission_date = project.get("submissionDate", "")
    publication_date = project.get("publicationDate", "")

    # Build URL
    url = f"https://www.ebi.ac.uk/pride/archive/projects/{accession}"

    # Extract citation
    citation = ""
    if references:
        first_ref = references[0]
        # Handle both dict and string references
        if isinstance(first_ref, dict):
            authors = first_ref.get("authors", [])
            ref_title = first_ref.get("title", "")
            journal = first_ref.get("journal", "")
            year = first_ref.get("year", "")
            doi = first_ref.get("doi", "")

            citation_parts = []
            if authors:
                citation_parts.append(
                    ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
                )
            if year:
                citation_parts.append(f"({year})")
            if ref_title:
                citation_parts.append(ref_title)
            if journal:
                citation_parts.append(journal)
            if doi:
                citation_parts.append(f"DOI: {doi}")

            citation = ". ".join(citation_parts)
        elif isinstance(first_ref, str):
            citation = first_ref

    return {
        "dataset_id": accession,
        "title": title,
        "instrument": instrument_str,
        "software": software_str,
        "n_samples": n_samples,
        "submission_date": submission_date,
        "publication_date": publication_date,
        "url": url,
        "citation": citation,
    }


def matches_criteria(dataset: dict) -> bool:
    """Check if dataset matches target criteria.

    Parameters
    ----------
    dataset : dict
        Dataset information

    Returns
    -------
    bool
        True if matches criteria
    """
    # Check instrument
    instrument = dataset.get("instrument", "").lower()
    has_target_instrument = any(target.lower() in instrument for target in TARGET_INSTRUMENTS)

    # Check software
    software = dataset.get("software", "").lower()
    has_target_software = any(target.lower() in software for target in TARGET_SOFTWARE)

    # Check if it's single-cell related (from title)
    title = dataset.get("title", "").lower()
    is_single_cell = any(
        kw in title for kw in ["single-cell", "single cell", "scp", "scope", "cytof"]
    )

    # Accept if it has target instrument OR is single-cell related
    # (we want to be inclusive to not miss good datasets)
    return has_target_instrument or is_single_cell or has_target_software


def filter_datasets(datasets: list[dict], verbose: bool = False) -> list[dict]:
    """Filter datasets by instrument and software criteria.

    Parameters
    ----------
    datasets : list
        List of dataset dictionaries
    verbose : bool
        Enable verbose logging

    Returns
    -------
    list
        Filtered list of datasets
    """
    filtered = []
    seen = set()  # Track unique accessions

    for dataset in datasets:
        accession = dataset.get("dataset_id", "")

        # Skip duplicates
        if accession in seen:
            continue

        # Check criteria
        if matches_criteria(dataset):
            filtered.append(dataset)
            seen.add(accession)
            if verbose:
                logger.info(f"✓ {accession}: {dataset['title'][:60]}...")
        else:
            if verbose:
                logger.debug(f"✗ {accession}: Does not match criteria")

    return filtered


def save_candidates(candidates: list[dict], output_path: str) -> None:
    """Save candidates to CSV file.

    Parameters
    ----------
    candidates : list
        List of candidate datasets
    output_path : str
        Output CSV file path
    """
    if not candidates:
        logger.warning("No candidates to save")
        return

    fieldnames = [
        "dataset_id",
        "title",
        "instrument",
        "software",
        "n_samples",
        "submission_date",
        "publication_date",
        "url",
        "citation",
    ]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    logger.info(f"Saved {len(candidates)} candidates to {output_path}")


def main() -> int:
    """Main function.

    Returns
    -------
    int
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Search PRIDE database for single-cell proteomics datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic search
    python -m scptensor.datasets.scripts.search_datasets

    # Custom output file with verbose logging
    python -m scptensor.datasets.scripts.search_datasets -o my_candidates.csv -v

    # Use specific keywords
    python -m scptensor.datasets.scripts.search_datasets --keywords "Orbitrap Astral" "DIA-NN"
        """,
    )
    parser.add_argument(
        "--output",
        "-o",
        default="candidates.csv",
        help="Output CSV file path (default: candidates.csv)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        help="Custom search keywords (default: built-in keyword list)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Maximum pages to search per keyword (default: 5)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Number of results per page (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Search but do not save results",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Use custom keywords or default list
    keywords = args.keywords if args.keywords else SEARCH_KEYWORDS

    logger.info("=" * 70)
    logger.info("PRIDE Single-Cell Proteomics Dataset Search")
    logger.info("=" * 70)
    logger.info(f"Keywords: {', '.join(keywords)}")
    logger.info(f"Target instruments: {', '.join(TARGET_INSTRUMENTS)}")
    logger.info(f"Target software: {', '.join(TARGET_SOFTWARE)}")
    logger.info(f"Output file: {args.output}")
    logger.info("=" * 70)

    # Collect all datasets
    all_datasets = []
    seen_accessions = set()

    for keyword in keywords:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Searching keyword: {keyword}")
        logger.info("=" * 70)

        # Search multiple pages
        for page in range(args.max_pages):
            projects = search_pride(keyword, page=page, size=args.page_size)

            if not projects:
                logger.warning(f"No results for keyword '{keyword}' page {page}")
                break

            if len(projects) == 0:
                logger.info(f"No more results for '{keyword}' after page {page}")
                break

            logger.info(f"Processing {len(projects)} projects from page {page}")

            # Process each project
            for project in projects:
                accession = project.get("accession", "")

                # Skip duplicates
                if accession in seen_accessions:
                    continue

                # Extract dataset info
                dataset_info = extract_dataset_info(project)
                all_datasets.append(dataset_info)
                seen_accessions.add(accession)

                if args.verbose:
                    logger.debug(f"  Found: {accession} - {dataset_info['title'][:50]}...")

            # If we got fewer results than requested, we've reached the end
            if len(projects) < args.page_size:
                logger.info(f"Reached last page (got {len(projects)} < {args.page_size} results)")
                break

            # Be nice to the API
            time.sleep(0.5)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Total datasets collected: {len(all_datasets)}")
    logger.info("=" * 70)

    # Filter datasets
    logger.info("\nFiltering datasets by criteria...")
    candidates = filter_datasets(all_datasets, verbose=args.verbose)

    logger.info(f"\n{'=' * 70}")
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total datasets found: {len(all_datasets)}")
    logger.info(f"Candidates after filtering: {len(candidates)}")

    # Display top candidates
    if candidates:
        logger.info(f"\nTop {min(10, len(candidates))} candidates:")
        for i, candidate in enumerate(candidates[:10], 1):
            logger.info(f"\n{i}. {candidate['dataset_id']}: {candidate['title'][:70]}...")
            logger.info(f"   Instrument: {candidate['instrument']}")
            logger.info(f"   Software: {candidate['software'][:50]}...")
            logger.info(f"   URL: {candidate['url']}")

    # Save results
    if not args.dry_run and candidates:
        logger.info(f"\nSaving candidates to {args.output}...")
        save_candidates(candidates, args.output)
    elif args.dry_run:
        logger.info("\n[DRY RUN] Not saving results")
    else:
        logger.warning("\nNo candidates to save")

    logger.info("\n" + "=" * 70)
    logger.info("Search complete!")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
