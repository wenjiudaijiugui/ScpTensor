#!/usr/bin/env python3
"""Download single-cell proteomics datasets from PRIDE/MassIVE.

This script downloads DIA-NN result files (report.tsv) from PRIDE or MassIVE
repositories and generates metadata.json files.

Usage:
    python -m scptensor.datasets.scripts.download_dataset --id PXD0xxxx
    python -m scptensor.datasets.scripts.download_dataset --from candidates.csv

Examples:
    # Download a single dataset
    python -m scptensor.datasets.scripts.download_dataset -i PXD046355

    # Download from CSV list
    python -m scptensor.datasets.scripts.download_dataset -f candidates.csv

    # Custom output directory
    python -m scptensor.datasets.scripts.download_dataset -i PXD046355 -o ./my_data
"""

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from ftplib import FTP
from pathlib import Path
from typing import Any

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Base directories
DATASETS_DIR = Path(__file__).parent.parent
PRIDE_DIR = DATASETS_DIR / "pride"
MASSIVE_DIR = DATASETS_DIR / "massive"

# PRIDE FTP and HTTP servers
PRIDE_FTP = "ftp.pride.ebi.ac.uk"
PRIDE_HTTP = "https://ftp.pride.ebi.ac.uk"

# Request settings
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
CHUNK_SIZE = 8192  # Download chunk size

# DIA-NN result file patterns
DIANN_PATTERNS = [
    "report.tsv",
    "report.parquet",
    "diann_report.tsv",
    "DIA-NN.report.tsv",
]


def make_request_with_retry(
    url: str, params: dict | None = None, retry_count: int = 0
) -> requests.Response | None:
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
    requests.Response or None
        Response object or None if failed
    """
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        if retry_count < MAX_RETRIES:
            logger.warning(f"Request failed (attempt {retry_count + 1}/{MAX_RETRIES}): {e}")
            logger.info(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            return make_request_with_retry(url, params, retry_count + 1)
        else:
            logger.error(f"Request failed after {MAX_RETRIES} attempts: {e}")
            return None


def calculate_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file.

    Parameters
    ----------
    file_path : Path
        Path to file

    Returns
    -------
    str
        SHA256 checksum (hex string)
    """
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Parameters
    ----------
    size_bytes : int
        Size in bytes

    Returns
    -------
    str
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def download_file_http(url: str, output_path: Path, verbose: bool = False) -> bool:
    """Download file using HTTP with progress display.

    Parameters
    ----------
    url : str
        URL to download
    output_path : Path
        Output file path
    verbose : bool
        Enable verbose output

    Returns
    -------
    bool
        True if download successful
    """
    logger.info(f"Downloading from: {url}")
    logger.info(f"Output file: {output_path}")

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()

        # Get file size
        total_size = int(response.headers.get("content-length", 0))
        if total_size > 0:
            logger.info(f"File size: {format_file_size(total_size)}")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress
        downloaded = 0
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Progress update
                    if verbose and total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(
                            f"Progress: {progress:.1f}% ({format_file_size(downloaded)}/{format_file_size(total_size)})"
                        )

        logger.info(f"Download complete: {output_path}")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        # Clean up partial download
        if output_path.exists():
            output_path.unlink()
        return False


def download_file_ftp(
    ftp_path: str, output_path: Path, host: str = PRIDE_FTP, verbose: bool = False
) -> bool:
    """Download file using FTP.

    Parameters
    ----------
    ftp_path : str
        Path on FTP server
    output_path : Path
        Output file path
    host : str
        FTP host
    verbose : bool
        Enable verbose output

    Returns
    -------
    bool
        True if download successful
    """
    logger.info(f"Connecting to FTP: {host}")
    logger.info(f"Downloading: {ftp_path}")

    try:
        with FTP(host) as ftp:
            ftp.login()  # Anonymous login

            # Get file size
            try:
                total_size = ftp.size(ftp_path)
                if total_size:
                    logger.info(f"File size: {format_file_size(total_size)}")
            except Exception:
                total_size = 0

            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            downloaded = [0]  # Use list for closure

            def callback(data: bytes) -> None:
                """Callback for FTP download."""
                with output_path.open("ab") as f:
                    f.write(data)
                downloaded[0] += len(data)

                # Progress update
                if verbose and total_size > 0:
                    progress = (downloaded[0] / total_size) * 100
                    logger.info(
                        f"Progress: {progress:.1f}% ({format_file_size(downloaded[0])}/{format_file_size(total_size)})"
                    )

            # Clear output file
            if output_path.exists():
                output_path.unlink()

            ftp.retrbinary(f"RETR {ftp_path}", callback)

            logger.info(f"Download complete: {output_path}")
            return True

    except Exception as e:
        logger.error(f"FTP download failed: {e}")
        # Clean up partial download
        if output_path.exists():
            output_path.unlink()
        return False


def find_diann_file_in_pride(pride_id: str, verbose: bool = False) -> str | None:
    """Find DIA-NN report file in PRIDE project.

    Parameters
    ----------
    pride_id : str
        PRIDE project ID (e.g., PXD046355)
    verbose : bool
        Enable verbose output

    Returns
    -------
    str or None
        Path to DIA-NN file on FTP server, or None if not found
    """
    logger.info(f"Searching for DIA-NN file in {pride_id}...")

    # PRIDE FTP directory structure
    ftp_base = f"/pride/data/archive/{pride_id[:3]}/{pride_id[:7]}/{pride_id}"

    try:
        with FTP(PRIDE_FTP) as ftp:
            ftp.login()

            # List files in project directory
            try:
                files = []
                ftp.cwd(ftp_base)

                # Recursively search for DIA-NN files
                def search_directory(current_dir: str = "") -> None:
                    """Recursively search for DIA-NN files."""
                    try:
                        items = ftp.nlst(current_dir)
                        for item in items:
                            # Check if it's a DIA-NN file
                            item_name = Path(item).name
                            if any(pattern in item_name for pattern in DIANN_PATTERNS):
                                files.append(f"{ftp_base}/{item}")

                            # Try to list as directory
                            try:
                                ftp.cwd(f"{ftp_base}/{item}")
                                ftp.cwd(ftp_base)  # Go back
                                search_directory(item)
                            except Exception:
                                pass  # Not a directory
                    except Exception as e:
                        if verbose:
                            logger.debug(f"Error listing directory {current_dir}: {e}")

                search_directory()

                if files:
                    logger.info(f"Found {len(files)} DIA-NN file(s)")
                    for f in files:
                        logger.info(f"  - {f}")

                    # Return first match (prefer report.tsv)
                    for pattern in DIANN_PATTERNS:
                        for f in files:
                            if pattern in f:
                                return f

                    return files[0]
                else:
                    logger.warning(f"No DIA-NN file found in {pride_id}")
                    return None

            except Exception as e:
                logger.error(f"Error accessing PRIDE FTP: {e}")
                return None

    except Exception as e:
        logger.error(f"FTP connection failed: {e}")
        return None


def get_pride_project_metadata(pride_id: str) -> dict[str, Any]:
    """Get project metadata from PRIDE API.

    Parameters
    ----------
    pride_id : str
        PRIDE project ID

    Returns
    -------
    dict
        Project metadata
    """
    url = f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{pride_id}"

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Could not fetch project metadata: {e}")
        return {}


def create_metadata(
    pride_id: str,
    file_path: Path,
    source: str = "PRIDE",
    project_info: dict | None = None,
) -> dict[str, Any]:
    """Create metadata.json for dataset.

    Parameters
    ----------
    pride_id : str
        PRIDE project ID
    file_path : Path
        Path to downloaded file
    source : str
        Data source (PRIDE or MassIVE)
    project_info : dict, optional
        Additional project information

    Returns
    -------
    dict
        Metadata dictionary
    """
    # Get file stats
    file_stats = file_path.stat()
    checksum = calculate_checksum(file_path)

    # Get project metadata from PRIDE API
    pride_metadata = get_pride_project_metadata(pride_id)

    # Extract instrument
    instruments = pride_metadata.get("instrumentNames", [])
    instrument = "; ".join(instruments) if instruments else "Unknown"

    # Extract title
    title = pride_metadata.get("title", pride_id)

    # Count samples and proteins (estimate from file)
    n_samples = "Unknown"
    n_proteins = "Unknown"

    if file_path.suffix == ".tsv":
        try:
            with file_path.open("r") as f:
                # First line is header
                header = f.readline().strip().split("\t")

                # Count lines (samples)
                line_count = sum(1 for _ in f)
                n_samples = str(line_count)

                # Try to count unique proteins
                if "Protein.Names" in header or "Proteins" in header:
                    protein_col = "Protein.Names" if "Protein.Names" in header else "Proteins"
                    proteins = set()
                    f.seek(0)
                    next(f)  # Skip header
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) > header.index(protein_col):
                            protein_names = parts[header.index(protein_col)].split(";")
                            proteins.update(protein_names)
                    n_proteins = str(len(proteins))

        except Exception as e:
            logger.warning(f"Could not parse file for statistics: {e}")

    # Build metadata
    metadata = {
        "id": pride_id,
        "source": source,
        "title": title,
        "instrument": instrument,
        "software": "DIA-NN",
        "n_samples": n_samples,
        "n_proteins": n_proteins,
        "download_date": datetime.now().isoformat(),
        "file": {
            "path": str(file_path.name),
            "size_bytes": file_stats.st_size,
            "size_human": format_file_size(file_stats.st_size),
            "checksum_sha256": checksum,
        },
        "url": f"https://www.ebi.ac.uk/pride/archive/projects/{pride_id}",
    }

    # Add project info if available
    if project_info:
        metadata["project_info"] = project_info

    return metadata


def download_from_pride(
    pride_id: str, output_dir: Path, verbose: bool = False, use_http: bool = True
) -> bool:
    """Download dataset from PRIDE.

    Parameters
    ----------
    pride_id : str
        PRIDE project ID (e.g., PXD046355)
    output_dir : Path
        Output directory
    verbose : bool
        Enable verbose output
    use_http : bool
        Use HTTP instead of FTP

    Returns
    -------
    bool
        True if download successful
    """
    logger.info("=" * 70)
    logger.info(f"Downloading from PRIDE: {pride_id}")
    logger.info("=" * 70)

    # Create output directory
    project_dir = output_dir / pride_id
    project_dir.mkdir(parents=True, exist_ok=True)

    # Find DIA-NN file
    diann_file_path = find_diann_file_in_pride(pride_id, verbose)

    if not diann_file_path:
        logger.error(f"Could not find DIA-NN file for {pride_id}")
        return False

    # Output file path
    output_file = project_dir / Path(diann_file_path).name

    # Download file
    if use_http:
        # Convert FTP path to HTTP URL
        http_url = f"{PRIDE_HTTP}{diann_file_path}"
        success = download_file_http(http_url, output_file, verbose)
    else:
        success = download_file_ftp(diann_file_path, output_file, verbose=verbose)

    if not success:
        logger.error(f"Failed to download {pride_id}")
        return False

    # Create metadata
    logger.info("Generating metadata...")
    metadata = create_metadata(pride_id, output_file, source="PRIDE")

    # Save metadata
    metadata_file = project_dir / "metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved to: {metadata_file}")
    logger.info("=" * 70)
    logger.info(f"Download complete: {pride_id}")
    logger.info("=" * 70)

    return True


def read_dataset_list(csv_path: Path) -> list[str]:
    """Read dataset IDs from CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file

    Returns
    -------
    list[str]
        List of dataset IDs
    """
    dataset_ids = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset_id = row.get("dataset_id", "")
            if dataset_id:
                dataset_ids.append(dataset_id)

    return dataset_ids


def main() -> int:
    """Main function.

    Returns
    -------
    int
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Download single-cell proteomics datasets from PRIDE/MassIVE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download a single dataset
    python -m scptensor.datasets.scripts.download_dataset -i PXD046355

    # Download from CSV list
    python -m scptensor.datasets.scripts.download_dataset -f candidates.csv

    # Custom output directory with verbose output
    python -m scptensor.datasets.scripts.download_dataset -i PXD046355 -o ./my_data -v

    # Use FTP instead of HTTP
    python -m scptensor.datasets.scripts.download_dataset -i PXD046355 --ftp
        """,
    )

    parser.add_argument(
        "--id",
        "-i",
        help="Dataset ID (e.g., PXD046355)",
    )
    parser.add_argument(
        "--from",
        "-f",
        dest="from_file",
        help="CSV file with dataset IDs (reads 'dataset_id' column)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help=f"Output directory (default: {PRIDE_DIR})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--ftp",
        action="store_true",
        help="Use FTP instead of HTTP for downloads",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find files but do not download",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Determine dataset IDs
    dataset_ids = []

    if args.id:
        dataset_ids = [args.id]
    elif args.from_file:
        csv_path = Path(args.from_file)
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return 1

        dataset_ids = read_dataset_list(csv_path)
        logger.info(f"Loaded {len(dataset_ids)} dataset IDs from {csv_path}")
    else:
        logger.error("Please specify --id or --from")
        parser.print_help()
        return 1

    # Determine output directory
    output_dir = args.output if args.output else PRIDE_DIR

    logger.info("=" * 70)
    logger.info("ScpTensor Dataset Downloader")
    logger.info("=" * 70)
    logger.info(f"Datasets to download: {len(dataset_ids)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Download method: {'FTP' if args.ftp else 'HTTP'}")
    logger.info("=" * 70)

    # Download each dataset
    success_count = 0
    failed_datasets = []

    for i, dataset_id in enumerate(dataset_ids, 1):
        logger.info(f"\n[{i}/{len(dataset_ids)}] Processing: {dataset_id}")

        if args.dry_run:
            # Just find the file
            diann_file = find_diann_file_in_pride(dataset_id, verbose=args.verbose)
            if diann_file:
                logger.info(f"[DRY RUN] Would download: {diann_file}")
                success_count += 1
            else:
                logger.warning(f"[DRY RUN] No DIA-NN file found for {dataset_id}")
                failed_datasets.append(dataset_id)
        else:
            # Actually download
            success = download_from_pride(
                dataset_id,
                output_dir,
                verbose=args.verbose,
                use_http=not args.ftp,
            )

            if success:
                success_count += 1
            else:
                failed_datasets.append(dataset_id)

        # Be nice to the server
        if i < len(dataset_ids):
            time.sleep(1)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total datasets: {len(dataset_ids)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {len(failed_datasets)}")

    if failed_datasets:
        logger.info("\nFailed datasets:")
        for dataset_id in failed_datasets:
            logger.info(f"  - {dataset_id}")

    logger.info("=" * 70)

    return 0 if success_count == len(dataset_ids) else 1


if __name__ == "__main__":
    sys.exit(main())
