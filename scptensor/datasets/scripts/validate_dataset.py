#!/usr/bin/env python3
"""Validate single-cell proteomics datasets.

This script validates dataset integrity, format, and metadata for downloaded
datasets from PRIDE/MassIVE repositories.

Validation includes:
- Required files existence (report.tsv, metadata.json)
- TSV file format and required columns
- Metadata fields completeness
- File checksum verification
- Data consistency checks

Usage:
    python -m scptensor.datasets.scripts.validate_dataset --id PXD0xxxx
    python -m scptensor.datasets.scripts.validate_dataset --all
    python -m scptensor.datasets.scripts.validate_dataset --id PXD0xxxx --update-registry

Examples:
    # Validate a single dataset
    python -m scptensor.datasets.scripts.validate_dataset -i PXD046355

    # Validate all datasets with verbose output
    python -m scptensor.datasets.scripts.validate_dataset --all -v

    # Validate and update registry
    python -m scptensor.datasets.scripts.validate_dataset -i PXD046355 --update-registry
"""

import argparse
import csv
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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
REGISTRY_FILE = DATASETS_DIR / "registry.json"

# Required metadata fields
REQUIRED_METADATA_FIELDS = [
    "id",
    "source",
    "title",
    "instrument",
    "software",
    "n_samples",
    "n_proteins",
    "download_date",
    "file",
]

# Required fields in metadata["file"]
REQUIRED_FILE_METADATA = [
    "path",
    "size_bytes",
    "checksum_sha256",
]

# Required TSV columns for DIA-NN report
REQUIRED_TSV_COLUMNS = [
    "Protein.Group",
    "Protein.Groups",
    "Protein.Ids",
    "Quantity",
    "Precursor.Quantity",
    "Run",
    "Modified.Sequence",
]

# Alternative column names (at least one from each group must be present)
COLUMN_ALTERNATIVES = {
    "protein": ["Protein.Group", "Protein.Groups", "Protein.Ids"],
    "quantity": ["Quantity", "Precursor.Quantity"],
}

# Minimum file sizes (bytes)
MIN_TSV_SIZE = 1024  # 1 KB
MIN_JSON_SIZE = 100  # 100 bytes

# Checksum chunk size
CHUNK_SIZE = 8192


class ValidationResult:
    """Container for validation results."""

    def __init__(self, dataset_id: str):
        """Initialize validation result.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        """
        self.dataset_id = dataset_id
        self.is_valid = True
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: dict[str, Any] = {}

    def add_error(self, message: str) -> None:
        """Add validation error.

        Parameters
        ----------
        message : str
            Error message
        """
        self.errors.append(message)
        self.is_valid = False
        logger.error(f"[{self.dataset_id}] {message}")

    def add_warning(self, message: str) -> None:
        """Add validation warning.

        Parameters
        ----------
        message : str
            Warning message
        """
        self.warnings.append(message)
        logger.warning(f"[{self.dataset_id}] {message}")

    def add_info(self, key: str, value: Any) -> None:
        """Add validation info.

        Parameters
        ----------
        key : str
            Info key
        value : Any
            Info value
        """
        self.info[key] = value

    def __str__(self) -> str:
        """Return string representation.

        Returns
        -------
        str
            Summary string
        """
        status = "VALID" if self.is_valid else "INVALID"
        result = f"[{self.dataset_id}] {status}"
        if self.errors:
            result += f" ({len(self.errors)} errors)"
        if self.warnings:
            result += f" ({len(self.warnings)} warnings)"
        return result


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


def find_dataset_dir(dataset_id: str) -> Path | None:
    """Find dataset directory.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g., PXD046355)

    Returns
    -------
    Path or None
        Path to dataset directory or None if not found
    """
    # Check PRIDE directory first
    pride_path = PRIDE_DIR / dataset_id
    if pride_path.exists() and pride_path.is_dir():
        return pride_path

    # Check MassIVE directory
    massive_path = MASSIVE_DIR / dataset_id
    if massive_path.exists() and massive_path.is_dir():
        return massive_path

    return None


def validate_files_exist(
    dataset_dir: Path, result: ValidationResult
) -> tuple[Path | None, Path | None]:
    """Check that required files exist.

    Parameters
    ----------
    dataset_dir : Path
        Dataset directory path
    result : ValidationResult
        Validation result container

    Returns
    -------
    tuple[Path | None, Path | None]
        Paths to TSV and JSON files (or None if not found)
    """
    tsv_file = None
    json_file = None

    # Find TSV file
    tsv_files = list(dataset_dir.glob("*.tsv"))
    if not tsv_files:
        result.add_error("No TSV file found (expected report.tsv)")
    elif len(tsv_files) > 1:
        # Prefer report.tsv
        for f in tsv_files:
            if f.name == "report.tsv":
                tsv_file = f
                break
        if not tsv_file:
            tsv_file = tsv_files[0]
            result.add_warning(f"Multiple TSV files found, using: {tsv_file.name}")
    else:
        tsv_file = tsv_files[0]

    # Find metadata.json
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        result.add_error("metadata.json not found")
    else:
        json_file = metadata_path

    # Check file sizes
    if tsv_file:
        size = tsv_file.stat().st_size
        if size < MIN_TSV_SIZE:
            result.add_error(
                f"TSV file too small: {format_file_size(size)} (minimum: {format_file_size(MIN_TSV_SIZE)})"
            )
        result.add_info("tsv_file", str(tsv_file.name))
        result.add_info("tsv_size", format_file_size(size))

    if json_file:
        size = json_file.stat().st_size
        if size < MIN_JSON_SIZE:
            result.add_error(f"JSON file too small: {format_file_size(size)}")
        result.add_info("json_file", str(json_file.name))

    return tsv_file, json_file


def validate_tsv_format(file_path: Path, result: ValidationResult) -> dict[str, Any] | None:
    """Validate TSV file format and extract statistics.

    Parameters
    ----------
    file_path : Path
        Path to TSV file
    result : ValidationResult
        Validation result container

    Returns
    -------
    dict or None
        TSV statistics or None if validation failed
    """
    stats: dict[str, Any] = {
        "n_rows": 0,
        "n_columns": 0,
        "columns": [],
        "has_protein_column": False,
        "has_quantity_column": False,
    }

    try:
        with file_path.open("r", encoding="utf-8", newline="") as f:
            # Read header
            reader = csv.reader(f, delimiter="\t")
            try:
                header = next(reader)
            except StopIteration:
                result.add_error("TSV file is empty (no header)")
                return None

            stats["n_columns"] = len(header)
            stats["columns"] = header

            # Check protein columns
            for protein_col in COLUMN_ALTERNATIVES["protein"]:
                if protein_col in header:
                    stats["has_protein_column"] = True
                    stats["protein_column"] = protein_col
                    break

            if not stats["has_protein_column"]:
                result.add_warning(
                    f"No protein column found. Expected one of: {COLUMN_ALTERNATIVES['protein']}"
                )

            # Check quantity columns
            for quantity_col in COLUMN_ALTERNATIVES["quantity"]:
                if quantity_col in header:
                    stats["has_quantity_column"] = True
                    stats["quantity_column"] = quantity_col
                    break

            if not stats["has_quantity_column"]:
                result.add_warning(
                    f"No quantity column found. Expected one of: {COLUMN_ALTERNATIVES['quantity']}"
                )

            # Count rows
            row_count = 0
            for _ in reader:
                row_count += 1

            stats["n_rows"] = row_count
            result.add_info("n_rows", row_count)
            result.add_info("n_columns", len(header))

            if row_count == 0:
                result.add_warning("TSV file has no data rows (header only)")

    except UnicodeDecodeError as e:
        result.add_error(f"TSV file encoding error: {e}")
        return None
    except csv.Error as e:
        result.add_error(f"TSV parsing error: {e}")
        return None
    except Exception as e:
        result.add_error(f"Error reading TSV file: {e}")
        return None

    return stats


def validate_metadata(
    metadata: dict[str, Any], result: ValidationResult, tsv_stats: dict[str, Any] | None
) -> None:
    """Validate metadata.json fields.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary
    result : ValidationResult
        Validation result container
    tsv_stats : dict or None
        TSV statistics for cross-validation
    """
    # Check required top-level fields
    missing_fields = []
    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            missing_fields.append(field)

    if missing_fields:
        result.add_error(f"Missing required metadata fields: {', '.join(missing_fields)}")

    # Validate file metadata
    if "file" in metadata:
        file_info = metadata["file"]
        if not isinstance(file_info, dict):
            result.add_error("'file' field must be a dictionary")
        else:
            missing_file_fields = []
            for field in REQUIRED_FILE_METADATA:
                if field not in file_info:
                    missing_file_fields.append(field)

            if missing_file_fields:
                result.add_error(
                    f"Missing required file metadata: {', '.join(missing_file_fields)}"
                )

    # Cross-validate with TSV stats
    if tsv_stats and "n_samples" in metadata:
        try:
            n_samples = int(metadata["n_samples"])
            if tsv_stats["n_rows"] > 0 and n_samples != tsv_stats["n_rows"]:
                result.add_warning(
                    f"Sample count mismatch: metadata says {n_samples}, TSV has {tsv_stats['n_rows']} rows"
                )
        except (ValueError, TypeError):
            result.add_warning(f"Invalid n_samples value in metadata: {metadata['n_samples']}")

    # Validate download_date format
    if "download_date" in metadata:
        try:
            datetime.fromisoformat(metadata["download_date"])
        except (ValueError, TypeError):
            result.add_warning(f"Invalid download_date format: {metadata['download_date']}")

    # Add metadata info
    result.add_info("source", metadata.get("source", "Unknown"))
    result.add_info("instrument", metadata.get("instrument", "Unknown"))
    result.add_info("software", metadata.get("software", "Unknown"))


def validate_checksum(file_path: Path, expected: str, result: ValidationResult) -> bool:
    """Verify SHA256 checksum.

    Parameters
    ----------
    file_path : Path
        Path to file
    expected : str
        Expected checksum
    result : ValidationResult
        Validation result container

    Returns
    -------
    bool
        True if checksum matches
    """
    logger.info(f"Calculating checksum for {file_path.name}...")

    actual = calculate_checksum(file_path)

    if actual != expected:
        result.add_error(f"Checksum mismatch: expected {expected[:16]}..., got {actual[:16]}...")
        return False

    result.add_info("checksum_valid", True)
    logger.info("Checksum verified successfully")
    return True


def load_registry() -> dict[str, Any]:
    """Load registry.json.

    Returns
    -------
    dict
        Registry data
    """
    if not REGISTRY_FILE.exists():
        return {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "datasets": [],
            "statistics": {
                "total_datasets": 0,
                "by_instrument": {},
                "by_source": {},
            },
        }

    with REGISTRY_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def update_registry(dataset_info: dict[str, Any], metadata: dict[str, Any]) -> bool:
    """Update registry.json with validated dataset.

    Parameters
    ----------
    dataset_info : dict
        Dataset information from validation
    metadata : dict
        Dataset metadata

    Returns
    -------
    bool
        True if update successful
    """
    try:
        registry = load_registry()

        # Check if dataset already exists
        existing_idx = None
        for i, ds in enumerate(registry["datasets"]):
            if ds.get("id") == metadata.get("id"):
                existing_idx = i
                break

        # Create registry entry
        entry = {
            "id": metadata.get("id"),
            "source": metadata.get("source"),
            "title": metadata.get("title"),
            "instrument": metadata.get("instrument"),
            "software": metadata.get("software"),
            "n_samples": metadata.get("n_samples"),
            "n_proteins": metadata.get("n_proteins"),
            "validated": True,
            "validation_date": datetime.now().isoformat(),
            "file_size": metadata.get("file", {}).get("size_human", "Unknown"),
        }

        # Update or add entry
        if existing_idx is not None:
            registry["datasets"][existing_idx] = entry
            logger.info(f"Updated existing registry entry for {metadata.get('id')}")
        else:
            registry["datasets"].append(entry)
            logger.info(f"Added new registry entry for {metadata.get('id')}")

        # Update statistics
        registry["last_updated"] = datetime.now().isoformat()
        registry["statistics"]["total_datasets"] = len(registry["datasets"])

        # Count by instrument
        by_instrument: dict[str, int] = {}
        for ds in registry["datasets"]:
            inst = ds.get("instrument", "Unknown")
            by_instrument[inst] = by_instrument.get(inst, 0) + 1
        registry["statistics"]["by_instrument"] = by_instrument

        # Count by source
        by_source: dict[str, int] = {}
        for ds in registry["datasets"]:
            src = ds.get("source", "Unknown")
            by_source[src] = by_source.get(src, 0) + 1
        registry["statistics"]["by_source"] = by_source

        # Save registry
        with REGISTRY_FILE.open("w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

        logger.info(f"Registry updated: {REGISTRY_FILE}")
        return True

    except Exception as e:
        logger.error(f"Failed to update registry: {e}")
        return False


def validate_dataset(dataset_id: str, verbose: bool = False) -> ValidationResult:
    """Validate a single dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g., PXD046355)
    verbose : bool
        Enable verbose output

    Returns
    -------
    ValidationResult
        Validation result
    """
    result = ValidationResult(dataset_id)

    logger.info("=" * 70)
    logger.info(f"Validating dataset: {dataset_id}")
    logger.info("=" * 70)

    # Find dataset directory
    dataset_dir = find_dataset_dir(dataset_id)
    if not dataset_dir:
        result.add_error(f"Dataset directory not found for {dataset_id}")
        logger.error(f"Dataset not found in {PRIDE_DIR} or {MASSIVE_DIR}")
        return result

    logger.info(f"Dataset directory: {dataset_dir}")

    # Step 1: Check files exist
    logger.info("\n[Step 1] Checking required files...")
    tsv_file, json_file = validate_files_exist(dataset_dir, result)

    if not tsv_file or not json_file:
        logger.error("Missing required files, cannot continue validation")
        return result

    # Step 2: Validate TSV format
    logger.info("\n[Step 2] Validating TSV format...")
    tsv_stats = validate_tsv_format(tsv_file, result)

    if verbose and tsv_stats:
        logger.info(
            f"  Columns ({tsv_stats['n_columns']}): {', '.join(tsv_stats['columns'][:5])}..."
        )
        logger.info(f"  Rows: {tsv_stats['n_rows']}")

    # Step 3: Load and validate metadata
    logger.info("\n[Step 3] Validating metadata...")
    try:
        with json_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        validate_metadata(metadata, result, tsv_stats)
    except json.JSONDecodeError as e:
        result.add_error(f"Invalid JSON in metadata.json: {e}")
        return result
    except Exception as e:
        result.add_error(f"Error reading metadata.json: {e}")
        return result

    # Step 4: Verify checksum
    logger.info("\n[Step 4] Verifying file checksum...")
    if "file" in metadata and "checksum_sha256" in metadata.get("file", {}):
        expected_checksum = metadata["file"]["checksum_sha256"]
        validate_checksum(tsv_file, expected_checksum, result)
    else:
        result.add_warning("No checksum in metadata, skipping verification")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info(f"VALIDATION RESULT: {result}")
    logger.info("=" * 70)

    if result.errors:
        logger.info("\nErrors:")
        for error in result.errors:
            logger.info(f"  - {error}")

    if result.warnings:
        logger.info("\nWarnings:")
        for warning in result.warnings:
            logger.info(f"  - {warning}")

    # Store metadata for registry update
    result.add_info("metadata", metadata)

    return result


def validate_all_datasets(verbose: bool = False) -> list[ValidationResult]:
    """Validate all datasets in pride/ and massive/ directories.

    Parameters
    ----------
    verbose : bool
        Enable verbose output

    Returns
    -------
    list[ValidationResult]
        List of validation results
    """
    results: list[ValidationResult] = []
    dataset_ids: list[str] = []

    # Collect all dataset IDs
    for source_dir in [PRIDE_DIR, MASSIVE_DIR]:
        if source_dir.exists():
            for dataset_dir in source_dir.iterdir():
                if dataset_dir.is_dir() and (dataset_dir / "metadata.json").exists():
                    dataset_ids.append(dataset_dir.name)

    logger.info("=" * 70)
    logger.info(f"Found {len(dataset_ids)} datasets to validate")
    logger.info("=" * 70)

    for dataset_id in dataset_ids:
        result = validate_dataset(dataset_id, verbose=verbose)
        results.append(result)

    return results


def main() -> int:
    """Main function.

    Returns
    -------
    int
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Validate single-cell proteomics datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate a single dataset
    python -m scptensor.datasets.scripts.validate_dataset -i PXD046355

    # Validate all datasets with verbose output
    python -m scptensor.datasets.scripts.validate_dataset --all -v

    # Validate and update registry
    python -m scptensor.datasets.scripts.validate_dataset -i PXD046355 --update-registry
        """,
    )

    parser.add_argument(
        "--id",
        "-i",
        help="Dataset ID to validate (e.g., PXD046355)",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Validate all datasets",
    )
    parser.add_argument(
        "--update-registry",
        "-u",
        action="store_true",
        help="Update registry.json after successful validation",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Determine what to validate
    if args.id:
        # Validate single dataset
        result = validate_dataset(args.id, verbose=args.verbose)

        # Update registry if requested and validation passed
        if args.update_registry and result.is_valid and "metadata" in result.info:
            logger.info("\nUpdating registry...")
            update_registry(result.info, result.info["metadata"])

        return 0 if result.is_valid else 1

    elif args.all:
        # Validate all datasets
        results = validate_all_datasets(verbose=args.verbose)

        # Summary
        valid_count = sum(1 for r in results if r.is_valid)
        invalid_count = len(results) - valid_count

        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total datasets: {len(results)}")
        logger.info(f"Valid: {valid_count}")
        logger.info(f"Invalid: {invalid_count}")

        if invalid_count > 0:
            logger.info("\nInvalid datasets:")
            for r in results:
                if not r.is_valid:
                    logger.info(f"  - {r.dataset_id}: {len(r.errors)} errors")

        # Update registry if requested
        if args.update_registry and valid_count > 0:
            logger.info("\nUpdating registry...")
            for r in results:
                if r.is_valid and "metadata" in r.info:
                    update_registry(r.info, r.info["metadata"])

        logger.info("=" * 70)

        return 0 if invalid_count == 0 else 1

    else:
        logger.error("Please specify --id or --all")
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
