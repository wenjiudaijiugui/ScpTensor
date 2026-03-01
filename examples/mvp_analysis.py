#!/usr/bin/env python3
"""MVP Analysis Script for ScpTensor.

This script demonstrates a complete single-cell proteomics analysis workflow
using the ScpTensor framework, from data loading through dimensionality reduction.

Example Usage:
    # Basic analysis with DIA-NN BGS format
    python examples/mvp_analysis.py --data scptensor/datasets/pride/PXD049211/report.tsv --format diann-bgs

    # Custom output directory with verbose logging
    python examples/mvp_analysis.py --data data/report.tsv --format diann-parquet --output my_analysis --verbose

    # Skip visualization for faster processing
    python examples/mvp_analysis.py --data data/report.tsv --format diann-tsv --no-viz

Author: ScpTensor Team
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Literal

import numpy as np


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="MVP Analysis Script for ScpTensor - Complete SCP Analysis Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data report.tsv --format diann-bgs
  %(prog)s --data report.parquet --format diann-parquet --output results --verbose
  %(prog)s --data report.tsv --format diann-tsv --no-viz --n-components 20

Supported Formats:
  diann-bgs        DIA-NN BGS TSV format (protein group matrix)
  diann-tsv        DIA-NN TSV matrix format
  diann-parquet    DIA-NN Parquet report format
  spectronaut-tsv  Spectronaut TSV pivot format
        """,
    )

    # Required arguments
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input data file (TSV, Parquet, or CSV directory)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["diann-bgs", "diann-tsv", "diann-parquet", "spectronaut-tsv"],
        default="diann-bgs",
        help="Data format (default: diann-bgs)",
    )

    # Optional arguments
    parser.add_argument(
        "--assay-name",
        type=str,
        default="proteins",
        help="Name for the assay (default: proteins)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="mvp_results",
        help="Output directory for results (default: mvp_results)",
    )

    parser.add_argument(
        "--log-base",
        type=float,
        default=2.0,
        help="Base for logarithmic transformation (default: 2.0)",
    )

    parser.add_argument(
        "--log-offset",
        type=float,
        default=1.0,
        help="Offset for logarithmic transformation (default: 1.0)",
    )

    parser.add_argument(
        "--n-components",
        type=int,
        default=15,
        help="Number of PCA components (default: 15)",
    )

    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def load_data(
    data_path: str,
    data_format: str,
    assay_name: str,
    verbose: bool = False,
) -> tuple:
    """Load single-cell proteomics data from file.

    Parameters
    ----------
    data_path : str
        Path to input data file.
    data_format : str
        Data format identifier.
    assay_name : str
        Name for the assay.
    verbose : bool, optional
        Print detailed information. Default is False.

    Returns
    -------
    tuple
        (container, format_used) where container is the loaded ScpContainer
        and format_used is the actual format applied.

    Raises
    ------
    FileNotFoundError
        If input file does not exist.
    ValueError
        If data format is not supported.
    """
    # Import here to avoid issues if scptensor is not installed
    try:
        from scptensor import ScpContainer
        from scptensor.io import load_csv, read_diann, read_pivot_report
    except ImportError as e:
        raise ImportError(
            "ScpTensor is not installed. Please install with: pip install scptensor"
        ) from e

    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if verbose:
        print(f"Loading data from: {data_path}")
        print(f"Format specified: {data_format}")

    # Format mapping and handling
    def _load_pivot(p):
        """Load pivot format data (DIA-NN BGS or Spectronaut)."""
        return read_pivot_report(p, assay_name=assay_name, protein_agg=False)

    format_handlers = {
        "diann-bgs": ("DIA-NN BGS format (pivot report)", _load_pivot),
        "diann-tsv": ("DIA-NN TSV format (protein matrix)", lambda p: read_diann(p, assay_name=assay_name)),
        "diann-parquet": ("DIA-NN Parquet format (main report)", lambda p: read_diann(p, assay_name=assay_name)),
        "spectronaut-tsv": ("Spectronaut TSV format (pivot report)", _load_pivot),
    }

    if data_format not in format_handlers:
        raise ValueError(
            f"Unsupported format: {data_format}. "
            f"Supported formats: {list(format_handlers.keys())}"
        )

    format_desc, handler = format_handlers[data_format]

    try:
        if verbose:
            print(f"Using {format_desc}...")

        container = handler(path)

        if verbose:
            print("Data loaded successfully")

        return container, format_desc

    except Exception as e:
        raise ValueError(
            f"Failed to load data with format '{data_format}': {e}"
        ) from e


def print_dataset_summary(container, verbose: bool = False) -> dict:
    """Print and return dataset summary statistics.

    Parameters
    ----------
    container : ScpContainer
        Loaded data container.
    verbose : bool, optional
        Print detailed information. Default is False.

    Returns
    -------
    dict
        Summary statistics including n_samples, n_features, missing_rate.
    """
    from scptensor import get_sparsity_ratio

    n_samples = container.n_samples
    n_features = container.n_features

    # Calculate missing rate for each assay
    missing_rates = {}
    for assay_name, assay in container.assays.items():
        if "X" in assay.layers or "MaxLFQ" in assay.layers:
            layer_name = "MaxLFQ" if "MaxLFQ" in assay.layers else "X"
            matrix = assay.layers[layer_name]
            sparsity = get_sparsity_ratio(matrix.X)
            missing_rates[assay_name] = sparsity

    # Overall missing rate (use first assay)
    overall_missing = next(iter(missing_rates.values())) if missing_rates else 0.0

    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Number of samples:      {n_samples:,}")
    print(f"Number of features:     {n_features:,}")
    print(f"Missing value rate:     {overall_missing:.2%}")
    print(f"Number of assays:       {len(container.assays)}")
    print(f"Assay names:            {', '.join(container.assays.keys())}")

    if verbose:
        print("\nDetailed Missing Rate by Assay:")
        for assay_name, rate in missing_rates.items():
            print(f"  {assay_name}: {rate:.2%}")

        print("\nSample Metadata Columns:")
        print(f"  {', '.join(container.obs.columns)}")

        first_assay = next(iter(container.assays.values()))
        print("\nFeature Metadata Columns:")
        print(f"  {', '.join(first_assay.var.columns)}")

    print("=" * 60 + "\n")

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "missing_rate": overall_missing,
        "missing_rates_by_assay": missing_rates,
    }


def apply_log_transform(
    container,
    base: float = 2.0,
    offset: float = 1.0,
    verbose: bool = False,
):
    """Apply logarithmic transformation to the data.

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    base : float, optional
        Logarithm base. Default is 2.0.
    offset : float, optional
        Offset added before log transform. Default is 1.0.
    verbose : bool, optional
        Print detailed information. Default is False.

    Returns
    -------
    ScpContainer
        Container with transformed data.
    """
    from scptensor import log_transform

    if verbose:
        print(f"Applying log transformation (base={base}, offset={offset})...")

    # Determine source layer name
    first_assay_name = next(iter(container.assays.keys()))
    first_assay = container.assays[first_assay_name]

    # Use MaxLFQ if available, otherwise use X
    source_layer = "MaxLFQ" if "MaxLFQ" in first_assay.layers else "X"

    try:
        container = log_transform(
            container,
            assay_name=first_assay_name,
            source_layer=source_layer,
            new_layer_name="log",
            base=base,
            offset=offset,
        )

        if verbose:
            print("Log transformation completed")
            print(f"New layer 'log' created in assay '{first_assay_name}'")

        return container

    except Exception as e:
        raise RuntimeError(f"Log transformation failed: {e}") from e


def generate_qc_visualizations(
    container,
    output_dir: Path,
    verbose: bool = False,
) -> list[Path]:
    """Generate QC visualizations.

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    output_dir : Path
        Output directory for plots.
    verbose : bool, optional
        Print detailed information. Default is False.

    Returns
    -------
    list[Path]
        List of generated plot file paths.
    """
    from scptensor.viz.recipes import qc_completeness

    output_files = []

    if verbose:
        print("Generating QC visualizations...")

    # Get first assay name
    first_assay_name = next(iter(container.assays.keys()))

    # Missing values heatmap
    try:
        missing_plot_path = output_dir / "qc_missing_heatmap.png"

        # Use log layer if available, otherwise use MaxLFQ or X
        first_assay = container.assays[first_assay_name]
        layer_to_plot = "log" if "log" in first_assay.layers else (
            "MaxLFQ" if "MaxLFQ" in first_assay.layers else "X"
        )

        # Note: qc_completeness uses 'layer' parameter, not 'layer_name'
        fig = qc_completeness(
            container,
            assay_name=first_assay_name,
            layer=layer_to_plot,
        )

        fig.figure.savefig(missing_plot_path, dpi=300, bbox_inches="tight")
        output_files.append(missing_plot_path)

        if verbose:
            print(f"  Saved: {missing_plot_path}")

    except Exception as e:
        warnings.warn(f"Failed to generate missing values heatmap: {e}")

    return output_files


def run_pca(
    container,
    n_components: int = 15,
    verbose: bool = False,
) -> tuple:
    """Run Principal Component Analysis.

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    n_components : int, optional
        Number of principal components. Default is 15.
        Will be automatically adjusted if > min(n_samples, n_features).
    verbose : bool, optional
        Print detailed information. Default is False.

    Returns
    -------
    tuple
        (container, actual_n_components) where container has PCA results
        and actual_n_components is the number of components computed.
    """
    from scptensor import reduce_pca

    first_assay_name = next(iter(container.assays.keys()))
    # For PCA, max components is min(n_samples, n_features)
    # But due to centering, effective max is n_samples - 1
    max_components = min(container.n_samples - 1, container.n_features)

    # Ensure at least 1 component
    max_components = max(1, max_components)

    # Adjust n_components if necessary
    actual_n_components = min(n_components, max_components)

    if verbose:
        print(f"Running PCA with {actual_n_components} components...")
        if actual_n_components < n_components:
            print(f"  Note: Requested {n_components} components but limited to {max_components} by data dimensions")

    try:
        container = reduce_pca(
            container,
            assay_name=first_assay_name,
            base_layer="log",
            new_assay_name="pca",
            n_components=actual_n_components,
            random_state=42,  # For reproducibility
        )

        if verbose:
            print(f"PCA completed with {actual_n_components} components")
            # PCA results are stored in a new assay called "pca"
            if "pca" in container.assays:
                print("PCA results stored in new assay 'pca'")

        return container, actual_n_components

    except Exception as e:
        raise RuntimeError(f"PCA failed: {e}") from e


def generate_pca_visualization(
    container,
    output_dir: Path,
    verbose: bool = False,
) -> Path | None:
    """Generate PCA embedding visualization.

    Parameters
    ----------
    container : ScpContainer
        Input data container with PCA results.
    output_dir : Path
        Output directory for plots.
    verbose : bool, optional
        Print detailed information. Default is False.

    Returns
    -------
    Path | None
        Path to generated plot, or None if failed.
    """
    pca_plot_path = output_dir / "pca_embedding.png"

    try:
        if verbose:
            print("Generating PCA visualization...")

        # PCA is stored in the "pca" assay with "X" layer
        # We'll create a simple scatter plot using matplotlib
        import matplotlib.pyplot as plt

        if "pca" not in container.assays:
            warnings.warn("PCA assay not found in container")
            return None

        pca_assay = container.assays["pca"]
        if "X" not in pca_assay.layers:
            warnings.warn("X layer not found in PCA assay")
            return None

        # Get PCA coordinates
        pca_matrix = pca_assay.layers["X"].X
        n_components = pca_matrix.shape[1]

        if n_components < 2:
            warnings.warn(f"PCA has only {n_components} component(s), need at least 2 for visualization")
            return None

        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_matrix[:, 0], pca_matrix[:, 1], alpha=0.6, s=50)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Embedding")

        # Add variance explained if available
        if "explained_variance_ratio" in pca_assay.var.columns:
            var_ratios = pca_assay.var["explained_variance_ratio"].to_numpy()
            if len(var_ratios) >= 2:
                plt.xlabel(f"PC1 ({var_ratios[0]*100:.1f}%)")
                plt.ylabel(f"PC2 ({var_ratios[1]*100:.1f}%)")

        plt.tight_layout()
        plt.savefig(pca_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        if verbose:
            print(f"  Saved: {pca_plot_path}")

        return pca_plot_path

    except Exception as e:
        warnings.warn(f"Failed to generate PCA visualization: {e}")
        return None


def save_results(
    container,
    output_dir: Path,
    verbose: bool = False,
) -> Path:
    """Save analysis results to NPZ format.

    Parameters
    ----------
    container : ScpContainer
        Data container with analysis results.
    output_dir : Path
        Output directory.
    verbose : bool, optional
        Print detailed information. Default is False.

    Returns
    -------
    Path
        Path to saved NPZ file.
    """
    from scptensor import save_npz

    output_path = output_dir / "container.npz"

    if verbose:
        print(f"Saving results to: {output_path}")

    try:
        save_npz(container, output_path)

        if verbose:
            print("Results saved successfully")

        return output_path

    except Exception as e:
        raise RuntimeError(f"Failed to save results: {e}") from e


def write_analysis_summary(
    output_dir: Path,
    summary_stats: dict,
    container,
    format_used: str,
    plot_files: list[Path],
    args: argparse.Namespace,
    actual_n_components: int = None,
):
    """Write analysis summary to text file.

    Parameters
    ----------
    output_dir : Path
        Output directory.
    summary_stats : dict
        Dataset summary statistics.
    container : ScpContainer
        Final data container.
    format_used : str
        Data format description.
    plot_files : list[Path]
        Generated plot file paths.
    args : argparse.Namespace
        Command-line arguments.
    """
    summary_path = output_dir / "analysis_summary.txt"

    with open(summary_path, "w") as f:
        f.write("ScpTensor MVP Analysis Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write("Input Parameters\n")
        f.write("-" * 40 + "\n")
        f.write(f"Data file:          {args.data}\n")
        f.write(f"Format:             {args.format}\n")
        f.write(f"Format used:        {format_used}\n")
        f.write(f"Assay name:         {args.assay_name}\n")
        f.write(f"Output directory:   {args.output}\n")
        f.write("\n")

        f.write("Transformation Parameters\n")
        f.write("-" * 40 + "\n")
        f.write(f"Log base:           {args.log_base}\n")
        f.write(f"Log offset:         {args.log_offset}\n")
        if actual_n_components is not None:
            f.write(f"PCA components:     {actual_n_components}\n")
        else:
            f.write(f"PCA components:     {args.n_components}\n")
        f.write("\n")

        f.write("Dataset Summary\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of samples:      {summary_stats['n_samples']:,}\n")
        f.write(f"Number of features:     {summary_stats['n_features']:,}\n")
        f.write(f"Missing value rate:     {summary_stats['missing_rate']:.2%}\n")
        f.write(f"Number of assays:       {len(container.assays)}\n")
        f.write("\n")

        f.write("Analysis Steps Completed\n")
        f.write("-" * 40 + "\n")
        f.write("[x] Data loading\n")
        f.write("[x] Log transformation\n")
        f.write("[x] PCA dimensionality reduction\n")
        f.write(f"[x] Result export (NPZ format)\n")
        if not args.no_viz:
            f.write(f"[x] QC visualization ({len(plot_files)} plots)\n")
        else:
            f.write("[ ] QC visualization (skipped)\n")
        f.write("\n")

        if plot_files:
            f.write("Generated Visualizations\n")
            f.write("-" * 40 + "\n")
            for plot_path in plot_files:
                f.write(f"  - {plot_path.name}\n")
            f.write("\n")

        f.write("Output Files\n")
        f.write("-" * 40 + "\n")
        f.write(f"  - container.npz (processed data)\n")
        f.write(f"  - analysis_summary.txt (this file)\n")
        if not args.no_viz:
            f.write(f"  - qc_missing_heatmap.png\n")
            f.write(f"  - pca_embedding.png\n")
        f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("Analysis completed successfully\n")


def main() -> int:
    """Main analysis workflow.

    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    # Suppress warnings for cleaner output (unless verbose)
    warnings.filterwarnings("ignore")

    # Parse arguments
    args = parse_arguments()

    try:
        print("\nScpTensor MVP Analysis Pipeline")
        print("=" * 60)

        # Step 1: Load data
        print("\n[Step 1/6] Loading data...")
        container, format_used = load_data(
            data_path=args.data,
            data_format=args.format,
            assay_name=args.assay_name,
            verbose=args.verbose,
        )

        # Step 2: Print dataset summary
        print("[Step 2/6] Computing dataset summary...")
        summary_stats = print_dataset_summary(container, verbose=args.verbose)

        # Step 3: Apply log transformation
        print("[Step 3/6] Applying log transformation...")
        container = apply_log_transform(
            container,
            base=args.log_base,
            offset=args.log_offset,
            verbose=args.verbose,
        )

        # Step 4: Run PCA
        print("[Step 4/6] Running PCA...")
        container, actual_n_components = run_pca(
            container,
            n_components=args.n_components,
            verbose=args.verbose,
        )

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 5: Generate visualizations (if not skipped)
        plot_files = []
        if not args.no_viz:
            print("[Step 5/6] Generating visualizations...")
            plot_files.extend(
                generate_qc_visualizations(container, output_dir, args.verbose)
            )
            pca_plot = generate_pca_visualization(container, output_dir, args.verbose)
            if pca_plot:
                plot_files.append(pca_plot)
        else:
            print("[Step 5/6] Skipping visualizations (--no-viz flag)")

        # Step 6: Save results
        print("[Step 6/6] Saving results...")
        save_results(container, output_dir, args.verbose)

        # Write analysis summary
        write_analysis_summary(
            output_dir=output_dir,
            summary_stats=summary_stats,
            container=container,
            format_used=format_used,
            plot_files=plot_files,
            args=args,
            actual_n_components=actual_n_components,
        )

        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print(f"Results saved to: {output_dir}/")
        print(f"  - container.npz")
        print(f"  - analysis_summary.txt")
        if not args.no_viz:
            print(f"  - qc_missing_heatmap.png")
            print(f"  - pca_embedding.png")
        print("=" * 60 + "\n")

        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
