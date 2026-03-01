#!/usr/bin/env python3
"""ScpTensor AutoSelect Example - Complete Analysis Pipeline.

This example demonstrates the automatic method selection feature
across all analysis stages.

The AutoSelector system evaluates multiple methods at each stage
and selects the best performing one based on data quality metrics.
"""

from scptensor import ScpDataGenerator
from scptensor.autoselect import AutoSelector


def main() -> None:
    """Run the autoselect example pipeline."""
    # Create test data using the data generator
    print("Creating test container...")
    generator = ScpDataGenerator(
        n_samples=100,
        n_features=50,
        missing_rate=0.3,
    )
    container = generator.generate()

    # Create auto selector
    # - stages: list of analysis stages to run
    # - keep_all: if False, only keeps the best method result
    print("\nCreating AutoSelector...")
    selector = AutoSelector(
        stages=["normalize", "impute"],
        keep_all=False,
    )

    # Run automatic selection
    # This will:
    # 1. Test all normalization methods on the raw data
    # 2. Select the best normalization method
    # 3. Test all imputation methods on the normalized data
    # 4. Select the best imputation method
    print("\nRunning automatic method selection...")
    result_container, report = selector.run(
        container,
        assay_name="proteins",
        initial_layer="raw",
    )

    # Print summary
    print("\n" + report.summary())
    print()

    # Export report in multiple formats
    print("Saving reports...")
    report.save("tmp/autoselect_report.md")
    report.save("tmp/autoselect_report.json", format="json")
    report.save("tmp/autoselect_scores.csv", format="csv")
    print("\nReports saved to tmp/ directory")

    # Access individual stage results
    if "normalization" in report.stages:
        norm_report = report.stages["normalization"]
        print("\nNormalization stage:")
        print(f"  Best method: {norm_report.best_method}")
        if norm_report.best_result:
            print(f"  Overall score: {norm_report.best_result.overall_score:.4f}")
            print(f"  Result layer: {norm_report.best_result.layer_name}")

    if "imputation" in report.stages:
        impute_report = report.stages["imputation"]
        print("\nImputation stage:")
        print(f"  Best method: {impute_report.best_method}")
        if impute_report.best_result:
            print(f"  Overall score: {impute_report.best_result.overall_score:.4f}")
            print(f"  Result layer: {impute_report.best_result.layer_name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
