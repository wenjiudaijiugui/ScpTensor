#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REVIEW_GLOB = "docs/review_*.md"
MANIFEST_DEFAULT = "docs/review_manifest_20260312.json"
REVIEW_TAXONOMY_MARKER = "说明：本节统一沿用全仓库资源分型。"
REVIEW_RECHECK_MARKER = "二次核查补充"
README_TAXONOMY_HEADER = "## Evidence Taxonomy"
BENCHMARK_RESOURCE_HEADER = "## Resource Roles"
REQUIRED_TAXONOMY_LABELS = [
    "论文证据",
    "数据入口",
    "模块规范 / 软件文档",
    "资源包",
]
MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate review manifest, review docs, and benchmark doc entrypoints."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root to validate.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest path. Defaults to docs/review_manifest_20260312.json under repo root.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(read_text(path))


def _normalize_target(target: str) -> str:
    target = target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    return target


def _find_absolute_filesystem_links(text: str) -> list[str]:
    issues: list[str] = []
    for match in MARKDOWN_LINK_PATTERN.finditer(text):
        target = _normalize_target(match.group(1))
        if not target or target.startswith(("http://", "https://", "mailto:", "#", "//")):
            continue
        if WINDOWS_ABSOLUTE_PATTERN.match(target) or target.startswith("file://"):
            issues.append(target)
            continue
        if target.startswith("/"):
            issues.append(target)
    return issues


def validate_manifest_schema(manifest: Any, manifest_path: Path) -> list[str]:
    issues: list[str] = []
    prefix = manifest_path.as_posix()
    if not isinstance(manifest, dict):
        return [f"{prefix}: manifest root must be a JSON object"]

    required_keys = {"generated_on", "scope_glob", "taxonomy", "reviews"}
    missing_keys = sorted(required_keys - set(manifest))
    if missing_keys:
        issues.append(f"{prefix}: missing top-level keys: {', '.join(missing_keys)}")

    if manifest.get("scope_glob") != REVIEW_GLOB:
        issues.append(
            f"{prefix}: scope_glob must be '{REVIEW_GLOB}', got {manifest.get('scope_glob')!r}"
        )

    taxonomy = manifest.get("taxonomy")
    if not isinstance(taxonomy, dict):
        issues.append(f"{prefix}: taxonomy must be an object")
        taxonomy = {}

    missing_taxonomy = [label for label in REQUIRED_TAXONOMY_LABELS if label not in taxonomy]
    if missing_taxonomy:
        issues.append(f"{prefix}: taxonomy missing required labels: {', '.join(missing_taxonomy)}")

    reviews = manifest.get("reviews")
    if not isinstance(reviews, list):
        issues.append(f"{prefix}: reviews must be a list")
        return issues

    seen_files: set[str] = set()
    valid_roles = set(taxonomy)
    for index, entry in enumerate(reviews):
        entry_prefix = f"{prefix}: reviews[{index}]"
        if not isinstance(entry, dict):
            issues.append(f"{entry_prefix} must be an object")
            continue

        file_value = entry.get("file")
        if not isinstance(file_value, str) or not file_value:
            issues.append(f"{entry_prefix}.file must be a non-empty string")
        else:
            if file_value in seen_files:
                issues.append(f"{entry_prefix}.file duplicates {file_value}")
            seen_files.add(file_value)
            if not Path(file_value).match(REVIEW_GLOB):
                issues.append(f"{entry_prefix}.file must match '{REVIEW_GLOB}': {file_value}")

        if not isinstance(entry.get("focus"), str) or not entry["focus"].strip():
            issues.append(f"{entry_prefix}.focus must be a non-empty string")

        resource_types = entry.get("resource_types")
        if not isinstance(resource_types, list) or not resource_types:
            issues.append(f"{entry_prefix}.resource_types must be a non-empty list")
        else:
            for role in resource_types:
                if not isinstance(role, str) or not role.strip():
                    issues.append(f"{entry_prefix}.resource_types contains an empty value")
                elif valid_roles and role not in valid_roles:
                    issues.append(f"{entry_prefix}.resource_types contains unknown role '{role}'")

        stable_entrypoints = entry.get("stable_entrypoints")
        if not isinstance(stable_entrypoints, list):
            issues.append(f"{entry_prefix}.stable_entrypoints must be a list")
            continue

        for item_index, item in enumerate(stable_entrypoints):
            item_prefix = f"{entry_prefix}.stable_entrypoints[{item_index}]"
            if not isinstance(item, dict):
                issues.append(f"{item_prefix} must be an object")
                continue
            for field in ("role", "label", "url"):
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    issues.append(f"{item_prefix}.{field} must be a non-empty string")
            role = item.get("role")
            if isinstance(role, str) and valid_roles and role not in valid_roles:
                issues.append(f"{item_prefix}.role uses unknown taxonomy label '{role}'")
            url = item.get("url")
            if isinstance(url, str) and not url.startswith(("http://", "https://")):
                issues.append(f"{item_prefix}.url must be http/https: {url!r}")

        notes = entry.get("notes")
        if notes is not None and not isinstance(notes, str):
            issues.append(f"{entry_prefix}.notes must be a string when present")

    return issues


def _review_entries_by_file(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("reviews", []):
        if isinstance(entry, dict) and isinstance(entry.get("file"), str):
            result[entry["file"]] = entry
    return result


def validate_repo(repo_root: Path, manifest_path: Path | None = None) -> list[str]:
    repo_root = repo_root.resolve()
    manifest_path = (manifest_path or repo_root / MANIFEST_DEFAULT).resolve()
    issues: list[str] = []

    if not manifest_path.exists():
        return [f"missing manifest: {manifest_path}"]

    manifest = load_manifest(manifest_path)
    issues.extend(validate_manifest_schema(manifest, manifest_path))

    review_entries = _review_entries_by_file(manifest)
    manifest_files = sorted(review_entries)
    actual_review_files = sorted(
        path.relative_to(repo_root).as_posix() for path in repo_root.glob(REVIEW_GLOB)
    )

    if manifest_files != actual_review_files:
        missing_from_manifest = sorted(set(actual_review_files) - set(manifest_files))
        missing_from_repo = sorted(set(manifest_files) - set(actual_review_files))
        if missing_from_manifest:
            issues.append("manifest missing review files: " + ", ".join(missing_from_manifest))
        if missing_from_repo:
            issues.append(
                "manifest references review files not found in repo: "
                + ", ".join(missing_from_repo)
            )

    docs_to_check = [
        Path("docs/README.md"),
        Path("benchmark/README.md"),
        *[Path(item) for item in actual_review_files],
        *sorted(path.relative_to(repo_root) for path in repo_root.glob("benchmark/*/README.md")),
    ]

    for relative_path in docs_to_check:
        path = repo_root / relative_path
        if not path.exists():
            issues.append(f"missing documentation file: {relative_path.as_posix()}")
            continue
        text = read_text(path)
        absolute_links = _find_absolute_filesystem_links(text)
        if absolute_links:
            issues.append(
                f"{relative_path.as_posix()}: contains absolute filesystem links: "
                + ", ".join(sorted(set(absolute_links)))
            )

    for review_path in actual_review_files:
        relative_path = Path(review_path)
        text = read_text(repo_root / relative_path)
        if REVIEW_TAXONOMY_MARKER not in text:
            issues.append(f"{review_path}: missing taxonomy marker '{REVIEW_TAXONOMY_MARKER}'")
        if REVIEW_RECHECK_MARKER not in text:
            issues.append(f"{review_path}: missing '{REVIEW_RECHECK_MARKER}' section")
        entry = review_entries.get(review_path)
        if entry is None:
            continue
        stable_entrypoints = entry.get("stable_entrypoints", [])
        if isinstance(stable_entrypoints, list):
            for item in stable_entrypoints:
                if not isinstance(item, dict):
                    continue
                url = item.get("url")
                if isinstance(url, str) and url not in text:
                    issues.append(f"{review_path}: missing stable entrypoint URL '{url}'")

    entrypoint_docs = [Path("docs/README.md"), Path("benchmark/README.md")]
    for relative_path in entrypoint_docs:
        path = repo_root / relative_path
        if not path.exists():
            continue
        text = read_text(path)
        if README_TAXONOMY_HEADER not in text:
            issues.append(f"{relative_path.as_posix()}: missing '{README_TAXONOMY_HEADER}'")
        if Path(MANIFEST_DEFAULT).name not in text:
            issues.append(
                f"{relative_path.as_posix()}: missing manifest reference '{Path(MANIFEST_DEFAULT).name}'"
            )

    benchmark_readmes = sorted(
        path.relative_to(repo_root) for path in repo_root.glob("benchmark/*/README.md")
    )
    for relative_path in benchmark_readmes:
        text = read_text(repo_root / relative_path)
        if BENCHMARK_RESOURCE_HEADER not in text:
            issues.append(f"{relative_path.as_posix()}: missing '{BENCHMARK_RESOURCE_HEADER}'")
        for label in REQUIRED_TAXONOMY_LABELS:
            if label not in text:
                issues.append(f"{relative_path.as_posix()}: missing taxonomy label '{label}'")

    return issues


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root
    manifest_path = args.manifest or (repo_root / MANIFEST_DEFAULT)
    issues = validate_repo(repo_root=repo_root, manifest_path=manifest_path)
    if issues:
        print(f"FAIL docs contract validation: {len(issues)} issue(s)")
        for issue in issues:
            print(f"- {issue}")
        return 1

    review_count = len(list(repo_root.glob(REVIEW_GLOB)))
    benchmark_readme_count = len(list(repo_root.glob("benchmark/*/README.md")))
    print(
        "PASS docs contract validation: "
        f"reviews={review_count} benchmark_readmes={benchmark_readme_count} manifest={manifest_path.relative_to(repo_root).as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
