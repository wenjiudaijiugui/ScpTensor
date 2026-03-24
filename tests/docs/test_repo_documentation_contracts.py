"""Repository-level documentation and workflow drift contracts."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github/workflows"
WORKFLOW_README = WORKFLOW_DIR / "README.md"
PATH_SUFFIXES = (".md", ".json", ".yml", ".yaml", ".py", ".ipynb", ".toml")
INDEX_DOCS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs/README.md",
    REPO_ROOT / "docs/user_workflows.md",
    REPO_ROOT / "benchmark/README.md",
    WORKFLOW_README,
)
COMMAND_SURFACES = (
    [REPO_ROOT / "README.md"]
    + sorted((REPO_ROOT / "docs").rglob("*.md"))
    + sorted((REPO_ROOT / "benchmark").rglob("*.md"))
    + sorted(WORKFLOW_DIR.glob("*.yml"))
)

MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
CODE_SPAN_PATTERN = re.compile(r"`([^`\n]+)`")
WORKFLOW_EXTRA_PATTERN = re.compile(r"--extra\s+([A-Za-z0-9_-]+)")
EDITABLE_EXTRA_PATTERN = re.compile(r'uv pip install -e "\.\[([^\]]+)\]"')
BARE_LOCAL_PYTHON_PATTERN = re.compile(
    r"^\s*(?:[A-Z_][A-Z0-9_]*=\S+\s+)*python(?:3)?\s+(?:benchmark/|scripts/)",
)
NO_PROJECT_LOCAL_PYTHON_PATTERN = re.compile(
    r"^\s*(?:[A-Z_][A-Z0-9_]*=\S+\s+)*uv run --no-project python\s+"
    r"(?:benchmark/|scripts/)",
)


def _load_pyproject() -> dict[str, object]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _looks_like_path(candidate: str, *, from_code_span: bool) -> bool:
    if candidate.startswith(("http://", "https://", "mailto:", "#", "/", "~")):
        return False
    if any(char in candidate for char in "*?{}[]"):
        return False
    if from_code_span and (" " in candidate or "/" not in candidate):
        return candidate == "LICENSE"
    return "/" in candidate or candidate.endswith(PATH_SUFFIXES) or candidate == "LICENSE"


def _resolve_reference(doc_path: Path, reference: str) -> Path:
    stripped = reference.strip()
    stripped = stripped[1:-1] if stripped.startswith("<") and stripped.endswith(">") else stripped
    stripped = stripped.split("#", 1)[0]
    relative_path = Path(stripped)

    candidate = (doc_path.parent / relative_path).resolve()
    if candidate.exists():
        return candidate
    return (REPO_ROOT / relative_path).resolve()


def _iter_relative_references(doc_path: Path) -> list[tuple[str, Path]]:
    text = doc_path.read_text(encoding="utf-8")
    references: list[tuple[str, Path]] = []

    for raw in MARKDOWN_LINK_PATTERN.findall(text):
        if not _looks_like_path(raw, from_code_span=False):
            continue
        references.append((raw, _resolve_reference(doc_path, raw)))

    for raw in CODE_SPAN_PATTERN.findall(text):
        if not _looks_like_path(raw, from_code_span=True):
            continue
        references.append((raw, _resolve_reference(doc_path, raw)))

    return references


def _workflow_job_ids(workflow_path: Path) -> set[str]:
    job_ids: set[str] = set()
    in_jobs = False

    for line in workflow_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("jobs:"):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        if not line.strip() or line.startswith("  #"):
            continue
        if not line.startswith("  "):
            break
        if line.startswith("  ") and not line.startswith("    ") and line.rstrip().endswith(":"):
            job_ids.add(line.strip()[:-1])

    return job_ids


def test_docs_ci_validator_uses_uv_run_python() -> None:
    workflow_text = (WORKFLOW_DIR / "docs-ci.yml").read_text(encoding="utf-8")
    readme_text = WORKFLOW_README.read_text(encoding="utf-8")

    assert "uses: astral-sh/setup-uv@v4" in workflow_text
    assert "uv run python scripts/docs/validate_review_manifest.py" in workflow_text
    assert "`uv run python scripts/docs/validate_review_manifest.py`" in readme_text


def test_workflow_readme_covers_current_workflow_files_and_jobs() -> None:
    readme_text = WORKFLOW_README.read_text(encoding="utf-8")
    workflow_files = sorted(WORKFLOW_DIR.glob("*.yml"))

    missing_workflows = [path.name for path in workflow_files if path.name not in readme_text]
    assert missing_workflows == []

    missing_jobs: list[str] = []
    for workflow_path in workflow_files:
        for job_id in sorted(_workflow_job_ids(workflow_path)):
            if job_id not in readme_text:
                missing_jobs.append(f"{workflow_path.name}:{job_id}")

    assert missing_jobs == []


def test_top_level_docs_reference_existing_local_files() -> None:
    missing: list[str] = []

    for doc_path in INDEX_DOCS:
        for raw, resolved in _iter_relative_references(doc_path):
            if not resolved.exists():
                rel_doc = doc_path.relative_to(REPO_ROOT)
                missing.append(f"{rel_doc}: {raw}")

    assert missing == []


def test_user_workflow_guide_is_indexed_from_root_and_docs_readmes() -> None:
    root_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    docs_text = (REPO_ROOT / "docs/README.md").read_text(encoding="utf-8")

    assert "docs/user_workflows.md" in root_text
    assert "user_workflows.md" in docs_text


def test_documented_optional_extras_exist() -> None:
    pyproject = _load_pyproject()
    optional = set(pyproject["project"]["optional-dependencies"])

    referenced_extras: set[str] = set()
    for path in [REPO_ROOT / "README.md", *sorted(WORKFLOW_DIR.glob("*.yml"))]:
        text = path.read_text(encoding="utf-8")
        referenced_extras.update(WORKFLOW_EXTRA_PATTERN.findall(text))
        for group in EDITABLE_EXTRA_PATTERN.findall(text):
            referenced_extras.update(part.strip() for part in group.split(","))

    assert referenced_extras <= optional


def test_repo_local_scripts_are_documented_with_uv_run_python() -> None:
    violations: list[str] = []

    for path in COMMAND_SURFACES:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if BARE_LOCAL_PYTHON_PATTERN.search(line):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
            if NO_PROJECT_LOCAL_PYTHON_PATTERN.search(line):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")

    assert violations == []
