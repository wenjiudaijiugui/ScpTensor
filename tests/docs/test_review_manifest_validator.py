from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_validator_module() -> object:
    module_path = Path(__file__).resolve().parents[2] / "scripts/docs/validate_review_manifest.py"
    spec = importlib.util.spec_from_file_location("validate_review_manifest", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_repo_fixture(tmp_path: Path, *, break_review_url: bool = False) -> tuple[Path, Path]:
    manifest_path = tmp_path / "docs/review_manifest_20260312.json"
    review_path = tmp_path / "docs/review_example_20260312.md"
    stable_url = "https://example.org/stable-entry"
    review_body = (
        "# Example Review\n\n"
        "## 3. Sources\n\n"
        "说明：本节统一沿用全仓库资源分型。除特别标注外，单篇文献条目默认记为 "
        "`论文证据`；官方软件/手册页记为 `模块规范 / 软件文档`；"
        "具体 accession 或 dataset page 记为 `数据入口`；"
        "可脚本化分发包记为 `资源包`。\n\n"
        "- Stable URL: {stable_url}\n\n"
        "### 3.7 二次核查补充（资源分型、稳定入口与场景边界）\n\n"
        "- checked\n"
    ).format(stable_url="https://example.org/other" if break_review_url else stable_url)
    write_file(review_path, review_body)

    write_file(
        tmp_path / "docs/README.md",
        """# Docs

## Evidence Taxonomy

- `论文证据`
- `数据入口`
- `模块规范 / 软件文档`
- `资源包`

Manifest: `review_manifest_20260312.json`
""",
    )
    write_file(
        tmp_path / "benchmark/README.md",
        """# Benchmark

## Evidence Taxonomy

- `论文证据`
- `数据入口`
- `模块规范 / 软件文档`
- `资源包`

Manifest: `review_manifest_20260312.json`
""",
    )
    write_file(
        tmp_path / "benchmark/aggregation/README.md",
        """# Aggregation

## Resource Roles

- `论文证据`
- `数据入口`
- `模块规范 / 软件文档`
- `资源包`
""",
    )

    manifest = {
        "generated_on": "2026-03-12",
        "scope_glob": "docs/review_*.md",
        "taxonomy": {
            "论文证据": "papers",
            "数据入口": "datasets",
            "模块规范 / 软件文档": "docs",
            "资源包": "packages",
            "本地实现上下文": "local alignment",
        },
        "reviews": [
            {
                "file": "docs/review_example_20260312.md",
                "focus": "example focus",
                "resource_types": ["论文证据", "模块规范 / 软件文档"],
                "stable_entrypoints": [
                    {
                        "role": "模块规范 / 软件文档",
                        "label": "Example stable page",
                        "url": stable_url,
                    },
                ],
                "notes": "example notes",
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return tmp_path, manifest_path


def test_validate_repo_success(tmp_path: Path) -> None:
    module = load_validator_module()
    repo_root, manifest_path = make_repo_fixture(tmp_path)
    issues = module.validate_repo(repo_root=repo_root, manifest_path=manifest_path)
    assert issues == []


def test_validate_repo_reports_missing_stable_entrypoint(tmp_path: Path) -> None:
    module = load_validator_module()
    repo_root, manifest_path = make_repo_fixture(tmp_path, break_review_url=True)
    issues = module.validate_repo(repo_root=repo_root, manifest_path=manifest_path)
    assert any("missing stable entrypoint URL" in issue for issue in issues)
