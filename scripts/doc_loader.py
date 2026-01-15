#!/usr/bin/env python3
"""ScpTensor Design Document Loader.

Progressive loading system for design documentation. Load specific sections
on-demand instead of entire documents to reduce context overhead.

Usage:
    python3 doc_loader.py <DOC_NAME> <ACTION> [args]

Examples:
    python3 doc_loader.py MASTER 1-50
    python3 doc_loader.py ARCHITECTURE search "normalization"
    python3 doc_loader.py ROADMAP outline
    python3 doc_loader.py MASTER section executive --output saved.md
"""

from __future__ import annotations

import re
import sys
from enum import Enum
from pathlib import Path
from typing import Final, ClassVar

# Document registry with relative paths
DOCS: Final = {
    "MASTER": "docs/design/MASTER.md",
    "ARCHITECTURE": "docs/design/ARCHITECTURE.md",
    "ROADMAP": "docs/design/ROADMAP.md",
    "MIGRATION": "docs/design/MIGRATION.md",
    "API_REFERENCE": "docs/design/API_REFERENCE.md",
    "INDEX": "docs/design/INDEX.md",
}

# Predefined section mappings for quick access
SECTION_MAP: Final = {
    "MASTER": {
        "executive": (1, 50),
        "architecture": (51, 150),
        "priority": (151, 250),
        "ecosystem": (251, 350),
        "maintenance": (351, 400),
        "metrics": (401, 450),
    },
    "ARCHITECTURE": {
        "modules": (1, 100),
        "structures": (101, 200),
        "patterns": (201, 300),
        "apis": (301, 450),
        "integration": (451, 550),
        "errors": (551, 650),
        "performance": (651, 750),
        "extensions": (751, 850),
        "testing": (851, 950),
    },
    "ROADMAP": {
        "summary": (1, 50),
        "priorities": (51, 150),
        "milestones": (151, 250),
        "dependencies": (251, 350),
        "sprints": (351, 500),
        "risks": (501, 600),
    },
    "MIGRATION": {
        "quickstart": (1, 50),
        "breaking": (51, 150),
        "compatibility": (151, 250),
        "strategies": (251, 350),
        "steps": (351, 500),
        "faq": (501, 600),
    },
    "API_REFERENCE": {
        "toc": (1, 80),
        "core": (81, 200),
        "normalization": (201, 350),
        "impute": (351, 500),
        "integration": (501, 600),
        "qc": (601, 700),
        "dim_reduction": (701, 800),
    },
}


class AnsiColor(Enum):
    """ANSI color codes for terminal output."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def __str__(self) -> str:
        return self.value


def print_color(text: str, color: AnsiColor | str = AnsiColor.RESET) -> None:
    """Print colored text to terminal.

    Args:
        text: Text to print
        color: Color name or AnsiColor enum
    """
    if isinstance(color, str):
        color = AnsiColor[color.upper()]

    print(f"{color}{text}{AnsiColor.RESET}")


def resolve_doc_path(doc_name: str) -> Path:
    """Resolve document name to absolute file path.

    Args:
        doc_name: Document identifier (e.g., 'MASTER', 'ARCHITECTURE')

    Returns:
        Absolute path to the document

    Raises:
        SystemExit: If document not found
    """
    key = doc_name.upper()
    if key not in DOCS:
        print_color(f"Unknown document: '{doc_name}'", AnsiColor.RED)
        print_color(f"Available: {', '.join(DOCS)}", AnsiColor.CYAN)
        sys.exit(1)

    script_dir = Path(__file__).parent.resolve()
    doc_path = (script_dir.parent / DOCS[key]).resolve()

    if not doc_path.exists():
        print_color(f"Document not found: {doc_path}", AnsiColor.RED)
        sys.exit(1)

    return doc_path


def read_document(doc_path: Path) -> list[str]:
    """Read document into memory efficiently.

    Args:
        doc_path: Path to document

    Returns:
        List of lines

    Raises:
        SystemExit: If file cannot be read
    """
    try:
        return doc_path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as e:
        print_color(f"Error reading {doc_path}: {e}", AnsiColor.RED)
        sys.exit(1)


def load_lines(doc_path: Path, start: int, end: int) -> str:
    """Load specific line range from document.

    Args:
        doc_path: Path to document
        start: Start line number (1-indexed)
        end: End line number (1-indexed, inclusive)

    Returns:
        Content of specified line range

    Raises:
        SystemExit: If line range is invalid
    """
    lines = read_document(doc_path)
    total_lines = len(lines)

    if not (1 <= start <= end <= total_lines):
        print_color(
            f"Invalid range: {start}-{end} (document has {total_lines} lines)",
            AnsiColor.RED,
        )
        sys.exit(1)

    return "".join(lines[start - 1 : end])


def load_all(doc_path: Path) -> str:
    """Load entire document.

    Args:
        doc_path: Path to document

    Returns:
        Full document content
    """
    return doc_path.read_text(encoding="utf-8")


def search_in_document(
    doc_path: Path,
    pattern: str,
    *,
    is_regex: bool = False,
    max_results: int = 10,
) -> str:
    """Search document for keyword or regex pattern.

    Args:
        doc_path: Path to document
        pattern: Search pattern (keyword or regex)
        is_regex: If True, treat pattern as regex
        max_results: Maximum matches to return

    Returns:
        Formatted search results with context
    """
    lines = read_document(doc_path)
    matches: list[str] = []

    flags = re.IGNORECASE if not is_regex else 0
    try:
        regex = re.compile(pattern, flags) if not is_regex else re.compile(pattern)
    except re.error as e:
        print_color(f"Invalid regex: {e}", AnsiColor.RED)
        sys.exit(1)

    for i, line in enumerate(lines, 1):
        if regex.search(line):
            ctx_start = max(0, i - 4)
            ctx_end = min(len(lines), i + 3)
            context = "".join(
                f"{j + 1:4d} | {ln}" for j, ln in enumerate(lines[ctx_start:ctx_end], ctx_start + 1)
            )
            matches.append(f"Line {i}:\n{context}\n")

            if len(matches) >= max_results:
                break

    if not matches:
        return f"No matches for: '{pattern}'"

    total_suffix = f"\n... (first {max_results} of {len(matches)})" if len(matches) >= max_results else ""
    return f"Found {len(matches)} matches for '{pattern}':\n\n" + "\n".join(matches) + total_suffix


def get_outline(doc_path: Path) -> str:
    """Extract document structure (headings with line numbers).

    Args:
        doc_path: Path to document

    Returns:
        Formatted outline
    """
    lines = read_document(doc_path)
    outline: list[str] = []

    for i, line in enumerate(lines, 1):
        if line.startswith("#"):
            level = len(re.match(r"^#+", line).group())  # type: ignore[arg-type]
            heading = line.lstrip("#").strip()
            indent = "  " * (level - 1)
            outline.append(f"{i:4d} | {indent}{'#' * level} {heading}")

    if not outline:
        return "No headings found"

    return f"Outline: {doc_path.name}\n\n" + "\n".join(outline)


def count_lines(doc_path: Path) -> str:
    """Count lines and report file size.

    Args:
        doc_path: Path to document

    Returns:
        Formatted statistics
    """
    lines = read_document(doc_path)
    size_kb = doc_path.stat().st_size / 1024

    return f"Document: {doc_path.name}\nLines: {len(lines)}\nSize: {size_kb:.1f} KB"


def parse_range(range_str: str) -> tuple[int, int] | None:
    """Parse line range string.

    Args:
        range_str: Range like '1-50' or 'all'

    Returns:
        Tuple of (start, end) or None for 'all'

    Raises:
        SystemExit: If format is invalid
    """
    range_str = range_str.strip()
    if range_str.lower() == "all":
        return None

    if not (match := re.match(r"^(\d+)-(\d+)$", range_str)):
        print_color(f"Invalid range: '{range_str}'. Use 'START-END' or 'all'", AnsiColor.RED)
        sys.exit(1)

    start, end = int(match.group(1)), int(match.group(2))

    if start > end:
        print_color(f"Start ({start}) > end ({end})", AnsiColor.RED)
        sys.exit(1)

    if start < 1:
        print_color(f"Line numbers start at 1, got {start}", AnsiColor.RED)
        sys.exit(1)

    return start, end


def load_section(doc_name: str, section_name: str) -> str:
    """Load predefined section by name.

    Args:
        doc_name: Document identifier
        section_name: Section name (e.g., 'executive', 'modules')

    Returns:
        Section content

    Raises:
        SystemExit: If section not found
    """
    doc_upper = doc_name.upper()

    if doc_upper not in SECTION_MAP:
        print_color(f"No sections defined for '{doc_name}'", AnsiColor.RED)
        sys.exit(1)

    sections = SECTION_MAP[doc_upper]
    if section_name not in sections:
        print_color(
            f"Unknown section '{section_name}'. Available: {', '.join(sections)}",
            AnsiColor.YELLOW,
        )
        sys.exit(1)

    start, end = sections[section_name]
    return load_lines(resolve_doc_path(doc_name), start, end)


def save_content(content: str, output_path: Path) -> None:
    """Save content to file.

    Args:
        content: Text to save
        output_path: Destination path
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        print_color(f"Saved: {output_path}", AnsiColor.GREEN)
    except OSError as e:
        print_color(f"Save failed: {e}", AnsiColor.RED)
        sys.exit(1)


def print_usage() -> None:
    """Display comprehensive usage information."""
    print_color("╔══════════════════════════════════════════════════════════════════╗", AnsiColor.BOLD)
    print_color("║         ScpTensor Design Document Loader - User Guide            ║", AnsiColor.BOLD)
    print_color("╚══════════════════════════════════════════════════════════════════╝", AnsiColor.BOLD)
    print()
    print(__doc__)
    print("=" * 70)
    print_color("DOCUMENTS:", AnsiColor.CYAN)
    for name, path in sorted(DOCS.items()):
        print(f"  {name:15s} {path}")
    print()
    print("=" * 70)
    print_color("ACTIONS:", AnsiColor.CYAN)
    print("  <START>-<END>    Load line range (e.g., '1-50', '100-150')")
    print("  all              Load entire document")
    print("  section <name>   Load predefined section")
    print("  search <kw>      Keyword search")
    print("  regex <pat>      Regex search")
    print("  outline          Document structure")
    print("  count            Line count and size")
    print("  help             Show this message")
    print()
    print("=" * 70)
    print_color("OPTIONS:", AnsiColor.CYAN)
    print("  --output <file>  Save to file")
    print()
    print("=" * 70)
    print_color("EXAMPLES:", AnsiColor.CYAN)
    print("  python3 doc_loader.py MASTER 1-50")
    print("  python3 doc_loader.py ARCHITECTURE search 'normalization'")
    print("  python3 doc_loader.py MASTER section executive")
    print("  python3 doc_loader.py ROADMAP outline")
    print("  python3 doc_loader.py MASTER 1-50 --output saved.md")
    print("  python3 doc_loader.py API_REFERENCE regex 'def.*normalize'")
    print("=" * 70)


def parse_args(args: list[str]) -> tuple[str, str, Path | None]:
    """Parse command line arguments.

    Args:
        args: Command line arguments (sys.argv[1:])

    Returns:
        Tuple of (doc_name, action, output_path)

    Raises:
        SystemExit: If arguments are invalid
    """
    if not args or args[0] in {"help", "--help", "-h"}:
        print_usage()
        sys.exit(0)

    if len(args) < 2:
        print_color("Too few arguments", AnsiColor.RED)
        print("Usage: doc_loader.py <DOC> <ACTION> [options]")
        sys.exit(1)

    doc_name, action = args[0], args[1]
    output_path = None

    if "--output" in args:
        idx = args.index("--output")
        if idx + 1 >= len(args):
            print_color("--output requires a path", AnsiColor.RED)
            sys.exit(1)
        output_path = Path(args[idx + 1])

    return doc_name, action, output_path


def main() -> None:
    """Main entry point."""
    doc_name, action, output_path = parse_args(sys.argv[1:])
    doc_path = resolve_doc_path(doc_name)

    # Dispatch action
    match action.lower():
        case "all":
            result = load_all(doc_path)
            metadata = f"[Loaded: {doc_path.name}]"

        case "section":
            if len(sys.argv) < 4:
                print_color("section requires a name", AnsiColor.RED)
                sys.exit(1)
            section_name = sys.argv[3]
            result = load_section(doc_name, section_name)
            metadata = f"[Section: {section_name} from {doc_path.name}]"

        case "search":
            if len(sys.argv) < 4:
                print_color("search requires a keyword", AnsiColor.RED)
                sys.exit(1)
            result = search_in_document(doc_path, sys.argv[3])
            metadata = f"[Searched: {doc_path.name}]"

        case "regex":
            if len(sys.argv) < 4:
                print_color("regex requires a pattern", AnsiColor.RED)
                sys.exit(1)
            result = search_in_document(doc_path, sys.argv[3], is_regex=True)
            metadata = f"[Regex: {doc_path.name}]"

        case "outline":
            result = get_outline(doc_path)
            metadata = f"[Outline: {doc_path.name}]"

        case "count":
            result = count_lines(doc_path)
            metadata = ""

        case _:
            # Try as line range
            range_result = parse_range(action)
            if range_result is None:
                result = load_all(doc_path)
                metadata = f"[Loaded: {doc_path.name}]"
            else:
                start, end = range_result
                result = load_lines(doc_path, start, end)
                metadata = f"[Lines {start}-{end} from {doc_path.name}]"

    # Output
    print(result)
    if metadata:
        print()
        print_color("---", AnsiColor.CYAN)
        print_color(metadata, AnsiColor.GREEN)

    if output_path:
        save_content(result, output_path)
        print_color(f"[Content: {len(result)} chars]", AnsiColor.BLUE)


if __name__ == "__main__":
    main()
