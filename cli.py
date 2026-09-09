"""
cli.py — PlagiarismIQ Command-Line Interface
=============================================
Usage
-----
  python cli.py path/to/docs/ [OPTIONS]
  python cli.py file1.txt file2.pdf file3.docx [OPTIONS]

Options
-------
  --threshold FLOAT      Suspicion threshold (default 0.75)
  --output-dir DIR       Where to save reports (default: ./reports)
  --format {json,csv,html,all}  Output format(s) (default: all)
  --model MODEL          Embedding model name (default: all-MiniLM-L6-v2)
  --weights S T J        Semantic, TF-IDF, Jaccard weights (must sum to 1.0)
  --no-cache             Disable embedding cache
  --quiet                Suppress progress output
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from plagiarism_engine import (
    EmbeddingEngine,
    ReportWriter,
    SimilarityAnalyzer,
    extract_all,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
CYAN  = "\033[96m"
DIM   = "\033[2m"


def _print(*args, quiet: bool = False, **kwargs):
    if not quiet:
        print(*args, **kwargs)


def _gather_files(inputs: list[str]) -> list[Path]:
    """Resolve inputs (files or directories) to a deduplicated list of paths."""
    paths: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for ext in ("*.txt", "*.pdf", "*.docx"):
                paths.extend(p.glob(ext))
        elif p.exists():
            paths.append(p)
        else:
            # Try glob
            matched = [Path(m) for m in glob.glob(inp)]
            paths.extend(matched)
    # Deduplicate while preserving order
    seen: set[Path] = set()
    result: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="plagiarism-checker",
        description="PlagiarismIQ — AI-powered ensemble plagiarism detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="PATH",
        help="Files or directories to analyze.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        metavar="FLOAT",
        help="Ensemble score threshold for flagging suspicious pairs (default: 0.75).",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        metavar="DIR",
        help="Output directory for reports (default: reports/).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "html", "all"],
        default="all",
        help="Report format (default: all).",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        metavar="MODEL",
        help="HuggingFace sentence-transformer model (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--weights",
        nargs=3,
        type=float,
        metavar=("SEMANTIC", "TFIDF", "JACCARD"),
        default=[0.55, 0.30, 0.15],
        help="Algorithm weights, must sum to 1.0 (default: 0.55 0.30 0.15).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args(argv)

    quiet = args.quiet

    _print(f"\n{BOLD}{CYAN}🔍 PlagiarismIQ v2.0{RESET}", quiet=quiet)
    _print(f"{DIM}Multi-algorithm ensemble plagiarism detection{RESET}\n", quiet=quiet)

    # Gather files
    paths = _gather_files(args.inputs)
    if len(paths) < 2:
        print(f"{RED}Error: at least 2 readable documents required.{RESET}")
        return 1

    _print(f"📂 Found {BOLD}{len(paths)}{RESET} documents:", quiet=quiet)
    for p in paths:
        _print(f"  · {p.name}", quiet=quiet)
    _print("", quiet=quiet)

    # Extract
    _print("📄 Extracting text…", quiet=quiet)
    t0 = time.perf_counter()
    texts_dict = extract_all(paths)
    names = list(texts_dict.keys())
    texts = list(texts_dict.values())
    _print(f"   {GREEN}✓{RESET} {len(names)} documents extracted ({time.perf_counter()-t0:.2f}s)\n", quiet=quiet)

    # Embed
    _print(f"🤖 Embedding with '{args.model}'…", quiet=quiet)
    t1 = time.perf_counter()
    cache_dir = None if args.no_cache else Path(".plagiarism_cache")
    engine = EmbeddingEngine(model_name=args.model, cache_dir=cache_dir)
    embeddings = engine.encode(texts)
    _print(f"   {GREEN}✓{RESET} Embeddings ready ({time.perf_counter()-t1:.2f}s)\n", quiet=quiet)

    # Analyze
    _print("📐 Running similarity analysis…", quiet=quiet)
    t2 = time.perf_counter()
    weights = tuple(args.weights)
    total_w = sum(weights)
    if abs(total_w - 1.0) > 0.01:
        weights = tuple(w / total_w for w in weights)
        _print(f"   {YELLOW}⚠{RESET}  Weights normalised to sum to 1.0", quiet=quiet)

    analyzer = SimilarityAnalyzer(weights=weights, threshold=args.threshold)
    results = analyzer.analyze(names, texts, embeddings)
    mat = analyzer.similarity_matrix(names, results)
    suspicious = analyzer.flag_suspicious(results)
    _print(f"   {GREEN}✓{RESET} Analysis complete ({time.perf_counter()-t2:.2f}s)\n", quiet=quiet)

    # Print summary
    _print(f"{BOLD}Summary{RESET}", quiet=quiet)
    _print(f"  Documents:        {len(names)}", quiet=quiet)
    _print(f"  Pairs compared:   {len(results)}", quiet=quiet)
    _print(f"  Suspicious pairs: {RED if suspicious else GREEN}{len(suspicious)}{RESET}", quiet=quiet)
    max_score = max((r.ensemble_score for r in results), default=0.0)
    _print(f"  Max similarity:   {max_score:.3f}", quiet=quiet)
    _print("", quiet=quiet)

    if suspicious:
        _print(f"{BOLD}{RED}⚠️  Suspicious Pairs:{RESET}", quiet=quiet)
        for r in suspicious:
            _print(
                f"  {r.doc_a}  ↔  {r.doc_b}   "
                f"{BOLD}{r.ensemble_score:.3f}{RESET}  {r.risk_level}",
                quiet=quiet,
            )
        _print("", quiet=quiet)

    # Reports
    out_dir = Path(args.output_dir)
    writer = ReportWriter()
    fmt = args.format

    _print(f"💾 Saving reports to '{out_dir}/'…", quiet=quiet)
    if fmt in ("json", "all"):
        p = writer.save(writer.to_json(results, names, texts), out_dir / "plagiarism_report.json")
        _print(f"   {GREEN}✓{RESET} {p}", quiet=quiet)
    if fmt in ("csv", "all"):
        p = writer.save(writer.to_csv(results), out_dir / "plagiarism_report.csv")
        _print(f"   {GREEN}✓{RESET} {p}", quiet=quiet)
    if fmt in ("html", "all"):
        p = writer.save(writer.to_html(results, names, texts, mat), out_dir / "plagiarism_report.html")
        _print(f"   {GREEN}✓{RESET} {p}", quiet=quiet)

    _print(f"\n{GREEN}{BOLD}✅ Done!{RESET}\n", quiet=quiet)
    return 0 if not suspicious else 2


if __name__ == "__main__":
    sys.exit(main())
