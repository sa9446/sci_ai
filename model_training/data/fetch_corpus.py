"""Builds the Phase A pretraining corpus as plain .txt files under data/raw/.

Four sources, each fully functional without any API key:

  1. arXiv abstracts   — export.arxiv.org public Atom API (physics/math/cs)
  2. Wikipedia STEM     — public MediaWiki API, category-member crawl + extracts
  3. OpenStax textbooks — NOT auto-scraped (site structure isn't stable enough
                          to hardcode reliably); drop CC-licensed book text
                          files yourself under data/raw/openstax/*.txt and
                          this script will just leave them in place.
  4. Scientific code    — scans a local directory of already-cloned,
                          permissively-licensed repos and keeps only files
                          that import numpy/scipy/sympy, so you control
                          licensing by choosing which repos to clone.

Run data/prepare.py afterwards to tokenize + pack everything in data/raw/
into train.bin/val.bin.
"""
from __future__ import annotations

import argparse
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_CATEGORIES = ["physics", "math", "cs.LG", "astro-ph", "cond-mat", "quant-ph"]

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_STEM_CATEGORIES = [
    "Category:Physics", "Category:Mathematics", "Category:Chemistry",
    "Category:Astronomy", "Category:Thermodynamics", "Category:Quantum mechanics",
]

SCI_CODE_IMPORT_PATTERN = re.compile(r"^\s*(import|from)\s+(numpy|scipy|sympy)\b", re.MULTILINE)


def fetch_arxiv_abstracts(out_path: Path, max_results_per_category: int = 2000) -> None:
    lines: list[str] = []
    for category in ARXIV_CATEGORIES:
        start = 0
        page_size = 200
        fetched = 0
        while fetched < max_results_per_category:
            params = {
                "search_query": f"cat:{category}",
                "start": start,
                "max_results": min(page_size, max_results_per_category - fetched),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            resp = requests.get(ARXIV_API, params=params, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            if not entries:
                break
            for entry in entries:
                summary = entry.find("atom:summary", ns)
                if summary is not None and summary.text:
                    lines.append(summary.text.strip().replace("\n", " "))
            fetched += len(entries)
            start += len(entries)
            time.sleep(3)  # arXiv API rate-limit courtesy delay
        print(f"  arXiv[{category}]: {fetched} abstracts")

    out_path.write_text("\n\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(lines)} arXiv abstracts to {out_path}")


def fetch_wikipedia_stem(out_path: Path, max_pages_per_category: int = 300) -> None:
    session = requests.Session()
    texts: list[str] = []
    seen_titles: set[str] = set()

    for category in WIKIPEDIA_STEM_CATEGORIES:
        cmcontinue = None
        collected = 0
        while collected < max_pages_per_category:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category,
                "cmlimit": min(100, max_pages_per_category - collected),
                "cmtype": "page",
                "format": "json",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            resp = session.get(WIKIPEDIA_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            members = data.get("query", {}).get("categorymembers", [])
            if not members:
                break

            titles = [m["title"] for m in members if m["title"] not in seen_titles]
            seen_titles.update(titles)
            if titles:
                extract_params = {
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "titles": "|".join(titles),
                    "format": "json",
                }
                eresp = session.get(WIKIPEDIA_API, params=extract_params, timeout=30)
                eresp.raise_for_status()
                pages = eresp.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    extract = page.get("extract", "")
                    if extract:
                        texts.append(extract)

            collected += len(members)
            cmcontinue = data.get("continue", {}).get("cmcontinue")
            if not cmcontinue:
                break
            time.sleep(1)
        print(f"  Wikipedia[{category}]: {collected} pages")

    out_path.write_text("\n\n".join(texts), encoding="utf-8")
    print(f"Wrote {len(texts)} Wikipedia STEM articles to {out_path}")


def scan_scientific_code(repos_dir: Path, out_path: Path) -> None:
    """Keeps .py files (from repos YOU clone into repos_dir) that import
    numpy/scipy/sympy — you are responsible for only cloning permissively
    licensed repos here (MIT/BSD/Apache)."""
    if not repos_dir.exists():
        print(f"  {repos_dir} does not exist yet — skipping code corpus. "
              f"Clone some MIT/BSD/Apache-licensed scientific-Python repos there first.")
        return

    chunks: list[str] = []
    for py_file in repos_dir.rglob("*.py"):
        try:
            text = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if SCI_CODE_IMPORT_PATTERN.search(text):
            chunks.append(text)

    out_path.write_text("\n\n# ---\n\n".join(chunks), encoding="utf-8")
    print(f"Wrote {len(chunks)} scientific-code files to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch the Phase A pretraining corpus into data/raw/")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "raw")
    p.add_argument("--skip-arxiv", action="store_true")
    p.add_argument("--skip-wikipedia", action="store_true")
    p.add_argument("--code-repos-dir", type=Path, default=Path(__file__).parent / "raw" / "code_repos",
                    help="Directory where you've manually cloned permissively-licensed sci-Python repos")
    p.add_argument("--arxiv-max-per-category", type=int, default=2000,
                    help="Abstracts fetched per ARXIV_CATEGORIES entry. The default (2000, ~12K abstracts "
                         "total, roughly a few million tokens) is too small to pretrain a 110M-param model "
                         "without overfitting within a few hundred iterations — go much larger (10000+) for "
                         "a real Phase A run.")
    p.add_argument("--wikipedia-max-pages-per-category", type=int, default=300,
                    help="Pages fetched per WIKIPEDIA_STEM_CATEGORIES entry. Same undersized-default caveat "
                         "as --arxiv-max-per-category.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_arxiv:
        print("Fetching arXiv abstracts ...")
        fetch_arxiv_abstracts(args.out_dir / "arxiv_abstracts.txt", max_results_per_category=args.arxiv_max_per_category)

    if not args.skip_wikipedia:
        print("Fetching Wikipedia STEM articles ...")
        fetch_wikipedia_stem(args.out_dir / "wikipedia_stem.txt", max_pages_per_category=args.wikipedia_max_pages_per_category)

    print("Scanning local scientific code repos ...")
    scan_scientific_code(args.code_repos_dir, args.out_dir / "scientific_code.txt")

    openstax_dir = args.out_dir / "openstax"
    if openstax_dir.exists() and any(openstax_dir.glob("*.txt")):
        print(f"Found {len(list(openstax_dir.glob('*.txt')))} OpenStax text files already in place — good.")
    else:
        print(f"NOTE: no OpenStax text found under {openstax_dir}. "
              f"Download CC-licensed textbook text manually and drop .txt files there.")


if __name__ == "__main__":
    main()
