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
# "physics" and "math" alone are archive groupings, not searchable cat: tags —
# arXiv's cat: search silently returns 0 results for them (confirmed against
# the live API; totalResults=0 for both), so they're replaced with actual
# leaf category codes below. The other four were already valid leaf/archive
# codes and verified to return real results.
ARXIV_CATEGORIES = [
    "physics.gen-ph", "physics.class-ph",  # replaces the dead "physics" entry
    "math.CA", "math.NA",                  # replaces the dead "math" entry
    "cs.LG", "astro-ph", "cond-mat", "quant-ph",
]

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
# Wikipedia's API rejects requests with no/generic User-Agent (403 "Please
# set a user-agent and respect our robot policy") — requests' default UA
# doesn't satisfy it.
WIKIPEDIA_USER_AGENT = "sci_ai_engine-corpus-fetch/1.0 (https://github.com/sa9446/sci_ai)"
WIKIPEDIA_STEM_CATEGORIES = [
    "Category:Physics", "Category:Mathematics", "Category:Chemistry",
    "Category:Astronomy", "Category:Thermodynamics", "Category:Quantum mechanics",
]

SCI_CODE_IMPORT_PATTERN = re.compile(r"^\s*(import|from)\s+(numpy|scipy|sympy)\b", re.MULTILINE)


def _get_with_retry(session: requests.Session, url: str, params: dict, timeout: int = 30,
                     retries: int = 4, backoff: float = 5.0) -> requests.Response:
    """arXiv's API is known to intermittently 500, especially at deep
    pagination offsets (confirmed live: start=10000 on a valid category threw
    a 500). A single transient failure shouldn't kill a 30+ minute fetch run."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            last_exc = exc
            wait = backoff * (2 ** attempt)
            print(f"    request failed ({exc}) — retry {attempt + 1}/{retries} in {wait:.0f}s")
            time.sleep(wait)
    raise last_exc  # noqa: RSE102 — re-raising the last captured exception is intentional


def fetch_arxiv_abstracts(out_path: Path, max_results_per_category: int = 2000) -> None:
    session = requests.Session()
    lines: list[str] = []

    def _flush() -> None:
        # Written after every category (not just once at the end) so a late
        # failure — even after retries are exhausted — doesn't discard
        # everything fetched before it.
        out_path.write_text("\n\n".join(lines), encoding="utf-8")

    for category in ARXIV_CATEGORIES:
        start = 0
        page_size = 200
        fetched = 0
        try:
            while fetched < max_results_per_category:
                params = {
                    "search_query": f"cat:{category}",
                    "start": start,
                    "max_results": min(page_size, max_results_per_category - fetched),
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                }
                resp = _get_with_retry(session, ARXIV_API, params)
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
        except Exception as exc:  # noqa: BLE001 — keep whatever other categories can still complete
            print(f"  arXiv[{category}]: FAILED after {fetched} abstracts ({exc}) — moving on")
            _flush()
            continue
        print(f"  arXiv[{category}]: {fetched} abstracts")
        _flush()

    print(f"Wrote {len(lines)} arXiv abstracts to {out_path}")


def fetch_wikipedia_stem(out_path: Path, max_pages_per_category: int = 300) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": WIKIPEDIA_USER_AGENT})
    texts: list[str] = []
    seen_titles: set[str] = set()

    def _flush() -> None:
        # Written after every category (not just once at the end) so a late
        # failure doesn't discard everything fetched before it.
        out_path.write_text("\n\n".join(texts), encoding="utf-8")

    for category in WIKIPEDIA_STEM_CATEGORIES:
        cmcontinue = None
        collected = 0
        try:
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
                resp = _get_with_retry(session, WIKIPEDIA_API, params)
                data = resp.json()
                members = data.get("query", {}).get("categorymembers", [])
                if not members:
                    break

                titles = [m["title"] for m in members if m["title"] not in seen_titles]
                seen_titles.update(titles)
                # The extracts API only allows batching multiple pages per call
                # (exlimit > 1) when exintro=1 (intro section only) — requesting
                # full article text is hard-capped at 1 page per call regardless
                # of exlimit ("exlimit was too large for a whole article extracts
                # request, lowered to 1", confirmed against the live API). Intro
                # text is shorter per page, but that's what makes batching (and
                # therefore a corpus of thousands of pages) practical at all.
                for i in range(0, len(titles), 20):
                    title_chunk = titles[i:i + 20]
                    extract_params = {
                        "action": "query",
                        "prop": "extracts",
                        "explaintext": 1,
                        "exintro": 1,
                        "exlimit": "max",
                        "titles": "|".join(title_chunk),
                        "format": "json",
                    }
                    eresp = _get_with_retry(session, WIKIPEDIA_API, extract_params)
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
        except Exception as exc:  # noqa: BLE001 — keep whatever other categories can still complete
            print(f"  Wikipedia[{category}]: FAILED after {collected} pages ({exc}) — moving on")
            _flush()
            continue
        print(f"  Wikipedia[{category}]: {collected} pages")
        _flush()

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
