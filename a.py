#!/usr/bin/env python3
# Python 3.9+ (works on macOS)
import os
import re
import time
import json
import argparse
import warnings
from typing import Optional, List, Set, Dict
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# ---- Silence urllib3 LibreSSL warning on some macOS builds ----
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass


BASE = "https://www.rev.com"
TRANSCRIPT_PREFIX = "/transcripts/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; rev-transcript-scraper/1.0)"
}


def slugify(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        path = "index"
    safe = path.replace("/", "__")
    safe = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", safe)
    return safe


def fetch(session: requests.Session, url: str) -> requests.Response:
    return session.get(url, headers=HEADERS, timeout=30)


def save_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def append_line(path: str, line: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def find_page_param(soup: BeautifulSoup) -> Optional[str]:
    """
    Rev category pages often paginate with a param like:
        ?05aad4b1_page=2
    We auto-detect the '<hash>_page' part from links on the page.
    """
    for a in soup.select("a[href*='_page=']"):
        href = a.get("href", "") or ""
        m = re.search(r"([a-z0-9]{8}_page)=\d+", href)
        if m:
            return m.group(1)
    return None


def extract_transcript_links(soup: BeautifulSoup) -> Set[str]:
    out: Set[str] = set()
    for a in soup.select(f"a[href^='{TRANSCRIPT_PREFIX}']"):
        href = a.get("href")
        if href:
            out.add(urljoin(BASE, href.split("#")[0]))
    return out


def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def looks_like_speaker_timestamp(line: str) -> bool:
    """
    Rev transcripts often format speaker lines like:
        Donald Trump (00:02):
        Speaker 1 (12:34):
    """
    return bool(re.match(r"^.{2,80}\(\s*\d{1,2}:\d{2}(:\d{2})?\s*\)\s*:\s*$", line))


def extract_transcript_text(html: str) -> Optional[str]:
    """
    Extract transcript content if the transcript is present in the HTML.
    This does not bypass paywalls; it only parses what is served.
    """
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("article") or soup.find("main") or soup.body
    if not container:
        return None

    blocks = container.find_all(["p", "li", "h2", "h3", "blockquote"])
    lines: List[str] = []

    if len(blocks) >= 10:
        for b in blocks:
            t = b.get_text(" ", strip=True)
            if t:
                lines.append(t)
    else:
        text = container.get_text("\n", strip=True)
        text = normalize_whitespace(text)
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    start = None
    for i, ln in enumerate(lines):
        if looks_like_speaker_timestamp(ln):
            start = i
            break

    if start is None:
        return None

    transcript = "\n\n".join(lines[start:])
    transcript = normalize_whitespace(transcript)

    # sanity check: should be decently long
    if len(transcript) < 1200:
        return None

    return transcript


def collect_all_links_incremental(
    start_url: str,
    out_links_path: str,
    delay: float = 1.0,
    patience: int = 3,
    max_pages: int = 2000,
) -> List[str]:
    """
    Crawl start_url and keep paginating via the discovered '*_page' query param.
    Writes links.txt incrementally so you see progress immediately.
    Stops when:
      - 404 on a page
      - no new links for 'patience' consecutive pages
      - hits max_pages
    """
    all_links: Set[str] = set()
    ordered: List[str] = []

    # Start fresh each run
    save_text(out_links_path, "")

    with requests.Session() as session:
        # Page 1
        r0 = fetch(session, start_url)
        if r0.status_code != 200:
            raise RuntimeError(f"Failed to load start page: HTTP {r0.status_code}")

        soup0 = BeautifulSoup(r0.text, "html.parser")
        param = find_page_param(soup0)

        links0 = extract_transcript_links(soup0)
        added0 = 0
        for u in sorted(links0):
            if u not in all_links:
                all_links.add(u)
                ordered.append(u)
                append_line(out_links_path, u)
                added0 += 1

        print(f"[page 1] added {added0} | total {len(ordered)}", flush=True)

        if not param:
            print("No pagination param found (_page). Stopping after page 1.", flush=True)
            return ordered

        no_new = 0

        for page in range(2, max_pages + 1):
            page_url = f"{start_url}?{param}={page}"
            r = fetch(session, page_url)

            if r.status_code == 404:
                print(f"[page {page}] 404 -> pagination ended", flush=True)
                break

            if r.status_code != 200:
                no_new += 1
                print(f"[page {page}] HTTP {r.status_code} (no_new={no_new})", flush=True)
                if no_new >= patience:
                    print("Stopping due to repeated non-200 responses.", flush=True)
                    break
                time.sleep(delay)
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            links = extract_transcript_links(soup)

            added = 0
            for u in sorted(links):
                if u not in all_links:
                    all_links.add(u)
                    ordered.append(u)
                    append_line(out_links_path, u)
                    added += 1

            if added == 0:
                no_new += 1
            else:
                no_new = 0

            print(f"[page {page}] added {added} | total {len(ordered)} | no_new={no_new}", flush=True)

            if no_new >= patience:
                print("No new links for several pages -> stopping.", flush=True)
                break

            time.sleep(delay)

    return ordered


def download_and_extract(
    urls: List[str],
    out_dir: str,
    delay: float = 1.0
) -> None:
    raw_dir = os.path.join(out_dir, "raw_html")
    txt_dir = os.path.join(out_dir, "extracted_txt")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    manifest: List[Dict] = []

    with requests.Session() as session:
        for i, url in enumerate(urls, 1):
            slug = slugify(url)
            html_path = os.path.join(raw_dir, f"{slug}.html")
            txt_path = os.path.join(txt_dir, f"{slug}.txt")

            try:
                r = fetch(session, url)
                status = r.status_code
                save_text(html_path, r.text if r.text else "")

                entry = {
                    "i": i,
                    "url": url,
                    "http_status": status,
                    "saved_html": html_path,
                    "saved_txt": None,
                    "transcript_found": False,
                    "error": None
                }

                if status == 200 and r.text:
                    transcript = extract_transcript_text(r.text)
                    if transcript:
                        save_text(txt_path, transcript)
                        entry["saved_txt"] = txt_path
                        entry["transcript_found"] = True
                    else:
                        entry["error"] = "No transcript text found in HTML (or heuristics failed)"
                else:
                    entry["error"] = f"HTTP {status}"

                manifest.append(entry)

                if i % 10 == 0:
                    ok = sum(1 for x in manifest if x.get("transcript_found"))
                    print(f"[download] {i}/{len(urls)} done | extracted {ok}", flush=True)

            except Exception as e:
                manifest.append({
                    "i": i,
                    "url": url,
                    "http_status": -1,
                    "saved_html": html_path,
                    "saved_txt": None,
                    "transcript_found": False,
                    "error": str(e)
                })

            time.sleep(delay)

    save_text(os.path.join(out_dir, "manifest.json"), json.dumps(manifest, indent=2))
    ok = sum(1 for x in manifest if x.get("transcript_found"))
    print(f"Download complete. Extracted transcripts: {ok}/{len(urls)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory (use an absolute path if unsure)")
    ap.add_argument("--start", required=True, help="Start page, e.g. https://www.rev.com/category/donald-trump")
    ap.add_argument("--delay", type=float, default=1.0, help="Delay between requests")
    ap.add_argument("--max-pages", type=int, default=2000, help="Safety cap for pagination pages")
    ap.add_argument("--patience", type=int, default=3, help="Stop after this many pages with no new links")
    ap.add_argument("--download", action="store_true", help="Download + extract after collecting links")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    links_path = os.path.join(out_dir, "links.txt")

    print(f"Writing output to: {out_dir}", flush=True)
    print(f"Collecting transcript links from: {args.start}", flush=True)

    links = collect_all_links_incremental(
        start_url=args.start,
        out_links_path=links_path,
        delay=args.delay,
        patience=args.patience,
        max_pages=args.max_pages
    )

    print(f"Finished link collection. Total links: {len(links)}", flush=True)
    print(f"Links saved to: {links_path}", flush=True)

    if args.download and links:
        print("Starting download + extraction...", flush=True)
        download_and_extract(links, out_dir=out_dir, delay=args.delay)


if __name__ == "__main__":
    main()

