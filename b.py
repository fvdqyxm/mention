#!/usr/bin/env python3
import os, re, time, base64, argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# ---- optional ssl warning silence
try:
    import warnings
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

# -----------------------
# Kalshi auth
# -----------------------
def load_private_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def sign_request(private_key, timestamp_ms: str, method: str, path: str) -> str:
    path_wo_query = path.split("?")[0]
    msg = f"{timestamp_ms}{method.upper()}{path_wo_query}".encode("utf-8")
    sig = private_key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")

class KalshiClient:
    def __init__(self, base_url: str, api_key_id: str, private_key_path: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.api_key_id = api_key_id
        self.private_key = load_private_key(private_key_path)
        self.timeout = timeout
        self.session = requests.Session()

    def _headers(self, method: str, path: str) -> Dict[str, str]:
        ts = str(int(time.time() * 1000))
        sig = sign_request(self.private_key, ts, method, path)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig,
        }

    def get(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = self.base_url + path
        r = self.session.get(url, headers=self._headers("GET", path), params=params, timeout=self.timeout)
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code} {r.reason} for {r.url}\n{(r.text or '')[:800]}", response=r)
        return r.json()

# -----------------------
# Transcript parsing
# -----------------------
DATE_RE = re.compile(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})")
TOKEN_RE = re.compile(r"[a-z']+")

def parse_date_from_filename(fn: str) -> Optional[datetime]:
    m = DATE_RE.search(fn)
    if not m:
        return None
    mm, dd, yy = map(int, m.groups())
    if yy < 100:
        yy += 2000
    try:
        return datetime(yy, mm, dd)
    except ValueError:
        return None

def normalize_text(text: str) -> str:
    """
    Lowercase, remove timestamps, collapse to word tokens joined by single spaces.
    Used for phrase-matching.
    """
    text = (text or "").lower()
    text = re.sub(r"\(\s*\d{1,2}:\d{2}(:\d{2})?\s*\)\s*:\s*", " ", text)
    toks = TOKEN_RE.findall(text)
    return " ".join(toks)

def tokenize_set_from_normalized(norm: str) -> set:
    return set(norm.split())

def load_speeches_presence(txt_dir: str) -> pd.DataFrame:
    rows = []
    for fn in os.listdir(txt_dir):
        if not fn.endswith(".txt"):
            continue
        dt = parse_date_from_filename(fn)
        if not dt:
            continue
        with open(os.path.join(txt_dir, fn), "r", encoding="utf-8") as f:
            text = f.read()
        norm = normalize_text(text)
        rows.append({
            "date": dt,
            "file": fn,
            "norm": norm,
            "words": tokenize_set_from_normalized(norm),
        })
    if not rows:
        raise RuntimeError(f"No dated transcripts found in {txt_dir}.")
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df

# -----------------------
# Market title parsing: extract quoted phrase + phrase-alternatives
# -----------------------
QUOTE_RE = re.compile(r'"([^"]+)"')

def extract_phrase_alternatives_from_title(title: str) -> Optional[List[List[str]]]:
    """
    For markets like:
      Will Trump say "Crypto / Bitcoin" before Feb 2, 2026?
    We extract inside quotes => "Crypto / Bitcoin"

    We then split into alternatives:
      [["crypto"], ["bitcoin"]]

    For multiword chunks like "golden dome":
      [["golden", "dome"]]
    and we require words to appear together IN ORDER (phrase match).
    """
    if not title:
        return None
    m = QUOTE_RE.search(title)
    if not m:
        return None

    inside = m.group(1).lower().strip()
    # split into alternatives on separators: / , or |.
    parts = re.split(r"\s*/\s*|\s*,\s*|\s+or\s+|\s*\|\s*", inside)

    alts: List[List[str]] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        toks = TOKEN_RE.findall(p)
        if not toks:
            continue
        alts.append(toks)

    # Deduplicate alternatives by tuple
    seen = set()
    uniq = []
    for a in alts:
        t = tuple(a)
        if t not in seen:
            seen.add(t)
            uniq.append(a)
    return uniq if uniq else None

def phrase_present_in_speech(norm: str, wordset: set, alts: List[List[str]], match_mode: str) -> bool:
    """
    match_mode:
      - "any": speech counts if ANY alternative is present
      - "all": speech counts only if ALL alternatives are present
    Presence definition:
      - single-word alt => token in wordset
      - multiword alt => exact phrase appears in normalized string with word boundaries
    """
    def alt_present(alt: List[str]) -> bool:
        if len(alt) == 1:
            return alt[0] in wordset
        phrase = " ".join(alt)
        # strict phrase in normalized token stream
        return f" {phrase} " in f" {norm} "

    if match_mode == "all":
        return all(alt_present(a) for a in alts)
    return any(alt_present(a) for a in alts)

# -----------------------
# Prob model: per-speech -> until expiry
# -----------------------
def parse_iso_dt(s) -> Optional[datetime]:
    if not s:
        return None
    s = str(s)
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def recency_weighted_p_speech(
    speech_df: pd.DataFrame,
    alts: List[List[str]],
    as_of_utc: datetime,
    lookback_days: int = 60,
    half_life_days: float = 14.0,
    match_mode: str = "any",
) -> Optional[float]:
    cutoff = as_of_utc.replace(tzinfo=None)
    start = cutoff - pd.Timedelta(days=lookback_days)
    sub = speech_df[(speech_df["date"] < cutoff) & (speech_df["date"] >= start)]
    if sub.empty:
        return None

    ages = (cutoff - sub["date"]).dt.days.astype(float).to_numpy()
    lam = 0.69314718056 / max(1e-9, half_life_days)
    w = np.exp(-lam * ages)

    present = sub.apply(lambda r: phrase_present_in_speech(r["norm"], r["words"], alts, match_mode), axis=1).astype(float).to_numpy()
    p = float((w * present).sum() / w.sum())
    return max(0.0, min(1.0, p))

def expected_speeches_before(
    speech_df: pd.DataFrame,
    as_of_utc: datetime,
    expiry_utc: datetime,
    cadence_lookback_days: int = 30
) -> float:
    now = as_of_utc.replace(tzinfo=None)
    expiry = expiry_utc.replace(tzinfo=None)
    if expiry <= now:
        return 0.0
    start = now - pd.Timedelta(days=cadence_lookback_days)
    recent = speech_df[(speech_df["date"] < now) & (speech_df["date"] >= start)]
    rate_per_day = len(recent) / max(1.0, cadence_lookback_days)
    horizon_days = (expiry - now).days + (expiry - now).seconds / 86400.0
    return max(0.0, rate_per_day * horizon_days)

def p_until_expiry(p_per_speech: float, expected_n_speeches: float) -> float:
    return 1.0 - (1.0 - p_per_speech) ** max(0.0, expected_n_speeches)

def implied_prob(m: Dict) -> Optional[float]:
    yes_bid = m.get("yes_bid")
    yes_ask = m.get("yes_ask")
    last_price = m.get("last_price")
    def ok(x): return isinstance(x, (int, float)) and x is not None and x >= 0
    if ok(yes_bid) and ok(yes_ask) and yes_ask and yes_ask > 0:
        return ((float(yes_bid) + float(yes_ask)) / 2.0) / 100.0
    if ok(last_price) and last_price and last_price > 0:
        return float(last_price) / 100.0
    if ok(yes_bid) and yes_bid and yes_bid > 0:
        return float(yes_bid) / 100.0
    return None

# -----------------------
# Event filters
# -----------------------
MONTH_WORDS = set([
    "january","february","march","april","may","june","july","august","september","october","november","december"
])

def is_month_or_week_bucket(title: str) -> bool:
    t = (title or "").lower()
    if "this week" in t or "this month" in t:
        return True
    if re.search(r"\b(in|before)\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b", t):
        return True
    if re.search(r"\bwhat will trump say in\b", t):
        return True
    return False

def is_specific_speech_event(title: str) -> bool:
    t = (title or "").lower()
    # we want: "during ..." / "remarks ..." / "interview ..." / "address ..." etc
    if "during" in t:
        return True
    for kw in ["remarks", "interview", "address", "announcement", "press briefing", "meeting", "roundtable", "wef", "state of the union"]:
        if kw in t:
            return True
    return False

def is_before_style_market(market_title: str) -> bool:
    # This catches: Will Trump say "X" before Feb 2, 2026?
    t = (market_title or "").lower()
    return t.startswith("will trump say") and " before " in t

# -----------------------
# Pagination
# -----------------------
def iter_events(client: KalshiClient, limit=200, max_pages=80) -> List[Dict]:
    out, cursor = [], None
    for page in range(1, max_pages + 1):
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        data = client.get("/trade-api/v2/events", params=params)
        batch = data.get("events", []) or []
        out.extend(batch)
        print(f"[events page {page}] fetched {len(batch)} (total {len(out)})", flush=True)
        cursor = data.get("cursor")
        if not cursor:
            break
    return out

def iter_markets_for_event(client: KalshiClient, event_ticker: str, limit=500, max_pages=30) -> List[Dict]:
    out, cursor = [], None
    for page in range(1, max_pages + 1):
        params = {"limit": limit, "event_ticker": event_ticker}
        if cursor:
            params["cursor"] = cursor
        data = client.get("/trade-api/v2/markets", params=params)
        batch = data.get("markets", []) or []
        out.extend(batch)
        cursor = data.get("cursor")
        if not cursor:
            break
    return out

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt_dir", default="out/extracted_txt")
    ap.add_argument("--base_url", default=os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com"))
    ap.add_argument("--api_key_id", default=os.getenv("KALSHI_API_KEY_ID", ""))
    ap.add_argument("--private_key", default=os.getenv("KALSHI_PRIVATE_KEY_PATH", ""))

    ap.add_argument("--event_query", default="trump say", help="substring match on event title")
    ap.add_argument("--max_event_pages", type=int, default=120)

    # Filters (defaults are what you want)
    ap.add_argument("--only_specific_speeches", action="store_true", help="Only use 'during <speech>' style events.")
    ap.add_argument("--skip_before", action="store_true", help="Skip 'Will Trump say ... before <date>' markets.")
    ap.add_argument("--skip_month_week", action="store_true", help="Skip month/week bucket events.")
    ap.add_argument("--only_trumpsay", action="store_true", help="Only consider event tickers starting with KXTRUMP")
    # Model params
    ap.add_argument("--lookback_days", type=int, default=120)
    ap.add_argument("--half_life_days", type=float, default=21.0)
    ap.add_argument("--cadence_lookback_days", type=int, default=30)

    ap.add_argument("--match_mode", choices=["any","all"], default="any")
    ap.add_argument("--max_horizon_days", type=float, default=14.0, help="Ignore events closing too far out.")
    ap.add_argument("--min_expected_speeches", type=float, default=0.25)

    ap.add_argument("--min_edge", type=float, default=0.01)
    ap.add_argument("--min_volume", type=int, default=0)

    ap.add_argument("--show_events", action="store_true")
    ap.add_argument("--show_samples", action="store_true")
    ap.add_argument("--out_csv", default="mispricings_specificspeech.csv")
    args = ap.parse_args()

    if not args.api_key_id or not args.private_key:
        raise SystemExit("Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH env vars.")

    print("Loading transcripts...", flush=True)
    speech_df = load_speeches_presence(args.txt_dir)
    print(f"Loaded {len(speech_df)} speeches: {speech_df['date'].min().date()} → {speech_df['date'].max().date()}", flush=True)

    client = KalshiClient(args.base_url, args.api_key_id, args.private_key)

    print("Fetching events...", flush=True)
    events = iter_events(client, max_pages=args.max_event_pages)

    q = args.event_query.lower().strip()
    matched = []
    for e in events:
        title = (e.get("title") or "")
        if q not in title.lower():
            continue
        if args.only_trumpsay and not str(e.get("event_ticker","")).startswith("KXTRUMP"):
            continue
        if args.skip_month_week and is_month_or_week_bucket(title):
            continue
        if args.only_specific_speeches and not is_specific_speech_event(title):
            continue
        matched.append(e)

    if not matched:
        raise SystemExit("No matching events after filters. Try removing --only_specific_speeches or --skip_month_week.")

    if args.show_events:
        print("\nMatched events:", flush=True)
        for e in matched[:40]:
            print("-", e.get("event_ticker"), "|", e.get("title"))
        print(f"(total matched: {len(matched)})\n", flush=True)

    now_utc = datetime.now(timezone.utc)

    rows = []
    dbg = {
        "events_matched": len(matched),
        "markets_seen": 0,
        "skipped_before": 0,
        "skipped_nonbinary": 0,
        "no_phrase_extracted": 0,
        "skipped_horizon": 0,
        "skipped_expected_speeches": 0,
        "skipped_mp_none": 0,
        "below_edge": 0,
        "below_volume": 0,
    }

    for e in matched:
        et = e.get("event_ticker")
        etitle = e.get("title") or ""
        markets = iter_markets_for_event(client, et)

        if args.show_samples:
            for m in markets[:12]:
                print(f"[sample] {etitle} | {m.get('title')}", flush=True)

        for m in markets:
            dbg["markets_seen"] += 1

            vol = m.get("volume") or 0
            if vol < args.min_volume:
                dbg["below_volume"] += 1
                continue

            title = m.get("title") or ""

            if args.skip_before and is_before_style_market(title):
                dbg["skipped_before"] += 1
                continue

            mp = implied_prob(m)
            if mp is None:
                dbg["skipped_mp_none"] += 1
                continue

            # Phrase markets have quoted phrases; specific-speech markets might not.
            # For specific-speech events, we still only trade the quoted-phrase markets.
            alts = extract_phrase_alternatives_from_title(title)
            if not alts:
                dbg["no_phrase_extracted"] += 1
                continue

            expiry = parse_iso_dt(m.get("close_time")) or parse_iso_dt(e.get("close_time"))
            if expiry is None:
                continue

            horizon_days = (expiry - now_utc).total_seconds() / 86400.0
            if horizon_days > args.max_horizon_days:
                dbg["skipped_horizon"] += 1
                continue
            if horizon_days <= 0:
                dbg["skipped_horizon"] += 1
                continue

            p_speech = recency_weighted_p_speech(
                speech_df=speech_df,
                alts=alts,
                as_of_utc=now_utc,
                lookback_days=args.lookback_days,
                half_life_days=args.half_life_days,
                match_mode=args.match_mode
            )
            if p_speech is None:
                continue

            N = expected_speeches_before(
                speech_df=speech_df,
                as_of_utc=now_utc,
                expiry_utc=expiry,
                cadence_lookback_days=args.cadence_lookback_days
            )

            if N < args.min_expected_speeches:
                dbg["skipped_expected_speeches"] += 1
                continue

            model_p = p_until_expiry(p_speech, N)
            edge = float(model_p) - float(mp)
            if abs(edge) < args.min_edge:
                dbg["below_edge"] += 1
                continue

            # pack phrase alternatives into a readable string
            phrase_repr = " / ".join([" ".join(a) for a in alts])

            rows.append({
                "event_ticker": et,
                "event_title": etitle[:160],
                "market_ticker": m.get("ticker"),
                "market_title": title[:180],
                "phrase_alts": phrase_repr[:180],
                "market_p": round(float(mp), 4),
                "p_per_speech": round(float(p_speech), 4),
                "exp_speeches": round(float(N), 3),
                "model_p": round(float(model_p), 4),
                "edge": round(float(edge), 4),
                "yes_bid": m.get("yes_bid"),
                "yes_ask": m.get("yes_ask"),
                "volume": vol,
                "close_time": m.get("close_time"),
                "horizon_days": round(float(horizon_days), 3),
            })

    if not rows:
        print("\nNo candidates. Debug breakdown:", flush=True)
        for k, v in dbg.items():
            print(f"  {k}={v}")
        print("\nMost common reasons in practice:")
        print("  - your matched events are 'during speech' but the markets are MULTI-outcome (no yes_bid/yes_ask)")
        print("  - or they have no quoted phrases (so we skip)")
        print("\nTry:")
        print("  --show_samples   (to see what the market titles actually look like)")
        print("  --only_specific_speeches OFF (to include KXTRUMPSAY weekly buckets)")
        return

    df = pd.DataFrame(rows).sort_values("edge", ascending=False)
    df.to_csv(args.out_csv, index=False)

    print(f"\nSaved: {args.out_csv}")
    print("\nTop candidates:")
    cols = ["market_ticker","phrase_alts","market_p","p_per_speech","exp_speeches","model_p","edge","volume","horizon_days","market_title"]
    print(df.head(30)[cols].to_string(index=False))

if __name__ == "__main__":
    main()

