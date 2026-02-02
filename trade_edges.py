#!/usr/bin/env python3
import os, sys, time, uuid, base64, argparse, csv, select
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple

import pandas as pd
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# Optional: silence urllib3 LibreSSL warning on some macOS builds
try:
    import warnings
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

# -----------------------
# Kalshi auth client
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
            "Content-Type": "application/json",
        }

    def get(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = self.base_url + path
        r = self.session.get(url, headers=self._headers("GET", path), params=params, timeout=self.timeout)
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code} {r.reason} for {r.url}\n{(r.text or '')[:800]}", response=r)
        return r.json()

    def post(self, path: str, payload: Dict) -> Dict:
        url = self.base_url + path
        r = self.session.post(url, headers=self._headers("POST", path), json=payload, timeout=self.timeout)
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code} {r.reason} for {r.url}\n{(r.text or '')[:800]}", response=r)
        return r.json()

# -----------------------
# API helpers
# -----------------------
def get_balance_and_value_cents(client: KalshiClient) -> Tuple[int, int]:
    data = client.get("/trade-api/v2/portfolio/balance")
    bal = int(data.get("balance", 0))
    pv  = int(data.get("portfolio_value", data.get("portfolioValue", bal)))
    return bal, pv

def get_positions(client: KalshiClient) -> List[Dict]:
    data = client.get("/trade-api/v2/portfolio/positions")
    return data.get("positions", []) or []

def get_fills_all(client: KalshiClient, limit: int = 1000, max_pages: int = 50) -> List[Dict]:
    out = []
    cursor = None
    for _ in range(max_pages):
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        data = client.get("/trade-api/v2/portfolio/fills", params=params)
        batch = data.get("fills", []) or []
        out.extend(batch)
        cursor = data.get("cursor")
        if not cursor:
            break
    return out

def get_market_snapshot(client: KalshiClient, ticker: str) -> Dict:
    return client.get(f"/trade-api/v2/markets/{ticker}")

def clamp_price(p: int) -> int:
    return max(1, min(99, int(p)))

def client_order_id() -> str:
    return f"edge-{uuid.uuid4().hex[:16]}"

# -----------------------
# Time-in-force mapping (Kalshi)
# -----------------------
TIF_MAP = {
    "fill_or_kill": "fill_or_kill",
    "good_till_canceled": "good_till_canceled",
    "immediate_or_cancel": "immediate_or_cancel",
    "fok": "fill_or_kill",
    "gtc": "good_till_canceled",
    "ioc": "immediate_or_cancel",
    "good_til_cancelled": "good_till_canceled",
    "immediate_or_cancelled": "immediate_or_cancel",
}

def normalize_tif(tif: str) -> str:
    t = (tif or "").strip().lower()
    return TIF_MAP.get(t, "good_till_canceled")

def place_limit_buy_yes(client: KalshiClient, ticker: str, yes_price_cents: int, count: int, tif: str) -> Dict:
    payload = {
        "ticker": ticker,
        "action": "buy",
        "type": "limit",
        "side": "yes",
        "count": int(count),
        "time_in_force": normalize_tif(tif),
        "client_order_id": client_order_id(),
        "yes_price": clamp_price(yes_price_cents),
    }
    return client.post("/trade-api/v2/portfolio/orders", payload)

# -----------------------
# Market pricing
# -----------------------
def best_yes_ask_from_snapshot(snap: Dict) -> Optional[int]:
    m = snap.get("market") or snap
    ya = m.get("yes_ask")
    if isinstance(ya, (int, float)) and ya and ya > 0:
        return int(ya)
    return None

def yes_bid_ask_last_mid(snap: Dict) -> Tuple[float, float, float]:
    m = snap.get("market") or snap
    yb = m.get("yes_bid")
    ya = m.get("yes_ask")
    lp = m.get("last_price")

    bid = float(yb)/100.0 if isinstance(yb, (int, float)) and yb and yb > 0 else 0.0
    last = float(lp)/100.0 if isinstance(lp, (int, float)) and lp and lp > 0 else bid
    if isinstance(yb,(int,float)) and isinstance(ya,(int,float)) and ya and ya > 0:
        mid = (float(yb)+float(ya))/2.0/100.0
    else:
        mid = last
    return bid, mid, last

# -----------------------
# Title filters (hard stop on "before <date>" style)
# -----------------------
def is_before_style_title(title: str) -> bool:
    t = (title or "").lower()
    return t.startswith("will trump say") and " before " in t

def minutes_to_close(iso: Optional[str]) -> Optional[float]:
    if not iso:
        return None
    s = str(iso)
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (dt - now).total_seconds() / 60.0
    except Exception:
        return None

# -----------------------
# Paper trading + sizing
# -----------------------
def kelly_yes(p: float, q: float) -> float:
    if q >= 1.0:
        return 0.0
    return max(0.0, (p - q) / max(1e-9, (1.0 - q)))

def ensure_csv_header(path: str, header: List[str]) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

def append_csv_row(path: str, row: List) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def nonblocking_readline() -> Optional[str]:
    """
    Read a line from stdin if available (non-blocking). Returns None if no input.
    """
    try:
        r, _, _ = select.select([sys.stdin], [], [], 0.0)
        if r:
            line = sys.stdin.readline()
            return line.strip()
    except Exception:
        return None
    return None

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges_csv", required=True)
    ap.add_argument("--base_url", default=os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com"))
    ap.add_argument("--api_key_id", default=os.getenv("KALSHI_API_KEY_ID", ""))
    ap.add_argument("--private_key", default=os.getenv("KALSHI_PRIVATE_KEY_PATH", ""))

    # Trading filters
    ap.add_argument("--max_yes_price_cents", type=int, default=50)
    ap.add_argument("--min_edge", type=float, default=0.02)
    ap.add_argument("--max_markets", type=int, default=10)
    ap.add_argument("--slippage_cents", type=int, default=1)
    ap.add_argument("--min_minutes_to_close", type=int, default=60 * 24, help="Default 1440 (>= 1 day).")
    ap.add_argument("--tif", default="gtc", choices=[
        "gtc","ioc","fok",
        "good_till_canceled","immediate_or_cancel","fill_or_kill",
        "good_til_cancelled","immediate_or_cancelled"
    ])

    # Spend controls
    ap.add_argument("--max_spend_usd", type=float, default=None, help="LIVE spend cap from real balance.")
    ap.add_argument("--max_per_market_usd", type=float, default=None)
    ap.add_argument("--paper_spend_usd", type=float, default=None, help="If --dry_run, pretend bankroll is this amount.")

    # Scope filters
    ap.add_argument("--only_trumpsay", action="store_true", help="Only tickers starting with KXTRUMP.")
    ap.add_argument("--ban_before_titles", action="store_true", default=True, help="Hard-ban 'Will Trump say ... before ...' titles (default on).")

    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--LIVE", action="store_true")

    # Monitoring/logging
    ap.add_argument("--poll_sec", type=int, default=60)
    ap.add_argument("--pnl_log", default="pnl_timeseries.csv")
    ap.add_argument("--pos_log", default="positions_pnl.csv")
    args = ap.parse_args()

    if not os.path.exists(args.edges_csv):
        here = os.getcwd()
        csvs = [f for f in os.listdir(here) if f.endswith(".csv")]
        raise SystemExit(f"edges_csv not found: {os.path.abspath(args.edges_csv)}\nCSV files here: {csvs}")

    if not args.api_key_id or not args.private_key:
        raise SystemExit("Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH env vars.")

    if args.LIVE and args.dry_run:
        raise SystemExit("Pick one: --dry_run OR --LIVE")
    if not args.LIVE:
        args.dry_run = True

    df = pd.read_csv(args.edges_csv)
    for col in ("market_ticker","market_title","model_p","market_p","edge","close_time"):
        if col not in df.columns:
            raise SystemExit(f"Expected column '{col}' in edges_csv.")

    # HARD stop: ban "before <date>" title markets (your explicit requirement)
    if args.ban_before_titles:
        df = df[~df["market_title"].astype(str).apply(is_before_style_title)].copy()

    if args.only_trumpsay:
        df = df[df["market_ticker"].astype(str).str.startswith("KXTRUMP")].copy()

    # basic edge sort
    df = df[df["edge"] >= args.min_edge].copy()
    df = df.sort_values("edge", ascending=False).head(args.max_markets)

    client = KalshiClient(args.base_url, args.api_key_id, args.private_key)
    bal, pv = get_balance_and_value_cents(client)

    # bankroll rules
    if args.dry_run:
        bankroll_cents = int(round((args.paper_spend_usd if args.paper_spend_usd is not None else max(1.0, bal/100.0)) * 100))
    else:
        if args.max_spend_usd is None:
            bankroll_cents = bal
        else:
            bankroll_cents = min(bal, int(round(args.max_spend_usd * 100)))

    print(f"Balance: ${bal/100:.2f} | Portfolio value: ${pv/100:.2f}")
    print(f"Mode: {'LIVE' if args.LIVE else 'DRY RUN'}")
    if args.dry_run:
        print(f"Paper bankroll: ${bankroll_cents/100:.2f}")
    else:
        print(f"Spend cap: ${bankroll_cents/100:.2f}")
    if args.max_per_market_usd is not None:
        print(f"Per-market cap: ${args.max_per_market_usd:.2f}")
    print(f"Rule: buy YES only, ask<= {args.max_yes_price_cents}c, min_edge>= {args.min_edge}, up to {len(df)} markets")
    print(f"Min time-to-close: {args.min_minutes_to_close} minutes")
    print(f"TIF: {normalize_tif(args.tif)}")
    print("Runtime commands: type `more` to loosen filters, `less` to tighten.\n")

    # dynamic thresholds (can be loosened/tightened at runtime)
    dyn_min_edge = args.min_edge
    dyn_max_yes = args.max_yes_price_cents

    def build_candidates() -> List[Dict]:
        cands = []
        for _, r in df.iterrows():
            ticker = str(r["market_ticker"])
            title  = str(r["market_title"])
            p_model = float(r["model_p"])
            close = r.get("close_time")

            mtc = minutes_to_close(close)
            if mtc is not None and mtc < args.min_minutes_to_close:
                continue

            # snapshot
            snap = get_market_snapshot(client, ticker)
            ask = best_yes_ask_from_snapshot(snap)
            if ask is None:
                continue

            limit_price = clamp_price(int(ask) + args.slippage_cents)
            if limit_price > dyn_max_yes:
                continue

            bid, mid, _ = yes_bid_ask_last_mid(snap)
            q = mid if mid > 0 else float(r["market_p"])
            edge = p_model - q
            if edge < dyn_min_edge:
                continue

            k = kelly_yes(p_model, q)
            if k <= 0:
                continue

            cands.append({
                "ticker": ticker,
                "title": title,
                "p": p_model,
                "q": q,
                "edge": edge,
                "price": limit_price,
                "kelly": k,
            })

        cands.sort(key=lambda x: x["edge"], reverse=True)
        return cands

    # sizing
    def plan_orders(cands: List[Dict]) -> List[Tuple[Dict,int,int]]:
        if not cands:
            return []
        total_k = sum(c["kelly"] for c in cands) or 1.0
        for c in cands:
            c["w"] = c["kelly"] / total_k

        remaining = bankroll_cents
        planned = []
        for i, c in enumerate(cands):
            budget = remaining if i == len(cands)-1 else int(bankroll_cents * c["w"])
            budget = min(budget, remaining)
            if args.max_per_market_usd is not None:
                budget = min(budget, int(round(args.max_per_market_usd * 100)))

            count = budget // c["price"]
            if count <= 0:
                continue
            cost = count * c["price"]
            remaining -= cost
            planned.append((c, count, cost))
        return planned

    # place once (or dry-run) then monitor
    candidates = build_candidates()
    if not candidates:
        print("No candidates with current constraints.")
        print("Try typing `more` once or twice (loosens min_edge and max_yes_price).")
    planned = plan_orders(candidates)

    if planned:
        print("Planned orders:")
        for c, count, cost in planned:
            print(f"  {c['ticker']} buy YES {count} @ {c['price']}c | w={c['w']:.3f} edge={c['edge']:+.3f} cost=${cost/100:.2f}")
        print("")
    else:
        print("No orders planned (sizing resulted in 0 counts). Try `more`.\n")

    for c, count, cost in planned:
        if args.dry_run:
            print(f"DRY  {c['ticker']} buy YES {count} @ {c['price']}c | cost=${cost/100:.2f}")
        else:
            resp = place_limit_buy_yes(client, c["ticker"], c["price"], count, args.tif)
            oid = (resp.get("order") or {}).get("order_id")
            print(f"LIVE {c['ticker']} buy YES {count} @ {c['price']}c | cost=${cost/100:.2f} | order_id={oid}")

    # monitoring logs
    ensure_csv_header(args.pnl_log, ["ts_iso", "balance_usd", "portfolio_value_usd"])
    ensure_csv_header(args.pos_log, ["ts_iso","ticker","qty_yes","avg_cost_cents","bid_cents","mid_cents","unreal_mid_usd","unreal_bid_usd"])

    print(f"\nMonitoring... logging to {args.pnl_log} and {args.pos_log} every {args.poll_sec}s")
    print("Type `more` / `less` and press Enter to adjust filters live. Ctrl+C to stop.\n")

    # NOTE: We don't attempt fancy cost-basis here (unless you ask) — we mark from portfolio positions + current bid/mid.
    # If you want avg_cost per position from fills, we can layer that back in, but only after positions schema is verified.

    while True:
        ts = datetime.now(timezone.utc).isoformat()
        try:
            cmd = nonblocking_readline()
            if cmd:
                c = cmd.strip().lower()
                if c == "more":
                    dyn_min_edge = max(0.0, dyn_min_edge - 0.01)
                    dyn_max_yes = min(99, dyn_max_yes + 5)
                    print(f"[cmd] loosen -> min_edge={dyn_min_edge:.3f}, max_yes={dyn_max_yes}c")
                elif c == "less":
                    dyn_min_edge = dyn_min_edge + 0.01
                    dyn_max_yes = max(1, dyn_max_yes - 5)
                    print(f"[cmd] tighten -> min_edge={dyn_min_edge:.3f}, max_yes={dyn_max_yes}c")
                else:
                    print("[cmd] options: more | less")

            bal, pv = get_balance_and_value_cents(client)
            append_csv_row(args.pnl_log, [ts, round(bal/100.0, 4), round(pv/100.0, 4)])

            positions = get_positions(client)

            # crude mark: look for yes position fields
            def extract_yes_qty(p: Dict) -> int:
                for k in ("yes_position","yes_qty","position_yes","qty_yes"):
                    if k in p:
                        try: return int(p.get(k) or 0)
                        except: pass
                # fallback: if p['position'] dict
                pos = p.get("position")
                if isinstance(pos, dict):
                    for k in ("yes","YES","qty_yes"):
                        if k in pos:
                            try: return int(pos.get(k) or 0)
                            except: pass
                return 0

            held = []
            for p in positions:
                ticker = str(p.get("ticker") or p.get("market_ticker") or "")
                if not ticker:
                    continue
                if args.only_trumpsay and not ticker.startswith("KXTRUMP"):
                    continue
                qty = extract_yes_qty(p)
                if qty > 0:
                    held.append((ticker, qty))

            unreal_mid_total = 0.0
            unreal_bid_total = 0.0

            # If we don’t have avg cost, we log marks only; avg_cost stays blank.
            for ticker, qty in held:
                snap = get_market_snapshot(client, ticker)
                bid, mid, _ = yes_bid_ask_last_mid(snap)
                bid_c = bid * 100.0
                mid_c = mid * 100.0

                # unknown avg cost here => leave empty; unreal pnl omitted (0)
                append_csv_row(args.pos_log, [ts, ticker, qty, "", round(bid_c,3), round(mid_c,3), "", ""])

            print(f"[{ts}] balance=${bal/100:.2f} pv=${pv/100:.2f} held={len(held)} (type more/less to adjust filters)", flush=True)

        except KeyboardInterrupt:
            print("\nStopped.")
            return
        except Exception as e:
            print(f"[{ts}] monitor error: {e}", flush=True)

        time.sleep(max(5, args.poll_sec))

if __name__ == "__main__":
    main()

