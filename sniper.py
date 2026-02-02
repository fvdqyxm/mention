#!/usr/bin/env python3
"""
===========================================================
KALSHI TRUMP-MENTIONS EDGE TRADER (REST AUTH + CONSOLE)
===========================================================

Goal:
- ONLY trade Trump "Will Trump say ..." markets (KXTRUMP...) using model edge.
- Poll REST snapshots, compute live mid, compare to model_p, place limit BUY YES.
- Designed to never look "blank": status, reject reasons, shadow mode.

How to run (paper):
  export KALSHI_API_KEY_ID="..."
  export KALSHI_PRIVATE_KEY_PATH="/full/path/to/kalshi-private.key"
  python3 sniper_trump_mentions.py --edges_csv mispricings_week.csv --dry_run

How to run (live):
  python3 sniper_trump_mentions.py --edges_csv mispricings_week.csv --LIVE

Console commands while running:
  more            -> loosen thresholds
  less            -> tighten thresholds
  shadow on/off   -> log WOULD_HAVE_TRADED candidates even if not trading
  live on/off     -> toggle live trading (safety: starts OFF unless --LIVE)
  status          -> print current constraints + counts
  quit            -> exit

Notes:
- This script uses REST endpoints only (no websocket required).
- Auth/signing follows Kalshi docs: timestamp + METHOD + path_without_query, RSA-PSS SHA256, salt_length=DIGEST_LENGTH.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import select
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


# -----------------------
# Defaults / knobs
# -----------------------

DEFAULT_BASE_URL = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com").rstrip("/")
REST_BASE = DEFAULT_BASE_URL + "/trade-api/v2"

# Hard scope: only these markets
ONLY_TICKER_PREFIX = "KXTRUMP"
REQUIRE_WILL_TRUMP_SAY_PREFIX = True  # set False if you want broader KXTRUMP events

# Safety filters
BAN_BEFORE_STYLE_TITLES = True  # "Will Trump say ... before <date>" are weird bucketed markets
FILTER_ASK_100 = True           # ignore yes_ask == 100 (1.00)

# Dynamic gating based on time-to-close (minutes)
# As markets get closer to close, we loosen min_edge a bit and allow slightly higher prices.
MTC_TIERS = [
    # (max_minutes_to_close, edge_multiplier, max_yes_price_cents_add)
    (60,   0.70,  8),   # within 1h: allow smaller edge, slightly higher prices
    (180,  0.80,  5),   # within 3h
    (720,  0.90,  2),   # within 12h
    (1440, 1.00,  0),   # within 1 day
]

# Base trading constraints (can be adjusted with `more/less`)
BASE_MIN_EDGE = 0.020
BASE_MAX_YES_PRICE_CENTS = 95       # "high nineties" sniping
BASE_MIN_MINUTES_TO_CLOSE = 0       # allow near close (your stated goal)
BASE_SLIPPAGE_CENTS = 1
BASE_MAX_MARKETS = 8

# Spend controls
DEFAULT_PAPER_BANKROLL_USD = 250.0
DEFAULT_MAX_PER_MARKET_USD = 50.0

# Polling & logging
POLL_SEC = 8
STATUS_EVERY_SEC = 10
REJECT_SAMPLE_N = 10

PNL_LOG = "pnl_timeseries.csv"
POS_LOG = "positions_pnl.csv"


# ===========================================================
# Utilities
# ===========================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def nonblocking_readline() -> Optional[str]:
    try:
        r, _, _ = select.select([sys.stdin], [], [], 0.0)
        if r:
            return sys.stdin.readline().strip()
    except Exception:
        return None
    return None

def ensure_csv_header(path: str, header: List[str]) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

def append_csv_row(path: str, row: List) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def clamp_price(cents: int) -> int:
    return max(1, min(99, int(cents)))

def is_before_style_title(title: str) -> bool:
    t = (title or "").lower()
    return t.startswith("will trump say") and " before " in t

def minutes_to_close(close_iso: Optional[str]) -> Optional[float]:
    if not close_iso:
        return None
    s = str(close_iso)
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (dt - datetime.now(timezone.utc)).total_seconds() / 60.0
    except Exception:
        return None

def pick_mtc_tier(mtc_min: Optional[float]) -> Tuple[float, int]:
    """
    Return (edge_multiplier, max_yes_add_cents) based on how close we are.
    """
    if mtc_min is None:
        return 1.0, 0
    for max_m, mult, add in MTC_TIERS:
        if mtc_min <= max_m:
            return mult, add
    return 1.0, 0


# ===========================================================
# Auth (RSA-PSS)
# ===========================================================

def _load_private_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def _sign_pss_base64(private_key, message: str) -> str:
    sig = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")

def build_auth_headers(api_key_id: str, private_key_path: str, method: str, path: str) -> Dict[str, str]:
    api_key_id = (api_key_id or "").strip()
    private_key_path = (private_key_path or "").strip()
    if not api_key_id or not private_key_path:
        raise SystemExit("Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH env vars.")
    if not os.path.exists(private_key_path):
        raise SystemExit(f"Private key file not found: {private_key_path}")

    pk = _load_private_key(private_key_path)
    ts_ms = str(int(time.time() * 1000))

    path_wo_query = path.split("?")[0]
    msg = f"{ts_ms}{method.upper()}{path_wo_query}"
    sig = _sign_pss_base64(pk, msg)

    return {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": sig,
    }

class KalshiClient:
    def __init__(self, base_url_root: str, api_key_id: str, private_key_path: str, timeout: int = 10):
        self.root = base_url_root.rstrip("/")
        self.api_key_id = api_key_id
        self.private_key_path = private_key_path
        self.timeout = timeout
        self.session = requests.Session()

    def _headers(self, method: str, path: str) -> Dict[str, str]:
        return build_auth_headers(self.api_key_id, self.private_key_path, method, path)

    def get(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = self.root + path
        r = self.session.get(url, headers=self._headers("GET", path), params=params, timeout=self.timeout)
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code} {r.reason} for {r.url}\n{(r.text or '')[:800]}", response=r)
        return r.json()

    def post(self, path: str, payload: Dict) -> Dict:
        url = self.root + path
        r = self.session.post(url, headers=self._headers("POST", path), json=payload, timeout=self.timeout)
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code} {r.reason} for {r.url}\n{(r.text or '')[:800]}", response=r)
        return r.json()


# ===========================================================
# Trading / pricing helpers
# ===========================================================

def get_market_snapshot(client: KalshiClient, ticker: str) -> Dict:
    # Kalshi: GET /trade-api/v2/markets/{ticker}
    return client.get(f"/trade-api/v2/markets/{ticker}")

def best_yes_ask_from_snapshot(snap: Dict) -> Optional[int]:
    m = snap.get("market") or snap
    ya = m.get("yes_ask")
    if isinstance(ya, (int, float)) and ya is not None and ya > 0:
        return int(ya)
    return None

def yes_bid_ask_last_mid(snap: Dict) -> Tuple[float, float, float]:
    m = snap.get("market") or snap
    yb = m.get("yes_bid")
    ya = m.get("yes_ask")
    lp = m.get("last_price")

    bid = float(yb) / 100.0 if isinstance(yb, (int, float)) and yb and yb > 0 else 0.0
    last = float(lp) / 100.0 if isinstance(lp, (int, float)) and lp and lp > 0 else bid
    if isinstance(yb, (int, float)) and isinstance(ya, (int, float)) and ya and ya > 0:
        mid = (float(yb) + float(ya)) / 2.0 / 100.0
    else:
        mid = last
    return bid, mid, last

def normalize_tif(tif: str) -> str:
    t = (tif or "").lower().strip()
    mapping = {
        "gtc": "good_till_canceled",
        "good_till_canceled": "good_till_canceled",
        "good_til_cancelled": "good_till_canceled",
        "good_til_canceled": "good_till_canceled",
        "ioc": "immediate_or_cancel",
        "immediate_or_cancel": "immediate_or_cancel",
        "immediate_or_cancelled": "immediate_or_cancel",
        "fok": "fill_or_kill",
        "fill_or_kill": "fill_or_kill",
    }
    return mapping.get(t, "good_till_canceled")

def client_order_id() -> str:
    return f"trumpedge-{int(time.time()*1000)}"

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


# ===========================================================
# Core planner
# ===========================================================

@dataclass
class DynConfig:
    min_edge: float = BASE_MIN_EDGE
    max_yes_price_cents: int = BASE_MAX_YES_PRICE_CENTS
    min_minutes_to_close: int = BASE_MIN_MINUTES_TO_CLOSE
    slippage_cents: int = BASE_SLIPPAGE_CENTS
    max_markets: int = BASE_MAX_MARKETS

    shadow: bool = True
    live: bool = False

    paper_bankroll_usd: float = DEFAULT_PAPER_BANKROLL_USD
    max_per_market_usd: float = DEFAULT_MAX_PER_MARKET_USD

def load_edges_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    REQUIRED_COLS = ["market_ticker", "market_title", "model_p", "market_p", "edge", "close_time"]
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise SystemExit(f"edges_csv missing required column: {c}")

    # Hard scope filters
    df["market_ticker"] = df["market_ticker"].astype(str)
    df["market_title"] = df["market_title"].astype(str)

    df = df[df["market_ticker"].str.startswith(ONLY_TICKER_PREFIX)].copy()

    if REQUIRE_WILL_TRUMP_SAY_PREFIX:
        df = df[df["market_title"].str.lower().str.startswith("will trump say")].copy()

    if BAN_BEFORE_STYLE_TITLES:
        df = df[~df["market_title"].apply(is_before_style_title)].copy()

    # clean numeric
    df["model_p"] = pd.to_numeric(df["model_p"], errors="coerce")
    df["market_p"] = pd.to_numeric(df["market_p"], errors="coerce")
    df["edge"] = pd.to_numeric(df["edge"], errors="coerce")

    df = df.dropna(subset=["market_ticker", "market_title", "model_p", "market_p", "edge"]).reset_index(drop=True)
    return df

def get_portfolio_balance_usd(client: KalshiClient) -> Optional[float]:
    try:
        j = client.get("/trade-api/v2/portfolio/balance")
        bal = (j.get("balance") or {}).get("available_cash")
        if isinstance(bal, (int, float)):
            return float(bal) / 100.0
    except Exception:
        return None
    return None

def build_candidates(client: KalshiClient, df: pd.DataFrame, cfg: DynConfig) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Returns (candidates_sorted, reject_reason_counts)
    Each candidate includes:
      ticker,title,p,q,edge,price,mtc,limit_price
    """
    rejects: Dict[str, int] = {}
    cands: List[Dict] = []

    def rej(r: str) -> None:
        rejects[r] = rejects.get(r, 0) + 1

    for _, r in df.iterrows():
        ticker = str(r["market_ticker"])
        title = str(r["market_title"])
        p_model = float(r["model_p"])
        close_time = r.get("close_time")

        mtc = minutes_to_close(close_time)
        if mtc is not None and mtc < cfg.min_minutes_to_close:
            rej("mtc_too_small")
            continue

        # dynamic loosening as we get close
        edge_mult, max_yes_add = pick_mtc_tier(mtc)
        dyn_min_edge = cfg.min_edge * edge_mult
        dyn_max_yes = cfg.max_yes_price_cents + max_yes_add

        # snapshot
        try:
            snap = get_market_snapshot(client, ticker)
        except Exception:
            rej("snapshot_fail")
            continue

        ask = best_yes_ask_from_snapshot(snap)
        if ask is None:
            rej("no_ask")
            continue

        if FILTER_ASK_100 and int(ask) >= 100:
            rej("ask_is_100")
            continue

        limit_price = clamp_price(int(ask) + cfg.slippage_cents)
        if limit_price > dyn_max_yes:
            rej("price_too_high")
            continue

        bid, mid, _last = yes_bid_ask_last_mid(snap)
        q = mid if mid > 0 else float(r["market_p"])
        edge = p_model - q
        if edge < dyn_min_edge:
            rej("edge_too_small")
            continue

        cands.append({
            "ticker": ticker,
            "title": title,
            "p": p_model,
            "q": q,
            "edge": edge,
            "price": limit_price,
            "ask": int(ask),
            "bid": int(round(bid * 100)),
            "mid": q,
            "mtc": mtc,
            "dyn_min_edge": dyn_min_edge,
            "dyn_max_yes": dyn_max_yes,
        })

    cands.sort(key=lambda x: (x["edge"]), reverse=True)
    return cands[: max(1, cfg.max_markets * 3)], rejects

def plan_orders(cands: List[Dict], bankroll_usd: float, cfg: DynConfig) -> List[Tuple[Dict, int, float]]:
    """
    Simple sizing:
    - equal-weight budgets capped per market
    - buys YES with count = floor(budget / price)
    Returns list of (candidate, count, cost_usd)
    """
    if not cands:
        return []

    # cap number of markets
    cands = cands[: cfg.max_markets]

    per_mkt_budget = min(cfg.max_per_market_usd, bankroll_usd / max(1, len(cands)))
    orders = []
    for c in cands:
        price_cents = int(c["price"])
        if price_cents <= 0:
            continue
        budget_cents = int(round(per_mkt_budget * 100))
        count = budget_cents // price_cents
        if count <= 0:
            continue
        cost_cents = count * price_cents
        orders.append((c, int(count), cost_cents / 100.0))

    # prioritize higher edge if budgets end up small
    orders.sort(key=lambda x: x[0]["edge"], reverse=True)
    return orders

def print_status(df: pd.DataFrame, cfg: DynConfig, last_rejects: Dict[str, int]) -> None:
    log(
        f"STATUS | universe={len(df)} "
        f"| min_edge={cfg.min_edge:.3f} max_yes={cfg.max_yes_price_cents}c slippage={cfg.slippage_cents}c "
        f"| max_markets={cfg.max_markets} shadow={cfg.shadow} live={cfg.live}"
    )
    if last_rejects:
        top = sorted(last_rejects.items(), key=lambda kv: kv[1], reverse=True)[:REJECT_SAMPLE_N]
        log("REJECTS (sample): " + ", ".join([f"{k}:{v}" for k, v in top]))

def handle_command(cmd: str, cfg: DynConfig) -> bool:
    """
    Returns True if should quit.
    """
    c = (cmd or "").strip().lower()
    if not c:
        return False

    if c in ("quit", "exit", "q"):
        return True

    if c == "more":
        # Loosen: smaller required edge, allow slightly higher price, scan more markets
        cfg.min_edge = max(0.003, cfg.min_edge * 0.85)
        cfg.max_yes_price_cents = min(99, cfg.max_yes_price_cents + 2)
        cfg.max_markets = min(20, cfg.max_markets + 1)
        log(f"MORE -> min_edge={cfg.min_edge:.3f}, max_yes={cfg.max_yes_price_cents}c, max_markets={cfg.max_markets}")
        return False

    if c == "less":
        # Tighten: larger required edge, lower max price, fewer markets
        cfg.min_edge = min(0.20, cfg.min_edge * 1.15)
        cfg.max_yes_price_cents = max(50, cfg.max_yes_price_cents - 2)
        cfg.max_markets = max(2, cfg.max_markets - 1)
        log(f"LESS -> min_edge={cfg.min_edge:.3f}, max_yes={cfg.max_yes_price_cents}c, max_markets={cfg.max_markets}")
        return False

    if c.startswith("shadow"):
        if "on" in c:
            cfg.shadow = True
        elif "off" in c:
            cfg.shadow = False
        log(f"shadow={cfg.shadow}")
        return False

    if c.startswith("live"):
        if "on" in c:
            cfg.live = True
        elif "off" in c:
            cfg.live = False
        log(f"live={cfg.live}")
        return False

    if c == "status":
        log("Type `more`, `less`, `shadow on/off`, `live on/off`, `quit`.")
        return False

    log(f"Unknown command: {cmd} (try: more/less/shadow on/off/live on/off/status/quit)")
    return False


# ===========================================================
# Main loop
# ===========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges_csv", required=True, help="mispricings_week.csv / mispricings_2026-01.csv etc")
    ap.add_argument("--base_url", default=DEFAULT_BASE_URL)

    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--LIVE", action="store_true")

    # override knobs if desired
    ap.add_argument("--min_edge", type=float, default=BASE_MIN_EDGE)
    ap.add_argument("--max_yes_price_cents", type=int, default=BASE_MAX_YES_PRICE_CENTS)
    ap.add_argument("--min_minutes_to_close", type=int, default=BASE_MIN_MINUTES_TO_CLOSE)
    ap.add_argument("--slippage_cents", type=int, default=BASE_SLIPPAGE_CENTS)
    ap.add_argument("--max_markets", type=int, default=BASE_MAX_MARKETS)

    ap.add_argument("--paper_bankroll_usd", type=float, default=DEFAULT_PAPER_BANKROLL_USD)
    ap.add_argument("--max_per_market_usd", type=float, default=DEFAULT_MAX_PER_MARKET_USD)

    ap.add_argument("--poll_sec", type=int, default=POLL_SEC)
    ap.add_argument("--status_every_sec", type=int, default=STATUS_EVERY_SEC)
    ap.add_argument("--pnl_log", default=PNL_LOG)
    ap.add_argument("--pos_log", default=POS_LOG)
    args = ap.parse_args()

    api_key_id = (os.getenv("KALSHI_API_KEY_ID", "") or "").strip()
    priv_key = (os.getenv("KALSHI_PRIVATE_KEY_PATH", "") or "").strip()
    if not api_key_id or not priv_key:
        raise SystemExit("Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH env vars.")

    if args.LIVE and args.dry_run:
        raise SystemExit("Pick one: --dry_run OR --LIVE")
    if not args.LIVE:
        args.dry_run = True

    cfg = DynConfig(
        min_edge=float(args.min_edge),
        max_yes_price_cents=int(args.max_yes_price_cents),
        min_minutes_to_close=int(args.min_minutes_to_close),
        slippage_cents=int(args.slippage_cents),
        max_markets=int(args.max_markets),
        shadow=True,
        live=bool(args.LIVE),
        paper_bankroll_usd=float(args.paper_bankroll_usd),
        max_per_market_usd=float(args.max_per_market_usd),
    )

    client = KalshiClient(args.base_url, api_key_id, priv_key)

    # smoke test auth
    try:
        _ = client.get("/trade-api/v2/communications/id")
        log("✅ Auth OK.")
    except Exception as e:
        raise SystemExit(f"Auth failed: {e}")

    # load edges universe
    df = load_edges_df(args.edges_csv)
    if df.empty:
        raise SystemExit("No rows after Trump-market filters. Check that your edges CSV contains KXTRUMP markets.")

    log(f"Loaded edges universe: {len(df)} rows (ticker prefix={ONLY_TICKER_PREFIX})")
    log("Type commands anytime: more / less / shadow on/off / live on/off / status / quit")

    ensure_csv_header(args.pnl_log, ["ts_iso", "mode", "balance_usd"])
    ensure_csv_header(args.pos_log, ["ts_iso", "mode", "ticker", "price_cents", "count", "edge", "p_model", "mid", "mtc_min", "title"])

    last_status = 0.0
    last_rejects: Dict[str, int] = {}

    while True:
        # commands
        cmd = nonblocking_readline()
        if cmd:
            if handle_command(cmd, cfg):
                log("Stopping.")
                break

        # bankroll
        if cfg.live and not args.dry_run:
            bal = get_portfolio_balance_usd(client)
            bankroll = bal if (bal is not None) else cfg.paper_bankroll_usd
        else:
            bankroll = cfg.paper_bankroll_usd

        # build candidates from live snapshots
        cands, rejects = build_candidates(client, df, cfg)
        last_rejects = rejects

        # status
        now = time.time()
        if (now - last_status) >= args.status_every_sec:
            last_status = now
            print_status(df, cfg, last_rejects)
            if cands:
                top = cands[: min(6, len(cands))]
                log("TOP (by edge):")
                for c in top:
                    mtc = c["mtc"]
                    log(
                        f"  {c['ticker']} price={c['price']}c ask={c['ask']}c "
                        f"edge={c['edge']:+.3f} (min={c['dyn_min_edge']:.3f}) "
                        f"p={c['p']:.3f} mid={c['mid']:.3f} mtc={None if mtc is None else round(mtc,1)} "
                        f"| {c['title'][:70]}"
                    )
            else:
                log("No candidates passing filters right now.")

        # plan orders
        planned = plan_orders(cands, bankroll, cfg)

        # shadow mode: log would-have
        if cfg.shadow and planned:
            for c, count, cost_usd in planned[: min(3, len(planned))]:
                log(
                    f"WOULD_HAVE_TRADED {c['ticker']} BUY YES {count} @ {c['price']}c "
                    f"cost=${cost_usd:.2f} edge={c['edge']:+.3f} mtc={None if c['mtc'] is None else round(c['mtc'],1)}"
                )

        # execute (only if live enabled)
        mode = "DRY" if args.dry_run or not cfg.live else "LIVE"
        if planned and (mode == "LIVE"):
            for c, count, cost_usd in planned:
                try:
                    resp = place_limit_buy_yes(client, c["ticker"], c["price"], count, tif="ioc")
                    oid = (resp.get("order") or {}).get("order_id")
                    log(f"LIVE ORDER {c['ticker']} YES {count} @ {c['price']}c | cost=${cost_usd:.2f} | order_id={oid}")
                    append_csv_row(args.pos_log, [
                        now_iso(), mode, c["ticker"], c["price"], count,
                        round(c["edge"], 4), round(c["p"], 4), round(c["mid"], 4),
                        None if c["mtc"] is None else round(c["mtc"], 2),
                        c["title"]
                    ])
                except Exception as e:
                    log(f"Order failed for {c['ticker']}: {type(e).__name__}: {e}")

        # log pnl/balance snapshot (simple)
        append_csv_row(args.pnl_log, [now_iso(), mode, round(bankroll, 2)])

        time.sleep(max(1, int(args.poll_sec)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Stopped by user.")
#!/usr/bin/env python3
"""
===========================================================
KALSHI TRUMP-MENTIONS EDGE TRADER (REST AUTH + CONSOLE)
===========================================================

Goal:
- ONLY trade Trump "Will Trump say ..." markets (KXTRUMP...) using model edge.
- Poll REST snapshots, compute live mid, compare to model_p, place limit BUY YES.
- Designed to never look "blank": status, reject reasons, shadow mode.

How to run (paper):
  export KALSHI_API_KEY_ID="..."
  export KALSHI_PRIVATE_KEY_PATH="/full/path/to/kalshi-private.key"
  python3 sniper_trump_mentions.py --edges_csv mispricings_week.csv --dry_run

How to run (live):
  python3 sniper_trump_mentions.py --edges_csv mispricings_week.csv --LIVE

Console commands while running:
  more            -> loosen thresholds
  less            -> tighten thresholds
  shadow on/off   -> log WOULD_HAVE_TRADED candidates even if not trading
  live on/off     -> toggle live trading (safety: starts OFF unless --LIVE)
  status          -> print current constraints + counts
  quit            -> exit

Notes:
- This script uses REST endpoints only (no websocket required).
- Auth/signing follows Kalshi docs: timestamp + METHOD + path_without_query, RSA-PSS SHA256, salt_length=DIGEST_LENGTH.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import select
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


# -----------------------
# Defaults / knobs
# -----------------------

DEFAULT_BASE_URL = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com").rstrip("/")
REST_BASE = DEFAULT_BASE_URL + "/trade-api/v2"

# Hard scope: only these markets
ONLY_TICKER_PREFIX = "KXTRUMP"
REQUIRE_WILL_TRUMP_SAY_PREFIX = True  # set False if you want broader KXTRUMP events

# Safety filters
BAN_BEFORE_STYLE_TITLES = True  # "Will Trump say ... before <date>" are weird bucketed markets
FILTER_ASK_100 = True           # ignore yes_ask == 100 (1.00)

# Dynamic gating based on time-to-close (minutes)
# As markets get closer to close, we loosen min_edge a bit and allow slightly higher prices.
MTC_TIERS = [
    # (max_minutes_to_close, edge_multiplier, max_yes_price_cents_add)
    (60,   0.70,  8),   # within 1h: allow smaller edge, slightly higher prices
    (180,  0.80,  5),   # within 3h
    (720,  0.90,  2),   # within 12h
    (1440, 1.00,  0),   # within 1 day
]

# Base trading constraints (can be adjusted with `more/less`)
BASE_MIN_EDGE = 0.020
BASE_MAX_YES_PRICE_CENTS = 95       # "high nineties" sniping
BASE_MIN_MINUTES_TO_CLOSE = 0       # allow near close (your stated goal)
BASE_SLIPPAGE_CENTS = 1
BASE_MAX_MARKETS = 8

# Spend controls
DEFAULT_PAPER_BANKROLL_USD = 250.0
DEFAULT_MAX_PER_MARKET_USD = 50.0

# Polling & logging
POLL_SEC = 8
STATUS_EVERY_SEC = 10
REJECT_SAMPLE_N = 10

PNL_LOG = "pnl_timeseries.csv"
POS_LOG = "positions_pnl.csv"


# ===========================================================
# Utilities
# ===========================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def nonblocking_readline() -> Optional[str]:
    try:
        r, _, _ = select.select([sys.stdin], [], [], 0.0)
        if r:
            return sys.stdin.readline().strip()
    except Exception:
        return None
    return None

def ensure_csv_header(path: str, header: List[str]) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

def append_csv_row(path: str, row: List) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def clamp_price(cents: int) -> int:
    return max(1, min(99, int(cents)))

def is_before_style_title(title: str) -> bool:
    t = (title or "").lower()
    return t.startswith("will trump say") and " before " in t

def minutes_to_close(close_iso: Optional[str]) -> Optional[float]:
    if not close_iso:
        return None
    s = str(close_iso)
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (dt - datetime.now(timezone.utc)).total_seconds() / 60.0
    except Exception:
        return None

def pick_mtc_tier(mtc_min: Optional[float]) -> Tuple[float, int]:
    """
    Return (edge_multiplier, max_yes_add_cents) based on how close we are.
    """
    if mtc_min is None:
        return 1.0, 0
    for max_m, mult, add in MTC_TIERS:
        if mtc_min <= max_m:
            return mult, add
    return 1.0, 0


# ===========================================================
# Auth (RSA-PSS)
# ===========================================================

def _load_private_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def _sign_pss_base64(private_key, message: str) -> str:
    sig = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")

def build_auth_headers(api_key_id: str, private_key_path: str, method: str, path: str) -> Dict[str, str]:
    api_key_id = (api_key_id or "").strip()
    private_key_path = (private_key_path or "").strip()
    if not api_key_id or not private_key_path:
        raise SystemExit("Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH env vars.")
    if not os.path.exists(private_key_path):
        raise SystemExit(f"Private key file not found: {private_key_path}")

    pk = _load_private_key(private_key_path)
    ts_ms = str(int(time.time() * 1000))

    path_wo_query = path.split("?")[0]
    msg = f"{ts_ms}{method.upper()}{path_wo_query}"
    sig = _sign_pss_base64(pk, msg)

    return {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": sig,
    }

class KalshiClient:
    def __init__(self, base_url_root: str, api_key_id: str, private_key_path: str, timeout: int = 10):
        self.root = base_url_root.rstrip("/")
        self.api_key_id = api_key_id
        self.private_key_path = private_key_path
        self.timeout = timeout
        self.session = requests.Session()

    def _headers(self, method: str, path: str) -> Dict[str, str]:
        return build_auth_headers(self.api_key_id, self.private_key_path, method, path)

    def get(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = self.root + path
        r = self.session.get(url, headers=self._headers("GET", path), params=params, timeout=self.timeout)
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code} {r.reason} for {r.url}\n{(r.text or '')[:800]}", response=r)
        return r.json()

    def post(self, path: str, payload: Dict) -> Dict:
        url = self.root + path
        r = self.session.post(url, headers=self._headers("POST", path), json=payload, timeout=self.timeout)
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code} {r.reason} for {r.url}\n{(r.text or '')[:800]}", response=r)
        return r.json()


# ===========================================================
# Trading / pricing helpers
# ===========================================================

def get_market_snapshot(client: KalshiClient, ticker: str) -> Dict:
    # Kalshi: GET /trade-api/v2/markets/{ticker}
    return client.get(f"/trade-api/v2/markets/{ticker}")

def best_yes_ask_from_snapshot(snap: Dict) -> Optional[int]:
    m = snap.get("market") or snap
    ya = m.get("yes_ask")
    if isinstance(ya, (int, float)) and ya is not None and ya > 0:
        return int(ya)
    return None

def yes_bid_ask_last_mid(snap: Dict) -> Tuple[float, float, float]:
    m = snap.get("market") or snap
    yb = m.get("yes_bid")
    ya = m.get("yes_ask")
    lp = m.get("last_price")

    bid = float(yb) / 100.0 if isinstance(yb, (int, float)) and yb and yb > 0 else 0.0
    last = float(lp) / 100.0 if isinstance(lp, (int, float)) and lp and lp > 0 else bid
    if isinstance(yb, (int, float)) and isinstance(ya, (int, float)) and ya and ya > 0:
        mid = (float(yb) + float(ya)) / 2.0 / 100.0
    else:
        mid = last
    return bid, mid, last

def normalize_tif(tif: str) -> str:
    t = (tif or "").lower().strip()
    mapping = {
        "gtc": "good_till_canceled",
        "good_till_canceled": "good_till_canceled",
        "good_til_cancelled": "good_till_canceled",
        "good_til_canceled": "good_till_canceled",
        "ioc": "immediate_or_cancel",
        "immediate_or_cancel": "immediate_or_cancel",
        "immediate_or_cancelled": "immediate_or_cancel",
        "fok": "fill_or_kill",
        "fill_or_kill": "fill_or_kill",
    }
    return mapping.get(t, "good_till_canceled")

def client_order_id() -> str:
    return f"trumpedge-{int(time.time()*1000)}"

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


# ===========================================================
# Core planner
# ===========================================================

@dataclass
class DynConfig:
    min_edge: float = BASE_MIN_EDGE
    max_yes_price_cents: int = BASE_MAX_YES_PRICE_CENTS
    min_minutes_to_close: int = BASE_MIN_MINUTES_TO_CLOSE
    slippage_cents: int = BASE_SLIPPAGE_CENTS
    max_markets: int = BASE_MAX_MARKETS

    shadow: bool = True
    live: bool = False

    paper_bankroll_usd: float = DEFAULT_PAPER_BANKROLL_USD
    max_per_market_usd: float = DEFAULT_MAX_PER_MARKET_USD

def load_edges_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    REQUIRED_COLS = ["market_ticker", "market_title", "model_p", "market_p", "edge", "close_time"]
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise SystemExit(f"edges_csv missing required column: {c}")

    # Hard scope filters
    df["market_ticker"] = df["market_ticker"].astype(str)
    df["market_title"] = df["market_title"].astype(str)

    df = df[df["market_ticker"].str.startswith(ONLY_TICKER_PREFIX)].copy()

    if REQUIRE_WILL_TRUMP_SAY_PREFIX:
        df = df[df["market_title"].str.lower().str.startswith("will trump say")].copy()

    if BAN_BEFORE_STYLE_TITLES:
        df = df[~df["market_title"].apply(is_before_style_title)].copy()

    # clean numeric
    df["model_p"] = pd.to_numeric(df["model_p"], errors="coerce")
    df["market_p"] = pd.to_numeric(df["market_p"], errors="coerce")
    df["edge"] = pd.to_numeric(df["edge"], errors="coerce")

    df = df.dropna(subset=["market_ticker", "market_title", "model_p", "market_p", "edge"]).reset_index(drop=True)
    return df

def get_portfolio_balance_usd(client: KalshiClient) -> Optional[float]:
    try:
        j = client.get("/trade-api/v2/portfolio/balance")
        bal = (j.get("balance") or {}).get("available_cash")
        if isinstance(bal, (int, float)):
            return float(bal) / 100.0
    except Exception:
        return None
    return None

def build_candidates(client: KalshiClient, df: pd.DataFrame, cfg: DynConfig) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Returns (candidates_sorted, reject_reason_counts)
    Each candidate includes:
      ticker,title,p,q,edge,price,mtc,limit_price
    """
    rejects: Dict[str, int] = {}
    cands: List[Dict] = []

    def rej(r: str) -> None:
        rejects[r] = rejects.get(r, 0) + 1

    for _, r in df.iterrows():
        ticker = str(r["market_ticker"])
        title = str(r["market_title"])
        p_model = float(r["model_p"])
        close_time = r.get("close_time")

        mtc = minutes_to_close(close_time)
        if mtc is not None and mtc < cfg.min_minutes_to_close:
            rej("mtc_too_small")
            continue

        # dynamic loosening as we get close
        edge_mult, max_yes_add = pick_mtc_tier(mtc)
        dyn_min_edge = cfg.min_edge * edge_mult
        dyn_max_yes = cfg.max_yes_price_cents + max_yes_add

        # snapshot
        try:
            snap = get_market_snapshot(client, ticker)
        except Exception:
            rej("snapshot_fail")
            continue

        ask = best_yes_ask_from_snapshot(snap)
        if ask is None:
            rej("no_ask")
            continue

        if FILTER_ASK_100 and int(ask) >= 100:
            rej("ask_is_100")
            continue

        limit_price = clamp_price(int(ask) + cfg.slippage_cents)
        if limit_price > dyn_max_yes:
            rej("price_too_high")
            continue

        bid, mid, _last = yes_bid_ask_last_mid(snap)
        q = mid if mid > 0 else float(r["market_p"])
        edge = p_model - q
        if edge < dyn_min_edge:
            rej("edge_too_small")
            continue

        cands.append({
            "ticker": ticker,
            "title": title,
            "p": p_model,
            "q": q,
            "edge": edge,
            "price": limit_price,
            "ask": int(ask),
            "bid": int(round(bid * 100)),
            "mid": q,
            "mtc": mtc,
            "dyn_min_edge": dyn_min_edge,
            "dyn_max_yes": dyn_max_yes,
        })

    cands.sort(key=lambda x: (x["edge"]), reverse=True)
    return cands[: max(1, cfg.max_markets * 3)], rejects

def plan_orders(cands: List[Dict], bankroll_usd: float, cfg: DynConfig) -> List[Tuple[Dict, int, float]]:
    """
    Simple sizing:
    - equal-weight budgets capped per market
    - buys YES with count = floor(budget / price)
    Returns list of (candidate, count, cost_usd)
    """
    if not cands:
        return []

    # cap number of markets
    cands = cands[: cfg.max_markets]

    per_mkt_budget = min(cfg.max_per_market_usd, bankroll_usd / max(1, len(cands)))
    orders = []
    for c in cands:
        price_cents = int(c["price"])
        if price_cents <= 0:
            continue
        budget_cents = int(round(per_mkt_budget * 100))
        count = budget_cents // price_cents
        if count <= 0:
            continue
        cost_cents = count * price_cents
        orders.append((c, int(count), cost_cents / 100.0))

    # prioritize higher edge if budgets end up small
    orders.sort(key=lambda x: x[0]["edge"], reverse=True)
    return orders

def print_status(df: pd.DataFrame, cfg: DynConfig, last_rejects: Dict[str, int]) -> None:
    log(
        f"STATUS | universe={len(df)} "
        f"| min_edge={cfg.min_edge:.3f} max_yes={cfg.max_yes_price_cents}c slippage={cfg.slippage_cents}c "
        f"| max_markets={cfg.max_markets} shadow={cfg.shadow} live={cfg.live}"
    )
    if last_rejects:
        top = sorted(last_rejects.items(), key=lambda kv: kv[1], reverse=True)[:REJECT_SAMPLE_N]
        log("REJECTS (sample): " + ", ".join([f"{k}:{v}" for k, v in top]))

def handle_command(cmd: str, cfg: DynConfig) -> bool:
    """
    Returns True if should quit.
    """
    c = (cmd or "").strip().lower()
    if not c:
        return False

    if c in ("quit", "exit", "q"):
        return True

    if c == "more":
        # Loosen: smaller required edge, allow slightly higher price, scan more markets
        cfg.min_edge = max(0.003, cfg.min_edge * 0.85)
        cfg.max_yes_price_cents = min(99, cfg.max_yes_price_cents + 2)
        cfg.max_markets = min(20, cfg.max_markets + 1)
        log(f"MORE -> min_edge={cfg.min_edge:.3f}, max_yes={cfg.max_yes_price_cents}c, max_markets={cfg.max_markets}")
        return False

    if c == "less":
        # Tighten: larger required edge, lower max price, fewer markets
        cfg.min_edge = min(0.20, cfg.min_edge * 1.15)
        cfg.max_yes_price_cents = max(50, cfg.max_yes_price_cents - 2)
        cfg.max_markets = max(2, cfg.max_markets - 1)
        log(f"LESS -> min_edge={cfg.min_edge:.3f}, max_yes={cfg.max_yes_price_cents}c, max_markets={cfg.max_markets}")
        return False

    if c.startswith("shadow"):
        if "on" in c:
            cfg.shadow = True
        elif "off" in c:
            cfg.shadow = False
        log(f"shadow={cfg.shadow}")
        return False

    if c.startswith("live"):
        if "on" in c:
            cfg.live = True
        elif "off" in c:
            cfg.live = False
        log(f"live={cfg.live}")
        return False

    if c == "status":
        log("Type `more`, `less`, `shadow on/off`, `live on/off`, `quit`.")
        return False

    log(f"Unknown command: {cmd} (try: more/less/shadow on/off/live on/off/status/quit)")
    return False


# ===========================================================
# Main loop
# ===========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges_csv", required=True, help="mispricings_week.csv / mispricings_2026-01.csv etc")
    ap.add_argument("--base_url", default=DEFAULT_BASE_URL)

    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--LIVE", action="store_true")

    # override knobs if desired
    ap.add_argument("--min_edge", type=float, default=BASE_MIN_EDGE)
    ap.add_argument("--max_yes_price_cents", type=int, default=BASE_MAX_YES_PRICE_CENTS)
    ap.add_argument("--min_minutes_to_close", type=int, default=BASE_MIN_MINUTES_TO_CLOSE)
    ap.add_argument("--slippage_cents", type=int, default=BASE_SLIPPAGE_CENTS)
    ap.add_argument("--max_markets", type=int, default=BASE_MAX_MARKETS)

    ap.add_argument("--paper_bankroll_usd", type=float, default=DEFAULT_PAPER_BANKROLL_USD)
    ap.add_argument("--max_per_market_usd", type=float, default=DEFAULT_MAX_PER_MARKET_USD)

    ap.add_argument("--poll_sec", type=int, default=POLL_SEC)
    ap.add_argument("--status_every_sec", type=int, default=STATUS_EVERY_SEC)
    ap.add_argument("--pnl_log", default=PNL_LOG)
    ap.add_argument("--pos_log", default=POS_LOG)
    args = ap.parse_args()

    api_key_id = (os.getenv("KALSHI_API_KEY_ID", "") or "").strip()
    priv_key = (os.getenv("KALSHI_PRIVATE_KEY_PATH", "") or "").strip()
    if not api_key_id or not priv_key:
        raise SystemExit("Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH env vars.")

    if args.LIVE and args.dry_run:
        raise SystemExit("Pick one: --dry_run OR --LIVE")
    if not args.LIVE:
        args.dry_run = True

    cfg = DynConfig(
        min_edge=float(args.min_edge),
        max_yes_price_cents=int(args.max_yes_price_cents),
        min_minutes_to_close=int(args.min_minutes_to_close),
        slippage_cents=int(args.slippage_cents),
        max_markets=int(args.max_markets),
        shadow=True,
        live=bool(args.LIVE),
        paper_bankroll_usd=float(args.paper_bankroll_usd),
        max_per_market_usd=float(args.max_per_market_usd),
    )

    client = KalshiClient(args.base_url, api_key_id, priv_key)

    # smoke test auth
    try:
        _ = client.get("/trade-api/v2/communications/id")
        log("✅ Auth OK.")
    except Exception as e:
        raise SystemExit(f"Auth failed: {e}")

    # load edges universe
    df = load_edges_df(args.edges_csv)
    if df.empty:
        raise SystemExit("No rows after Trump-market filters. Check that your edges CSV contains KXTRUMP markets.")

    log(f"Loaded edges universe: {len(df)} rows (ticker prefix={ONLY_TICKER_PREFIX})")
    log("Type commands anytime: more / less / shadow on/off / live on/off / status / quit")

    ensure_csv_header(args.pnl_log, ["ts_iso", "mode", "balance_usd"])
    ensure_csv_header(args.pos_log, ["ts_iso", "mode", "ticker", "price_cents", "count", "edge", "p_model", "mid", "mtc_min", "title"])

    last_status = 0.0
    last_rejects: Dict[str, int] = {}

    while True:
        # commands
        cmd = nonblocking_readline()
        if cmd:
            if handle_command(cmd, cfg):
                log("Stopping.")
                break

        # bankroll
        if cfg.live and not args.dry_run:
            bal = get_portfolio_balance_usd(client)
            bankroll = bal if (bal is not None) else cfg.paper_bankroll_usd
        else:
            bankroll = cfg.paper_bankroll_usd

        # build candidates from live snapshots
        cands, rejects = build_candidates(client, df, cfg)
        last_rejects = rejects

        # status
        now = time.time()
        if (now - last_status) >= args.status_every_sec:
            last_status = now
            print_status(df, cfg, last_rejects)
            if cands:
                top = cands[: min(6, len(cands))]
                log("TOP (by edge):")
                for c in top:
                    mtc = c["mtc"]
                    log(
                        f"  {c['ticker']} price={c['price']}c ask={c['ask']}c "
                        f"edge={c['edge']:+.3f} (min={c['dyn_min_edge']:.3f}) "
                        f"p={c['p']:.3f} mid={c['mid']:.3f} mtc={None if mtc is None else round(mtc,1)} "
                        f"| {c['title'][:70]}"
                    )
            else:
                log("No candidates passing filters right now.")

        # plan orders
        planned = plan_orders(cands, bankroll, cfg)

        # shadow mode: log would-have
        if cfg.shadow and planned:
            for c, count, cost_usd in planned[: min(3, len(planned))]:
                log(
                    f"WOULD_HAVE_TRADED {c['ticker']} BUY YES {count} @ {c['price']}c "
                    f"cost=${cost_usd:.2f} edge={c['edge']:+.3f} mtc={None if c['mtc'] is None else round(c['mtc'],1)}"
                )

        # execute (only if live enabled)
        mode = "DRY" if args.dry_run or not cfg.live else "LIVE"
        if planned and (mode == "LIVE"):
            for c, count, cost_usd in planned:
                try:
                    resp = place_limit_buy_yes(client, c["ticker"], c["price"], count, tif="ioc")
                    oid = (resp.get("order") or {}).get("order_id")
                    log(f"LIVE ORDER {c['ticker']} YES {count} @ {c['price']}c | cost=${cost_usd:.2f} | order_id={oid}")
                    append_csv_row(args.pos_log, [
                        now_iso(), mode, c["ticker"], c["price"], count,
                        round(c["edge"], 4), round(c["p"], 4), round(c["mid"], 4),
                        None if c["mtc"] is None else round(c["mtc"], 2),
                        c["title"]
                    ])
                except Exception as e:
                    log(f"Order failed for {c['ticker']}: {type(e).__name__}: {e}")

        # log pnl/balance snapshot (simple)
        append_csv_row(args.pnl_log, [now_iso(), mode, round(bankroll, 2)])

        time.sleep(max(1, int(args.poll_sec)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Stopped by user.")

