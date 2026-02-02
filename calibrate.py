#!/usr/bin/env python3
import json, math, argparse
import pandas as pd
from datetime import datetime, timezone

def logit(p):
    p = max(1e-6, min(1-1e-6, float(p)))
    return math.log(p/(1-p))

def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1/(1+ez)
    ez = math.exp(z)
    return ez/(1+ez)

def train_sgd(X, y, lr=0.05, l2=1e-3, epochs=40):
    d = len(X[0])
    w = [0.0]*d
    b = 0.0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            z = b + sum(wj*xj for wj,xj in zip(w, xi))
            p = sigmoid(z)
            err = (p - yi)
            b -= lr * err
            for j in range(d):
                w[j] -= lr * (err*xi[j] + l2*w[j])
    return w, b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market_csv", default="market_timeseries.csv")
    ap.add_argument("--model_csv", default="model_timeseries.csv")
    ap.add_argument("--resolved_csv", default="resolved_outcomes.csv")
    ap.add_argument("--out_params", default="calibration_params.json")
    args = ap.parse_args()

    mkt = pd.read_csv(args.market_csv)
    mdl = pd.read_csv(args.model_csv)
    res = pd.read_csv(args.resolved_csv)

    mkt["ts"] = pd.to_datetime(mkt["ts_iso"], utc=True, errors="coerce")
    mdl["ts"] = pd.to_datetime(mdl["ts_iso"], utc=True, errors="coerce")
    res["ts"] = pd.to_datetime(res["ts_iso"], utc=True, errors="coerce")

    mkt = mkt.dropna(subset=["ts","ticker"]).sort_values(["ticker","ts"])
    mdl = mdl.dropna(subset=["ts","ticker","p_model"]).sort_values(["ticker","ts"])
    res = res.dropna(subset=["ts","ticker","outcome_yes"])

    if mkt.empty or mdl.empty or res.empty:
        raise SystemExit("Need non-empty market_timeseries, model_timeseries, and resolved_outcomes CSVs.")

    # Merge model into market with asof per ticker
    merged = []
    for ticker, g in mkt.groupby("ticker"):
        gm = mdl[mdl["ticker"] == ticker]
        if gm.empty:
            continue
        g = g.sort_values("ts")
        gm = gm.sort_values("ts")
        j = pd.merge_asof(g, gm, on="ts", direction="backward", tolerance=pd.Timedelta("7D"))
        j["ticker"] = ticker
        merged.append(j)

    if not merged:
        raise SystemExit("No overlaps between market_timeseries and model_timeseries.")
    df = pd.concat(merged, ignore_index=True)

    # Compute 1h momentum per ticker from mid_prob
    df["mid_prob"] = pd.to_numeric(df["mid_prob"], errors="coerce")
    df = df.dropna(subset=["mid_prob","p_model"])
    df = df.sort_values(["ticker","ts"])

    # Create lagged mid_prob 1h ago (approx: last sample <= ts-1h)
    df["ts_minus_1h"] = df["ts"] - pd.Timedelta("1H")
    mom = []
    for ticker, g in df.groupby("ticker"):
        g2 = g[["ts","mid_prob"]].rename(columns={"ts":"ts_lag","mid_prob":"mid_lag"}).sort_values("ts_lag")
        g = g.sort_values("ts")
        j = pd.merge_asof(
            g.sort_values("ts_minus_1h"),
            g2,
            left_on="ts_minus_1h",
            right_on="ts_lag",
            direction="backward"
        )
        m = (j["mid_prob"] - j["mid_lag"]).fillna(0.0)
        mom.append(m)
    df["momentum_1h"] = pd.concat(mom, ignore_index=True)

    # Build examples: for each resolved ticker, take last row before resolution ts
    X, y = [], []
    for _, rr in res.iterrows():
        ticker = rr["ticker"]
        y_i = int(rr["outcome_yes"])
        t_res = rr["ts"]
        hist = df[(df["ticker"] == ticker) & (df["ts"] <= t_res)].sort_values("ts")
        if hist.empty:
            continue
        row = hist.iloc[-1]

        mid = float(row["mid_prob"])
        p_model = float(row["p_model"])
        spread = float(row["spread"]) if not pd.isna(row.get("spread")) else 0.0
        volume = float(row["volume"]) if not pd.isna(row.get("volume")) else 0.0
        mom1h = float(row["momentum_1h"]) if not pd.isna(row.get("momentum_1h")) else 0.0

        # tau_days from close_time if present
        tau_days = 0.0
        ct = row.get("close_time")
        if isinstance(ct, str) and ct:
            try:
                s = ct
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                close_dt = datetime.fromisoformat(s)
                if close_dt.tzinfo is None:
                    close_dt = close_dt.replace(tzinfo=timezone.utc)
                tau = close_dt - row["ts"].to_pydatetime()
                tau_days = max(0.0, tau.total_seconds()/86400.0)
            except Exception:
                tau_days = 0.0

        xi = [
            1.0,               # const feature
            logit(p_model),
            logit(mid),
            tau_days,
            spread,
            math.log1p(volume),
            mom1h,
        ]
        X.append(xi)
        y.append(y_i)

    if len(X) == 0:
        raise SystemExit("No usable resolved examples (need markets to resolve first).")

    w, b = train_sgd(X, y, lr=0.05, l2=1e-3, epochs=50)

    params = {
        "features": ["const","logit_p_model","logit_mid","tau_days","spread","log1p_volume","momentum_1h"],
        "w": w,
        "b": b,
        "trained_on": len(X),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(args.out_params, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print(f"Saved {args.out_params} trained_on={len(X)}")

if __name__ == "__main__":
    main()

