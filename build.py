"""Theme Monitor: pulls prices, computes per-stock metrics, writes docs/index.html.

Run:  python build.py          (live data via yfinance)
      python build.py --demo   (synthetic prices, for testing the page)
"""
import csv, json, sys, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
BENCHMARKS = ["SPY", "QQQ"]
# Leader score blend (edit to taste; must sum to 1)
WEIGHTS = {"r1w": 0.2, "r1m": 0.5, "r3m": 0.3}
SPARK_DAYS = 63  # ~3 months


def load_universe():
    with open(ROOT / "data" / "tickers.csv", newline="") as f:
        return list(csv.DictReader(f))


def fetch(tickers, demo=False):
    if demo:
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=320)
        rng = np.random.default_rng(7)
        drift = rng.normal(0.0005, 0.001, len(tickers))
        rets = rng.normal(drift, 0.025, (len(idx), len(tickers)))
        close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=tickers)
        vol = pd.DataFrame(rng.integers(1e5, 5e6, (len(idx), len(tickers))), index=idx, columns=tickers)
        return close, vol
    import yfinance as yf
    df = yf.download(tickers, period="15mo", auto_adjust=True, progress=False, threads=True)
    return df["Close"], df["Volume"]


def pct(a, b):
    return None if (b is None or pd.isna(a) or pd.isna(b) or b == 0) else round((a / b - 1) * 100, 2)


def metrics(s, v):
    s = s.dropna()
    if len(s) < 30:
        return None
    last = s.iloc[-1]
    def back(n):
        return s.iloc[-1 - n] if len(s) > n else np.nan
    prior_year = s[s.index.year < s.index[-1].year]
    ytd_base = prior_year.iloc[-1] if len(prior_year) else s.iloc[0]
    hi52 = s.iloc[-252:].max()
    vv = v.reindex(s.index).dropna()
    relvol = round(float(vv.iloc[-1] / vv.iloc[-21:-1].mean()), 2) if len(vv) > 21 and vv.iloc[-21:-1].mean() > 0 else None
    spark = s.iloc[-SPARK_DAYS:]
    return {
        "close": round(float(last), 2),
        "r1d": pct(last, back(1)), "r1w": pct(last, back(5)), "r1m": pct(last, back(21)),
        "r3m": pct(last, back(63)), "ytd": pct(last, ytd_base),
        "a50": bool(last > s.iloc[-50:].mean()) if len(s) >= 50 else None,
        "a200": bool(last > s.iloc[-200:].mean()) if len(s) >= 200 else None,
        "off_hi": pct(last, hi52),
        "relvol": relvol,
        "spark": [round(float(x / spark.iloc[0]), 4) for x in spark],
    }


def main():
    demo = "--demo" in sys.argv
    uni = load_universe()
    tickers = sorted({u["ticker"] for u in uni} | set(BENCHMARKS))
    close, vol = fetch(tickers, demo)
    as_of = close.dropna(how="all").index[-1].strftime("%Y-%m-%d")

    cache, missing = {}, []
    for t in tickers:
        m = metrics(close[t], vol[t]) if t in close else None
        if m is None:
            missing.append(t)
        cache[t] = m

    stocks = []
    for u in uni:
        m = cache.get(u["ticker"])
        if m:
            stocks.append({**u, **m})
    bench = {b: {k: cache[b][k] for k in ("r1d", "r1w", "r1m", "r3m", "ytd", "spark")} for b in BENCHMARKS if cache.get(b)}

    payload = {
        "as_of": as_of,
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "weights": WEIGHTS, "bench": bench, "stocks": stocks,
        "missing": sorted(set(missing) - set(BENCHMARKS)),
        "demo": demo,
    }

    # daily snapshot for future trend work
    hist = ROOT / "data" / "history"
    hist.mkdir(exist_ok=True)
    pd.DataFrame([{k: s[k] for k in ("theme", "ticker", "close", "r1d", "r1w", "r1m", "r3m", "ytd")} for s in stocks]) \
        .to_csv(hist / f"{as_of}.csv", index=False)

    tpl = (ROOT / "template.html").read_text()
    html = tpl.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    (ROOT / "docs" / "index.html").write_text(html)
    print(f"as_of {as_of}: {len(stocks)} rows, {len(missing)} missing: {missing}")


if __name__ == "__main__":
    main()
