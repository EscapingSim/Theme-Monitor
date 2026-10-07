"""Theme Monitor nightly build.

Pages written to docs/:
  index.html   - theme leaderboard (your theme universe, data/tickers.csv)
  setups.html  - Summary, Minervini, Zanger, Qullamaggie and Kell screens (broad market)

Run:  python build.py          (live data via yfinance)
      python build.py --demo   (synthetic prices, for testing the pages)
"""
import csv, io, json, sys, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
BENCHMARKS = ["SPY", "QQQ"]

# ---- settings you can edit ----
WEIGHTS = {"r1w": 0.2, "r1m": 0.5, "r3m": 0.3}   # theme/leader score blend
SPARK_DAYS = 63
MIN_PRICE = 10            # screens: ignore stocks under this price
MIN_DOLLAR_VOL = 20e6     # screens: 50-day avg $ volume floor
LEADING_THEMES = 5        # top-N themes count as "leading"
CHUNK = 150               # tickers per download batch
Q_MIN_ADR = 4.0           # Qullamaggie: min 20-day average daily range %
EP_MIN_GAP = 10.0         # Qullamaggie: episodic pivot min gap-up %
EP_MIN_RELVOL = 3.0       # Qullamaggie: episodic pivot min relative volume
# -------------------------------

WIKI = {
    "S&P 500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol", "Security", "GICS Sector"),
    "S&P 400": ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Symbol", "Security", "GICS Sector"),
    "S&P 600": ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "Symbol", "Security", "GICS Sector"),
}


def load_themes():
    with open(DATA / "tickers.csv", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("ticker")]


def load_broad(demo):
    """S&P 1500 (from Wikipedia) + optional data/extra_universe.csv. Cached for resilience."""
    cache = DATA / "universe_cache.csv"
    rows = {}
    if demo:
        for i in range(400):
            rows[f"DEMO{i:03d}"] = ("Demo Company %d" % i, "Demo")
    else:
        import requests
        ok = True
        for name, (url, sym, sec, gics) in WIKI.items():
            try:
                html = requests.get(url, headers={"User-Agent": "theme-monitor/1.0 (personal dashboard)"}, timeout=30).text
                tbl = next(t for t in pd.read_html(io.StringIO(html)) if sym in t.columns)
                for _, r in tbl.iterrows():
                    rows[str(r[sym]).replace(".", "-").strip()] = (str(r[sec]), str(r.get(gics, "")))
            except Exception as e:
                ok = False
                print(f"universe fetch failed for {name}: {e}")
        if ok and rows:
            pd.DataFrame([(k, *v) for k, v in rows.items()], columns=["ticker", "name", "sector"]).to_csv(cache, index=False)
        elif cache.exists():
            print("using cached universe")
            for _, r in pd.read_csv(cache).iterrows():
                rows.setdefault(r["ticker"], (r["name"], r["sector"]))
    extra = DATA / "extra_universe.csv"
    if extra.exists():
        for _, r in pd.read_csv(extra).iterrows():
            rows.setdefault(str(r["ticker"]).strip(), (str(r.get("name", "")), str(r.get("sector", ""))))
    return rows


def fetch(tickers, demo=False):
    if demo:
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=320)
        rng = np.random.default_rng(7)
        n = len(tickers)
        drift = rng.normal(0.0006, 0.0012, n)
        rets = rng.normal(drift, 0.022, (len(idx), n))
        c = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=tickers)
        spread = np.abs(rng.normal(0.012, 0.006, c.shape))
        h, l = c * (1 + spread), c * (1 - spread)
        v = pd.DataFrame(rng.integers(3e5, 8e6, c.shape), index=idx, columns=tickers).astype(float)
        o = c.shift(1) * (1 + rng.normal(0, 0.005, c.shape))
        gap = rng.random(n) < 0.02                      # a few demo gap-ups for EP testing
        o.iloc[-1, gap] = c.iloc[-2, gap] * 1.15
        c.iloc[-1, gap] = c.iloc[-2, gap] * 1.18; h.iloc[-1, gap] = c.iloc[-1, gap] * 1.01
        v.iloc[-1, gap] = v.iloc[-1, gap] * 6
        return {"Open": o, "Close": c, "High": h, "Low": l, "Volume": v}
    import yfinance as yf
    parts = {k: [] for k in ("Open", "Close", "High", "Low", "Volume")}
    for i in range(0, len(tickers), CHUNK):
        batch = tickers[i:i + CHUNK]
        try:
            df = yf.download(batch, period="15mo", auto_adjust=True, progress=False, threads=True, group_by="column")
            for k in parts:
                x = df[k]
                if isinstance(x, pd.Series):
                    x = x.to_frame(batch[0])
                parts[k].append(x)
        except Exception as e:
            print(f"batch {i} failed: {e}")
    return {k: pd.concat(v, axis=1) for k, v in parts.items()}


def pct(a, b):
    return None if (b is None or pd.isna(a) or pd.isna(b) or b == 0) else round((a / b - 1) * 100, 2)


def theme_metrics(s, v):
    s = s.dropna()
    if len(s) < 30:
        return None
    last = s.iloc[-1]
    back = lambda n: s.iloc[-1 - n] if len(s) > n else np.nan
    prior = s[s.index.year < s.index[-1].year]
    ytd_base = prior.iloc[-1] if len(prior) else s.iloc[0]
    vv = v.reindex(s.index).dropna()
    base = vv.iloc[-21:-1].mean() if len(vv) > 21 else 0
    spark = s.iloc[-SPARK_DAYS:]
    return {
        "close": round(float(last), 2),
        "r1d": pct(last, back(1)), "r1w": pct(last, back(5)), "r1m": pct(last, back(21)),
        "r3m": pct(last, back(63)), "ytd": pct(last, ytd_base),
        "a50": bool(last > s.iloc[-50:].mean()) if len(s) >= 50 else None,
        "a200": bool(last > s.iloc[-200:].mean()) if len(s) >= 200 else None,
        "off_hi": pct(last, s.iloc[-252:].max()),
        "relvol": round(float(vv.iloc[-1] / base), 2) if base > 0 else None,
        "spark": [round(float(x / spark.iloc[0]), 4) for x in spark],
    }


TT_LABELS = ["Above 150D & 200D", "150D > 200D", "200D rising 1M", "50D > 150D & 200D",
             "Above 50D", "30%+ above 52W low", "Within 25% of 52W high", "RS 70+"]


def screen_metrics(o, c, h, l, v):
    """Raw per-stock values for all screens (RS rating filled in later)."""
    c = c.dropna()
    if len(c) < 253:
        return None
    o, h, l, v = o.reindex(c.index), h.reindex(c.index), l.reindex(c.index), v.reindex(c.index).fillna(0)
    px = float(c.iloc[-1])
    m = lambda n, off=0: float(c.iloc[len(c) - n - off:len(c) - off].mean())
    s50, s150, s200 = m(50), m(150), m(200)
    s200_1m, s50_prev = m(200, 21), m(50, 1)
    lo52, hi52 = float(l.iloc[-252:].min()), float(h.iloc[-252:].max())
    avgv = float(v.iloc[-51:-1].mean())
    dvol = float((c * v).iloc[-50:].mean())
    roc = lambda n: px / float(c.iloc[-1 - n]) - 1
    rs_raw = 0.4 * roc(63) + 0.2 * roc(126) + 0.2 * roc(189) + 0.2 * roc(252)

    pivot = float(h.iloc[-51:-1].max())                      # prior 50-day high
    segs = [h.iloc[-60 + 15 * i:-60 + 15 * (i + 1) or None] for i in range(4)]
    lsegs = [l.iloc[-60 + 15 * i:-60 + 15 * (i + 1) or None] for i in range(4)]
    depths = [float((a.max() - b.min()) / a.max()) for a, b in zip(segs, lsegs)]
    contractions = 0                                          # consecutive shrinking pullbacks ending now
    for i in range(3, 0, -1):
        if depths[i] < depths[i - 1]:
            contractions += 1
        else:
            break
    tight10 = float((h.iloc[-10:].max() - l.iloc[-10:].min()) / px)
    dryup = float(v.iloc[-10:].mean() / avgv) if avgv else None
    relvol = float(v.iloc[-1] / avgv) if avgv else None
    r15 = roc(15)

    prev = float(c.iloc[-2])
    # ---- Qullamaggie ----
    s10, s20 = m(10), m(20)
    adr = float(((h / l) - 1).iloc[-20:].mean() * 100)
    r1m, r3m, r6m = roc(21), roc(63), roc(126)
    hi20 = float(h.iloc[-21:-1].max())                         # prior 20-day high (consolidation high)
    hi10 = float(h.iloc[-11:-1].max())
    rng10 = float((h.iloc[-10:].max() - l.iloc[-10:].min()) / px * 100)
    higher_lows = float(l.iloc[-5:].min()) >= float(l.iloc[-10:-5].min())
    q_leader = r1m >= 0.25 or r3m >= 0.50 or r6m >= 1.0
    q_trend = px > s10 and px > s20 and s10 > s20 > s50
    q_setup = q_leader and q_trend and adr >= Q_MIN_ADR and rng10 <= 3 * adr and higher_lows and 0 <= (hi20 / px - 1) <= 0.05
    q_brk = q_leader and adr >= Q_MIN_ADR and px > hi10 and prev <= hi10 and relvol is not None and relvol >= 1.5 and s10 > s20
    gap = (float(o.iloc[-1]) / prev - 1) * 100 if not pd.isna(o.iloc[-1]) else 0.0
    ep = gap >= EP_MIN_GAP and relvol is not None and relvol >= EP_MIN_RELVOL and px >= float(o.iloc[-1]) * 0.98
    neglected = (prev / float(c.iloc[-64]) - 1) < 0.30         # not already extended before the gap
    # ---- Oliver Kell (10/20 EMA price cycle) ----
    e10, e20 = c.ewm(span=10, adjust=False).mean(), c.ewm(span=20, adjust=False).mean()
    E10, E20 = float(e10.iloc[-1]), float(e20.iloc[-1])
    spread = abs(E10 - E20) / px * 100
    below5 = int((c.iloc[-6:-1] < e10.iloc[-6:-1]).sum())
    k_pop = px > E10 and px > E20 and below5 >= 3 and spread <= 3 and (px / prev - 1) >= 0.02 and relvol is not None and relvol >= 1.2
    s50_10 = m(50, 10)
    k_up = E10 > E20 and px > s50 and s50 > s50_10
    k_cross = k_up and float(l.iloc[-1]) <= E10 * 1.005 and px >= E20 and float((c.iloc[-10:-1] / e20.iloc[-10:-1]).max()) >= 1.05
    hi30 = float(h.iloc[-31:-1].max()); lo30 = float(l.iloc[-31:-1].min())
    k_bnb = k_up and px > hi30 and prev <= hi30 and (hi30 / lo30 - 1) <= 0.25 and relvol is not None and relvol >= 1.4
    k_exh = px >= E10 * (1 + 2.5 * adr / 100) and roc(5) >= 0.15
    k_drop = px < E10 and px < E20 and prev >= float(e20.iloc[-2]) and float(e10.iloc[-2]) > float(e20.iloc[-2])

    tt = [px > s150 and px > s200, s150 > s200, s200 > s200_1m, s50 > s150 and s50 > s200,
          px > s50, px >= 1.3 * lo52, px >= 0.75 * hi52]          # 8th (RS) added later
    return {
        "close": round(px, 2), "rs_raw": rs_raw, "tt": tt,
        "r1d": round((px / float(c.iloc[-2]) - 1) * 100, 2),
        "off_hi": round((px / hi52 - 1) * 100, 1), "above_lo": round((px / lo52 - 1) * 100, 1),
        "ext50": round((px / s50 - 1) * 100, 1),
        "pivot": round(pivot, 2), "to_pivot": round((pivot / px - 1) * 100, 1),
        "contr": int(contractions), "depths": [round(d * 100, 1) for d in depths],
        "tight": round(tight10 * 100, 1), "dryup": None if dryup is None else round(dryup, 2),
        "relvol": None if relvol is None else round(relvol, 2),
        "dvol": round(dvol / 1e6, 1),
        "liquid": px >= MIN_PRICE and dvol >= MIN_DOLLAR_VOL,
        "breakout": px > pivot and float(c.iloc[-2]) <= pivot and relvol is not None and relvol >= 1.5,
        "near": 0 <= (pivot / px - 1) <= 0.05,
        "above50": px > s50,
        "climax": (px / s50 - 1) >= 0.25 and r15 >= 0.20,
        "break50": px < s50 and float(c.iloc[-2]) >= s50_prev,
        "adr": round(adr, 1), "r1m": round(r1m * 100, 1), "r3m": round(r3m * 100, 1), "r6m": round(r6m * 100, 1),
        "rng10": round(rng10, 1), "to_hi20": round((hi20 / px - 1) * 100, 1), "gap": round(gap, 1),
        "q_setup": bool(q_setup), "q_brk": bool(q_brk), "ep": bool(ep), "neglected": bool(neglected),
        "ext10": round((px / E10 - 1) * 100, 1), "spread": round(spread, 1),
        "k_pop": bool(k_pop), "k_cross": bool(k_cross), "k_bnb": bool(k_bnb), "k_exh": bool(k_exh), "k_drop": bool(k_drop),
    }


def main():
    demo = "--demo" in sys.argv
    themes = load_themes()
    broad = load_broad(demo)
    theme_tk = {u["ticker"] for u in themes}
    tickers = sorted(theme_tk | set(broad) | set(BENCHMARKS))
    px = fetch(tickers, demo)
    close, vol = px["Close"], px["Volume"]
    as_of = close.dropna(how="all").index[-1].strftime("%Y-%m-%d")
    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ---------- theme page ----------
    tm, missing = {}, []
    for t in sorted(theme_tk | set(BENCHMARKS)):
        tm[t] = theme_metrics(close[t], vol[t]) if t in close else None
        if tm[t] is None and t not in BENCHMARKS:
            missing.append(t)
    stocks = [{**u, **tm[u["ticker"]]} for u in themes if tm.get(u["ticker"])]
    bench = {b: {k: tm[b][k] for k in ("r1d", "r1w", "r1m", "r3m", "ytd", "spark")} for b in BENCHMARKS if tm.get(b)}
    write("template.html", "index.html", {"as_of": as_of, "generated": generated, "weights": WEIGHTS,
          "bench": bench, "stocks": stocks, "missing": sorted(missing), "demo": demo})

    # leading themes (equal-weight, all exposures)
    def blend(d):
        vals = [(d[k], w) for k, w in WEIGHTS.items() if d.get(k) is not None]
        return sum(a * w for a, w in vals) / sum(w for _, w in vals) if vals else None
    tscore = {}
    for s in stocks:
        b = blend(s)
        if b is not None:
            tscore.setdefault(s["theme"], []).append(b)
    ranked = sorted(tscore, key=lambda k: -np.mean(tscore[k]))
    lead = set(ranked[:LEADING_THEMES])
    themes_of = {}
    for u in themes:
        themes_of.setdefault(u["ticker"], [])
        if u["theme"] not in themes_of[u["ticker"]]:
            themes_of[u["ticker"]].append(u["theme"])
    names = {u["ticker"]: u["company"] for u in themes}

    # ---------- setups page ----------
    sm = {}
    for t in tickers:
        if t in BENCHMARKS or t not in close:
            continue
        try:
            r = screen_metrics(px["Open"][t], close[t], px["High"][t], px["Low"][t], vol[t])
        except Exception:
            r = None
        if r:
            sm[t] = r
    raw = pd.Series({t: r["rs_raw"] for t, r in sm.items()})
    rs = (raw.rank(pct=True) * 98 + 1).round().astype(int)
    rows = []
    for t, r in sm.items():
        r["rs"] = int(rs[t])
        r["tt"].append(r["rs"] >= 70)
        r["tt_n"] = int(sum(r["tt"]))
        r["tt_miss"] = [TT_LABELS[i] for i, ok in enumerate(r["tt"]) if not ok]
        r["vcp"] = r["contr"] >= 2 and r["tight"] <= 10 and (r["dryup"] or 9) <= 0.85 and 0 <= r["to_pivot"] <= 10
        r["zan_near"] = r["near"] and r["above50"] and r["rs"] >= 80
        r["zan_break"] = r["breakout"] and r["above50"] and r["rs"] >= 70
        r["warn"] = (r["climax"] or r["break50"]) and r["rs"] >= 70
        r["k_pop"] = r["k_pop"] and r["rs"] >= 60
        r["k_cross"] = r["k_cross"] and r["rs"] >= 70
        r["k_bnb"] = r["k_bnb"] and r["rs"] >= 70
        r["k_warn"] = (r["k_exh"] or r["k_drop"]) and r["rs"] >= 70
        sig = {"Minervini TT 8/8": r["tt_n"] == 8, "VCP-like": r["vcp"] and r["tt_n"] >= 7,
               "Zanger breakout": r["zan_break"], "Zanger near pivot": r["zan_near"],
               "Qulla setup": r["q_setup"], "Qulla breakout": r["q_brk"], "Episodic pivot": r["ep"],
               "Kell wedge pop": r["k_pop"], "Kell EMA crossback": r["k_cross"], "Kell base n' break": r["k_bnb"]}
        r["signals"] = [k for k, v in sig.items() if v]
        r["warnings"] = [k for k, v in {"Climax watch": r["climax"] and r["rs"] >= 70, "Closed below 50D": r["break50"] and r["rs"] >= 70,
                         "Exhaustion extension": r["k_exh"] and r["rs"] >= 70, "Wedge drop": r["k_drop"] and r["rs"] >= 70}.items() if v]
        r["n_sig"] = len(r["signals"])
        if not r["liquid"] or not (r["tt_n"] >= 7 or r["signals"] or r["warnings"]):
            continue
        th = themes_of.get(t, [])
        rows.append({
            "ticker": t, "name": names.get(t) or broad.get(t, ("", ""))[0], "sector": broad.get(t, ("", ""))[1],
            "themes": th, "lead": any(x in lead for x in th),
            **{k: r[k] for k in ("close", "r1d", "rs", "tt_n", "tt_miss", "off_hi", "above_lo", "ext50", "pivot",
                                 "to_pivot", "contr", "depths", "tight", "dryup", "relvol", "dvol", "vcp",
                                 "zan_near", "zan_break", "climax", "break50", "warn",
                                 "adr", "r1m", "r3m", "r6m", "rng10", "to_hi20", "gap", "q_setup", "q_brk", "ep", "neglected",
                                 "ext10", "spread", "k_pop", "k_cross", "k_bnb", "k_exh", "k_drop", "k_warn",
                                 "signals", "warnings", "n_sig")},
        })
    rows.sort(key=lambda x: (-x["n_sig"], -x["rs"]))
    write("template_setups.html", "setups.html", {
        "as_of": as_of, "generated": generated, "rows": rows, "universe": len(sm),
        "lead_themes": ranked[:LEADING_THEMES], "min_price": MIN_PRICE, "min_dvol": MIN_DOLLAR_VOL / 1e6, "demo": demo,
        "q_adr": Q_MIN_ADR, "ep_gap": EP_MIN_GAP, "ep_rv": EP_MIN_RELVOL})

    hist = DATA / "history"
    hist.mkdir(exist_ok=True)
    pd.DataFrame([{k: s[k] for k in ("theme", "ticker", "close", "r1d", "r1w", "r1m", "r3m", "ytd")} for s in stocks]) \
        .to_csv(hist / f"{as_of}_themes.csv", index=False)
    pd.DataFrame([{**{k: r[k] for k in ("ticker", "close", "rs", "tt_n", "n_sig")}, "signals": "; ".join(r["signals"]), "warnings": "; ".join(r["warnings"])} for r in rows]) \
        .to_csv(hist / f"{as_of}_setups.csv", index=False)
    print(f"as_of {as_of}: themes {len(stocks)} rows ({len(missing)} missing {missing}); "
          f"screened {len(sm)} stocks, {len(rows)} on setups page")


def write(tpl, out, payload):
    html = (ROOT / tpl).read_text().replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    (ROOT / "docs" / out).write_text(html)


if __name__ == "__main__":
    main()
