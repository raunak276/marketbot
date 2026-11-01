#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOLD LAB  -  automatic market research + thousands of scenario tests
=====================================================================
Ye script:
  1. Gold ka data laata hai (yfinance / TradingView CSV / synthetic test data)
  2. Market snapshot banata hai (trend, volatility, levels, DXY/yields correlation, news)
  3. ~2000 strategy scenarios test karta hai (5 strategies x parameters x SL/TP/hold)
  4. Overfitting se bachne ke liye train/test split use karta hai
  5. Best robust scenario ko Monte Carlo (trade bootstrap) + 100s fake price paths pe test karta hai
  6. Leverage sweep: 1x se 100x tak - kitne % paths me account udd jata hai

Install :  pip install numpy pandas matplotlib yfinance
Run     :  python gold_lab.py                       (yfinance, GC=F, 1h, 730d)
           python gold_lab.py --source csv --csv mydata.csv   (TradingView export)
           python gold_lab.py --source synth        (random data: sanity check, koi edge nahi milna chahiye)

DISCLAIMER: Backtest = past. Ye future ki guarantee nahi hai. Educational use only.
"""
import argparse
import itertools
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ====================================================================
# 1. DATA
# ====================================================================
def load_yf(ticker, interval, period):
    import yfinance as yf
    df = yf.download(ticker, interval=interval, period=period,
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance se data nahi aaya. Internet / ticker check karo.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    return df[["open", "high", "low", "close"]].dropna()


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in df.columns if c in ("time", "date", "datetime", "timestamp")), None)
    if tcol is None:
        raise RuntimeError("CSV me time/date column chahiye (time, date, datetime ya timestamp).")
    t = df[tcol]
    if np.issubdtype(t.dtype, np.number):
        df.index = pd.to_datetime(t, unit="ms" if t.iloc[0] > 1e11 else "s")
    else:
        df.index = pd.to_datetime(t)
    return df[["open", "high", "low", "close"]].astype(float).sort_index().dropna()


def load_synth(n=12000, seed=7):
    """Pure random walk with volatility clustering. Isme koi edge NAHI hai - sanity test ke liye."""
    rng = np.random.default_rng(seed)
    lv = np.zeros(n)
    for i in range(1, n):
        lv[i] = 0.98 * lv[i - 1] + 0.10 * rng.standard_normal()
    vol = 0.0012 * np.exp(lv)
    ret = vol * rng.standard_normal(n) + (rng.random(n) < 0.003) * rng.normal(0, 0.01, n)
    close = 3900 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.standard_normal((2, n))) * vol * 0.6
    high = np.maximum(open_, close) * (1 + wick[0])
    low = np.minimum(open_, close) * (1 - wick[1])
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


# ====================================================================
# 2. INDICATORS
# ====================================================================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ====================================================================
# 3. STRATEGIES  (signal bar close pe banta hai, entry NEXT bar open pe => no look-ahead)
#    return: int8 array  +1 long, -1 short, 0 nothing
# ====================================================================
def _sig(long_, short_):
    return (long_.fillna(False).astype(np.int8) - short_.fillna(False).astype(np.int8)).values


def sig_sr_reject(df, a, N, tol, wick):
    """Aapka setup: resistance/support touch + rejection candle (wick + color)."""
    o, h, l, c = df.open, df.high, df.low, df.close
    hi = h.rolling(N).max().shift(1)
    lo = l.rolling(N).min().shift(1)
    rng = (h - l).replace(0, np.nan)
    up_w = (h - np.maximum(o, c)) / rng
    lo_w = (np.minimum(o, c) - l) / rng
    short = (h >= hi - tol * a) & (c < hi) & (c < o) & (up_w >= wick)
    long_ = (l <= lo + tol * a) & (c > lo) & (c > o) & (lo_w >= wick)
    return _sig(long_, short)


def sig_breakout(df, a, N):
    c = df.close
    hi = df.high.rolling(N).max().shift(1)
    lo = df.low.rolling(N).min().shift(1)
    long_ = (c > hi) & (c.shift(1) <= hi.shift(1))
    short = (c < lo) & (c.shift(1) >= lo.shift(1))
    return _sig(long_, short)


def sig_ema(df, a, fast, slow):
    c = df.close
    f, s = ema(c, fast), ema(c, slow)
    long_ = (f > s) & (f.shift(1) <= s.shift(1))
    short = (f < s) & (f.shift(1) >= s.shift(1))
    return _sig(long_, short)


def sig_rsi(df, a, n, os_, ob):
    r = rsi(df.close, n)
    long_ = (r > os_) & (r.shift(1) <= os_)
    short = (r < ob) & (r.shift(1) >= ob)
    return _sig(long_, short)


def sig_squeeze(df, a, n, q):
    """Compression ke baad breakout (abhi aapke chart jaisi tight range)."""
    c = df.close
    m, sd = c.rolling(n).mean(), c.rolling(n).std()
    w = 4 * sd / m
    thr = w.rolling(100).quantile(q)
    sq = w.shift(1) <= thr.shift(1)
    long_ = sq & (c > m + 2 * sd)
    short = sq & (c < m - 2 * sd)
    return _sig(long_, short)


STRATS = {
    "sr_reject": (sig_sr_reject, {"N": [24, 48, 96, 168], "tol": [0.1, 0.25, 0.5], "wick": [0.3, 0.5]}),
    "breakout": (sig_breakout, {"N": [24, 48, 96, 168]}),
    "ema_trend": (sig_ema, {"fast": [8, 12, 20], "slow": [34, 55, 89, 144]}),
    "rsi_revert": (sig_rsi, {"n": [7, 14], "os_": [20, 25, 30], "ob": [70, 75, 80]}),
    "squeeze": (sig_squeeze, {"n": [20, 40], "q": [0.1, 0.2, 0.3]}),
}
RISK = {"sl_atr": [1.0, 1.5, 2.0, 3.0], "rr": [1.0, 1.5, 2.0, 3.0], "hold": [24, 72]}


# ====================================================================
# 4. BACKTEST ENGINE  (leverage, fees, slippage, liquidation, gap-aware, SL-first conservative)
# ====================================================================
def backtest(o, h, l, c, a, sig, sl_atr, rr, hold, lev, fee, slip, mmr=0.005):
    """Return (entry_index[], pnl_on_margin[]). pnl -1.0 = pura margin gaya (liquidation)."""
    n = len(c)
    ent, out = [], []
    next_free = 0
    for s in np.flatnonzero(sig):
        e = s + 1
        if e < next_free or e >= n:
            continue
        side = int(sig[s])
        av = a[s]
        if not av > 0:
            continue
        entry = o[e] * (1 + side * slip)
        d = sl_atr * av
        sl = entry - side * d
        tp = entry + side * d * rr
        liq = entry * (1 - side * (1.0 / lev - mmr))
        liq_binding = (liq >= sl) if side == 1 else (liq <= sl)
        stop = max(sl, liq) if side == 1 else min(sl, liq)
        last = min(n - 1, e + hold)
        xp, xi, isliq = c[last], last, False
        for j in range(e, last + 1):
            if side == 1:
                if l[j] <= stop:
                    xp = stop if j == e else min(o[j], stop)
                    xi, isliq = j, liq_binding
                    break
                if h[j] >= tp:
                    xp, xi = tp, j
                    break
            else:
                if h[j] >= stop:
                    xp = stop if j == e else max(o[j], stop)
                    xi, isliq = j, liq_binding
                    break
                if l[j] <= tp:
                    xp, xi = tp, j
                    break
        if isliq:
            r = -1.0
        else:
            xp_eff = xp * (1 - side * slip)
            r = lev * side * (xp_eff / entry - 1) - 2 * fee * lev
            r = max(r, -1.0)
        ent.append(e)
        out.append(r)
        next_free = xi + 1
    return np.array(ent, dtype=int), np.array(out, dtype=float)


def stats(p, frac):
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return dict(n=0, win=np.nan, avg=np.nan, pf=np.nan, ret=np.nan, mdd=np.nan, t=0.0, liq=0)
    eq = np.cumprod(1 + frac * p)
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    gain, loss = p[p > 0].sum(), -p[p < 0].sum()
    sd = p.std(ddof=1) if n > 2 else 0
    return dict(n=n, win=(p > 0).mean(), avg=p.mean(),
                pf=(gain / loss) if loss > 0 else np.inf,
                ret=eq[-1] - 1, mdd=(eq / peak - 1).min(),
                t=(p.mean() / sd * np.sqrt(n)) if sd > 0 else 0.0,
                liq=int((p <= -0.999).sum()))


# ====================================================================
# 5. MARKET SNAPSHOT (automatic research)
# ====================================================================
def snapshot(df, a, ticker, macro=True):
    c = df.close
    step = pd.Series(df.index).diff().median()
    bpd = max(1, int(round(pd.Timedelta("1D") / step))) if pd.notna(step) else 1
    last = c.iloc[-1]
    print("\n" + "=" * 70)
    print("MARKET SNAPSHOT")
    print("=" * 70)
    print(f"Last price        : {last:,.2f}   ({df.index[-1]})")
    for k, lab in ((bpd, "1D"), (5 * bpd, "5D"), (20 * bpd, "20D")):
        if len(c) > k:
            print(f"Change {lab:<3}        : {(last / c.iloc[-1 - k] - 1) * 100:+.2f}%")
    atr_pct = a / c * 100
    rank = (atr_pct <= atr_pct.iloc[-1]).mean() * 100
    print(f"ATR(14)           : {a.iloc[-1]:.2f}  ({atr_pct.iloc[-1]:.3f}% per bar, volatility percentile {rank:.0f}%)")
    r = rsi(c, 14).iloc[-1]
    print(f"RSI(14)           : {r:.1f}  " + ("(overbought zone)" if r > 70 else "(oversold zone)" if r < 30 else "(neutral)"))
    e20, e50 = ema(c, 20).iloc[-1], ema(c, 50).iloc[-1]
    trend = "UP" if last > e20 > e50 else "DOWN" if last < e20 < e50 else "MIXED / range"
    print(f"Trend (EMA20/50)  : {trend}   [EMA20={e20:,.2f}  EMA50={e50:,.2f}]")
    bbw = 4 * c.rolling(20).std() / c.rolling(20).mean()
    bb_rank = (bbw.iloc[-500:] <= bbw.iloc[-1]).mean() * 100
    print(f"Squeeze           : Bollinger width percentile {bb_rank:.0f}%  " +
          ("<- TIGHT compression, breakout aa sakta hai (direction unknown)" if bb_rank < 20 else ""))
    print("Key levels (recent high / low):")
    for k, lab in ((bpd, "1D"), (2 * bpd, "2D"), (7 * bpd, "7D")):
        if len(df) > k:
            hi, lo = df.high.iloc[-k:].max(), df.low.iloc[-k:].min()
            print(f"   {lab}: resistance {hi:,.2f} ({(hi / last - 1) * 100:+.2f}%)   support {lo:,.2f} ({(lo / last - 1) * 100:+.2f}%)")

    if not macro:
        return
    try:
        import yfinance as yf
        tick = ["GC=F", "DX-Y.NYB", "^TNX", "^GSPC"]
        d = yf.download(tick, period="1y", interval="1d", auto_adjust=False, progress=False)["Close"].dropna()
        rets = d.pct_change().dropna().iloc[-60:]
        print("\nGold ka 60-day correlation (daily returns):")
        names = {"DX-Y.NYB": "US Dollar Index", "^TNX": "US 10Y yield", "^GSPC": "S&P 500"}
        for k, nm in names.items():
            if k in rets:
                print(f"   vs {nm:<16}: {rets['GC=F'].corr(rets[k]):+.2f}")
        print("   (negative = dollar/yield upar to gold neeche, aam taur pe)")
    except Exception as e:
        print(f"\n[macro correlation skip: {type(e).__name__}]")
    try:
        import yfinance as yf
        news = yf.Ticker(ticker).news or []
        print("\nLatest headlines:")
        for nws in news[:6]:
            title = nws.get("title") or (nws.get("content") or {}).get("title")
            if title:
                print("   -", title)
    except Exception:
        pass


# ====================================================================
# 6. ROBUSTNESS: fake price paths + trade bootstrap
# ====================================================================
def make_path(df, rng, block=48):
    """Block-bootstrap: real bars ke tukde randomly jodke nayi price history, drift hata ke."""
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    pc = np.r_[c[0], c[:-1]]
    rel = np.c_[o / pc, h / pc, l / pc, c / pc]
    mu = np.log(c / pc).mean()
    rel = rel * np.exp(-mu)
    n = len(df)
    parts, tot = [], 0
    while tot < n:
        s = rng.integers(0, n - block)
        parts.append(rel[s:s + block])
        tot += block
    rel2 = np.vstack(parts)[:n]
    closes = c[0] * np.cumprod(rel2[:, 3])
    prevc = np.r_[c[0], closes[:-1]]
    return pd.DataFrame({"open": prevc * rel2[:, 0], "high": prevc * rel2[:, 1],
                         "low": prevc * rel2[:, 2], "close": closes}, index=df.index)


def mc_trades(pnls, frac, runs, rng):
    n = len(pnls)
    idx = rng.integers(0, n, (runs, n))
    eq = np.cumprod(1 + frac * pnls[idx], axis=1)
    eq = np.c_[np.ones(runs), eq]
    peak = np.maximum.accumulate(eq, axis=1)
    return eq, (eq / peak - 1).min(axis=1)


# ====================================================================
# 7. MAIN
# ====================================================================
def main():
    ap = argparse.ArgumentParser(description="Gold research + scenario lab")
    ap.add_argument("--source", choices=["yf", "csv", "synth"], default="yf")
    ap.add_argument("--ticker", default="GC=F", help="yfinance ticker (GC=F gold futures, XAUUSD=X spot)")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--period", default="730d")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--lev", type=float, default=5, help="scan ke time leverage")
    ap.add_argument("--fee", type=float, default=0.0006, help="per side taker fee (0.0006 = 0.06%%)")
    ap.add_argument("--slip", type=float, default=0.0002, help="slippage per side")
    ap.add_argument("--frac", type=float, default=1.0, help="equity ka kitna hissa margin banta hai (1.0 = pura)")
    ap.add_argument("--train", type=float, default=0.6, help="train data fraction")
    ap.add_argument("--mc-runs", type=int, default=5000)
    ap.add_argument("--paths", type=int, default=150, help="fake price paths ki sankhya")
    ap.add_argument("--capital", type=float, default=100, help="starting capital (Rs) sirf display ke liye")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-macro", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # ---- data
    if args.source == "yf":
        df = load_yf(args.ticker, args.interval, args.period)
    elif args.source == "csv":
        if not args.csv:
            sys.exit("--csv path do")
        df = load_csv(args.csv)
    else:
        df = load_synth()
        print("[SYNTHETIC random data - koi real edge nahi hona chahiye]")
    print(f"Data: {len(df)} bars  {df.index[0]} -> {df.index[-1]}")
    a_s = atr(df, 14)
    snapshot(df, a_s, args.ticker, macro=(args.source == "yf" and not args.no_macro))

    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    a = a_s.values
    n = len(df)
    split = int(n * args.train)

    # ---- scan
    n_sc = sum(np.prod([len(v) for v in g.values()]) for _, g in STRATS.values()) * \
        int(np.prod([len(v) for v in RISK.values()]))
    print("\n" + "=" * 70)
    print(f"SCENARIO SCAN: {n_sc} scenarios | leverage {args.lev:g}x | fee {args.fee * 100:.3f}%/side | train {args.train:.0%}")
    print("=" * 70)
    rows, store = [], {}
    for name, (fn, grid) in STRATS.items():
        keys = list(grid)
        for vals in itertools.product(*grid.values()):
            p = dict(zip(keys, vals))
            sig = fn(df, a_s, **p)
            for sl_atr, rr, hold in itertools.product(*RISK.values()):
                ent, pnl = backtest(o, h, l, c, a, sig, sl_atr, rr, hold, args.lev, args.fee, args.slip)
                tr, te = stats(pnl[ent < split], args.frac), stats(pnl[ent >= split], args.frac)
                sid = len(rows)
                rows.append(dict(id=sid, strat=name, params=",".join(f"{k}={v}" for k, v in p.items()),
                                 sl_atr=sl_atr, rr=rr, hold=hold,
                                 tr_n=tr["n"], tr_t=tr["t"], tr_ret=tr["ret"],
                                 te_n=te["n"], te_win=te["win"], te_pf=te["pf"], te_t=te["t"],
                                 te_ret=te["ret"], te_mdd=te["mdd"], te_liq=te["liq"]))
                store[sid] = (name, p, sl_atr, rr, hold)
    res = pd.DataFrame(rows)
    res.to_csv("gold_lab_results.csv", index=False)
    print(f"Done in {time.time() - t0:.0f}s  -> gold_lab_results.csv")

    valid = res[(res.tr_n >= 30) & (res.te_n >= 15)].copy()
    prof_te = (valid.te_ret > 0).mean() * 100 if len(valid) else 0
    print(f"Valid scenarios (enough trades): {len(valid)}")
    print(f"Test period me profitable: {prof_te:.0f}%  <- baseline. Fees ki wajah se random data me ye 50% se kam hota hai;"
          " 50% ke aas-paas ya kam = zyadatar scenarios me koi edge nahi")
    top = valid.sort_values("tr_t", ascending=False).head(12)
    cols = ["id", "strat", "params", "sl_atr", "rr", "hold", "tr_n", "tr_t", "te_n", "te_win", "te_pf", "te_ret", "te_mdd", "te_liq"]
    print("\nTop 12 by TRAIN score (aur unka TEST result - asli imtihaan):")
    print(top[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    surv = valid[(valid.tr_t > 1.0) & (valid.te_t > 1.0) & (valid.te_ret > 0) & (valid.te_pf > 1.1)].copy()
    chance = len(valid) * 0.025
    print(f"\nSURVIVORS (train aur test dono me kaam kiya): {len(surv)} / {len(valid)}")
    print(f"Sirf luck se bhi lagbhag {chance:.0f} aise aa sakte hain - isliye survivors ko bhi shak se dekho.")

    if len(surv):
        surv["score"] = np.minimum(surv.tr_t, surv.te_t)
        best_id = int(surv.sort_values("score", ascending=False).iloc[0].id)
        robust = True
    elif len(valid):
        best_id = int(top.iloc[0].id)
        robust = False
        print("\n>>> Koi robust scenario nahi mila. Aage ka part sirf DEMO hai (top-train scenario, NOT reliable).")
    else:
        sys.exit("Kaafi trades nahi mile. Zyada data / chhota timeframe try karo.")

    name, p, sl_atr, rr, hold = store[best_id]
    fn = STRATS[name][0]
    print(f"\nCHOSEN scenario #{best_id}: {name} {p} | SL={sl_atr}xATR  RR={rr}  hold={hold} bars | robust={robust}")

    sig = fn(df, a_s, **p)
    ent, pnl = backtest(o, h, l, c, a, sig, sl_atr, rr, hold, args.lev, args.fee, args.slip)
    test_pnl = pnl[ent >= split]
    st_all, st_te = stats(pnl, args.frac), stats(test_pnl, args.frac)
    print(f"Full  : trades={st_all['n']} win={st_all['win']:.0%} PF={st_all['pf']:.2f} return={st_all['ret']:+.0%} maxDD={st_all['mdd']:.0%}")
    print(f"Test  : trades={st_te['n']} win={st_te['win']:.0%} PF={st_te['pf']:.2f} return={st_te['ret']:+.0%} maxDD={st_te['mdd']:.0%}")

    # ---- trade bootstrap Monte Carlo
    base = test_pnl if len(test_pnl) >= 15 else pnl
    eq_mc, mdd_mc = mc_trades(base, args.frac, args.mc_runs, rng)
    fin = eq_mc[:, -1]
    ruin = (eq_mc.min(axis=1) <= 0.2).mean()
    print("\n" + "=" * 70)
    print(f"MONTE CARLO ({args.mc_runs} runs, trades reshuffle, {len(base)} trades each)")
    print("=" * 70)
    print(f"Final equity  5% / 50% / 95% : {args.capital * np.percentile(fin, 5):.0f} / "
          f"{args.capital * np.percentile(fin, 50):.0f} / {args.capital * np.percentile(fin, 95):.0f}   (start Rs {args.capital:g})")
    print(f"Max drawdown  median / worst5% : {np.median(mdd_mc):.0%} / {np.percentile(mdd_mc, 5):.0%}")
    print(f"P(profit) = {(fin > 1).mean():.0%}   P(account 80%+ kharab kabhi bhi) = {ruin:.0%}")

    # ---- fake price paths + leverage sweep
    levs = [1, 2, 3, 5, 10, 20, 50, 100]
    print("\n" + "=" * 70)
    print(f"LEVERAGE SWEEP on {args.paths} fake price paths (drift hata ke) + real data")
    print("=" * 70)
    paths = []
    for _ in range(args.paths):
        pdf = make_path(df, rng)
        pa = atr(pdf, 14)
        paths.append((pdf.open.values, pdf.high.values, pdf.low.values, pdf.close.values, pa.values,
                      fn(pdf, pa, **p)))
    sweep = []
    for lv in levs:
        finals, liqs = [], 0
        for (po, ph, pl, pc_, pa, ps) in paths:
            _, pp = backtest(po, ph, pl, pc_, pa, ps, sl_atr, rr, hold, lv, args.fee, args.slip)
            s = stats(pp, args.frac)
            finals.append(1 + s["ret"] if s["n"] else 1.0)
            liqs += s["liq"] > 0
        finals = np.array(finals)
        _, rp = backtest(o, h, l, c, a, sig, sl_atr, rr, hold, lv, args.fee, args.slip)
        sr = stats(rp, args.frac)
        sweep.append(dict(lev=lv, real_ret=sr["ret"], real_mdd=sr["mdd"], real_liq=sr["liq"],
                          path_median=np.median(finals), path_p_profit=(finals > 1).mean(),
                          path_p_wipe=(finals < 0.2).mean(), path_p_liq=liqs / len(paths)))
    sw = pd.DataFrame(sweep)
    print(sw.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nPadhne ka tareeka: path_p_wipe = fake histories ka kitna hissa jisme account 80%+ khatam;"
          "\npath_p_liq = kitne paths me kam se kam ek liquidation hui.")

    # ---- plot
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
            eq_real = np.cumprod(1 + args.frac * pnl)
            ax[0].plot(ent, args.capital * eq_real)
            ax[0].axvline(split, color="r", ls="--", label="train | test")
            ax[0].set_title("Real data equity (chosen scenario)")
            ax[0].legend()
            x = np.arange(eq_mc.shape[1])
            for lo_, hi_, al in ((5, 95, 0.2), (25, 75, 0.4)):
                ax[1].fill_between(x, args.capital * np.percentile(eq_mc, lo_, axis=0),
                                   args.capital * np.percentile(eq_mc, hi_, axis=0), alpha=al, color="tab:blue")
            ax[1].plot(x, args.capital * np.median(eq_mc, axis=0), color="k")
            ax[1].set_title("Monte Carlo fan (5-95%, 25-75%)")
            ax[2].bar([str(v) + "x" for v in levs], sw.path_p_wipe * 100, color="tab:red")
            ax[2].set_title("% fake paths jisme account 80%+ khatam")
            ax[2].set_ylabel("%")
            plt.tight_layout()
            plt.savefig("gold_lab_report.png", dpi=110)
            print("\nChart saved: gold_lab_report.png")
        except Exception as e:
            print(f"[plot skip: {e}]")

    print(f"\nTotal time {time.time() - t0:.0f}s")
    print("NOTE: Backtest fees/slippage assumptions pe chalta hai; weekend liquidity, gaps, funding fees real me alag ho sakte hain.")
    print("Pehle demo/paper trading. Ye financial advice nahi hai.")


if __name__ == "__main__":
    main()
