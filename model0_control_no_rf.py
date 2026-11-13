# %% [markdown]
# # Model 0 - CONTROL: rule-only VWAP, Bollinger and DUAL (no Random Forest)
# ASTRAQUANT X-2026 - Problem Statement 14 - Automated Bank NIFTY Options Trading
#
# Purpose: a control experiment. Every candidate signal from the indicator rules is traded
# (subject to the same risk rules); there is NO Random Forest filter. Comparing this file's
# results with models 1-3 measures exactly how much the Random Forest adds.
#
# Identical to models 1-3 in everything else:
#   - same indicator formulas, same entry rules (VWAP cross / Bollinger re-entry / DUAL pullback)
#   - same Black-Scholes ATM option pricing, +12% target / -6% stop, 12-bar max hold, 15:15 square-off
#   - same 1% risk per trade, max 3 trades/day, -2% daily loss cap, same costs and slippage
#   - same out-of-sample test window (the first 6 sessions are skipped so the dates match the RF models)
#
#   1. VWAP rule only      - CE on close crossing above VWAP, PE on crossing below
#   2. Bollinger rule only - CE on re-entry from lower band (z <= -2), PE on re-entry from upper band
#   3. DUAL rule only      - CE when above VWAP and z recovers from <= -0.5; PE is the mirror image
#
# Formulas (deck slide 10):
#   TP_i = (H_i + L_i + C_i)/3 ,  VWAP_t = sum(TP_i V_i)/sum(V_i)  (resets daily),  d_t = (C_t - VWAP_t)/VWAP_t
#   MA_t = (1/n) sum C_i ,  sigma_t = sqrt((1/n) sum (C_i - MA_t)^2) ,  U/L = MA_t +/- k sigma_t ,  n=20, k=2
#   BW_t = (U_t - L_t)/MA_t ,  z_t = (C_t - MA_t)/sigma_t

# %% Imports and settings
import datetime as dt
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 5-minute Bank NIFTY data: the copy in this repo, else the original Desktop/Astra copy
_REPO_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "banknifty_clean_5m.csv")
DATA_PATH = _REPO_CSV if os.path.exists(_REPO_CSV) else os.path.expanduser("~/Desktop/Astra/banknifty_clean_5m.csv")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_control")

# Indicators
BB_N, BB_K = 20, 2.0          # Bollinger window and width (bands, z-score)
Z_PB = 0.5                    # DUAL pullback depth: z must dip to -0.5 sigma (CE) / +0.5 sigma (PE)

# Option pricing (deck slide 7/10): ATM strike = nearest 100, sigma 15%, r 6%, 4 days to expiry
IV, RATE, DTE_DAYS, STRIKE_STEP = 0.15, 0.06, 4, 100
LOT_SIZE = 30                 # Bank NIFTY lot size (change if NSE revises it)

# Risk rules (deck slide 9)
CAPITAL = 500_000
RISK_PER_TRADE = 0.01         # 1% of capital at risk per trade
TARGET, STOP = 0.12, 0.06     # +12% target / -6% stop on premium
MAX_HOLD_BARS = 12            # 12 x 5 min = 1 hour
MAX_TRADES_PER_DAY = 3
DAILY_LOSS_CAP = 0.02         # stop for the day at -2% of capital
ENTRY_START, ENTRY_END, SQUARE_OFF = dt.time(9, 30), dt.time(14, 30), dt.time(15, 15)

# Costs (per order, NSE index options)
TICK, SLIP_TICKS = 0.05, 2
BROKERAGE = 20.0
STT_SELL = 0.001              # 0.1% of premium on sell side
EXCH_TXN = 0.0003503          # NSE transaction charge on premium
SEBI_FEE = 10 / 1e7
STAMP_BUY = 0.00003
GST = 0.18

# Test window - skip the first 6 sessions so dates match the RF models' out-of-sample period
WARMUP_SESSIONS = 6

os.makedirs(OUT_DIR, exist_ok=True)


# %% 1. Load and clean data
def load_data(path):
    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df.drop_duplicates("Datetime").sort_values("Datetime").reset_index(drop=True)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df["date"] = df["Datetime"].dt.date
    df["time"] = df["Datetime"].dt.time
    return df


# %% 2. Indicator engine
def add_indicators(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (tp * df["Volume"]).groupby(df["date"]).cumsum() / df["Volume"].groupby(df["date"]).cumsum()
    df["MA"] = df["Close"].rolling(BB_N).mean()
    df["SD"] = df["Close"].rolling(BB_N).std(ddof=0)
    df["UB"] = df["MA"] + BB_K * df["SD"]
    df["LB"] = df["MA"] - BB_K * df["SD"]
    df["BW"] = (df["UB"] - df["LB"]) / df["MA"]
    df["z"] = (df["Close"] - df["MA"]) / df["SD"]
    df["d"] = (df["Close"] - df["VWAP"]) / df["VWAP"]
    return df


def common_features(df):
    return pd.DataFrame({
        "vol_ratio": df["Volume"] / df["Volume"].rolling(20).mean(),
        "mins_since_open": df["Datetime"].dt.hour * 60 + df["Datetime"].dt.minute - (9 * 60 + 15),
        "ret_1bar": df["Close"].pct_change(),
    })


def vwap_features(df):
    return pd.DataFrame({
        "d_vwap": df["d"],
        "d_vwap_chg3": df["d"].diff(3),
        "vwap_slope6": df["VWAP"].pct_change(6),
    })


def bb_features(df):
    return pd.DataFrame({
        "z": df["z"],
        "BW": df["BW"],
        "BW_chg5": df["BW"].pct_change(5),
        "ma_slope6": df["MA"].pct_change(6),
    })


# %% 3. Layer 1 - candidate signals for each model
def entry_window(df):
    return (df["time"] >= ENTRY_START) & (df["time"] <= ENTRY_END)


def signals_vwap(df):
    above = df["Close"] > df["VWAP"]
    prev = above.shift(1, fill_value=False)
    s = (above & ~prev).astype(int) - (~above & prev).astype(int)
    return s.where(entry_window(df), 0)


def signals_bb(df):
    z, zp = df["z"], df["z"].shift(1)
    ce = (zp <= -BB_K) & (z > -BB_K)        # closed at/below lower band, now back inside
    pe = (zp >= BB_K) & (z < BB_K)          # closed at/above upper band, now back inside
    s = ce.astype(int) - pe.astype(int)
    return s.where(entry_window(df), 0)


def signals_dual(df):
    z, zp = df["z"], df["z"].shift(1)
    above = df["Close"] > df["VWAP"]
    ce = above & (zp <= -Z_PB) & (z > -Z_PB)    # uptrend (VWAP) + pullback ends (BB)
    pe = ~above & (zp >= Z_PB) & (z < Z_PB)     # downtrend (VWAP) + bounce ends (BB)
    s = ce.astype(int) - pe.astype(int)
    return s.where(entry_window(df), 0)


# %% 4. Option pricing (Black-Scholes, ATM strike)
def bs_price(S, K, T, kind):
    T = max(T, 1e-6)
    d1 = (np.log(S / K) + (RATE + 0.5 * IV ** 2) * T) / (IV * np.sqrt(T))
    d2 = d1 - IV * np.sqrt(T)
    if kind == "CE":
        return S * norm.cdf(d1) - K * np.exp(-RATE * T) * norm.cdf(d2)
    return K * np.exp(-RATE * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, kind):
    d1 = (np.log(S / K) + (RATE + 0.5 * IV ** 2) * T) / (IV * np.sqrt(T))
    return norm.cdf(d1) if kind == "CE" else norm.cdf(d1) - 1


def simulate_trade(df, i):
    """Enter at close of bar i, exit on target / stop / 12 bars / 15:15 / end of session."""
    kind = "CE" if df["signal"].iat[i] == 1 else "PE"
    S0 = df["Close"].iat[i]
    K = round(S0 / STRIKE_STEP) * STRIKE_STEP
    T0 = DTE_DAYS / 365
    entry = bs_price(S0, K, T0, kind)
    j, reason = i, "session_end"
    while j + 1 < len(df) and df["date"].iat[j + 1] == df["date"].iat[i]:
        j += 1
        T = T0 - (j - i) * 5 / (365 * 24 * 60)
        r = bs_price(df["Close"].iat[j], K, T, kind) / entry - 1
        if r >= TARGET:
            reason = "target"; break
        if r <= -STOP:
            reason = "stop"; break
        if j - i >= MAX_HOLD_BARS:
            reason = "time"; break
        if df["time"].iat[j] >= SQUARE_OFF:
            reason = "square_off"; break
    T = T0 - (j - i) * 5 / (365 * 24 * 60)
    exit_px = bs_price(df["Close"].iat[j], K, T, kind)
    return dict(kind=kind, strike=K, entry_premium=entry, exit_premium=exit_px,
                delta=bs_delta(S0, K, T0, kind), exit_idx=j, exit_reason=reason,
                label=int(exit_px / entry - 1 >= TARGET - 1e-9))


def trade_costs(buy_px, sell_px, qty):
    buy_val, sell_val = buy_px * qty, sell_px * qty
    exch = (buy_val + sell_val) * EXCH_TXN
    sebi = (buy_val + sell_val) * SEBI_FEE
    brok = 2 * BROKERAGE
    return brok + exch + sebi + sell_val * STT_SELL + buy_val * STAMP_BUY + GST * (brok + exch + sebi)


# %% 5. Rule-only backtest (no Random Forest) - every candidate is traded if risk rules allow
def run_model(name, df, signal_fn, feature_fns):
    d = df.copy()
    d["signal"] = signal_fn(d)
    # same candidate set as the RF models (bars where all their features exist)
    X = pd.concat([f(d) for f in feature_fns] + [common_features(d)], axis=1)
    cand = [i for i in np.where(d["signal"] != 0)[0] if X.iloc[i].notna().all()]

    sims = {i: simulate_trade(d, i) for i in cand}
    days = sorted(d["date"].unique())
    oos_start = days[WARMUP_SESSIONS]

    trades, preds, decide_times = [], [], []
    equity_by_day = {}
    counts = dict(train_initial=0, train_final=0, rf_approved=0, rf_rejected=0,
                  blocked_open_position=0, blocked_max_trades=0, blocked_daily_cap=0,
                  daily_cap_days=0, risk_pct=[])
    for day in days[WARMUP_SESSIONS:]:
        today = [i for i in cand if d["date"].iat[i] == day]
        day_pnl, n_today, busy_until = 0.0, 0, -1
        for i in today:
            t0 = time.perf_counter()
            preds.append((i, 1.0, sims[i]["label"]))      # control: every signal is "approved"
            counts["rf_approved"] += 1
            decide_times.append(time.perf_counter() - t0)
            if i <= busy_until:                           # risk rules block the trade
                counts["blocked_open_position"] += 1; continue
            if n_today >= MAX_TRADES_PER_DAY:
                counts["blocked_max_trades"] += 1; continue
            if day_pnl <= -DAILY_LOSS_CAP * CAPITAL:
                counts["blocked_daily_cap"] += 1; continue
            s = sims[i]
            buy = s["entry_premium"] + SLIP_TICKS * TICK
            sell = max(s["exit_premium"] - SLIP_TICKS * TICK, 0.05)
            lots = max(1, int(RISK_PER_TRADE * CAPITAL / (STOP * buy * LOT_SIZE)))
            qty = lots * LOT_SIZE
            counts["risk_pct"].append(STOP * buy * qty / CAPITAL * 100)
            costs = trade_costs(buy, sell, qty)
            pnl = (sell - buy) * qty - costs
            trades.append(dict(model=name, entry_time=d["Datetime"].iat[i],
                               exit_time=d["Datetime"].iat[s["exit_idx"]], side=s["kind"],
                               spot=d["Close"].iat[i], strike=s["strike"], delta=round(s["delta"], 2),
                               lots=lots, buy=round(buy, 2), sell=round(sell, 2),
                               exit_reason=s["exit_reason"], costs=round(costs, 2), pnl=round(pnl, 2)))
            day_pnl += pnl
            n_today += 1
            busy_until = s["exit_idx"]                    # one position at a time
        equity_by_day[day] = day_pnl
        counts["daily_cap_days"] += int(day_pnl <= -DAILY_LOSS_CAP * CAPITAL)

    tr = pd.DataFrame(trades)
    pr = pd.DataFrame(preds, columns=["idx", "p_hat", "y"])
    metrics = compute_metrics(name, tr, pr, equity_by_day, 0.0, decide_times, counts, len(cand))
    return dict(name=name, data=d, trades=tr, preds=pr, metrics=metrics,
                daily=pd.Series(equity_by_day))


# %% 6. Performance metrics
METRIC_SECTIONS = [
    ("Candidates (out-of-sample)", ["Test candidates (out-of-sample)", "Blocked by risk rules",
                                    "Trades taken", "CE / PE"]),
    ("Signal quality (no filter)", ["Accuracy rate %"]),   # = share of all signals that hit the target
    ("Profitability", ["Net P&L (Rs)", "ROI %", "Sharpe (ann.)", "Win rate %", "Profit factor",
                       "Avg win (Rs)", "Avg loss (Rs)", "Total costs (Rs)", "Costs % of gross P&L"]),
    ("Risk rates", ["Max drawdown %", "Volatility % (ann.)", "Risk per trade % (rule)",
                    "Avg actual risk per trade %", "Planned reward:risk", "Realised reward:risk",
                    "Break-even win rate %", "Target hit rate %", "Stop-loss hit rate %",
                    "Time / square-off exit rate %", "Max consecutive losses", "Daily loss cap hit (days)"]),
]


TAU = 0.5  # only used by compute_metrics; control signals all have p_hat = 1


def _pct(a, b, nd=1):
    return round(a / b * 100, nd) if b else 0.0


def compute_metrics(name, tr, pr, daily, train_time, infer_times, counts, n_cand):
    daily = pd.Series(daily, dtype=float)
    n = len(tr)
    pnl = tr["pnl"] if n else pd.Series(dtype=float)
    eq = CAPITAL + pnl.cumsum()
    peak = pd.concat([pd.Series([CAPITAL]), eq]).cummax().iloc[1:]
    max_dd = ((eq - peak) / peak).min() if n else 0.0
    rets = daily / CAPITAL
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0

    yt = pr["y"].astype(int) if len(pr) else pd.Series(dtype=int)
    yp = (pr["p_hat"] >= TAU).astype(int) if len(pr) else pd.Series(dtype=int)
    tp, fp = int(((yp == 1) & (yt == 1)).sum()), int(((yp == 1) & (yt == 0)).sum())
    tn, fn = int(((yp == 0) & (yt == 0)).sum()), int(((yp == 0) & (yt == 1)).sum())

    streak = max_streak = 0
    for v in pnl:
        streak = streak + 1 if v <= 0 else 0
        max_streak = max(max_streak, streak)
    reasons = tr["exit_reason"].value_counts() if n else pd.Series(dtype=int)
    gross = (pnl + tr["costs"]).sum() if n else 0.0
    blocked = counts["blocked_open_position"] + counts["blocked_max_trades"] + counts["blocked_daily_cap"]

    return {
        "Model": name,
        # candidates
        "Total candidate signals": n_cand,
        "Train candidates (initial)": counts["train_initial"],
        "Train candidates (final model)": counts["train_final"],
        "Test candidates (out-of-sample)": len(pr),
        # RF filter
        "RF approved signals": counts["rf_approved"],
        "RF rejected signals": counts["rf_rejected"],
        "RF approval rate %": _pct(counts["rf_approved"], len(pr)),
        "Blocked by risk rules": f"{blocked} (open pos {counts['blocked_open_position']}, "
                                 f"max/day {counts['blocked_max_trades']}, loss cap {counts['blocked_daily_cap']})",
        "Trades taken": n,
        "CE / PE": f"{(tr['side'] == 'CE').sum() if n else 0} / {(tr['side'] == 'PE').sum() if n else 0}",
        # prediction quality
        "Accuracy rate %": round(accuracy_score(yt, yp) * 100, 1) if len(pr) else 0.0,
        "Precision %": round(precision_score(yt, yp, zero_division=0) * 100, 1) if len(pr) else 0.0,
        "Recall %": round(recall_score(yt, yp, zero_division=0) * 100, 1) if len(pr) else 0.0,
        "F1-score": round(f1_score(yt, yp, zero_division=0), 3) if len(pr) else 0.0,
        "TP / FP / TN / FN": f"{tp} / {fp} / {tn} / {fn}",
        "Raw signal hit rate % (no RF)": _pct(int(yt.sum()), len(yt)),
        "RF-approved hit rate %": _pct(tp, tp + fp),
        # profitability
        "Net P&L (Rs)": round(pnl.sum(), 0),
        "ROI %": round(pnl.sum() / CAPITAL * 100, 2),
        "Sharpe (ann.)": round(sharpe, 2),
        "Win rate %": _pct(len(wins), n),
        "Profit factor": round(wins.sum() / -losses.sum(), 2) if losses.sum() < 0 else np.inf,
        "Avg win (Rs)": round(avg_win, 0),
        "Avg loss (Rs)": round(avg_loss, 0),
        "Total costs (Rs)": round(tr["costs"].sum(), 0) if n else 0,
        "Costs % of gross P&L": _pct(tr["costs"].sum(), abs(gross)) if n else 0.0,
        # risk rates
        "Max drawdown %": round(max_dd * 100, 2),
        "Volatility % (ann.)": round(rets.std() * np.sqrt(252) * 100, 2),
        "Risk per trade % (rule)": RISK_PER_TRADE * 100,
        "Avg actual risk per trade %": round(np.mean(counts["risk_pct"]), 2) if counts["risk_pct"] else 0.0,
        "Planned reward:risk": f"{TARGET / STOP:.1f} : 1",
        "Realised reward:risk": f"{avg_win / -avg_loss:.2f} : 1" if avg_loss < 0 else "n/a",
        "Break-even win rate %": _pct(-avg_loss, avg_win - avg_loss) if avg_loss < 0 else 0.0,
        "Target hit rate %": _pct(reasons.get("target", 0), n),
        "Stop-loss hit rate %": _pct(reasons.get("stop", 0), n),
        "Time / square-off exit rate %": _pct(n - reasons.get("target", 0) - reasons.get("stop", 0), n),
        "Max consecutive losses": max_streak,
        "Daily loss cap hit (days)": counts["daily_cap_days"],
        # efficiency
        "Train time (s)": round(train_time, 2),
        "Inference (ms/signal)": round(np.mean(infer_times) * 1000, 2) if infer_times else 0,
    }


def print_metrics(m):
    for title, keys in METRIC_SECTIONS:
        print(f"  -- {title}")
        for k in keys:
            print(f"     {k:<34}{m[k]}")


# %% 7. Visualisations
def plot_signals(res, path):
    d, tr = res["data"], res["trades"]
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(d))
    ax.plot(x, d["Close"], color="#222", lw=0.8, label="Bank NIFTY close")
    ax.plot(x, d["VWAP"], color="#1f77b4", lw=1, label="VWAP (daily)")
    ax.plot(x, d["MA"], color="#ff7f0e", lw=0.8, ls="--", label="BB middle (MA20)")
    ax.fill_between(x, d["LB"], d["UB"], color="#ff7f0e", alpha=0.12, label="Bollinger +/-2 sigma")
    if len(tr):
        pos = d.reset_index().set_index("Datetime")["index"]
        ce, pe = tr[tr["side"] == "CE"], tr[tr["side"] == "PE"]
        ax.scatter(pos[ce["entry_time"]], ce["spot"], marker="^", s=90, color="green", zorder=5, label="BUY CE")
        ax.scatter(pos[pe["entry_time"]], pe["spot"], marker="v", s=90, color="red", zorder=5, label="BUY PE")
    ticks = d.groupby("date").head(1).index
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(dd)[5:] for dd in d.loc[ticks, "date"]], rotation=45, fontsize=8)
    ax.set_title(f"{res['name']} - trading signals")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_comparison(results, path):
    colors = {"VWAP rule only": "#1f77b4", "Bollinger rule only": "#ff7f0e", "DUAL rule only": "#2ca02c"}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax = axes[0, 0]
    for r in results:
        tr = r["trades"]
        if len(tr):
            ax.step(pd.concat([pd.Series([tr["entry_time"].iloc[0]]), tr["exit_time"]]),
                    np.r_[CAPITAL, CAPITAL + tr["pnl"].cumsum()], where="post",
                    lw=2, color=colors[r["name"]], label=r["name"])
    ax.axhline(CAPITAL, color="grey", lw=0.8, ls=":")
    ax.set_title("Equity curve (out-of-sample)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    panels = [("ROI %", axes[0, 1]), ("Sharpe (ann.)", axes[0, 2]), ("Max drawdown %", axes[1, 0]),
              ("Win rate %", axes[1, 1]), ("Profit factor", axes[1, 2])]
    for metric, ax in panels:
        names = [r["name"] for r in results]
        vals = [r["metrics"][metric] for r in results]
        bars = ax.bar(range(3), vals, color=[colors[n] for n in names])
        ax.set_xticks(range(3)); ax.set_xticklabels(["VWAP", "Bollinger", "DUAL"])
        for b, v in zip(bars, vals):
            ax.annotate(f"{v}", (b.get_x() + b.get_width() / 2, v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=10)
        ax.set_title(metric); ax.grid(axis="y", alpha=0.3); ax.axhline(0, color="grey", lw=0.8)
    fig.suptitle("CONTROL (no Random Forest): VWAP vs Bollinger vs DUAL", fontsize=14)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# %% 8. Run the three rule-only models, then compare with the Random Forest versions
CONTROL_VS_RF = ["Trades taken", "Net P&L (Rs)", "ROI %", "Sharpe (ann.)",
                 "Max drawdown %", "Win rate %", "Profit factor", "Stop-loss hit rate %"]

if __name__ == "__main__":
    df = add_indicators(load_data(DATA_PATH))
    days = sorted(df["date"].unique())
    print(f"Data: {len(df)} bars, {len(days)} sessions")
    print(f"Test window: {days[WARMUP_SESSIONS]} -> {days[-1]} (same dates as the RF models)\n")

    results = [
        run_model("VWAP rule only", df, signals_vwap, [vwap_features]),
        run_model("Bollinger rule only", df, signals_bb, [bb_features]),
        run_model("DUAL rule only", df, signals_dual, [vwap_features, bb_features]),
    ]

    for r in results:
        print("=" * 70)
        print(f"  {r['name']} (no Random Forest)  -  output")
        print("=" * 70)
        print_metrics(r["metrics"])
        print()

    keys = [k for _, ks in METRIC_SECTIONS for k in ks]
    table = pd.DataFrame([r["metrics"] for r in results]).set_index("Model").T.loc[keys]
    print("=" * 70)
    print("  CONTROL COMPARISON (no Random Forest)")
    print("=" * 70)
    print(table.to_string())

    # Control vs treatment: run the RF versions from model3 (same folder) for a side-by-side view
    try:
        import model3_dual_vwap_bb_rf as rf
        rf_df = rf.add_indicators(rf.load_data(DATA_PATH))
        rf_results = [
            rf.run_model("VWAP + RF", rf_df, rf.signals_vwap, [rf.vwap_features]),
            rf.run_model("Bollinger + RF", rf_df, rf.signals_bb, [rf.bb_features]),
            rf.run_model("DUAL (VWAP + BB) + RF", rf_df, rf.signals_dual, [rf.vwap_features, rf.bb_features]),
        ]
        rows = {}
        for c, t, label in zip(results, rf_results, ["VWAP", "Bollinger", "DUAL"]):
            # like-for-like signal quality: share of ACTED-ON signals that hit the target
            # (control acts on every signal -> raw hit rate; RF acts on approved ones -> precision)
            rows[f"{label} | no RF"] = {"Hit rate of acted-on signals %": c["metrics"]["Accuracy rate %"],
                                        **{k: c["metrics"][k] for k in CONTROL_VS_RF}}
            rows[f"{label} | with RF"] = {"Hit rate of acted-on signals %": t["metrics"]["Precision %"],
                                          **{k: t["metrics"][k] for k in CONTROL_VS_RF}}
        vs = pd.DataFrame(rows)
        print("\n" + "=" * 70)
        print("  CONTROL vs RANDOM FOREST - what the RF filter adds")
        print("=" * 70)
        print(vs.to_string())
        vs.to_csv(os.path.join(OUT_DIR, "control_vs_rf.csv"))
    except ImportError:
        print("\n(model3_dual_vwap_bb_rf.py not found next to this file - skipping control vs RF table)")

    table.to_csv(os.path.join(OUT_DIR, "control_metrics.csv"))
    pd.concat([r["trades"] for r in results]).to_csv(os.path.join(OUT_DIR, "control_trades.csv"), index=False)
    plot_comparison(results, os.path.join(OUT_DIR, "control_dashboard.png"))
    for r, tag in zip(results, ["vwap", "bollinger", "dual"]):
        plot_signals(r, os.path.join(OUT_DIR, f"signals_{tag}_no_rf.png"))
    print(f"\nSaved tables and charts to {OUT_DIR}")
