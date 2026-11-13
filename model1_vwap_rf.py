# %% [markdown]
# # Model 1 - VWAP + Random Forest
# ASTRAQUANT X-2026 - Problem Statement 14 - Automated Bank NIFTY Options Trading
#
# Layer 1 (rule):  VWAP cross.  Close crosses ABOVE VWAP -> candidate BUY CE,
#                               close crosses BELOW VWAP -> candidate BUY PE.
# Layer 2 (ML):    Random Forest meta-filter, act only if p_hat >= tau.
#
# Formulas (deck slide 10):
#   TP_i   = (H_i + L_i + C_i) / 3
#   VWAP_t = sum(TP_i * V_i) / sum(V_i)     (i = first bar of the session, resets daily)
#   d_t    = (C_t - VWAP_t) / VWAP_t
#   Option premium: Black-Scholes, ATM strike = nearest 100, sigma 15%, r 6%, 4 days to expiry
#   RF:  p_hat_t = (1/B) sum_b h_b(x_t),  label y_t = 1 if premium hits +12% before -6% within 12 bars
#
# Same RF settings, option pricing, risk rules and costs as models 2 and 3, so results compare fairly.

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 5-minute Bank NIFTY data: the copy in this repo, else the original Desktop/Astra copy
_REPO_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "banknifty_clean_5m.csv")
DATA_PATH = _REPO_CSV if os.path.exists(_REPO_CSV) else os.path.expanduser("~/Desktop/Astra/banknifty_clean_5m.csv")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_vwap")

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

# Random Forest (identical for all three models)
RF_PARAMS = dict(n_estimators=300, max_depth=4, min_samples_leaf=3,
                 class_weight="balanced", random_state=42, n_jobs=1)
TAU = 0.50                    # act only if p_hat >= tau
WARMUP_SESSIONS = 6           # first 6 sessions are training-only; then walk-forward, retrain daily

os.makedirs(OUT_DIR, exist_ok=True)


# %% 1. Load and clean data
def load_data(path):
    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df.drop_duplicates("Datetime").sort_values("Datetime").reset_index(drop=True)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df["date"] = df["Datetime"].dt.date
    df["time"] = df["Datetime"].dt.time
    return df


# %% 2. Indicator engine - VWAP only
def add_indicators(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3                     # typical price
    df["VWAP"] = (tp * df["Volume"]).groupby(df["date"]).cumsum() / df["Volume"].groupby(df["date"]).cumsum()
    df["d"] = (df["Close"] - df["VWAP"]) / df["VWAP"]                 # distance from VWAP
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




# %% 3. Layer 1 - candidate signals
def entry_window(df):
    return (df["time"] >= ENTRY_START) & (df["time"] <= ENTRY_END)


def signals_vwap(df):
    above = df["Close"] > df["VWAP"]
    prev = above.shift(1, fill_value=False)
    s = (above & ~prev).astype(int) - (~above & prev).astype(int)
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


# %% 5. Layer 2 - Random Forest meta-filter (walk-forward) + risk-managed backtest
def run_model(name, df, signal_fn, feature_fns):
    d = df.copy()
    d["signal"] = signal_fn(d)
    X = pd.concat([f(d) for f in feature_fns] + [common_features(d)], axis=1)
    X["side"] = d["signal"]
    cand = [i for i in np.where(d["signal"] != 0)[0] if X.iloc[i].notna().all()]

    sims = {i: simulate_trade(d, i) for i in cand}
    y = pd.Series({i: sims[i]["label"] for i in cand}, dtype=int)
    days = sorted(d["date"].unique())

    trades, preds, model = [], [], None
    train_time, infer_times = 0.0, []
    equity_by_day = {}
    oos_start = days[WARMUP_SESSIONS]
    counts = dict(train_initial=sum(d["date"].iat[i] < oos_start for i in cand),
                  train_final=0, rf_approved=0, rf_rejected=0,
                  blocked_open_position=0, blocked_max_trades=0, blocked_daily_cap=0,
                  daily_cap_days=0, risk_pct=[])
    for day in days[WARMUP_SESSIONS:]:
        train_idx = [i for i in cand if d["date"].iat[i] < day]
        today = [i for i in cand if d["date"].iat[i] == day]
        if not today or y[train_idx].nunique() < 2:
            equity_by_day[day] = 0.0                      # no trades today still counts as a 0% day
            continue
        counts["train_final"] = len(train_idx)
        t0 = time.perf_counter()
        model = RandomForestClassifier(**RF_PARAMS).fit(X.loc[train_idx], y[train_idx])
        train_time += time.perf_counter() - t0

        day_pnl, n_today, busy_until = 0.0, 0, -1
        for i in today:
            t0 = time.perf_counter()
            p_hat = model.predict_proba(X.loc[[i]])[0, 1]
            infer_times.append(time.perf_counter() - t0)
            preds.append((i, p_hat, y[i]))
            if p_hat < TAU:
                counts["rf_rejected"] += 1                # RF rejects the signal
                continue
            counts["rf_approved"] += 1
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
                               p_hat=round(p_hat, 3), lots=lots, buy=round(buy, 2), sell=round(sell, 2),
                               exit_reason=s["exit_reason"], costs=round(costs, 2), pnl=round(pnl, 2)))
            day_pnl += pnl
            n_today += 1
            busy_until = s["exit_idx"]                    # one position at a time
        equity_by_day[day] = day_pnl
        counts["daily_cap_days"] += int(day_pnl <= -DAILY_LOSS_CAP * CAPITAL)

    tr = pd.DataFrame(trades)
    pr = pd.DataFrame(preds, columns=["idx", "p_hat", "y"])
    metrics = compute_metrics(name, tr, pr, equity_by_day, train_time, infer_times, counts, len(cand))
    return dict(name=name, data=d, trades=tr, preds=pr, metrics=metrics, model=model,
                features=list(X.columns), daily=pd.Series(equity_by_day))


# %% 6. Performance metrics
METRIC_SECTIONS = [
    ("Candidates (train / test)", ["Total candidate signals", "Train candidates (initial)",
                                   "Train candidates (final model)", "Test candidates (out-of-sample)"]),
    ("Random Forest filter", ["RF approved signals", "RF rejected signals", "RF approval rate %",
                              "Blocked by risk rules", "Trades taken", "CE / PE"]),
    ("Prediction quality", ["Accuracy rate %", "Precision %", "Recall %", "F1-score",
                            "TP / FP / TN / FN", "Raw signal hit rate % (no RF)", "RF-approved hit rate %"]),
    ("Profitability", ["Net P&L (Rs)", "ROI %", "Sharpe (ann.)", "Win rate %", "Profit factor",
                       "Avg win (Rs)", "Avg loss (Rs)", "Total costs (Rs)", "Costs % of gross P&L"]),
    ("Risk rates", ["Max drawdown %", "Volatility % (ann.)", "Risk per trade % (rule)",
                    "Avg actual risk per trade %", "Planned reward:risk", "Realised reward:risk",
                    "Break-even win rate %", "Target hit rate %", "Stop-loss hit rate %",
                    "Time / square-off exit rate %", "Max consecutive losses", "Daily loss cap hit (days)"]),
    ("Efficiency", ["Train time (s)", "Inference (ms/signal)"]),
]


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




def plot_equity(res, path):
    tr = res["trades"]
    fig, ax = plt.subplots(figsize=(10, 4))
    if len(tr):
        ax.step(pd.concat([pd.Series([tr["entry_time"].iloc[0]]), tr["exit_time"]]),
                np.r_[CAPITAL, CAPITAL + tr["pnl"].cumsum()], where="post", lw=2)
    ax.axhline(CAPITAL, color="grey", lw=0.8, ls=":")
    ax.set_title(f"{res['name']} - equity curve (out-of-sample)"); ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_importance(res, path):
    imp = pd.Series(res["model"].feature_importances_, index=res["features"]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot.barh(ax=ax, color="#2ca02c")
    ax.set_title(f"{res['name']} - Random Forest feature importance")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# %% 8. Run the model
if __name__ == "__main__":
    df = add_indicators(load_data(DATA_PATH))
    days = sorted(df["date"].unique())
    print(f"Data: {len(df)} bars, {len(days)} sessions")
    print(f"Walk-forward: warm-up {days[0]} -> {days[WARMUP_SESSIONS - 1]}, "
          f"out-of-sample {days[WARMUP_SESSIONS]} -> {days[-1]} (RF retrained every session)\n")

    res = run_model("VWAP + RF", df, signals_vwap, [vwap_features])

    print("=" * 60)
    print("  VWAP + RF  -  output")
    print("=" * 60)
    print_metrics(res["metrics"])

    print("\nTrades:")
    print(res["trades"].to_string(index=False) if len(res["trades"]) else "  none")

    pd.Series(res["metrics"]).to_csv(os.path.join(OUT_DIR, "metrics_vwap.csv"))
    res["trades"].to_csv(os.path.join(OUT_DIR, "trades_vwap.csv"), index=False)
    plot_signals(res, os.path.join(OUT_DIR, "signals_vwap.png"))
    plot_equity(res, os.path.join(OUT_DIR, "equity_vwap.png"))
    plot_importance(res, os.path.join(OUT_DIR, "feature_importance_vwap.png"))
    print(f"\nSaved tables and charts to {OUT_DIR}")
