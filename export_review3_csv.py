"""Export the DUAL (VWAP + Bollinger) + Random Forest results to the files the dashboard reads.

Writes
  data/review3_signals.csv   one row per 5-minute bar of the out-of-sample period, with indicators,
                             the rule decision, the walk-forward RF confidence, and - for every
                             candidate - the option strike, entry premium, target / stop premiums,
                             and where the trade exits (target / stop / time / 15:15).
  data/strategy_meta.json    every strategy setting (capital, lot size, risk rules, costs, ...)
                             plus the expected results, all read from model3_dual_vwap_bb_rf.py.

The dashboard must take ALL numbers from these two files. Nothing is copied from the slides.

Usage:  python3 export_review3_csv.py
"""
import json
import os

import numpy as np
import pandas as pd

import model3_dual_vwap_bb_rf as m

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data")


def build():
    df = m.add_indicators(m.load_data(m.DATA_PATH))
    days = sorted(df["date"].unique())
    oos_start = days[m.WARMUP_SESSIONS]

    rf_res = m.run_model("DUAL (VWAP + BB) + RF", df, m.signals_dual, [m.vwap_features, m.bb_features])
    d = rf_res["data"]
    p_by_idx = dict(zip(rf_res["preds"]["idx"], rf_res["preds"]["p_hat"]))

    # candidates the rule produces in the out-of-sample window (same set the control model trades)
    X = pd.concat([m.vwap_features(d), m.bb_features(d), m.common_features(d)], axis=1)
    oos = d["date"] >= oos_start
    cand = [i for i in np.where((d["signal"] != 0) & oos)[0] if X.iloc[i].notna().all()]

    out = d.loc[oos, ["Datetime", "Open", "High", "Low", "Close", "Volume", "VWAP", "MA", "UB", "LB", "z", "d"]].copy()
    out.columns = ["datetime", "open", "high", "low", "close", "volume", "vwap", "bb_mid", "bb_upper", "bb_lower", "z_score", "vwap_dist"]
    for c in ["decision", "rf_confidence", "strike", "entry_premium", "target_pct", "stop_pct", "target_premium",
              "stop_premium", "exit_datetime", "exit_premium", "exit_reason", "hit_target"]:
        out[c] = np.nan
    out["decision"] = "HOLD"
    out = out.astype({"exit_datetime": object, "exit_reason": object})

    for i in cand:
        s = m.simulate_trade(d, i)
        row = d.index[i]
        out.loc[row, "decision"] = "BUY_CE" if d["signal"].iat[i] == 1 else "BUY_PE"
        out.loc[row, "rf_confidence"] = round(float(p_by_idx[i]), 4) if i in p_by_idx else np.nan
        out.loc[row, "strike"] = s["strike"]
        out.loc[row, "entry_premium"] = round(s["entry_premium"], 4)
        out.loc[row, "target_pct"] = m.TARGET
        out.loc[row, "stop_pct"] = m.STOP
        out.loc[row, "target_premium"] = round(s["entry_premium"] * (1 + m.TARGET), 4)
        out.loc[row, "stop_premium"] = round(s["entry_premium"] * (1 - m.STOP), 4)
        out.loc[row, "exit_datetime"] = str(d["Datetime"].iat[s["exit_idx"]])
        out.loc[row, "exit_premium"] = round(s["exit_premium"], 4)
        out.loc[row, "exit_reason"] = s["exit_reason"]
        out.loc[row, "hit_target"] = s["label"]

    out["datetime"] = out["datetime"].astype(str)
    # keep full precision so indicators/features recomputed from this CSV match the model exactly
    return df, days, oos_start, out, rf_res


def replay(sig, conf, slip_ticks, use_rf=True):
    """Reference backtest that uses ONLY the CSV + meta settings (what the dashboard must reproduce)."""
    slip = slip_ticks * m.TICK
    t = pd.to_datetime(sig["datetime"])
    pos = {v: k for k, v in enumerate(sig["datetime"])}
    trades, busy_until = [], -1
    day_n, day_pnl = {}, {}
    for k, r in enumerate(sig.itertuples(index=False)):
        if r.decision == "HOLD":
            continue
        if use_rf and (pd.isna(r.rf_confidence) or r.rf_confidence < conf):
            continue
        day = r.datetime[:10]
        if k <= busy_until or day_n.get(day, 0) >= m.MAX_TRADES_PER_DAY or day_pnl.get(day, 0) <= -m.DAILY_LOSS_CAP * m.CAPITAL:
            continue
        buy = r.entry_premium + slip
        sell = max(r.exit_premium - slip, 0.05)
        lots = max(1, int(m.RISK_PER_TRADE * m.CAPITAL / (m.STOP * buy * m.LOT_SIZE)))
        qty = lots * m.LOT_SIZE
        costs = m.trade_costs(buy, sell, qty)
        pnl = (sell - buy) * qty - costs
        trades.append(pnl)
        day_n[day] = day_n.get(day, 0) + 1
        day_pnl[day] = day_pnl.get(day, 0) + pnl
        busy_until = pos[r.exit_datetime]
    return len(trades), round(sum(trades), 0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, days, oos_start, sig, rf_res = build()
    sig.to_csv(os.path.join(OUT_DIR, "review3_signals.csv"), index=False)

    ctl = __import__("model0_control_no_rf")
    cdf = ctl.add_indicators(ctl.load_data(ctl.DATA_PATH))
    ctl_res = ctl.run_model("DUAL rule only", cdf, ctl.signals_dual, [ctl.vwap_features, ctl.bb_features])

    meta = {
        "source": "model3_dual_vwap_bb_rf.py (all values read from the code, not from slides)",
        "data_file": os.path.basename(m.DATA_PATH),
        "strategy": "DUAL: VWAP gives direction, Bollinger z-score pullback gives timing, Random Forest approves",
        "indicators": {"bb_window": m.BB_N, "bb_k": m.BB_K, "dual_pullback_z": m.Z_PB, "vwap": "typical price, resets daily"},
        "option_pricing": {"model": "Black-Scholes ATM", "iv": m.IV, "rate": m.RATE, "days_to_expiry": m.DTE_DAYS, "strike_step": m.STRIKE_STEP,
                           "note": "Dataset has no real option chain; premiums are priced with Black-Scholes in the Python code."},
        "capital": m.CAPITAL, "lot_size": m.LOT_SIZE, "risk_per_trade": m.RISK_PER_TRADE,
        "target_pct": m.TARGET, "stop_pct": m.STOP, "max_hold_bars": m.MAX_HOLD_BARS,
        "max_trades_per_day": m.MAX_TRADES_PER_DAY, "daily_loss_cap_pct": m.DAILY_LOSS_CAP,
        "daily_loss_cap_rs": m.DAILY_LOSS_CAP * m.CAPITAL,
        "entry_window": [m.ENTRY_START.strftime("%H:%M"), m.ENTRY_END.strftime("%H:%M")], "square_off": m.SQUARE_OFF.strftime("%H:%M"),
        "costs": {"tick": m.TICK, "slippage_ticks_default": m.SLIP_TICKS, "brokerage_per_order": m.BROKERAGE, "stt_sell": m.STT_SELL,
                  "exchange_txn": m.EXCH_TXN, "sebi_fee": m.SEBI_FEE, "stamp_buy": m.STAMP_BUY, "gst": m.GST},
        "rf": {"params": {k: v for k, v in m.RF_PARAMS.items() if k != "n_jobs"}, "tau_default": m.TAU,
               "label": f"1 if premium hits +{m.TARGET:.0%} before -{m.STOP:.0%} within {m.MAX_HOLD_BARS} bars"},
        "validation": {"method": "walk-forward, RF retrained before every session on earlier sessions only",
                       "warmup_sessions": m.WARMUP_SESSIONS, "train_start": str(days[0]),
                       "test_start": str(oos_start), "test_end": str(days[-1]), "sessions_total": len(days)},
        "expected": {
            "ai_model": {k: rf_res["metrics"][k] for k in ["Trades taken", "Net P&L (Rs)", "ROI %", "Sharpe (ann.)", "Max drawdown %",
                                                           "Win rate %", "Profit factor", "Accuracy rate %", "Precision %", "Recall %", "F1-score"]},
            "no_ml": {k: ctl_res["metrics"][k] for k in ["Trades taken", "Net P&L (Rs)", "ROI %", "Sharpe (ann.)", "Max drawdown %", "Win rate %", "Profit factor"]},
        },
    }
    with open(os.path.join(OUT_DIR, "strategy_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # self-check: the CSV alone must reproduce the code's results
    ai = replay(sig, m.TAU, m.SLIP_TICKS, True)
    noml = replay(sig, m.TAU, m.SLIP_TICKS, False)
    print(f"Wrote data/review3_signals.csv ({len(sig)} rows, {int((sig['decision'] != 'HOLD').sum())} candidates) and data/strategy_meta.json")
    print(f"CSV replay  AI model: {ai[0]} trades, Rs {ai[1]:,.0f}   (code: {rf_res['metrics']['Trades taken']} trades, Rs {rf_res['metrics']['Net P&L (Rs)']:,.0f})")
    print(f"CSV replay  No-ML   : {noml[0]} trades, Rs {noml[1]:,.0f}   (code: {ctl_res['metrics']['Trades taken']} trades, Rs {ctl_res['metrics']['Net P&L (Rs)']:,.0f})")


if __name__ == "__main__":
    main()
