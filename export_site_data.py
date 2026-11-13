"""Export the backtest results of all six models to website/data.js.

Runs the same engine as the model scripts:
  - model3_dual_vwap_bb_rf.py  -> VWAP + RF, Bollinger + RF, DUAL + RF
  - model0_control_no_rf.py    -> the same three rules without the Random Forest (control)
and writes one JavaScript file the website loads (works by double-clicking index.html, no server).

Usage:  python3 export_site_data.py
"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import model0_control_no_rf as ctl
import model3_dual_vwap_bb_rf as rf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "website", "data.js")


def r(x, nd=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return round(float(x), nd)


def clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if (np.isnan(v) or np.isinf(v)) else round(float(v), 4)
    return v


def export_model(key, family, use_rf, res, module):
    d = res["data"]
    pos = pd.Series(np.arange(len(d)), index=d["Datetime"])
    p_by_idx = dict(zip(res["preds"]["idx"], res["preds"]["p_hat"])) if len(res["preds"]) else {}
    traded = set(pos[res["trades"]["entry_time"]].tolist()) if len(res["trades"]) else set()

    candidates = []
    for i in sorted(p_by_idx):
        s = module.simulate_trade(d, i)
        p = float(p_by_idx[i])
        candidates.append(dict(
            i=int(i), side=int(d["signal"].iat[i]), p=r(p, 3) if use_rf else None,
            ok=bool(p >= module.TAU) if use_rf else True, traded=i in traded,
            y=int(s["label"]), xi=int(s["exit_idx"]), xr=s["exit_reason"],
            ep=r(s["entry_premium"]), xp=r(s["exit_premium"]), k=int(s["strike"]),
            ret=r((s["exit_premium"] / s["entry_premium"] - 1) * 100, 2),
        ))

    trades = []
    for _, t in res["trades"].iterrows():
        trades.append(dict(
            i=int(pos[t["entry_time"]]), xi=int(pos[t["exit_time"]]), side=t["side"], k=int(t["strike"]),
            lots=int(t["lots"]), buy=r(t["buy"]), sell=r(t["sell"]), costs=r(t["costs"]), pnl=r(t["pnl"]),
            p=r(t.get("p_hat"), 3) if use_rf else None, reason=t["exit_reason"], delta=r(t["delta"]),
        ))

    importance = []
    if use_rf and res.get("model") is not None:
        imp = pd.Series(res["model"].feature_importances_, index=res["features"]).sort_values(ascending=False)
        importance = [[k, r(v, 4)] for k, v in imp.items()]

    return dict(
        key=key, family=family, rf=use_rf, name=res["name"],
        metrics={k: clean(v) for k, v in res["metrics"].items()},
        candidates=candidates, trades=trades, importance=importance,
        daily=[[str(k), r(v)] for k, v in res["daily"].items()],
    )


def main():
    df = rf.add_indicators(rf.load_data(rf.DATA_PATH))
    days = sorted(df["date"].unique())
    oos_start = days[rf.WARMUP_SESSIONS]

    runs = [
        ("vwap_rf", "vwap", True, rf.run_model("VWAP + RF", df, rf.signals_vwap, [rf.vwap_features]), rf),
        ("bb_rf", "bb", True, rf.run_model("Bollinger + RF", df, rf.signals_bb, [rf.bb_features]), rf),
        ("dual_rf", "dual", True, rf.run_model("DUAL (VWAP + BB) + RF", df, rf.signals_dual,
                                               [rf.vwap_features, rf.bb_features]), rf),
    ]
    cdf = ctl.add_indicators(ctl.load_data(ctl.DATA_PATH))
    runs += [
        ("vwap_ctl", "vwap", False, ctl.run_model("VWAP rule only", cdf, ctl.signals_vwap, [ctl.vwap_features]), ctl),
        ("bb_ctl", "bb", False, ctl.run_model("Bollinger rule only", cdf, ctl.signals_bb, [ctl.bb_features]), ctl),
        ("dual_ctl", "dual", False, ctl.run_model("DUAL rule only", cdf, ctl.signals_dual,
                                                  [ctl.vwap_features, ctl.bb_features]), ctl),
    ]

    # chart times: IST wall-clock written as UTC seconds so the chart shows 09:15-15:25
    t = (df["Datetime"].dt.tz_localize(None).astype("int64") // 10**9).astype(int)
    bars = dict(
        t=t.tolist(),
        o=[r(x) for x in df["Open"]], h=[r(x) for x in df["High"]], l=[r(x) for x in df["Low"]],
        c=[r(x) for x in df["Close"]], v=[int(x) for x in df["Volume"]],
        vwap=[r(x) for x in df["VWAP"]], ma=[r(x) for x in df["MA"]],
        ub=[r(x) for x in df["UB"]], lb=[r(x) for x in df["LB"]],
        z=[r(x, 3) for x in df["z"]], d=[r(x * 100, 4) for x in df["d"]], bw=[r(x * 100, 4) for x in df["BW"]],
        day=[str(x) for x in df["date"]],
    )

    site = dict(
        meta=dict(
            generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
            bars=len(df), sessions=len(days), first=str(days[0]), last=str(days[-1]),
            oosStart=str(oos_start), oosStartIndex=int(np.argmax(df["date"].values == oos_start)),
            capital=rf.CAPITAL, lot=rf.LOT_SIZE, riskPerTrade=rf.RISK_PER_TRADE,
            target=rf.TARGET, stop=rf.STOP, maxHold=rf.MAX_HOLD_BARS, maxTrades=rf.MAX_TRADES_PER_DAY,
            dailyCap=rf.DAILY_LOSS_CAP, iv=rf.IV, rate=rf.RATE, dte=rf.DTE_DAYS, strikeStep=rf.STRIKE_STEP,
            bbN=rf.BB_N, bbK=rf.BB_K, zpb=rf.Z_PB, tau=rf.TAU, warmup=rf.WARMUP_SESSIONS,
            rfParams={k: v for k, v in rf.RF_PARAMS.items() if k != "n_jobs"},
            costs=dict(tick=rf.TICK, slipTicks=rf.SLIP_TICKS, brokerage=rf.BROKERAGE, stt=rf.STT_SELL,
                       exch=rf.EXCH_TXN, sebi=rf.SEBI_FEE, stamp=rf.STAMP_BUY, gst=rf.GST),
            entryStart="09:30", entryEnd="14:30", squareOff="15:15",
        ),
        sections=rf.METRIC_SECTIONS,
        bars=bars,
        models={key: export_model(key, fam, use_rf, res, mod) for key, fam, use_rf, res, mod in runs},
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("// Generated by export_site_data.py - do not edit by hand\n")
        f.write("window.SITE_DATA = ")
        json.dump(site, f, separators=(",", ":"))
        f.write(";\n")
    print(f"Wrote {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB)")
    for key, *_ in runs:
        m = site["models"][key]["metrics"]
        print(f"  {key:<9} trades={m['Trades taken']:<3} net P&L={m['Net P&L (Rs)']:<10} Sharpe={m['Sharpe (ann.)']}")


if __name__ == "__main__":
    main()
