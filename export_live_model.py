"""Train the final DUAL Random Forest on ALL sessions and save it for the live dashboard.

The backtest (export_review3_csv.py) uses walk-forward models, so its results stay out-of-sample.
For LIVE signals we need one model trained on everything we have, saved to disk, so the dashboard
can score new candidates without ever retraining.

Writes
  data/rf_model.joblib     the fitted RandomForestClassifier (scikit-learn version pinned in the card)
  data/model_card.json     feature order + exact formulas, training range, versions, and test fixtures:
                             - feature_fixture: feature rows + expected probability for real candidates
                             - repricing_check: confirms Black-Scholes re-pricing from the CSV
                               reproduces every exit in data/review3_signals.csv

Usage:  python3 export_live_model.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

import model3_dual_vwap_bb_rf as m

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data")

FEATURE_FORMULAS = {
    "d_vwap": "(close - vwap) / vwap",
    "d_vwap_chg3": "d_vwap - d_vwap.shift(3)",
    "vwap_slope6": "vwap.pct_change(6)",
    "z": "(close - ma20) / sd20   (sd20 = population std, ddof=0)",
    "BW": "(upper - lower) / ma20   (upper/lower = ma20 +/- 2*sd20)",
    "BW_chg5": "BW.pct_change(5)",
    "ma_slope6": "ma20.pct_change(6)",
    "vol_ratio": "volume / volume.rolling(20).mean()",
    "mins_since_open": "hour*60 + minute - 555   (minutes since 09:15 IST)",
    "ret_1bar": "close.pct_change()",
    "side": "+1 for a BUY_CE candidate, -1 for BUY_PE",
}


def reprice_from_csv(sig):
    """Re-simulate every candidate's exit using ONLY the CSV closes + Black-Scholes (what the dashboard does)."""
    t = pd.to_datetime(sig["datetime"])
    day = sig["datetime"].str[:10].values
    close = sig["close"].values
    hhmm = t.dt.strftime("%H:%M").values
    T0 = m.DTE_DAYS / 365
    bad = 0
    for i in np.where(sig["decision"] != "HOLD")[0]:
        kind = "CE" if sig["decision"].iat[i] == "BUY_CE" else "PE"
        K = round(close[i] / m.STRIKE_STEP) * m.STRIKE_STEP
        entry = m.bs_price(close[i], K, T0, kind)
        tgt, stp = sig["target_pct"].iat[i], sig["stop_pct"].iat[i]
        j, reason = i, "session_end"
        while j + 1 < len(sig) and day[j + 1] == day[i]:
            j += 1
            r = m.bs_price(close[j], K, T0 - (j - i) * 5 / 525600, kind) / entry - 1
            if r >= tgt: reason = "target"; break
            if r <= -stp: reason = "stop"; break
            if j - i >= m.MAX_HOLD_BARS: reason = "time"; break
            if hhmm[j] >= m.SQUARE_OFF.strftime("%H:%M"): reason = "square_off"; break
        exit_px = m.bs_price(close[j], K, T0 - (j - i) * 5 / 525600, kind)
        ok = (K == sig["strike"].iat[i] and abs(entry - sig["entry_premium"].iat[i]) < 0.01
              and abs(exit_px - sig["exit_premium"].iat[i]) < 0.01 and reason == sig["exit_reason"].iat[i]
              and str(t.iat[j]) == str(pd.Timestamp(sig["exit_datetime"].iat[i])))
        bad += not ok
    return bad


def main():
    df = m.add_indicators(m.load_data(m.DATA_PATH))
    d = df.copy()
    d["signal"] = m.signals_dual(d)
    X = pd.concat([m.vwap_features(d), m.bb_features(d), m.common_features(d)], axis=1)
    X["side"] = d["signal"]
    cand = [i for i in np.where(d["signal"] != 0)[0] if X.iloc[i].notna().all()]
    y = pd.Series({i: m.simulate_trade(d, i)["label"] for i in cand}, dtype=int)

    model = RandomForestClassifier(**m.RF_PARAMS).fit(X.loc[cand], y[cand])
    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(OUT_DIR, "rf_model.joblib"))

    # fixtures: the dashboard must compute the same features and get the same probability
    fixture_idx = cand[-5:]
    p = model.predict_proba(X.loc[fixture_idx])[:, 1]
    fixture = [{"datetime": str(d["Datetime"].iat[i]),
                "features": {k: round(float(v), 10) for k, v in X.loc[i].items()},
                "expected_probability": round(float(pp), 6)} for i, pp in zip(fixture_idx, p)]

    sig = pd.read_csv(os.path.join(OUT_DIR, "review3_signals.csv"))
    mismatches = reprice_from_csv(sig)

    card = {
        "model": "RandomForestClassifier - DUAL (VWAP + Bollinger) meta-filter",
        "file": "rf_model.joblib",
        "purpose": "LIVE scoring only. Backtest results come from walk-forward models (review3_signals.csv).",
        "trained_on": {"sessions": f"{df['date'].min()} -> {df['date'].max()}", "candidates": len(cand),
                       "positive_rate": round(float(y.mean()), 4)},
        "params": {k: v for k, v in m.RF_PARAMS.items()},
        "label": f"1 if the ATM premium hits +{m.TARGET:.0%} before -{m.STOP:.0%} within {m.MAX_HOLD_BARS} bars",
        "feature_order": list(X.columns),
        "feature_formulas": FEATURE_FORMULAS,
        "candidate_rule": {
            "BUY_CE": f"close > vwap AND z[t-1] <= -{m.Z_PB} AND z[t] > -{m.Z_PB} AND {m.ENTRY_START:%H:%M} <= time <= {m.ENTRY_END:%H:%M}",
            "BUY_PE": f"close < vwap AND z[t-1] >= {m.Z_PB} AND z[t] < {m.Z_PB} AND {m.ENTRY_START:%H:%M} <= time <= {m.ENTRY_END:%H:%M}",
        },
        "versions": {"scikit-learn": sklearn.__version__, "joblib": joblib.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "feature_fixture": fixture,
        "repricing_check": {"candidates": int((sig["decision"] != "HOLD").sum()), "mismatches": mismatches},
    }
    with open(os.path.join(OUT_DIR, "model_card.json"), "w") as f:
        json.dump(card, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    print(f"Saved data/rf_model.joblib (trained on {len(cand)} candidates, {df['date'].nunique()} sessions) and data/model_card.json")
    print(f"Re-pricing check: {card['repricing_check']['candidates']} candidates re-simulated from CSV closes, {mismatches} mismatches")


if __name__ == "__main__":
    main()
