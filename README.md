# LuminX — Automated Bank NIFTY Options Trading (ASTRAQUANT X 2026 · PS14)

**Strategy:** VWAP sets the direction, a Bollinger Band z-score pullback sets the timing, and a Random Forest approves or rejects each CE/PE signal.

Options are priced with Black-Scholes (ATM strike), with:
- a +12% target and −6% stop on the premium
- 1% of capital risked per trade, max 3 trades a day, and a −2% daily loss cap
- full NSE costs and slippage on every trade

## Results (out-of-sample, walk-forward, 22 Oct – 13 Nov 2026, after costs)

| Model | Trades | Net P&L | Sharpe | Max DD | Win rate |
|---|---|---|---|---|---|
| VWAP + RF | 20 | ₹4,380 | 0.34 | −7.18% | 35.0% |
| Bollinger + RF | 25 | ₹38,395 | 2.61 | −7.70% | 44.0% |
| **DUAL (VWAP + BB) + RF** | **19** | **₹1,00,362** | **7.24** | **−3.15%** | **57.9%** |

Control (same rules without the Random Forest): VWAP ₹12,148 · Bollinger ₹66,582 · DUAL ₹86,843.

These figures cover about 16 test sessions in one trending month, so read them as a ranking of the models, not as expected returns.

## Files

| File | What it does |
|---|---|
| `model1_vwap_rf.py` | VWAP + Random Forest |
| `model2_bollinger_rf.py` | Bollinger Bands + Random Forest |
| `model3_dual_vwap_bb_rf.py` | DUAL + Random Forest, prints the three-way comparison |
| `model0_control_no_rf.py` | the same three rules without the Random Forest (control experiment) |
| `export_site_data.py` | writes `website/data.js` for the LuminX website |
| `export_review3_csv.py` | writes `data/review3_signals.csv` + `data/strategy_meta.json` for the dashboard |
| `export_live_model.py` | writes `data/rf_model.joblib` + `data/model_card.json` for live scoring |
| `website/` | **LuminX** interactive site: paper-trading terminal, signals, backtest, method, Review 3 map |
| `COPILOT_MASTER_PROMPT.md` | build spec for the Flask version of the dashboard |
| `outputs_*/` | charts and tables produced by the model scripts |

All scripts read `banknifty_clean_5m.csv` from this folder.

## Run

```bash
pip install pandas numpy scipy scikit-learn matplotlib joblib
python3 model3_dual_vwap_bb_rf.py      # three-way comparison
python3 model0_control_no_rf.py        # control vs Random Forest
python3 export_site_data.py            # refresh the website data
```

**Open the website:** double-click `website/index.html` (it needs internet for fonts and chart libraries), or run:

```bash
python3 -m http.server 8765 --directory website
```

and visit http://localhost:8765.

All trading on the site is simulated on historical data. No real orders are placed.
