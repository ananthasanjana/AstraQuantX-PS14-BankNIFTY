import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==============================================================================
# STEP 1: LOAD DATA & COMPUTE ONLY DAILY-RESETTING VWAP
# ==============================================================================
df = pd.read_csv("banknifty_clean_5m.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Date"] = df["Datetime"].dt.date
df["Time_Str"] = df["Datetime"].dt.strftime("%H:%M")

# Typical Price & Cumulative Daily VWAP (resets every morning at 09:15 IST)
df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3.0
df["TP_Vol"] = df["TP"] * df["Volume"]
df["VWAP"] = df.groupby("Date")["TP_Vol"].cumsum() / df.groupby("Date")["Volume"].cumsum()

# VWAP-specific features
df["VWAP_Dist"] = (df["Close"] - df["VWAP"]) / df["VWAP"]
df["VWAP_Slope"] = df["VWAP"].diff()
df["Vol_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
df["Ret_1bar"] = df["Close"].pct_change()
df = df.dropna().reset_index(drop=True)

# ==============================================================================
# STEP 2: GENERATE CANDIDATE SIGNALS USING ONLY VWAP
# ==============================================================================
in_win = (df["Time_Str"] >= "09:30") & (df["Time_Str"] <= "14:30")
df["Sig_VWAP"] = 0
df.loc[(df["Close"] > df["VWAP"]) & (df["VWAP_Dist"] > 0.0015) & (df["VWAP_Slope"] > 0) & in_win, "Sig_VWAP"] = 1   # BUY CE
df.loc[(df["Close"] < df["VWAP"]) & (df["VWAP_Dist"] < -0.0015) & (df["VWAP_Slope"] < 0) & in_win, "Sig_VWAP"] = -1 # BUY PE

# ==============================================================================
# STEP 3: META-LABELLING (+30% TARGET VS -20% STOP WITHIN 12 BARS)
# ==============================================================================
labels = np.zeros(len(df), dtype=int)
pnl_list = np.zeros(len(df))

for i in range(len(df) - 12):
    sig = df.loc[i, "Sig_VWAP"]
    if sig == 0:
        continue
    col = "ATM_CE_Price" if sig == 1 else "ATM_PE_Price"
    ep = df.loc[i, col]
    exit_p = df.loc[i + 12, col]
    for fp in df.loc[i + 1 : i + 12, col].values:
        ret = (fp - ep) / ep
        if ret >= 0.30:
            labels[i] = 1
            exit_p = fp
            break
        elif ret <= -0.20:
            labels[i] = 0
            exit_p = fp
            break
    # Net P&L for 1 lot (30 units) minus Rs. 40 brokerage
    pnl_list[i] = ((exit_p - ep) * 30.0) - 40.0

df["Trade_PnL"] = pnl_list
df["Label"] = (df["Trade_PnL"] > 0).astype(int)

# ==============================================================================
# STEP 4: TRAIN RANDOM FOREST CLASSIFIER (CHRONOLOGICAL 60% TRAIN / 40% TEST)
# ==============================================================================
vwap_features = ["VWAP_Dist", "VWAP_Slope", "Vol_Ratio", "Ret_1bar"]
split = int(len(df) * 0.60)
cand = df[df["Sig_VWAP"] != 0]
train_c, test_c = cand[cand.index < split], cand[cand.index >= split]

rf_vwap = RandomForestClassifier(n_estimators=150, max_depth=4, class_weight="balanced", random_state=42)
rf_vwap.fit(train_c[vwap_features], train_c["Label"])

y_true = test_c["Label"]
y_pred = (rf_vwap.predict_proba(test_c[vwap_features])[:, 1] >= 0.50).astype(int)

# Use .copy() to prevent Pandas SettingWithCopy warnings
executed = test_c[y_pred == 1].copy()

# ==============================================================================
# STEP 5: CALCULATE ALL QUANTITATIVE & CLASSIFICATION METRICS
# ==============================================================================
initial_capital = 100000.0
net_profit = executed["Trade_PnL"].sum()
win_rate = (executed["Trade_PnL"] > 0).mean() * 100 if len(executed) > 0 else 0.0

if len(executed) > 0:
    equity = pd.concat([pd.Series([initial_capital]), initial_capital + executed["Trade_PnL"].cumsum()], ignore_index=True)
    max_dd = (((equity - equity.cummax()) / equity.cummax()) * 100.0).min()
    rets = executed["Trade_PnL"] / initial_capital
    risk_vol = rets.std() * 100.0
    sharpe = (rets.mean() / rets.std()) * np.sqrt(len(executed)) if rets.std() > 0 else 0.0
else:
    max_dd = risk_vol = sharpe = 0.0

print("==========================================================")
print("   EXPERIMENT 1: SINGLE VWAP INDICATOR + RANDOM FOREST    ")
print("==========================================================")
print(f"Test Candidates      : {len(test_c)}")
print(f"RF Approved Trades   : {len(executed)}")
print(f"Accuracy Rate        : {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision            : {precision_score(y_true, y_pred, zero_division=0):.3f}")
print(f"Recall               : {recall_score(y_true, y_pred, zero_division=0):.3f}")
print(f"F1-Score             : {f1_score(y_true, y_pred, zero_division=0):.3f}")
print(f"Strategy Win Rate    : {win_rate:.1f}%")
print(f"Risk Rate (Cap / Vol): 1.0% per trade / {risk_vol:.2f}% volatility")
print(f"Sharpe Ratio         : {sharpe:.2f}")
print(f"Maximum Drawdown     : {max_dd:.2f}%")
print(f"Net Profit (After ₹40): ₹{net_profit:.2f}")
print("==========================================================")

# ==============================================================================
# STEP 6: PLOT CUMULATIVE RETURNS GRAPH USING MATPLOTLIB
# ==============================================================================
if len(executed) > 0:
    executed["Cumulative_PnL"] = executed["Trade_PnL"].cumsum()
    
    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(executed["Datetime"]), executed["Cumulative_PnL"], label="VWAP + RF Strategy Equity Curve", color="blue", linewidth=2)
    plt.title("Cumulative Strategy P&L Over Time (Test Set)")
    plt.xlabel("Datetime")
    plt.ylabel("Cumulative Net P&L (₹)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    print("\nDisplaying cumulative returns plot window...")
    plt.show()
else:
    print("\nNo trades executed by the model on the test set to plot.")