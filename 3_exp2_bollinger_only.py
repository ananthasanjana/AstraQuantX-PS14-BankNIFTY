import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==============================================================================
# STEP 1: LOAD DATA & COMPUTE 20-PERIOD BOLLINGER BANDS
# ==============================================================================
df = pd.read_csv("banknifty_clean_5m.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Date"] = df["Datetime"].dt.date
df["Time_Str"] = df["Datetime"].dt.strftime("%H:%M")

n, k = 20, 1.5
df["MA20"] = df["Close"].rolling(n).mean()
df["STD20"] = df["Close"].rolling(n).std()
df["Upper_Band"] = df["MA20"] + (k * df["STD20"])
df["Lower_Band"] = df["MA20"] - (k * df["STD20"])
df["Z_Score"] = (df["Close"] - df["MA20"]) / df["STD20"]
df["BW"] = (df["Upper_Band"] - df["Lower_Band"]) / df["MA20"]
df["BW_Change"] = df["BW"].diff()

df["Vol_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
df["Ret_1bar"] = df["Close"].pct_change()
df = df.dropna().reset_index(drop=True)

# ==============================================================================
# STEP 2: GENERATE CANDIDATE SIGNALS USING BOLLINGER BANDS
# ==============================================================================
in_win = (df["Time_Str"] >= "09:30") & (df["Time_Str"] <= "14:30")
df["Sig_BB"] = 0
df.loc[(df["Z_Score"] >= k) & (df["BW_Change"] > 0) & in_win, "Sig_BB"] = 1    # BUY CE
df.loc[(df["Z_Score"] <= -k) & (df["BW_Change"] > 0) & in_win, "Sig_BB"] = -1  # BUY PE

# ==============================================================================
# STEP 3: META-LABELLING (+30% TARGET VS -20% STOP WITHIN 12 BARS / 15:15 EXIT)
# ==============================================================================
labels = np.zeros(len(df), dtype=int)
pnl_list = np.zeros(len(df))
exit_idx_list = np.arange(len(df))

for i in range(len(df) - 12):
    sig = df.loc[i, "Sig_BB"]
    if sig == 0:
        continue

    col = "ATM_CE_Price" if sig == 1 else "ATM_PE_Price"
    ep = df.loc[i, col]
    current_date = df.loc[i, "Date"]
    
    # Default exit at bar i + 12 (or earlier if day ends / 15:15 IST)
    actual_exit_idx = i + 12
    exit_p = df.loc[actual_exit_idx, col]

    for j in range(i + 1, i + 13):
        # Force intraday square-off at 15:15 IST or before next day
        if df.loc[j, "Date"] != current_date or df.loc[j, "Time_Str"] >= "15:15":
            actual_exit_idx = j if df.loc[j, "Date"] == current_date else j - 1
            exit_p = df.loc[actual_exit_idx, col]
            break

        fp = df.loc[j, col]
        ret = (fp - ep) / ep
        if ret >= 0.30:
            labels[i] = 1
            exit_p = fp
            actual_exit_idx = j
            break
        elif ret <= -0.20:
            labels[i] = 0
            exit_p = fp
            actual_exit_idx = j
            break

    exit_idx_list[i] = actual_exit_idx
    # Net P&L for 1 lot (30 units) minus Rs. 40 brokerage
    pnl_list[i] = ((exit_p - ep) * 30.0) - 40.0
df["Trade_PnL"] = pnl_list
df["Label"] = (df["Trade_PnL"] > 0).astype(int)
df["Exit_Idx"] = exit_idx_list

# ==============================================================================
# STEP 4: TRAIN RANDOM FOREST CLASSIFIER (LEAK-FREE 60% TRAIN / 40% TEST)
# ==============================================================================
bb_features = ["Z_Score", "BW", "BW_Change", "Vol_Ratio", "Ret_1bar"]
split = int(len(df) * 0.60)
cand = df[df["Sig_BB"] != 0].copy()

# Exclude training trades whose 12-bar exit window crosses into the test set
train_c = cand[cand["Exit_Idx"] < split]
test_c = cand[cand.index >= split].copy()

rf_bb = RandomForestClassifier(
    n_estimators=150, max_depth=4, class_weight="balanced", random_state=42
)
rf_bb.fit(train_c[bb_features], train_c["Label"])

y_true = test_c["Label"]
y_pred = (rf_bb.predict_proba(test_c[bb_features])[:, 1] >= 0.50).astype(int)
test_c["RF_Pred"] = y_pred

# Filter out overlapping trades (enforce 1 active trade at a time, max 3/day)
executed_rows = []
last_exit_idx = -1
daily_counts = {}

for idx, row in test_c[test_c["RF_Pred"] == 1].iterrows():
    if idx <= last_exit_idx:
        continue  # Skip while a position is already open
    d = row["Date"]
    if daily_counts.get(d, 0) >= 3:
        continue  # Enforce max 3 trades per day
    executed_rows.append(row)
    last_exit_idx = row["Exit_Idx"]
    daily_counts[d] = daily_counts.get(d, 0) + 1

executed = pd.DataFrame(executed_rows)

# ==============================================================================
# STEP 5: CALCULATE ALL QUANTITATIVE & CLASSIFICATION METRICS
# ==============================================================================
initial_capital = 100000.0
net_profit = executed["Trade_PnL"].sum() if len(executed) > 0 else 0.0
win_rate = (executed["Trade_PnL"] > 0).mean() * 100 if len(executed) > 0 else 0.0

if len(executed) > 0:
    executed["Cum_PnL"] = executed["Trade_PnL"].cumsum()
    executed["Cum_Return_Pct"] = (executed["Cum_PnL"] / initial_capital) * 100.0
    equity = pd.concat(
        [pd.Series([initial_capital]), initial_capital + executed["Cum_PnL"]],
        ignore_index=True,
    )
    max_dd = (((equity - equity.cummax()) / equity.cummax()) * 100.0).min()
    rets = executed["Trade_PnL"] / initial_capital
    risk_vol = rets.std() * 100.0
    sharpe = (rets.mean() / rets.std()) * np.sqrt(len(executed)) if rets.std() > 0 else 0.0
else:
    max_dd = risk_vol = sharpe = 0.0

print("==========================================================")
print(" EXPERIMENT 2: SINGLE BOLLINGER INDICATOR + RANDOM FOREST ")
print("==========================================================")
print(f"Test Candidates       : {len(test_c)}")
print(f"RF Approved Trades    : {len(executed)}")
print(f"Accuracy Rate         : {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision             : {precision_score(y_true, y_pred, zero_division=0):.3f}")
print(f"Recall                : {recall_score(y_true, y_pred, zero_division=0):.3f}")
print(f"F1-Score              : {f1_score(y_true, y_pred, zero_division=0):.3f}")
print(f"Strategy Win Rate     : {win_rate:.1f}%")
print(f"Risk Rate (Cap / Vol) : 1.0% per trade / {risk_vol:.2f}% volatility")
print(f"Sharpe Ratio          : {sharpe:.2f}")
print(f"Maximum Drawdown      : {max_dd:.2f}%")
print(f"Net Profit (After ₹40): ₹{net_profit:.2f}")
print("==========================================================")

# ==============================================================================
# STEP 6: PLOT CUMULATIVE RETURNS USING MATPLOTLIB
# ==============================================================================
if len(executed) > 0:
    plt.figure(figsize=(10, 5))
    plt.plot(
        executed["Datetime"],
        executed["Cum_Return_Pct"],
        marker="o",
        linewidth=2,
        color="#0d9488",
        label="BB + Random Forest Strategy",
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("Cumulative Returns (%) — Out-of-Sample Test Set", fontsize=13, fontweight="bold")
    plt.xlabel("Date / Time")
    plt.ylabel("Cumulative Return (%)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("cumulative_returns_bb_rf.png", dpi=300)
    plt.show()
else:
    print("No trades executed in the test set to plot.")