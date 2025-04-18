import pandas as pd
import numpy as np
from ta import momentum, trend, volume, volatility
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

import random
CapitalUSD = 1000
DataSize = 2000

# Load CSV (make sure path is correct)
data = pd.read_csv("SPYUSUSD2years.csv")

# Select subset
data = data.iloc[:1+DataSize].copy()

# Ensure lowercase column names and remove spaces
data.columns = data.columns.str.lower().str.replace(' ', '')
#print(data.head())


# add technical indicators

data['rsi'] = momentum.RSIIndicator(data['close'], window=14).rsi()
data['sma'] = trend.SMAIndicator(data['close'], window=14).sma_indicator()
data['ema'] = trend.EMAIndicator(data['close'], window=14).ema_indicator()
data['stoch_k'] = momentum.StochasticOscillator(data['high'], data['low'], data['close'], window=14).stoch()
macd = trend.MACD(data['close'])
data['macd'] = macd.macd()
data['ad'] = volume.AccDistIndexIndicator(data['high'], data['low'], data['close'], data['volume']).acc_dist_index()
data['obv'] = volume.OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
data['roc'] = momentum.ROCIndicator(data['close'], window=14).roc()
data['williams_r'] = momentum.WilliamsRIndicator(data['high'], data['low'], data['close'], lbp=14).williams_r()
data['disparity_index'] = 100 * ((data['close'] - data['ema']) / data['ema'])

data.dropna(inplace=True)
pd.set_option("display.max_columns", 50)
#print(data.head())
#print(data.shape)




# define stateID Function



def get_rsi_level(x):
    return "RSI_Low" if x < 30 else "RSI_High" if x > 70 else "RSI_Mid"

def get_macd_level(x):
    return "MACD_Pos" if x > 0 else "MACD_Neg"

def get_sma_ema_level(close, sma, ema):
    if close > sma and close > ema: return "AboveBothMA"
    elif close < sma and close < ema: return "BelowBothMA"
    else: return "BetweenMA"

def get_stoch_level(x):
    x *= 100
    return "Stoch_High" if x > 80 else "Stoch_Low" if x < 20 else "Stoch_Mid"

def get_ad_level(x): return "AD_Pos" if x > 0 else "AD_Neg"
def get_obv_level(x): return "OBV_Pos" if x > 0 else "OBV_Neg"

def get_roc_level(x):
    return "ROC_Up" if x > 0.001 else "ROC_Down" if x < -0.001 else "ROC_Flat"

def get_williams_level(x):
    x *= 100
    return "Will_Oversold" if x < -80 else "Will_Overbought" if x > -20 else "Will_Neutral"

def get_disparity_level(x):
    return "Disp_Pos" if x > 0.1 else "Disp_Neg" if x < -0.1 else "Disp_Neutral"



## create sate ID


def generate_state(row):
    return "|".join([
        get_rsi_level(row['rsi']),
        get_macd_level(row['macd']),
        get_sma_ema_level(row['close'], row['sma'], row['ema']),
        get_stoch_level(row['stoch_k']),
        get_ad_level(row['ad']),
        get_obv_level(row['obv']),
        get_roc_level(row['roc']),
        get_williams_level(row['williams_r']),
        get_disparity_level(row['disparity_index']),
    ])

data['StateID'] = data.apply(generate_state, axis=1)

#print(data.head())



# initialize Q table


actions = ['Buy', 'Sell', 'Hold']
unique_states = data['StateID'].unique()

Q_table = pd.DataFrame([(s, a, 0) for s in unique_states for a in actions],
                       columns=['State', 'Action', 'Q'])

#print(" Q table ",Q_table)



# Q learning simulation


#print(Q_table)




import numpy as np
import random

# Parameters
epsilon = 0.2  # 20% exploration

# Define possible actions
actions = ['Buy', 'Sell', 'Hold']

# Let's pick row i from data
i = 1  # first time step
current_state = data.iloc[i]['StateID']

# Epsilon-greedy action selection
if np.random.rand() < epsilon:
    chosen_action = random.choice(actions)  # Explore: random action
else:
    # Exploit: choose best action based on Q-table
    q_values = Q_table[Q_table['State'] == current_state]
    max_q = q_values['Q'].max()
    best_actions = q_values[q_values['Q'] == max_q]['Action'].tolist()
    chosen_action = random.choice(best_actions)  # Break ties randomly

# Show chosen action for this state
#print("Time:", data.iloc[i]['timestamp'])
#print("State:", current_state)
#print("Chosen Action:", chosen_action)



# hyperparameters , reward

# --- Hyperparameters ---
alpha = 0.2               # Learning rate
gamma = 0.85              # Discount factor
epsilon = 0.3             # Exploration rate
hold_reward_scale = 100   # Reward scale for holding
actions = ['Buy', 'Sell', 'Hold']

# --- Initialize ---
Q_table['Q'] = 0
position = False
entry_price = None
entry_index = None
entry_state = None

# --- Track total reward ---
total_reward = 0

# --- Main loop ---
for i in range(4, len(data) - 1):
    current_state = data.iloc[i]['StateID']
    next_state = data.iloc[i + 1]['StateID']
    current_price = data.iloc[i]['close']
    next_price = data.iloc[i + 1]['close']

    # ε-greedy policy
    if np.random.rand() < epsilon:
        action = random.choice(actions)
    else:
        q_values = Q_table[Q_table['State'] == current_state]
        max_q = q_values['Q'].max()
        best_actions = q_values[q_values['Q'] == max_q]['Action'].tolist()
        action = random.choice(best_actions)

    reward = 0

    # --- Buy ---
    if action == 'Buy' and not position:
        position = True
        entry_price = current_price
        entry_index = i
        entry_state = current_state
        reward = 0  # no immediate reward

    # --- Hold ---
    elif action == 'Hold' and position:
        reward = (current_price - entry_price) * hold_reward_scale

        idx = (Q_table['State'] == current_state) & (Q_table['Action'] == 'Hold')
        old_q_hold = Q_table.loc[idx, 'Q'].values[0]
        max_q_next = Q_table[Q_table['State'] == next_state]['Q'].max()

        Q_table.loc[idx, 'Q'] = old_q_hold + alpha * (reward + gamma * max_q_next - old_q_hold)

    # --- Sell ---
    elif action == 'Sell' and position:
        profit = current_price - entry_price
        reward = profit * 1000

        # Update Buy (entry) Q-value
        buy_idx = (Q_table['State'] == entry_state) & (Q_table['Action'] == 'Buy')
        old_q_buy = Q_table.loc[buy_idx, 'Q'].values[0]
        max_q_next_buy = Q_table[Q_table['State'] == current_state]['Q'].max()

        Q_table.loc[buy_idx, 'Q'] = old_q_buy + alpha * (reward + gamma * max_q_next_buy - old_q_buy)

        # Update Sell (exit) Q-value
        sell_idx = (Q_table['State'] == current_state) & (Q_table['Action'] == 'Sell')
        old_q_sell = Q_table.loc[sell_idx, 'Q'].values[0]
        max_q_next_sell = Q_table[Q_table['State'] == next_state]['Q'].max()

        Q_table.loc[sell_idx, 'Q'] = old_q_sell + alpha * (reward + gamma * max_q_next_sell - old_q_sell)

        # Reset position
        position = False
        entry_price = None
        entry_index = None
        entry_state = None
        print("i is ",i)

    # Accumulate reward
    total_reward += reward
pd.set_option("display.max_rows", 1000)  # set as needed
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 1000)
#print(Q_table)


# Example: show best actions for a specific state

# Best action per state (greedy policy)
Q_policy = (
    Q_table.sort_values(by=['State', 'Q'], ascending=[True, False])
    .groupby('State')
    .head(1)
    .reset_index(drop=True)
    .sort_values(by='Q', ascending=False)
)

# Show top policies
print(Q_policy.head(10))


### run backtest function



def run_backtest(Q_table, data, CapitalUSD=1000):
    position = False
    entry_price = None
    total_profit = 0
    trade_log = []

    # Create a numeric index
    data = data.copy()
    data['index'] = np.arange(len(data))

    i = 0
    while i < len(data):
        row = data.iloc[i]
        current_state = row['StateID']
        current_price = row['close']
        timestamp = row['timestamp']

        q_values = Q_table[Q_table['State'] == current_state]
        q_ordered = q_values.sort_values(by='Q', ascending=False)

        max_q = q_ordered.iloc[0]['Q']
        second_max_q = q_ordered.iloc[1]['Q'] if len(q_ordered) > 1 else max_q
        action_confidence = max_q - second_max_q

        best_actions = q_ordered[q_ordered['Q'] == max_q]['Action'].tolist()
        action = np.random.choice(best_actions)

        # --- Buy ---
        if action == "Buy" and not position:
            position = True
            entry_price = current_price
            entry_time = timestamp
            entry_index = i
            entry_confidence = action_confidence

        # --- Sell ---
        elif action == "Sell" and position:
            exit_price = current_price
            exit_time = timestamp
            exit_index = i

            price_change = exit_price - entry_price
            commission = (CapitalUSD / current_price) * (0.005 + 0.01)
            profit = (price_change / entry_price) * CapitalUSD - commission
            total_profit += profit

            trade_log.append({
                'EntryTime': entry_time,
                'ExitTime': exit_time,
                'EntryPrice': entry_price,
                'ExitPrice': exit_price,
                'Profit': profit,
                'EntryIndex': entry_index,
                'ExitIndex': exit_index,
                'ActionConfidence': entry_confidence
            })

            position = False
            entry_price = None
            i = exit_index + 1
            continue

        i += 1

    trade_df = pd.DataFrame(trade_log)
    trade_df['CumulativeProfit'] = trade_df['Profit'].cumsum()

    # Drawdown calculation
    if len(trade_df) > 1:
        cumulative_max = trade_df['CumulativeProfit'].cummax()
        drawdown = cumulative_max - trade_df['CumulativeProfit']
        max_dd = drawdown.max()
    else:
        max_dd = np.nan

    # Buy & Hold
    buy_hold_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * CapitalUSD

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(data['index'], data['close'], label='Price')
    if not trade_df.empty:
        plt.scatter(trade_df['EntryIndex'], trade_df['EntryPrice'], color='green', label='Buy', s=40)
        plt.scatter(trade_df['ExitIndex'], trade_df['ExitPrice'], color='red', label='Sell', s=40)
    plt.title("Trades on SPY")
    plt.xlabel("Time Index")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        'Total_Profit': total_profit,
        'Avg_Profit_Per_Trade': trade_df['Profit'].mean() if not trade_df.empty else 0,
        'Std_Dev_Profit': trade_df['Profit'].std() if not trade_df.empty else 0,
        'Max_Drawdown': max_dd,
        'Buy_and_Hold': buy_hold_return,
        'Number_of_Trades': len(trade_df),
        'Trade_Log': trade_df
    }



# usage
results = run_backtest(Q_table, data)
print("Total Profit:", results['Total_Profit'])
print("Buy & Hold Profit:", results['Buy_and_Hold'])
print("Number of Trades:", results['Number_of_Trades'])



# performance plot



# Add Trade Number column
results['Trade_Log']['TradeNumber'] = range(1, len(results['Trade_Log']) + 1)

# Set style
sns.set(style="whitegrid")

# Create bar plot
plt.figure(figsize=(12, 6))
barplot = sns.barplot(
    x="TradeNumber",
    y="Profit",
    data=results['Trade_Log'],
    palette=["green" if p > 0 else "red" for p in results['Trade_Log']['Profit']]
)

# Add labels to bars
for idx, row in results['Trade_Log'].iterrows():
    y = row['Profit']
    plt.text(row['TradeNumber'] - 1, y + (0.5 if y >= 0 else -0.5), f"{y:.2f}",
             ha='center', va='bottom' if y >= 0 else 'top', fontsize=8)

# Labels and title
plt.title("Profit/Loss per Trade")
plt.xlabel("Trade Number")
plt.ylabel("Profit (in price units or pips)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


#####plotting




# Get the trade log
log = results['Trade_Log'].copy()

# Basic stats
total_trades = len(log)
winning_trades = (log['Profit'] > 0).sum()
losing_trades = (log['Profit'] < 0).sum()
zero_trades = (log['Profit'] == 0).sum()

avg_profit = log['Profit'].mean()
std_profit = log['Profit'].std()
win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
loss_rate = (losing_trades / total_trades) * 100 if total_trades > 0 else 0
zero_rate = (zero_trades / total_trades) * 100 if total_trades > 0 else 0

# Summary table
trade_summary = pd.DataFrame({
    "Metric": [
        "Total Trades", "Winning Trades", "Losing Trades", "Zero P/L Trades",
        "Win Rate (%)", "Loss Rate (%)", "Avg Profit", "Std Dev Profit"
    ],
    "Value": [
        total_trades, winning_trades, losing_trades, zero_trades,
        round(win_rate, 2), round(loss_rate, 2), round(avg_profit, 5), round(std_profit, 5)
    ]
})

# Print summary
print(trade_summary)

### save plot

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ensure seaborn is set up
sns.set(style="whitegrid")

# Set simulation number
simulation = 1  # change this as needed

# Get trade log
log = results['Trade_Log'].copy()

# Add Outcome column
log['Outcome'] = log['Profit'].apply(lambda x: "Win" if x > 0 else "Loss" if x < 0 else "Zero")

# --- Plot Outcome Frequency ---
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=log, x='Outcome', palette="pastel", edgecolor="black")

# Add count labels on bars
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=9)

plt.title("Frequency of Trade Outcomes")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"Simulation{simulation}_outcomes.png")
plt.close()

# --- Plot Cumulative Profit ---
plt.figure(figsize=(8, 5))
plt.plot(log['CumulativeProfit'], color='blue', linewidth=2)
plt.axhline(0, color='gray', linestyle='--')
plt.title("Cumulative Profit (USD)")
plt.xlabel("Trade #")
plt.ylabel("Profit in USD")
plt.tight_layout()
plt.savefig(f"Simulation{simulation}_cumulative_profit.png")
plt.close()

## final output

# Initialize outside loop if needed
if 'horizontal_summaryG' not in globals():
    horizontal_summaryG = pd.DataFrame()

# Extract profitable and losing trades
profitable_trades = log[log['Profit'] > 0]['Profit']
losing_trades = log[log['Profit'] < 0]['Profit']

# Calculate metrics
avg_profit_size = profitable_trades.mean() if not profitable_trades.empty else 0
avg_loss_size = losing_trades.mean() if not losing_trades.empty else 0
total_profit_winners = profitable_trades.sum()
total_loss_losers = losing_trades.sum()

# Vertical summary (optional)
full_trade_summary = pd.DataFrame({
    "Metric": [
        "Data Size", "Capital Used", "Total Trades",
        "Winning Trades", "Losing Trades",
        "Total Profit (USD)", "Profit % of Capital",
        "Avg Profit Size", "Avg Loss Size",
        "Total Profit from Winners", "Total Loss from Losers"
    ],
    "Value": [
        len(data), CapitalUSD, total_trades,
        winning_trades, total_trades - winning_trades,
        round(results['Total_Profit'], 2),
        round(results['Total_Profit'] * 100 / CapitalUSD, 2),
        round(avg_profit_size, 2),
        round(avg_loss_size, 2),
        round(total_profit_winners, 2),
        round(total_loss_losers, 2)
    ]
})

# print(full_trade_summary)

# Horizontal summary
num_losing_trades = len(losing_trades)

horizontal_summary = pd.DataFrame([{
    "DataSize": len(data),
    "CapitalUsed": CapitalUSD,
    "TotalTrades": total_trades,
    "WinningTrades": winning_trades,
    "LosingTrades": total_trades - winning_trades,
    "TotalProfitUSD": round(results['Total_Profit'], 2),
    "ProfitPercent": round(results['Total_Profit'] * 100 / CapitalUSD, 2),
    "AvgProfitSize": round(avg_profit_size, 2),
    "AvgLossSize": round(avg_loss_size, 2),
    "TotalProfitFromWinners": round(total_profit_winners, 2),
    "TotalLossFromLosers": round(total_loss_losers, 2),
    "PerformanceRatePerOrder": round(results['Total_Profit'], 2) / total_trades if total_trades > 0 else 0,
    "PerformanceRatePerOrderPROF": round(total_profit_winners, 2) / winning_trades if winning_trades > 0 else 0,
    "PerformanceRatePerOrderLOss": -round(total_loss_losers, 2) / num_losing_trades if num_losing_trades > 0 else 0
}])

# View horizontal summary
print(horizontal_summary)

# Append to global summary
horizontal_summaryG = pd.concat([horizontal_summaryG, horizontal_summary], ignore_index=True)

# Increment simulation
simulation += 1
print("Simulation:", simulation)
