import pandas as pd
import numpy as np
from ta import momentum, trend, volume, volatility
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd
import matplotlib.pyplot as plt

import os
import random

CapitalUSD = 1000
DataSize = 6500

# Load CSV (make sure path is correct)
data = pd.read_csv("SPYUSUSD2years.csv")

# Select subset
data = data.iloc[:1+DataSize].copy()
print(data)
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


# Setup
CapitalUSD = 1000
actions = ['Buy', 'Sell', 'Hold']
os.makedirs("plots", exist_ok=True)
horizontal_summaryG = pd.DataFrame()

for simulation in range(1, 10):

    # --- Reset Q-table ---
    Q_table['Q'] = 0
    alpha = 0.2
    gamma = 0.85
    epsilon = 0.3
    hold_reward_scale = 100

    position = False
    entry_price = None
    entry_index = None
    entry_state = None
    total_reward = 0

    # --- Q-Learning Training ---
    for i in range(4, len(data) - 1):
        current_state = data.iloc[i]['StateID']
        next_state = data.iloc[i + 1]['StateID']
        current_price = data.iloc[i]['close']
        next_price = data.iloc[i + 1]['close']

        if np.random.rand() < epsilon:
            action = random.choice(actions)
        else:
            q_values = Q_table[Q_table['State'] == current_state]
            max_q = q_values['Q'].max()
            best_actions = q_values[q_values['Q'] == max_q]['Action'].tolist()
            action = random.choice(best_actions)

        reward = 0

        if action == 'Buy' and not position:
            position = True
            entry_price = current_price
            entry_index = i
            entry_state = current_state

        elif action == 'Hold' and position:
            reward = (current_price - entry_price) * hold_reward_scale
            idx = (Q_table['State'] == current_state) & (Q_table['Action'] == 'Hold')
            old_q = Q_table.loc[idx, 'Q'].values[0]
            max_q_next = Q_table[Q_table['State'] == next_state]['Q'].max()
            Q_table.loc[idx, 'Q'] = old_q + alpha * (reward + gamma * max_q_next - old_q)

        elif action == 'Sell' and position:
            profit = current_price - entry_price
            reward = profit * 1000
            # Update Buy
            buy_idx = (Q_table['State'] == entry_state) & (Q_table['Action'] == 'Buy')
            old_q_buy = Q_table.loc[buy_idx, 'Q'].values[0]
            max_q_buy = Q_table[Q_table['State'] == current_state]['Q'].max()
            Q_table.loc[buy_idx, 'Q'] = old_q_buy + alpha * (reward + gamma * max_q_buy - old_q_buy)
            # Update Sell
            sell_idx = (Q_table['State'] == current_state) & (Q_table['Action'] == 'Sell')
            old_q_sell = Q_table.loc[sell_idx, 'Q'].values[0]
            max_q_sell = Q_table[Q_table['State'] == next_state]['Q'].max()
            Q_table.loc[sell_idx, 'Q'] = old_q_sell + alpha * (reward + gamma * max_q_sell - old_q_sell)

            position = False
            entry_price = None
            entry_index = None
            entry_state = None

        total_reward += reward

    # --- Run Backtest ---
    def run_backtest(Q_table, data, CapitalUSD=1000):
        position = False
        entry_price = None
        total_profit = 0
        trade_log = []

        data = data.copy()
        data['index'] = np.arange(len(data))
        i = 0

        while i < len(data):
            row = data.iloc[i]
            state = row['StateID']
            current_price = row['close']
            timestamp = row['timestamp']

            q_values = Q_table[Q_table['State'] == state]
            q_ordered = q_values.sort_values(by='Q', ascending=False)
            max_q = q_ordered.iloc[0]['Q']
            second_max_q = q_ordered.iloc[1]['Q'] if len(q_ordered) > 1 else max_q
            action_confidence = max_q - second_max_q
            best_actions = q_ordered[q_ordered['Q'] == max_q]['Action'].tolist()
            action = np.random.choice(best_actions)

            # Determine capital coefficient based on Q differences
            q_buy = q_values[q_values['Action'] == 'Buy']['Q'].values[0]
            q_sell = q_values[q_values['Action'] == 'Sell']['Q'].values[0]
            q_hold = q_values[q_values['Action'] == 'Hold']['Q'].values[0]
            avg_other_q = np.mean([q for a, q in zip(['Buy', 'Sell', 'Hold'], [q_buy, q_sell, q_hold]) if a != action])
            confidence_ratio = abs(q_values[q_values['Action'] == action]['Q'].values[0] - avg_other_q)

            # Capital allocation based on confidence
            if confidence_ratio >= 0.5:
                capital_used = 2 * CapitalUSD  # $2000
            elif confidence_ratio < 0.2:
                capital_used = 0.5 * CapitalUSD  # $500
            else:
                capital_used = CapitalUSD  # $1000

            if action == "Buy" and not position:
                position = True
                entry_price = current_price
                entry_time = timestamp
                entry_index = i
                entry_confidence = action_confidence
                entry_capital = capital_used

            elif action == "Sell" and position:
                exit_price = current_price
                exit_time = timestamp
                exit_index = i
                price_change = exit_price - entry_price

                commission = (entry_capital / current_price) * (0.005 + 0.01)
                profit = (price_change / entry_price) * entry_capital - commission
                total_profit += profit

                trade_log.append({
                    'EntryTime': entry_time, 'ExitTime': exit_time,
                    'EntryPrice': entry_price, 'ExitPrice': exit_price,
                    'Profit': profit, 'EntryIndex': entry_index,
                    'ExitIndex': exit_index, 'ActionConfidence': entry_confidence,
                    'CapitalUsed': entry_capital
                })

                position = False
                entry_price = None
                i = exit_index + 1
                continue

            i += 1

        trade_df = pd.DataFrame(trade_log)
        trade_df['CumulativeProfit'] = trade_df['Profit'].cumsum()

        if len(trade_df) > 1:
            cumulative_max = trade_df['CumulativeProfit'].cummax()
            drawdown = cumulative_max - trade_df['CumulativeProfit']
            max_dd = drawdown.max()
        else:
            max_dd = np.nan

        # Just for reference: Buy & Hold with base capital
        buy_hold_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * CapitalUSD

        return {
            'Total_Profit': total_profit,
            'Avg_Profit_Per_Trade': trade_df['Profit'].mean() if not trade_df.empty else 0,
            'Std_Dev_Profit': trade_df['Profit'].std() if not trade_df.empty else 0,
            'Max_Drawdown': max_dd,
            'Buy_and_Hold': buy_hold_return,
            'Number_of_Trades': len(trade_df),
            'Trade_Log': trade_df
        }


    results = run_backtest(Q_table, data)
    log = results['Trade_Log'].copy()

    # --- Save Cumulative Profit Plot Only ---
    plt.figure(figsize=(8, 5))
    plt.plot(log['CumulativeProfit'], color='blue', linewidth=2)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f"Cumulative Profit (Simulation {simulation})")
    plt.xlabel("Trade #")
    plt.ylabel("Profit in USD")
    plt.tight_layout()
    plt.savefig(f"plots/Simulation{simulation}_cumulative_profit.png")
    plt.close()

    # --- Final Stats Collection ---
    profitable_trades = log[log['Profit'] > 0]['Profit']
    losing_trades = log[log['Profit'] < 0]['Profit']
    num_losing_trades = len(losing_trades)
    avg_profit_size = profitable_trades.mean() if not profitable_trades.empty else 0
    avg_loss_size = losing_trades.mean() if not losing_trades.empty else 0
    total_profit_winners = profitable_trades.sum()
    total_loss_losers = losing_trades.sum()

    total_trades = len(log)
    winning_trades = (log['Profit'] > 0).sum()

    horizontal_summary = pd.DataFrame([{
        "Simulation": simulation,
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

    horizontal_summaryG = pd.concat([horizontal_summaryG, horizontal_summary], ignore_index=True)
    print(f"Simulation {simulation} complete.")

# Final Output
print("\n--- Summary of All Simulations ---")
print(horizontal_summaryG)




