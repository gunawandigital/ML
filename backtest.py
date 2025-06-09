
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from feature_engineering import prepare_data

def select_features(df):
    """Select relevant features for backtesting"""
    feature_columns = [
        # Original features
        'EMA_9', 'EMA_21', 'EMA_50',
        'RSI_14', 'RSI_21',
        'HL_Ratio', 'OC_Ratio',
        'Return_1', 'Return_5', 'Return_15',
        'Volatility_5', 'Volatility_15',
        'EMA_Cross_9_21', 'EMA_Cross_21_50',
        'Price_Above_EMA9', 'Price_Above_EMA21', 'Price_Above_EMA50',
        
        # Advanced features
        'BB_Position', 'MACD', 'MACD_Histogram',
        'Stoch_K', 'Stoch_D',
        'Hour', 'DayOfWeek', 'IsLondonSession', 'IsNYSession',
        'ROC_5', 'ROC_10',
        
        # Smart Money Concept (SMC) features
        'BullishStructure', 'BearishStructure',
        'BullishOB', 'BearishOB', 'NearBullishOB', 'NearBearishOB',
        'BullishFVG', 'BearishFVG',
        'LiquiditySweepHigh', 'LiquiditySweepLow',
        'Premium', 'Discount', 'Equilibrium',
        'BOS_Bullish', 'BOS_Bearish', 'CHoCH'
    ]
    
    # Only include volume features if available
    if 'Volume_Ratio' in df.columns:
        feature_columns.append('Volume_Ratio')
    
    # Filter out columns that don't exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]
    
    return df[available_features]

def backtest_strategy(data_path=None, model_path='models/', 
                     initial_capital=10000, position_size=0.1, transaction_cost=0.0001):
    """Backtest the trading strategy using real MetaAPI data when available"""
    
    # Auto-select only real MetaAPI data
    if data_path is None:
        # Only use real MetaAPI data for backtesting
        if os.path.exists('data/xauusd_m15_real.csv'):
            data_path = 'data/xauusd_m15_real.csv'
            data_type = 'REAL MetaAPI'
        else:
            raise FileNotFoundError("No real MetaAPI data found! Please download real data first using option 2 in main menu.")
        
        print(f"📊 Backtesting with {data_type} data: {data_path}")
    
    # Load model and scaler
    model = joblib.load(f'{model_path}/random_forest_model.pkl')
    scaler = joblib.load(f'{model_path}/scaler.pkl')
    
    # Prepare data
    df = prepare_data(data_path)
    
    # Split data for backtesting (use last 70% for backtest)
    split_point = int(len(df) * 0.3)
    backtest_data = df.iloc[split_point:].copy()
    
    # Select features
    features = select_features(backtest_data)
    features_scaled = scaler.transform(features)
    
    # Make predictions
    predictions = model.predict(features_scaled)
    
    # Add predictions to backtest data
    backtest_data['Prediction'] = predictions
    backtest_data['Signal'] = backtest_data['Prediction'].map({-1: 'SELL', 0: 'HOLD', 1: 'BUY'})
    
    # Initialize backtesting variables
    capital = initial_capital
    position = 0  # 0: no position, 1: long, -1: short
    entry_price = 0
    trades = []
    equity_curve = []
    
    for i, (timestamp, row) in enumerate(backtest_data.iterrows()):
        current_price = row['Close']
        signal = row['Prediction']
        
        # Close existing position if signal changes or is HOLD
        if position != 0 and (signal != position or signal == 0):
            # Calculate profit/loss
            if position == 1:  # Close long position
                pnl = (current_price - entry_price) * (capital * position_size / entry_price)
            else:  # Close short position
                pnl = (entry_price - current_price) * (capital * position_size / entry_price)
            
            # Apply transaction cost
            pnl -= capital * position_size * transaction_cost
            
            capital += pnl
            
            trades.append({
                'exit_time': timestamp,
                'exit_price': current_price,
                'position': 'LONG' if position == 1 else 'SHORT',
                'pnl': pnl,
                'capital': capital
            })
            
            position = 0
        
        # Open new position
        if signal != 0 and position == 0:
            position = signal
            entry_price = current_price
            
        equity_curve.append({
            'timestamp': timestamp,
            'capital': capital,
            'price': current_price,
            'signal': signal
        })
    
    # Convert to DataFrame
    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)
    
    # Calculate performance metrics
    total_return = (capital - initial_capital) / initial_capital * 100
    
    if len(trades_df) > 0:
        win_trades = trades_df[trades_df['pnl'] > 0]
        lose_trades = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(win_trades) / len(trades_df) * 100
        avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
        avg_loss = lose_trades['pnl'].mean() if len(lose_trades) > 0 else 0
        profit_factor = abs(win_trades['pnl'].sum() / lose_trades['pnl'].sum()) if len(lose_trades) > 0 and lose_trades['pnl'].sum() != 0 else float('inf')
        
        # Calculate maximum drawdown
        equity_df['peak'] = equity_df['capital'].expanding().max()
        equity_df['drawdown'] = (equity_df['capital'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].min()
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        max_drawdown = 0
    
    # Print results
    print("=== BACKTEST RESULTS ===")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital: ${capital:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Win: ${avg_win:.2f}")
    print(f"Average Loss: ${avg_loss:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    
    return equity_df, trades_df

def plot_equity_curve(equity_df, backtest_data):
    """Plot equity curve and price chart"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot price chart with signals
    ax1.plot(backtest_data.index, backtest_data['Close'], label='XAUUSD Price', linewidth=1)
    
    # Plot buy/sell signals
    buy_signals = backtest_data[backtest_data['Prediction'] == 1]
    sell_signals = backtest_data[backtest_data['Prediction'] == -1]
    
    ax1.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', 
               s=50, alpha=0.7, label='BUY Signal')
    ax1.scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', 
               s=50, alpha=0.7, label='SELL Signal')
    
    ax1.set_title('XAUUSD Price with Trading Signals')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot equity curve
    ax2.plot(equity_df['timestamp'], equity_df['capital'], label='Equity Curve', 
             color='blue', linewidth=2)
    ax2.axhline(y=equity_df['capital'].iloc[0], color='gray', linestyle='--', 
               alpha=0.7, label='Initial Capital')
    
    ax2.fill_between(equity_df['timestamp'], equity_df['capital'], 
                    equity_df['capital'].iloc[0], alpha=0.3, color='blue')
    
    ax2.set_title('Equity Curve')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Capital ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    try:
        # Run backtest (will auto-select best data)
        equity_df, trades_df = backtest_strategy()
        
        # Use only real MetaAPI data for plotting
        if os.path.exists('data/xauusd_m15_real.csv'):
            selected_data = 'data/xauusd_m15_real.csv'
        else:
            raise FileNotFoundError("No real MetaAPI data found for plotting!")
        
        # Load backtest data for plotting
        df = prepare_data(selected_data)
        split_point = int(len(df) * 0.3)
        backtest_data = df.iloc[split_point:]
        
        # Plot results
        plot_equity_curve(equity_df, backtest_data)
        
        print("\nEquity curve plot saved as 'backtest_results.png'")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train.py first to create the model.")
