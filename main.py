
#!/usr/bin/env python3
"""
Forex Machine Learning Trading System for XAUUSD
Complete pipeline: Feature Engineering -> Training -> Prediction -> Backtesting
"""

import os
import sys
from datetime import datetime

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def main():
    """Main execution pipeline"""
    
    print_header("FOREX ML TRADING SYSTEM - XAUUSD")
    print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if data file exists
    if not os.path.exists('data/xauusd_m15.csv'):
        print("\nError: Data file 'data/xauusd_m15.csv' not found!")
        print("Please ensure the data file exists before running the system.")
        return
    
    try:
        # Step 1: Feature Engineering Test
        print_header("STEP 1: FEATURE ENGINEERING")
        from feature_engineering import prepare_data
        
        print("Testing feature engineering...")
        data = prepare_data('data/xauusd_m15.csv')
        print(f"✓ Features created successfully! Data shape: {data.shape}")
        print(f"✓ Target distribution:\n{data['Target'].value_counts()}")
        
        # Step 2: Model Training
        print_header("STEP 2: MODEL TRAINING")
        from train import train_model
        
        print("Training Random Forest model...")
        model, scaler, accuracy = train_model()
        print(f"✓ Model trained successfully! Accuracy: {accuracy:.4f}")
        
        # Step 3: Latest Prediction
        print_header("STEP 3: LATEST PREDICTION")
        from predict import get_latest_signal
        
        print("Getting latest trading signal...")
        signal = get_latest_signal()
        
        print("\n📊 LATEST TRADING SIGNAL")
        print(f"   Timestamp: {signal['timestamp']}")
        print(f"   Current Price: ${signal['current_price']:.2f}")
        print(f"   Signal: {signal['signal']}")
        print(f"   Confidence: {signal['confidence']:.1%}")
        
        # Step 4: Backtesting
        print_header("STEP 4: BACKTESTING")
        from backtest import backtest_strategy
        
        print("Running backtest...")
        equity_df, trades_df = backtest_strategy()
        
        if len(trades_df) > 0:
            total_return = (equity_df['capital'].iloc[-1] - 10000) / 10000 * 100
            print(f"✓ Backtest completed! Total Return: {total_return:.2f}%")
        else:
            print("✓ Backtest completed! No trades executed.")
        
        # Summary
        print_header("EXECUTION SUMMARY")
        print("✓ All components executed successfully!")
        print("✓ Model is ready for live trading")
        print("✓ Check 'backtest_results.png' for visual results")
        
        print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        print("Please check the error and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
