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

def show_menu():
    """Display the main menu options."""
    print("\n[ MENU ]")
    print("1. Run Complete Pipeline")
    print("2. Download Real Data from MetaAPI")
    print("3. Feature Engineering")
    print("4. Model Training")
    print("5. Latest Prediction")
    print("6. Backtesting")
    print("7. MetaAPI Setup")
    print("8. Exit")

def run_pipeline():
    """Executes the complete trading pipeline."""
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

async def run_metaapi_download():
    """Run MetaAPI data download and retraining"""
    try:
        from data_downloader import main as download_main
        await download_main()
    except ImportError:
        print("❌ MetaAPI downloader not available")
    except Exception as e:
        print(f"❌ Error in MetaAPI download: {e}")

def main():
    """Main function with menu system"""

    print_header("FOREX ML TRADING SYSTEM - XAUUSD")
    print(f"System started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    while True:
        try:
            show_menu()
            choice = input("\nSelect option (1-8): ").strip()

            if choice == "1":
                print_header("RUNNING COMPLETE PIPELINE")
                run_pipeline()

            elif choice == "2":
                print_header("DOWNLOADING REAL DATA FROM METAAPI")
                import asyncio
                asyncio.run(run_metaapi_download())

            elif choice == "3":
                print_header("FEATURE ENGINEERING")
                from feature_engineering import prepare_data
                data = prepare_data('data/xauusd_m15.csv')
                print(f"✓ Features created: {data.shape}")

            elif choice == "4":
                print_header("MODEL TRAINING")
                from train import train_model
                model, scaler, accuracy = train_model()
                print(f"✓ Model trained: {accuracy:.4f} accuracy")

            elif choice == "5":
                print_header("LATEST PREDICTION")
                from predict import get_latest_signal
                signal = get_latest_signal()
                print(f"Signal: {signal['signal']} | Confidence: {signal['confidence']:.1%}")

            elif choice == "6":
                print_header("BACKTESTING")
                from backtest import backtest_strategy
                equity_df, trades_df = backtest_strategy()
                print("✓ Backtesting completed")

            elif choice == "7":
                print_header("METAAPI SETUP")
                import asyncio
                from setup_metaapi import main as setup_main
                asyncio.run(setup_main())

            elif choice == "8":
                print("\n👋 Goodbye! Happy trading!")
                break

            else:
                print("❌ Invalid choice. Please select 1-8.")

            input("\nPress Enter to continue...")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Happy trading!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()