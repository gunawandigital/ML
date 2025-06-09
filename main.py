
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
    print("5. Advanced Model Comparison")
    print("6. Latest Prediction")
    print("7. Backtesting")
    print("8. MetaAPI Setup")
    print("9. 🚀 START LIVE TRADING")
    print("10. Exit")

def run_pipeline():
    """Executes the complete trading pipeline."""
    # Check for best available data file (prioritize combined > real > sample)
    data_files = [
        ('data/xauusd_m15_combined.csv', 'COMBINED (Real + Sample)'),
        ('data/xauusd_m15_real.csv', 'REAL MetaAPI'),
        ('data/xauusd_m15.csv', 'SAMPLE')
    ]
    
    selected_data = None
    data_type = None
    
    for data_path, desc in data_files:
        if os.path.exists(data_path):
            selected_data = data_path
            data_type = desc
            break
    
    if not selected_data:
        print("\nError: No data files found!")
        print("Please download data first using option 2 (Download Real Data from MetaAPI)")
        return
    
    print(f"\n📊 Using {data_type} data: {selected_data}")

    try:
        # Step 1: Feature Engineering Test
        print_header("STEP 1: FEATURE ENGINEERING")
        from feature_engineering import prepare_data

        print("Testing feature engineering...")
        data = prepare_data(selected_data)
        print(f"✓ Features created successfully! Data shape: {data.shape}")
        print(f"✓ Target distribution:\n{data['Target'].value_counts()}")

        # Step 2: Model Training
        print_header("STEP 2: MODEL TRAINING")
        from train import train_model

        print(f"Training Random Forest model with {data_type} data...")
        model, scaler, accuracy = train_model(data_path=selected_data)
        print(f"✓ Model trained successfully! Accuracy: {accuracy:.4f}")

        # Step 3: Latest Prediction
        print_header("STEP 3: LATEST PREDICTION")
        from predict import get_latest_signal

        print("Getting latest trading signal...")
        signal = get_latest_signal(data_path=selected_data)

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
            choice = input("\nSelect option (1-10): ").strip()

            if choice == "1":
                print_header("RUNNING COMPLETE PIPELINE")
                run_pipeline()

            elif choice == "2":
                print_header("DOWNLOADING REAL DATA FROM METAAPI")
                import asyncio
                asyncio.run(run_metaapi_download())

            elif choice == "3":
                print_header("FEATURE ENGINEERING")
                
                # Auto-select best data
                data_files = [
                    'data/xauusd_m15_combined.csv',
                    'data/xauusd_m15_real.csv', 
                    'data/xauusd_m15.csv'
                ]
                
                selected_data = None
                for data_path in data_files:
                    if os.path.exists(data_path):
                        selected_data = data_path
                        break
                
                if not selected_data:
                    print("❌ No data files found!")
                    continue
                
                print(f"Using data: {selected_data}")
                from feature_engineering import prepare_data
                data = prepare_data(selected_data)
                print(f"✓ Features created: {data.shape}")

            elif choice == "4":
                print_header("MODEL TRAINING")
                
                # Check for available data sources
                data_files = {
                    'combined': 'data/xauusd_m15_combined.csv',
                    'real': 'data/xauusd_m15_real.csv', 
                    'sample': 'data/xauusd_m15.csv'
                }
                
                available_files = {k: v for k, v in data_files.items() if os.path.exists(v)}
                
                if not available_files:
                    print("❌ No data files found!")
                    continue
                
                print("\nAvailable datasets:")
                for i, (key, path) in enumerate(available_files.items(), 1):
                    import pandas as pd
                    try:
                        df_info = pd.read_csv(path)
                        rows = len(df_info)
                        print(f"{i}. {key.upper()} data: {path} ({rows} rows)")
                    except:
                        print(f"{i}. {key.upper()} data: {path} (error reading)")
                
                print(f"{len(available_files)+1}. Auto-select best data")
                
                try:
                    choice_data = input(f"\nSelect dataset (1-{len(available_files)+1}): ").strip()
                    
                    if choice_data == str(len(available_files)+1):
                        # Auto-select: prioritize combined > real > sample
                        if 'combined' in available_files:
                            selected_data = available_files['combined']
                            data_type = 'COMBINED'
                        elif 'real' in available_files:
                            selected_data = available_files['real']
                            data_type = 'REAL'
                        else:
                            selected_data = available_files['sample']
                            data_type = 'SAMPLE'
                    else:
                        idx = int(choice_data) - 1
                        selected_data = list(available_files.values())[idx]
                        data_type = list(available_files.keys())[idx].upper()
                    
                    print(f"\n🎯 Training with {data_type} data: {selected_data}")
                    
                    from train import train_model
                    model, scaler, accuracy = train_model(data_path=selected_data)
                    print(f"✓ Model trained with {data_type} data: {accuracy:.4f} accuracy")
                    
                except (ValueError, IndexError):
                    print("❌ Invalid selection, using default data")
                    from train import train_model
                    model, scaler, accuracy = train_model()
                    print(f"✓ Model trained: {accuracy:.4f} accuracy")

            elif choice == "5":
                print_header("ADVANCED MODEL COMPARISON")
                
                # Check for available data
                data_files = [
                    'data/xauusd_m15_combined.csv',
                    'data/xauusd_m15_real.csv', 
                    'data/xauusd_m15.csv'
                ]
                
                selected_data = None
                for data_path in data_files:
                    if os.path.exists(data_path):
                        selected_data = data_path
                        break
                
                if not selected_data:
                    print("❌ No data files found!")
                    continue
                
                print(f"🎯 Comparing models with data: {selected_data}")
                
                try:
                    from train_advanced import compare_models_safe
                    print("🔍 Checking system dependencies...")
                    models, best_model = compare_models_safe(data_path=selected_data)
                    
                    if models and best_model:
                        print(f"\n🏆 Best performing model: {best_model}")
                        if len(models) == 1:
                            print("ℹ️  Note: Only Random Forest was tested due to system limitations")
                            print("   For full comparison, advanced libraries need OpenMP support")
                    else:
                        print("❌ Model comparison could not be completed")
                        
                except ImportError as e:
                    print(f"❌ Import error: {e}")
                    print("📦 Please ensure all packages are installed")
                except Exception as e:
                    print(f"❌ Unexpected error: {e}")
                    print("🔧 Please check your system configuration")

            elif choice == "6":
                print_header("LATEST PREDICTION")
                
                # Auto-select best data for prediction
                data_files = [
                    'data/xauusd_m15_combined.csv',
                    'data/xauusd_m15_real.csv', 
                    'data/xauusd_m15.csv'
                ]
                
                selected_data = None
                for data_path in data_files:
                    if os.path.exists(data_path):
                        selected_data = data_path
                        break
                
                from predict import get_latest_signal
                signal = get_latest_signal(data_path=selected_data)
                print(f"Signal: {signal['signal']} | Confidence: {signal['confidence']:.1%}")
                print(f"Using data: {selected_data}")

            elif choice == "7":
                print_header("BACKTESTING")
                from backtest import backtest_strategy
                equity_df, trades_df = backtest_strategy()
                print("✓ Backtesting completed")

            elif choice == "8":
                print_header("METAAPI SETUP")
                try:
                    import asyncio
                    from setup_metaapi import main as setup_main
                    asyncio.run(setup_main())
                except KeyboardInterrupt:
                    print("\n🛑 Setup interrupted by user")
                except Exception as e:
                    print(f"❌ Setup error: {e}")
                    print("💡 This might be due to network connectivity or configuration issues")

            elif choice == "9":
                print_header("🚀 STARTING LIVE AUTOMATED TRADING")
                try:
                    import subprocess
                    subprocess.run([sys.executable, "run_live_trading.py"])
                except KeyboardInterrupt:
                    print("\n🛑 Trading interrupted by user")
                except Exception as e:
                    print(f"❌ Trading error: {e}")

            elif choice == "10":
                print("\n👋 Goodbye! Happy trading!")
                break

            else:
                print("❌ Invalid choice. Please select 1-10.")

            input("\nPress Enter to continue...")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Happy trading!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
