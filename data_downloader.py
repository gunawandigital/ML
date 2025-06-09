
"""
MetaAPI Data Downloader
Download real historical data from MetaAPI for model training
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Optional

try:
    from metaapi_cloud_sdk import MetaApi
except ImportError:
    print("⚠️ MetaAPI SDK not installed. Install with: pip install metaapi-cloud-sdk")

from trading_config import TradingConfig

class MetaAPIDataDownloader:
    """Download historical data from MetaAPI"""
    
    def __init__(self, token: str, account_id: str):
        self.token = token
        self.account_id = account_id
        self.api = None
        self.account = None
        self.connection = None
        
    async def initialize(self):
        """Initialize MetaAPI connection"""
        try:
            self.api = MetaApi(self.token)
            self.account = await self.api.metatrader_account_api.get_account(self.account_id)
            
            # Deploy and wait for connection
            await self.account.deploy()
            await self.account.wait_connected()
            
            # Use RPC connection for historical data
            self.connection = self.account.get_rpc_connection()
            await self.connection.connect()
            await self.connection.wait_synchronized()
            
            print("✅ MetaAPI RPC connection established for data download")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize MetaAPI: {e}")
            return False
    
    async def download_historical_data(self, 
                                     symbol: str = "XAUUSD",
                                     timeframe: str = "M15",
                                     days_back: int = 30,
                                     save_path: str = "data/xauusd_m15_real.csv") -> pd.DataFrame:
        """Download historical OHLC data from MetaAPI"""
        
        try:
            print(f"📥 Downloading {days_back} days of {symbol} {timeframe} data...")
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            print(f"   Period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
            
            # Download data in chunks (MetaAPI has limits)
            all_candles = []
            chunk_days = 7  # Download 7 days at a time
            
            current_start = start_time
            while current_start < end_time:
                current_end = min(current_start + timedelta(days=chunk_days), end_time)
                
                print(f"   Downloading chunk: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
                
                try:
                    # Use the correct MetaAPI method for historical data
                    history = await self.connection.get_historical_candles(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_time=current_start,
                        end_time=current_end
                    )
                    
                    if history:
                        all_candles.extend(history)
                        print(f"     Downloaded {len(history)} candles")
                    else:
                        print(f"     No data available for this period")
                    
                    # Small delay to avoid rate limits
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    print(f"     Error downloading chunk: {e}")
                    # Continue with next chunk instead of stopping
                
                current_start = current_end
            
            if not all_candles:
                print("❌ No data downloaded")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for candle in all_candles:
                # MetaAPI candle format
                time_obj = candle['time']
                
                data.append({
                    'Date': time_obj.strftime('%Y-%m-%d'),
                    'Time': time_obj.strftime('%H:%M'),
                    'Open': candle['open'],
                    'High': candle['high'],
                    'Low': candle['low'],
                    'Close': candle['close'],
                    'Volume': candle.get('tickVolume', 0)
                })
            
            df = pd.DataFrame(data)
            
            # Remove duplicates and sort
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.drop_duplicates(subset=['DateTime']).sort_values('DateTime')
            df = df.drop('DateTime', axis=1)
            
            print(f"✅ Downloaded {len(df)} candles")
            print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"   Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
            
            # Save to file
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"✅ Data saved to: {save_path}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            return pd.DataFrame()
    
    async def update_training_data(self, 
                                 new_data_path: str = "data/xauusd_m15_real.csv",
                                 original_data_path: str = "data/xauusd_m15.csv",
                                 output_path: str = "data/xauusd_m15_combined.csv"):
        """Combine new real data with existing data"""
        
        try:
            print("🔄 Updating training dataset...")
            
            # Load new real data
            if os.path.exists(new_data_path):
                new_df = pd.read_csv(new_data_path)
                print(f"   New real data: {len(new_df)} rows")
            else:
                print("   No new real data found")
                return False
            
            # Load original data (if exists)
            if os.path.exists(original_data_path):
                original_df = pd.read_csv(original_data_path)
                print(f"   Original data: {len(original_df)} rows")
                
                # Combine datasets
                combined_df = pd.concat([original_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            # Remove duplicates
            combined_df['DateTime'] = pd.to_datetime(combined_df['Date'] + ' ' + combined_df['Time'])
            combined_df = combined_df.drop_duplicates(subset=['DateTime']).sort_values('DateTime')
            combined_df = combined_df.drop('DateTime', axis=1)
            
            # Save combined dataset
            combined_df.to_csv(output_path, index=False)
            
            print(f"✅ Combined dataset saved: {len(combined_df)} rows")
            print(f"   Saved to: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating training data: {e}")
            return False
    
    async def download_and_retrain(self, 
                                 days_back: int = 30,
                                 retrain_model: bool = True):
        """Complete workflow: download data and retrain model"""
        
        print("🚀 Starting data download and retraining workflow...")
        
        # Step 1: Download real data
        real_data = await self.download_historical_data(days_back=days_back)
        
        if real_data.empty:
            print("❌ Failed to download real data")
            return False
        
        # Step 2: Update training dataset
        success = await self.update_training_data()
        
        if not success:
            print("❌ Failed to update training data")
            return False
        
        # Step 3: Retrain model with new data
        if retrain_model:
            print("\n🤖 Retraining model with real data...")
            
            try:
                from train import train_model
                
                # Train with combined dataset
                model, scaler, accuracy = train_model(
                    data_path='data/xauusd_m15_combined.csv',
                    model_path='models/'
                )
                
                print(f"✅ Model retrained successfully!")
                print(f"   New accuracy: {accuracy:.4f}")
                
                # Test with latest prediction
                print("\n📊 Testing with latest real data...")
                from predict import get_latest_signal
                
                signal = get_latest_signal(data_path='data/xauusd_m15_combined.csv')
                print(f"   Latest signal: {signal['signal']} (confidence: {signal['confidence']:.1%})")
                
                return True
                
            except Exception as e:
                print(f"❌ Error retraining model: {e}")
                return False
        
        return True

async def main():
    """Main function to download real data and retrain"""
    
    config = TradingConfig()
    
    # Validate configuration
    errors = config.validate_config()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        print("\nPlease fix trading_config.py first!")
        return
    
    # Create downloader
    downloader = MetaAPIDataDownloader(
        token=config.META_API_TOKEN,
        account_id=config.ACCOUNT_ID
    )
    
    # Initialize connection
    if await downloader.initialize():
        
        print("\n" + "="*60)
        print("MetaAPI Data Download Options:")
        print("1. Download 7 days of data")
        print("2. Download 30 days of data") 
        print("3. Download 90 days of data")
        print("4. Custom days")
        print("="*60)
        
        try:
            choice = input("Choose option (1-4): ").strip()
            
            if choice == "1":
                days = 7
            elif choice == "2":
                days = 30
            elif choice == "3":
                days = 90
            elif choice == "4":
                days = int(input("Enter number of days: "))
            else:
                days = 30
                print("Using default: 30 days")
            
            # Download and retrain
            success = await downloader.download_and_retrain(
                days_back=days,
                retrain_model=True
            )
            
            if success:
                print("\n🎉 Data download and retraining completed successfully!")
                print("\nNext steps:")
                print("1. Test the retrained model")
                print("2. Run backtesting with new data")
                print("3. Start live trading with improved model")
            else:
                print("\n❌ Process failed. Check the logs above.")
                
        except KeyboardInterrupt:
            print("\n🛑 Process cancelled by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    else:
        print("❌ Failed to connect to MetaAPI")

if __name__ == "__main__":
    asyncio.run(main())
