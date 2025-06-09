#!/usr/bin/env python3
"""
MetaAPI Data Downloader for real-time XAUUSD data
Downloads historical data and combines with existing sample data
"""

import asyncio
import os
import pandas as pd
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from metaapi_cloud_sdk import MetaApi
except ImportError:
    logger.error("MetaAPI SDK not installed. Run: pip install metaapi-cloud-sdk")

class MetaAPIDataDownloader:
    """Download historical data from MetaAPI"""

    def __init__(self, token: str, account_id: str):
        self.token = token
        self.account_id = account_id
        self.api = None
        self.account = None

    async def initialize(self):
        """Initialize MetaAPI connection"""
        try:
            self.api = MetaApi(self.token)
            self.account = await self.api.metatrader_account_api.get_account(self.account_id)

            # Deploy account if needed
            if self.account.state != 'DEPLOYED':
                await self.account.deploy()

            # Wait for connection
            await self.account.wait_connected()
            logger.info("✅ MetaAPI connection established")
            return True

        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False

    async def download_historical_data(self, symbol: str = "XAUUSD", timeframe: str = "15m", days_back: int = 30):
        """Download historical data"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)

            logger.info(f"📊 Downloading {days_back} days of {symbol} data...")

            # Calculate approximate number of candles needed
            # For 15m timeframe: 96 candles per day (24*4)
            candles_per_day = {'1m': 1440, '5m': 288, '15m': 96, '30m': 48, '1h': 24, '4h': 6, '1d': 1}
            limit = min(candles_per_day.get(timeframe, 96) * days_back, 1000)  # MetaAPI limit is usually 1000

            # Get historical candles
            candles = await self.account.get_historical_candles(
                symbol, timeframe, start_time, limit
            )

            # Convert to DataFrame
            data = []
            for candle in candles:
                if isinstance(candle['time'], str):
                    time_obj = datetime.fromisoformat(candle['time'].replace('Z', '+00:00'))
                else:
                    time_obj = candle['time']

                data.append({
                    'Date': time_obj.strftime('%Y-%m-%d'),
                    'Time': time_obj.strftime('%H:%M:%S'),
                    'Open': candle['open'],
                    'High': candle['high'],
                    'Low': candle['low'],
                    'Close': candle['close'],
                    'Volume': candle.get('tickVolume', 0)
                })

            df = pd.DataFrame(data)
            logger.info(f"✅ Downloaded {len(df)} candles")
            return df

        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return pd.DataFrame()

    async def save_real_data_only(self, new_data: pd.DataFrame):
        """Save only real MetaAPI data (no sample data combination)"""
        try:
            # Save real data as primary dataset
            real_file = "data/xauusd_m15_real.csv"

            # Ensure consistent data types
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in new_data.columns:
                    new_data[col] = pd.to_numeric(new_data[col], errors='coerce')

            # Ensure Volume column exists and is numeric
            if 'Volume' not in new_data.columns:
                new_data['Volume'] = 0
            new_data['Volume'] = pd.to_numeric(new_data['Volume'], errors='coerce').fillna(0)

            # Remove duplicates based on Date and Time
            new_data = new_data.drop_duplicates(subset=['Date', 'Time'])

            # Sort by date and time with proper error handling
            try:
                new_data['datetime'] = pd.to_datetime(new_data['Date'] + ' ' + new_data['Time'], format='mixed', errors='coerce')
            except:
                # Fallback: try different parsing methods
                new_data['datetime'] = pd.to_datetime(new_data['Date'] + ' ' + new_data['Time'], errors='coerce')

            # Remove rows with invalid datetime
            new_data = new_data.dropna(subset=['datetime'])
            new_data = new_data.sort_values('datetime').drop('datetime', axis=1)

            # Save real data
            new_data.to_csv(real_file, index=False)
            logger.info(f"✅ Real MetaAPI data saved: {real_file} ({len(new_data)} rows)")

            return new_data

        except Exception as e:
            logger.error(f"❌ Save real data failed: {e}")
            return pd.DataFrame()

    async def download_and_retrain(self, days_back: int = 30):
        """Download data and retrain model"""
        try:
            # Download new data
            new_data = await self.download_historical_data(days_back=days_back)

            if new_data.empty:
                logger.error("❌ No data downloaded")
                return False

            # Save and combine data
            combined_data = await self.save_real_data_only(new_data)

            if combined_data.empty:
                logger.error("❌ Failed to process data")
                return False

            # Retrain model with new data
            logger.info("🤖 Retraining model with updated data...")

            try:
                from train import train_model

                # Use combined data if available, otherwise real data
                data_file = "data/xauusd_m15_real.csv"

                model, scaler, accuracy = train_model(data_path=data_file)
                logger.info(f"✅ Model retrained successfully! Accuracy: {accuracy:.4f}")

                return True

            except Exception as train_error:
                logger.error(f"❌ Retraining failed: {train_error}")
                return False

        except Exception as e:
            logger.error(f"❌ Download and retrain failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup connections"""
        try:
            if self.account:
                await self.account.undeploy()
            logger.info("✅ Connections cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")

async def main():
    """Main function for data download and retrain"""

    print("📊 MetaAPI Data Downloader & Model Retrain")
    print("=" * 60)

    from trading_config import TradingConfig
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
            success = await downloader.download_and_retrain(days_back=days)

            if success:
                print("\n🎉 Data download and model retraining completed!")
                print("✅ Your system is now updated with real market data")
            else:
                print("\n❌ Process failed. Please check the logs above")

        except (EOFError, KeyboardInterrupt):
            print("\n🛑 Download cancelled by user")
        except ValueError:
            print("❌ Invalid input. Please enter a number")

        finally:
            await downloader.cleanup()

    else:
        print("❌ Failed to connect to MetaAPI")
        print("💡 Please check your credentials and network connection")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Download interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")