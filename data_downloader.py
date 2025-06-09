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

            # Get historical candles
            candles = await self.account.get_historical_candles(
                symbol, timeframe, start_time, end_time
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

    async def save_and_combine_data(self, new_data: pd.DataFrame):
        """Save new data and combine with existing data"""
        try:
            # Save real data
            real_file = "data/xauusd_m15_real.csv"
            new_data.to_csv(real_file, index=False)
            logger.info(f"✅ Real data saved: {real_file}")

            # Combine with sample data if available
            sample_file = "data/xauusd_m15.csv"
            combined_file = "data/xauusd_m15_combined.csv"

            if os.path.exists(sample_file):
                sample_data = pd.read_csv(sample_file)

                # Combine datasets
                combined_data = pd.concat([sample_data, new_data], ignore_index=True)

                # Remove duplicates based on Date and Time
                combined_data = combined_data.drop_duplicates(subset=['Date', 'Time'])

                # Sort by date and time
                combined_data['datetime'] = pd.to_datetime(combined_data['Date'] + ' ' + combined_data['Time'])
                combined_data = combined_data.sort_values('datetime').drop('datetime', axis=1)

                # Save combined data
                combined_data.to_csv(combined_file, index=False)
                logger.info(f"✅ Combined data saved: {combined_file} ({len(combined_data)} rows)")

                return combined_data
            else:
                logger.warning("Sample data not found, using only real data")
                return new_data

        except Exception as e:
            logger.error(f"❌ Save/combine failed: {e}")
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
            combined_data = await self.save_and_combine_data(new_data)

            if combined_data.empty:
                logger.error("❌ Failed to process data")
                return False

            # Retrain model with new data
            logger.info("🤖 Retraining model with updated data...")

            try:
                from train import train_model

                # Use combined data if available, otherwise real data
                data_file = "data/xauusd_m15_combined.csv" if os.path.exists("data/xauusd_m15_combined.csv") else "data/xauusd_m15_real.csv"

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