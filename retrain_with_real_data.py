
#!/usr/bin/env python3
"""
Quick script to retrain model with real MetaAPI data
"""

import asyncio
import os
from datetime import datetime

async def quick_retrain(days_back=30):
    """Quick retraining with real data"""
    
    print("🚀 Quick Retrain with Real MetaAPI Data")
    print("=" * 50)
    
    try:
        from trading_config import TradingConfig
        from data_downloader import MetaAPIDataDownloader
        
        config = TradingConfig()
        
        # Validate config
        errors = config.validate_config()
        if errors:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        # Download real data
        downloader = MetaAPIDataDownloader(
            token=config.META_API_TOKEN,
            account_id=config.ACCOUNT_ID
        )
        
        if await downloader.initialize():
            print(f"📥 Downloading {days_back} days of real data...")
            
            success = await downloader.download_and_retrain(
                days_back=days_back,
                retrain_model=True
            )
            
            if success:
                print("✅ Retraining completed with real data!")
                
                # Quick performance test
                print("\n📊 Testing performance...")
                from predict import get_latest_signal
                
                signal = get_latest_signal(data_path='data/xauusd_m15_combined.csv')
                print(f"Latest signal: {signal['signal']} ({signal['confidence']:.1%})")
                
                return True
            else:
                print("❌ Retraining failed")
                return False
        else:
            print("❌ Failed to connect to MetaAPI")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    days = 30
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except:
            days = 30
    
    print(f"Retraining with {days} days of real data...")
    success = asyncio.run(quick_retrain(days))
    
    if success:
        print("\n🎉 Ready for improved trading!")
    else:
        print("\n❌ Retraining failed. Check configuration.")
