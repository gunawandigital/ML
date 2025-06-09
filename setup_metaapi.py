
"""
Setup script for MetaAPI integration
This script helps you set up the MetaAPI connection and test the integration
"""

import asyncio
import os
from datetime import datetime
import pandas as pd

async def test_metaapi_connection():
    """Test MetaAPI connection and basic functionality"""
    
    print("🚀 MetaAPI Integration Setup")
    print("=" * 50)
    
    # Check if MetaAPI SDK is installed
    try:
        from metaapi_cloud_sdk import MetaApi
        print("✅ MetaAPI SDK is installed")
    except ImportError:
        print("❌ MetaAPI SDK not found!")
        print("   Install with: pip install metaapi-cloud-sdk")
        return False
    
    # Check configuration
    from trading_config import TradingConfig
    
    config = TradingConfig()
    
    if config.META_API_TOKEN == "your_metaapi_token_here":
        print("❌ Please update META_API_TOKEN in trading_config.py")
        print("   Get your token from: https://app.metaapi.cloud/")
        return False
    
    if config.ACCOUNT_ID == "your_mt_account_id_here":
        print("❌ Please update ACCOUNT_ID in trading_config.py")
        print("   Use your MetaTrader account ID")
        return False
    
    try:
        # Initialize MetaAPI
        print(f"🔗 Connecting to MetaAPI...")
        api = MetaApi(config.META_API_TOKEN)
        account = await api.metatrader_account_api.get_account(config.ACCOUNT_ID)
        
        print(f"✅ Account found: {account.name}")
        print(f"   Type: {account.type}")
        print(f"   Server: {account.server}")
        
        # Deploy account
        print("⚙️ Deploying account...")
        await account.deploy()
        await account.wait_connected()
        
        # Create connection
        print("🔌 Creating connection...")
        connection = account.get_streaming_connection()
        await connection.connect()
        await connection.wait_synchronized()
        
        # Test basic functions
        print("📊 Testing account information...")
        account_info = await connection.get_account_information()
        print(f"   Balance: ${account_info['balance']:.2f}")
        print(f"   Equity: ${account_info['equity']:.2f}")
        print(f"   Currency: {account_info['currency']}")
        
        # Test symbol data
        print(f"📈 Testing {config.SYMBOL} data...")
        end_time = datetime.now()
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        candles = await connection.get_historical_candles(
            symbol=config.SYMBOL,
            timeframe=config.TIMEFRAME,
            start_time=start_time,
            limit=10
        )
        
        print(f"   Retrieved {len(candles)} candles")
        if candles:
            latest = candles[-1]
            print(f"   Latest price: ${latest['close']:.2f}")
            print(f"   Time: {latest['time']}")
        
        # Test positions
        print("📋 Checking positions...")
        positions = await connection.get_positions()
        print(f"   Open positions: {len(positions)}")
        
        # Close connection
        await connection.close()
        print("✅ MetaAPI connection test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

async def run_demo_trading():
    """Run a demo trading session (paper trading)"""
    
    print("\n🎯 Demo Trading Session")
    print("=" * 50)
    
    from metaapi_trader import MetaAPITrader
    from trading_config import TradingConfig
    
    config = TradingConfig()
    
    # Create trader instance with demo settings
    trader = MetaAPITrader(
        token=config.META_API_TOKEN,
        account_id=config.ACCOUNT_ID,
        symbol=config.SYMBOL,
        lot_size=config.LOT_SIZE,
        max_risk_per_trade=config.MAX_RISK_PER_TRADE,
        stop_loss_pips=config.STOP_LOSS_PIPS,
        take_profit_pips=config.TAKE_PROFIT_PIPS,
        confidence_threshold=config.CONFIDENCE_THRESHOLD
    )
    
    if await trader.initialize():
        print("✅ Trader initialized successfully!")
        
        # Generate a single signal for testing
        print("\n📊 Generating test signal...")
        signal = await trader.generate_trading_signal()
        
        print(f"   Signal: {signal['signal']}")
        print(f"   Confidence: {signal['confidence']:.1%}")
        print(f"   Current Price: ${signal['current_price']:.2f}")
        
        if signal['confidence'] >= config.CONFIDENCE_THRESHOLD:
            print(f"✅ Signal meets confidence threshold ({config.CONFIDENCE_THRESHOLD:.0%})")
            print("   Ready for live trading!")
        else:
            print(f"⚠️ Signal below confidence threshold ({config.CONFIDENCE_THRESHOLD:.0%})")
        
        # Show trading stats
        stats = trader.get_trading_stats()
        print(f"\n📈 Trading Stats: {stats}")
        
        return True
    else:
        print("❌ Failed to initialize trader")
        return False

def create_secrets_template():
    """Create a template for storing MetaAPI credentials securely"""
    
    secrets_template = '''
# MetaAPI Credentials - Add these to your Replit Secrets
# Go to: Secrets tab in your Repl sidebar

METAAPI_TOKEN=your_metaapi_token_here
METAAPI_ACCOUNT_ID=your_mt_account_id_here

# How to get your credentials:
# 1. Go to https://app.metaapi.cloud/
# 2. Sign up/login to MetaAPI
# 3. Create a new token in the Token Management section
# 4. Add your MetaTrader account (demo or live)
# 5. Copy the token and account ID to your Replit Secrets
'''
    
    with open('metaapi_secrets_template.txt', 'w') as f:
        f.write(secrets_template)
    
    print("📝 Created metaapi_secrets_template.txt")
    print("   Follow the instructions to set up your credentials")

async def main():
    """Main setup function"""
    
    print("🔧 MetaAPI Trading System Setup")
    print("=" * 60)
    
    # Create secrets template
    create_secrets_template()
    
    # Test connection
    connection_ok = await test_metaapi_connection()
    
    if connection_ok:
        # Run demo trading
        demo_ok = await run_demo_trading()
        
        if demo_ok:
            print("\n🎉 Setup completed successfully!")
            print("   Your automated trading system is ready!")
            print("\n📋 Next Steps:")
            print("   1. Review your trading configuration in trading_config.py")
            print("   2. Adjust risk parameters as needed")
            print("   3. Run 'python metaapi_trader.py' to start live trading")
            print("   4. Monitor logs for trading activity")
        else:
            print("\n⚠️ Demo trading test failed")
    else:
        print("\n❌ Setup failed - please check your credentials")

if __name__ == "__main__":
    asyncio.run(main())
