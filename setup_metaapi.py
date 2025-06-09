"""
Setup script for MetaAPI integration
This script helps you set up the MetaAPI connection and test the integration
"""

import asyncio
import os
from datetime import datetime
import pandas as pd

def setup_metaapi_account():
    """Interactive setup for MetaAPI account"""

    print("🚀 MetaAPI Setup Wizard")
    print("=" * 50)

    print("\n📋 Step 1: Get MetaAPI Account")
    print("1. Go to https://metaapi.cloud")
    print("2. Create a free account")
    print("3. Get your API token from dashboard")
    print("4. Connect your MetaTrader account")

    print("\n🔑 Step 2: Configuration")
    print("Edit trading_config.py and set:")
    print("- META_API_TOKEN: Your API token")
    print("- ACCOUNT_ID: Your MetaTrader account ID")

    # Check if config is already set
    try:
        from trading_config import TradingConfig
        config = TradingConfig()
        errors = config.validate_config()

        if not errors:
            print("\n✅ Configuration looks good!")
            return True
        else:
            print("\n❌ Configuration errors:")
            for error in errors:
                print(f"   - {error}")
            return False

    except Exception as e:
        print(f"\n❌ Configuration error: {e}")
        return False

async def test_metaapi_connection():
    """Test MetaAPI connection without real trading"""

    print("\n🧪 Testing MetaAPI Connection")
    print("=" * 50)

    try:
        from trading_config import TradingConfig
        config = TradingConfig()

        # Validate config first
        errors = config.validate_config()
        if errors:
            print("❌ Please fix configuration errors first:")
            for error in errors:
                print(f"   - {error}")
            return False

        # Try to import MetaAPI
        try:
            from metaapi_cloud_sdk import MetaApi
            print("✅ MetaAPI SDK imported successfully")
        except ImportError:
            print("❌ MetaAPI SDK not installed")
            print("   Run: pip install metaapi-cloud-sdk")
            return False

        # Test connection (without actual connection)
        print("✅ Configuration validated")
        print("✅ Ready for MetaAPI connection")
        print("\n💡 To test real connection, run the demo trading session")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
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
            print("   Waiting for better opportunity...")

        return True
    else:
        print("❌ Failed to initialize trader")
        return False

async def main():
    """Main setup function"""

    print("🤖 Forex ML Trading System - MetaAPI Setup")
    print("=" * 60)
    print(f"Setup started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Setup account
    print("\n" + "="*60)
    if not setup_metaapi_account():
        print("\n❌ Setup incomplete. Please fix configuration first.")
        return

    # Step 2: Test connection
    print("\n" + "="*60)
    if not await test_metaapi_connection():
        print("\n❌ Connection test failed.")
        return

    # Step 3: Run demo
    print("\n" + "="*60)
    choice = input("Run demo trading session? (y/n): ").lower()
    if choice == 'y':
        await run_demo_trading()

    print("\n🎉 Setup completed!")
    print("=" * 60)
    print("Next steps:")
    print("1. Verify your MetaAPI account is connected")
    print("2. Test with demo account first")
    print("3. When ready, switch to live account")
    print("4. Run automated trading with proper risk management")

if __name__ == "__main__":
    asyncio.run(main())