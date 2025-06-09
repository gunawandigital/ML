
#!/usr/bin/env python3
"""
Live Automated Trading Runner
Start automated trading with MetaAPI integration
"""

import asyncio
import signal
import sys
from datetime import datetime
from metaapi_trader import MetaAPITrader
from trading_config import TradingConfig

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n🛑 Trading stopped by user')
    sys.exit(0)

async def main():
    """Run automated trading"""
    
    print("🤖 FOREX ML AUTOMATED TRADING SYSTEM")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load configuration
    config = TradingConfig()
    
    # Validate config
    errors = config.validate_config()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        return
    
    print("\n📊 Trading Configuration:")
    print(f"   Symbol: {config.SYMBOL}")
    print(f"   Lot Size: {config.LOT_SIZE}")
    print(f"   Stop Loss: {config.STOP_LOSS_PIPS} pips")
    print(f"   Take Profit: {config.TAKE_PROFIT_PIPS} pips")
    print(f"   Confidence Threshold: {config.CONFIDENCE_THRESHOLD:.0%}")
    print(f"   Max Risk per Trade: {config.MAX_RISK_PER_TRADE:.1%}")
    print(f"   Trading Hours: {config.TRADING_START_HOUR}:00 - {config.TRADING_END_HOUR}:00")
    
    # Create trader
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
    
    # Initialize connection
    print("\n🔌 Initializing MetaAPI connection...")
    if await trader.initialize():
        print("✅ Connection established successfully!")
        
        # Confirm before starting
        print("\n⚠️ IMPORTANT NOTICE:")
        print("   - This will start LIVE automated trading")
        print("   - Real money trades will be executed")
        print("   - Press Ctrl+C to stop trading anytime")
        
        confirm = input("\nStart automated trading? (yes/no): ").lower()
        
        if confirm in ['yes', 'y']:
            print("\n🚀 Starting automated trading...")
            print("   Signal check interval: 5 minutes")
            print("   Press Ctrl+C to stop\n")
            
            try:
                # Start trading with 5-minute intervals
                await trader.start_trading(check_interval=300)
            except KeyboardInterrupt:
                print("\n🛑 Trading stopped by user")
            finally:
                await trader.cleanup()
                
                # Show trading stats
                stats = trader.get_trading_stats()
                print(f"\n📈 Trading Session Summary:")
                print(f"   Total Trades: {stats.get('total_trades', 0)}")
                if stats.get('total_trades', 0) > 0:
                    print(f"   Buy Trades: {stats.get('buy_trades', 0)}")
                    print(f"   Sell Trades: {stats.get('sell_trades', 0)}")
                    print(f"   Average Confidence: {stats.get('avg_confidence', 0):.1%}")
        else:
            print("❌ Trading cancelled by user")
            await trader.cleanup()
    else:
        print("❌ Failed to initialize connection")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Program interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
