
#!/usr/bin/env python3
"""
Auto Live Trading System
Automatically starts live trading with web monitoring dashboard
"""

import asyncio
import signal
import sys
import os
import threading
import time
from datetime import datetime
from flask import Flask, jsonify, render_template, request
from metaapi_trader import MetaAPITrader
from trading_config import TradingConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global trading state
trader_instance = None
trading_task = None
trading_active = False

# Flask app for monitoring
app = Flask(__name__)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global trading_active
    print('\n🛑 Auto trading stopped by user')
    trading_active = False
    sys.exit(0)

@app.route('/')
def dashboard():
    """Main dashboard"""
    return jsonify({
        'status': 'Auto Live Trading System',
        'trading_active': trading_active,
        'timestamp': datetime.now().isoformat(),
        'message': 'Trading system running automatically',
        'endpoints': {
            'status': '/status',
            'stop': '/stop',
            'stats': '/stats',
            'health': '/health'
        }
    })

@app.route('/status')
def status():
    """Get trading status"""
    global trader_instance, trading_active
    
    if trader_instance:
        stats = trader_instance.get_trading_stats()
    else:
        stats = {'total_trades': 0, 'message': 'Trader not initialized'}
    
    return jsonify({
        'trading_active': trading_active,
        'trader_initialized': trader_instance is not None,
        'stats': stats,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/stop', methods=['POST'])
def stop_trading():
    """Stop trading"""
    global trading_active, trader_instance
    
    trading_active = False
    if trader_instance:
        trader_instance.stop_trading()
    
    return jsonify({
        'success': True,
        'message': 'Trading stopped successfully',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/stats')
def get_stats():
    """Get trading statistics"""
    global trader_instance
    
    if trader_instance:
        stats = trader_instance.get_trading_stats()
        return jsonify({
            'success': True,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Trader not initialized',
            'timestamp': datetime.now().isoformat()
        })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Auto Live Trading System',
        'uptime': time.time(),
        'timestamp': datetime.now().isoformat()
    })

async def initialize_and_trade():
    """Initialize trader and start trading"""
    global trader_instance, trading_active
    
    print("🤖 AUTO LIVE TRADING SYSTEM")
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
        print("\n💡 Please add META_API_TOKEN and ACCOUNT_ID to Replit Secrets")
        return False
    
    print("\n📊 Trading Configuration:")
    print(f"   Symbol: {config.SYMBOL}")
    print(f"   Lot Size: {config.LOT_SIZE}")
    print(f"   Stop Loss: {config.STOP_LOSS_PIPS} pips")
    print(f"   Take Profit: {config.TAKE_PROFIT_PIPS} pips")
    print(f"   Confidence Threshold: {config.CONFIDENCE_THRESHOLD:.0%}")
    print(f"   Max Risk per Trade: {config.MAX_RISK_PER_TRADE:.1%}")
    
    # Create trader
    trader_instance = MetaAPITrader(
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
    if await trader_instance.initialize():
        print("✅ Connection established successfully!")
        
        # Auto-start trading
        print("\n🚀 AUTO-STARTING LIVE TRADING...")
        print("   Signal check interval: 5 minutes")
        print("   Web monitoring: http://localhost:5000")
        print("   Stop trading: POST /stop")
        print("\n" + "="*60)
        
        trading_active = True
        
        try:
            # Start trading with 5-minute intervals
            await trader_instance.start_trading(check_interval=300)
        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
        except Exception as e:
            print(f"\n❌ Trading error: {e}")
        finally:
            trading_active = False
            await trader_instance.cleanup()
            
            # Show trading stats
            if trader_instance:
                stats = trader_instance.get_trading_stats()
                print(f"\n📈 Auto Trading Session Summary:")
                print(f"   Total Trades: {stats.get('total_trades', 0)}")
                if stats.get('total_trades', 0) > 0:
                    print(f"   Buy Trades: {stats.get('buy_trades', 0)}")
                    print(f"   Sell Trades: {stats.get('sell_trades', 0)}")
                    print(f"   Average Confidence: {stats.get('avg_confidence', 0):.1%}")
    else:
        print("❌ Failed to initialize connection")
        trading_active = False
        return False
    
    return True

def run_flask_app():
    """Run Flask monitoring app"""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

async def main():
    """Main function"""
    
    # Start Flask monitoring in background thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    
    print("🌐 Web monitoring started on http://localhost:5000")
    
    # Wait a moment for Flask to start
    await asyncio.sleep(2)
    
    # Start trading
    await initialize_and_trade()

if __name__ == "__main__":
    try:
        # Set auto trading environment variable
        os.environ["AUTO_TRADING"] = "true"
        
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Auto trading system interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
