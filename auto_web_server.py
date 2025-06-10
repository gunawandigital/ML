
#!/usr/bin/env python3
"""
Auto Web Server with Integrated Live Trading
Combines web dashboard with automatic live trading
"""

from flask import Flask, jsonify, render_template
from threading import Thread
import asyncio
import logging
import time
import os
from datetime import datetime
from metaapi_trader import MetaAPITrader
from trading_config import TradingConfig

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global trading state
trader_instance = None
trading_active = False
trading_task = None

@app.route('/')
def dashboard():
    """Main dashboard"""
    return jsonify({
        'service': 'Auto Live Trading Web Server',
        'trading_active': trading_active,
        'timestamp': datetime.now().isoformat(),
        'trader_status': 'initialized' if trader_instance else 'not_initialized',
        'endpoints': {
            'status': '/api/status',
            'health': '/api/health',
            'stats': '/api/stats',
            'stop': '/api/stop (POST)',
            'start': '/api/start (POST)'
        },
        'message': '🤖 Auto trading system running in background'
    })

@app.route('/api/status')
def api_status():
    """API status endpoint"""
    global trader_instance, trading_active
    
    status_data = {
        'trading_active': trading_active,
        'trader_initialized': trader_instance is not None,
        'timestamp': datetime.now().isoformat(),
        'system_status': 'running'
    }
    
    if trader_instance:
        try:
            stats = trader_instance.get_trading_stats()
            status_data['trading_stats'] = stats
        except Exception as e:
            status_data['stats_error'] = str(e)
    
    return jsonify(status_data)

@app.route('/api/health')
def api_health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'Auto Live Trading System',
        'uptime': time.time(),
        'trading_active': trading_active,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/stats')
def api_stats():
    """Get trading statistics"""
    global trader_instance
    
    if trader_instance:
        try:
            stats = trader_instance.get_trading_stats()
            return jsonify({
                'success': True,
                'stats': stats,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    else:
        return jsonify({
            'success': False,
            'message': 'Trader not initialized',
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop trading"""
    global trading_active, trader_instance
    
    try:
        trading_active = False
        if trader_instance:
            trader_instance.stop_trading()
        
        return jsonify({
            'success': True,
            'message': 'Trading stopped successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

async def auto_trading_worker():
    """Background trading worker"""
    global trader_instance, trading_active
    
    logger.info("🤖 Starting auto trading worker...")
    
    # Load configuration
    config = TradingConfig()
    
    # Validate config
    errors = config.validate_config()
    if errors:
        logger.error("❌ Configuration errors:")
        for error in errors:
            logger.error(f"   - {error}")
        return
    
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
    logger.info("🔌 Initializing MetaAPI connection...")
    if await trader_instance.initialize():
        logger.info("✅ MetaAPI connection established!")
        
        trading_active = True
        logger.info("🚀 Auto trading started with 5-minute intervals")
        
        try:
            await trader_instance.start_trading(check_interval=300)
        except Exception as e:
            logger.error(f"❌ Trading error: {e}")
        finally:
            trading_active = False
            if trader_instance:
                await trader_instance.cleanup()
    else:
        logger.error("❌ Failed to initialize MetaAPI connection")

def start_trading_background():
    """Start trading in background thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(auto_trading_worker())
    except Exception as e:
        logger.error(f"Background trading error: {e}")
    finally:
        loop.close()

if __name__ == '__main__':
    # Set auto trading environment
    os.environ["AUTO_TRADING"] = "true"
    
    print("🤖 AUTO LIVE TRADING WEB SERVER")
    print("=" * 50)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🌐 Web interface: http://localhost:5000")
    print("🤖 Auto trading: ENABLED")
    print("=" * 50)
    
    # Start trading in background thread
    trading_thread = Thread(target=start_trading_background, daemon=True)
    trading_thread.start()
    logger.info("✅ Background trading thread started")
    
    # Start web server
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Auto trading web server stopped")
        trading_active = False
