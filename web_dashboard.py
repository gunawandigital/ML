
#!/usr/bin/env python3
"""
Web Dashboard for Forex Trading System
"""

from flask import Flask, render_template, jsonify, request
import asyncio
import threading
import json
import traceback
from datetime import datetime

app = Flask(__name__)

# Global variables
trader = None
trading_thread = None
trading_status = {
    'is_running': False,
    'last_signal': {
        'signal': 'HOLD',
        'confidence': 0.0,
        'current_price': 2000.00,
        'timestamp': datetime.now().isoformat()
    },
    'balance': 5000.0,
    'total_trades': 0,
    'error_message': None
}

async def run_trading_async():
    """Run trading in async context"""
    global trader, trading_status
    
    try:
        # Import here to handle missing dependencies gracefully
        from metaapi_trader import MetaAPITrader
        from trading_config import TradingConfig
        
        config = TradingConfig()
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
            trading_status['is_running'] = True
            trading_status['error_message'] = None
            
            # Trading loop
            while trading_status['is_running']:
                try:
                    signal = await trader.generate_trading_signal()
                    trading_status['last_signal'] = signal
                    trading_status['balance'] = await trader.get_account_balance()
                    
                    if 'error' not in signal:
                        await trader.check_and_close_positions()
                        if not trader.current_position:
                            await trader.place_order(signal)
                    
                    stats = trader.get_trading_stats()
                    trading_status['total_trades'] = stats.get('total_trades', 0)
                    
                    await asyncio.sleep(300)  # 5 minutes
                except Exception as e:
                    print(f"Trading loop error: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retry
                    
        else:
            trading_status['error_message'] = "Failed to initialize MetaAPI connection"
            
    except ImportError as e:
        trading_status['error_message'] = f"Missing dependencies: {e}"
    except Exception as e:
        trading_status['error_message'] = f"Trading error: {e}"
        print(f"Trading error details: {traceback.format_exc()}")
    finally:
        trading_status['is_running'] = False
        if trader:
            try:
                await trader.cleanup()
            except:
                pass

def run_trading_thread():
    """Run trading in separate thread"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_trading_async())
    except Exception as e:
        trading_status['error_message'] = f"Thread error: {e}"
        trading_status['is_running'] = False

@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current trading status"""
    return jsonify(trading_status)

@app.route('/api/start', methods=['POST'])
def start_trading():
    """Start automated trading"""
    global trading_thread
    
    try:
        if not trading_status['is_running']:
            trading_status['error_message'] = None
            trading_thread = threading.Thread(target=run_trading_thread)
            trading_thread.daemon = True
            trading_thread.start()
            return jsonify({'success': True, 'message': 'Trading started successfully'})
        else:
            return jsonify({'success': False, 'message': 'Trading is already running'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to start trading: {e}'})

@app.route('/api/stop', methods=['POST'])
def stop_trading():
    """Stop automated trading"""
    try:
        trading_status['is_running'] = False
        return jsonify({'success': True, 'message': 'Trading stopped successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to stop trading: {e}'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'trading_active': trading_status['is_running']
    })

if __name__ == '__main__':
    print("🚀 Starting Forex Trading Dashboard...")
    print("📊 Dashboard will be available at: http://0.0.0.0:5000")
    print("🔄 Use Ctrl+C to stop the server")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except Exception as e:
        print(f"❌ Dashboard startup error: {e}")
        print("💡 Please check if port 5000 is available")
