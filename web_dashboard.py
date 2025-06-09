
#!/usr/bin/env python3
"""
Web Dashboard for Forex Trading System
"""

from flask import Flask, render_template, jsonify, request
import asyncio
import threading
import json
from datetime import datetime
from metaapi_trader import MetaAPITrader
from trading_config import TradingConfig

app = Flask(__name__)
trader = None
trading_thread = None
trading_status = {
    'is_running': False,
    'last_signal': None,
    'balance': 5000.0,
    'total_trades': 0,
    'error_message': None
}

async def run_trading_async():
    """Run trading in async context"""
    global trader, trading_status
    
    try:
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
                
        else:
            trading_status['error_message'] = "Failed to initialize MetaAPI connection"
            
    except Exception as e:
        trading_status['error_message'] = str(e)
    finally:
        trading_status['is_running'] = False
        if trader:
            await trader.cleanup()

def run_trading_thread():
    """Run trading in separate thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_trading_async())

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
    
    if not trading_status['is_running']:
        trading_thread = threading.Thread(target=run_trading_thread)
        trading_thread.daemon = True
        trading_thread.start()
        return jsonify({'success': True, 'message': 'Trading started'})
    else:
        return jsonify({'success': False, 'message': 'Trading already running'})

@app.route('/api/stop', methods=['POST'])
def stop_trading():
    """Stop automated trading"""
    trading_status['is_running'] = False
    return jsonify({'success': True, 'message': 'Trading stopped'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
