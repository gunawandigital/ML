from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
import os
from datetime import datetime
import asyncio
from threading import Thread
import time

# Import trading modules
from trading_config import TradingConfig
from predict import get_latest_signal, load_model

app = Flask(__name__)

# Global variables for dashboard data
dashboard_data = {
    'status': 'Stopped',
    'balance': 0.0,
    'current_signal': None,
    'last_update': None,
    'trades_count': 0,
    'error_message': None
}

async def get_real_price():
    """Get real-time price from MetaAPI"""
    try:
        from metaapi_trader import MetaAPITrader
        from trading_config import TradingConfig
        
        config = TradingConfig()
        
        # Only try to get real price if credentials are properly configured
        if (config.META_API_TOKEN != "YOUR_METAAPI_TOKEN_HERE" and 
            config.ACCOUNT_ID != "YOUR_ACCOUNT_ID_HERE" and
            config.META_API_TOKEN and config.ACCOUNT_ID):
            
            trader = MetaAPITrader(
                token=config.META_API_TOKEN,
                account_id=config.ACCOUNT_ID,
                symbol="XAUUSD"
            )
            
            # Quick connection attempt with timeout
            if await asyncio.wait_for(trader.initialize(), timeout=15):
                df = await asyncio.wait_for(trader.get_real_time_data(count=1), timeout=10)
                await trader.cleanup()
                
                if not df.empty and 'Close' in df.columns:
                    return float(df['Close'].iloc[-1])
                    
    except Exception as e:
        print(f"Could not get real-time price: {e}")
    
    # Fallback to last known price from historical data
    try:
        if os.path.exists('data/xauusd_m15_real.csv'):
            import pandas as pd
            df = pd.read_csv('data/xauusd_m15_real.csv')
            if not df.empty and 'Close' in df.columns:
                return float(df['Close'].iloc[-1])
    except:
        pass
    
    # Final fallback - current XAUUSD market price estimate
    return 2650.0

def load_dashboard_data():
    """Load dashboard data safely"""
    global dashboard_data

    try:
        # Load trading config
        config = TradingConfig()

        # Check if model exists
        if os.path.exists('models/random_forest_model.pkl'):
            dashboard_data['model_status'] = 'Loaded'

            # Get latest signal if data exists
            try:
                if os.path.exists('data/xauusd_m15_real.csv'):
                    signal = get_latest_signal('data/xauusd_m15_real.csv')
                    
                    # Try to get real-time price
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        real_price = loop.run_until_complete(
                            asyncio.wait_for(get_real_price(), timeout=20)
                        )
                        loop.close()
                        
                        # Update signal with real price
                        signal['current_price'] = real_price
                        signal['timestamp'] = datetime.now()
                        signal['price_source'] = 'real-time'
                        
                    except Exception as price_error:
                        print(f"Using historical price: {price_error}")
                        signal['price_source'] = 'historical'
                    
                    dashboard_data['current_signal'] = signal
                else:
                    dashboard_data['current_signal'] = {
                        'signal': 'NO_DATA',
                        'confidence': 0.0,
                        'current_price': 2650.0,
                        'timestamp': datetime.now(),
                        'price_source': 'fallback'
                    }
            except Exception as e:
                dashboard_data['current_signal'] = {
                    'signal': 'ERROR',
                    'confidence': 0.0,
                    'current_price': 2650.0,
                    'timestamp': datetime.now(),
                    'error': str(e),
                    'price_source': 'fallback'
                }
        else:
            dashboard_data['model_status'] = 'Not Found'
            dashboard_data['current_signal'] = {
                'signal': 'NO_MODEL',
                'confidence': 0.0,
                'current_price': 2650.0,
                'timestamp': datetime.now(),
                'price_source': 'fallback'
            }

        # Set demo balance
        dashboard_data['balance'] = 5000.0
        dashboard_data['last_update'] = datetime.now()
        dashboard_data['error_message'] = None

    except Exception as e:
        dashboard_data['error_message'] = str(e)
        dashboard_data['last_update'] = datetime.now()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    load_dashboard_data()
    
    # Create dummy data for signals and performance
    signals = {'signal': 'BUY', 'confidence': 0.8}
    performance = {'roi': 0.1, 'drawdown': 0.05}

    # Check if live trading is running
    trading_status = "STOPPED"  # You can implement actual status check

    return render_template('dashboard.html', 
                         data=dashboard_data,
                         signals=signals, 
                         performance=performance,
                         trading_status=trading_status,
                         timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/api/status')
def api_status():
    """API endpoint for current status"""
    load_dashboard_data()
    return jsonify(dashboard_data)

@app.route('/api/signal')
def api_signal():
    """API endpoint for latest signal"""
    try:
        if os.path.exists('data/xauusd_m15_real.csv'):
            signal = get_latest_signal('data/xauusd_m15_real.csv')
            return jsonify({
                'success': True,
                'signal': signal
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No data available'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/config')
def api_config():
    """API endpoint for trading configuration"""
    try:
        config = TradingConfig()
        return jsonify({
            'success': True,
            'config': {
                'symbol': config.SYMBOL,
                'lot_size': config.LOT_SIZE,
                'confidence_threshold': config.CONFIDENCE_THRESHOLD,
                'stop_loss_pips': config.STOP_LOSS_PIPS,
                'take_profit_pips': config.TAKE_PROFIT_PIPS,
                'max_risk_per_trade': config.MAX_RISK_PER_TRADE
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/backtest')
def api_backtest():
    """API endpoint for backtest results"""
    try:
        # Check if backtest results exist
        if os.path.exists('backtest_results.png'):
            return jsonify({
                'success': True,
                'backtest_available': True,
                'message': 'Backtest results available'
            })
        else:
            return jsonify({
                'success': True,
                'backtest_available': False,
                'message': 'No backtest results found'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("🌐 Starting Forex Trading Dashboard...")
    print("📊 Dashboard will be available at: http://0.0.0.0:5000")
    print("🔄 Auto-refresh every 30 seconds")

    # Load initial data
    load_dashboard_data()

    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)