
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
                    dashboard_data['current_signal'] = signal
                else:
                    dashboard_data['current_signal'] = {
                        'signal': 'NO_DATA',
                        'confidence': 0.0,
                        'current_price': 0.0,
                        'timestamp': datetime.now()
                    }
            except Exception as e:
                dashboard_data['current_signal'] = {
                    'signal': 'ERROR',
                    'confidence': 0.0,
                    'current_price': 0.0,
                    'timestamp': datetime.now(),
                    'error': str(e)
                }
        else:
            dashboard_data['model_status'] = 'Not Found'
            dashboard_data['current_signal'] = {
                'signal': 'NO_MODEL',
                'confidence': 0.0,
                'current_price': 0.0,
                'timestamp': datetime.now()
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
    return render_template('dashboard.html', data=dashboard_data)

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
