
from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import asyncio
from threading import Thread
import time
import logging
import numpy as np
import traceback

# Import trading modules
from trading_config import TradingConfig
from predict import get_latest_signal, load_model

app = Flask(__name__)

# Setup proper logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for dashboard data
dashboard_data = {
    'status': 'Stopped',
    'balance': 0.0,
    'current_signal': None,
    'last_update': None,
    'trades_count': 0,
    'error_message': None,
    'live_trading_active': False,
    'last_trade_check': None,
    'system_health': {
        'model_loaded': False,
        'data_available': False,
        'connection_status': 'Unknown',
        'last_signal_time': None,
        'errors': []
    }
}

def check_live_trading_status():
    """Check if live trading is currently active"""
    try:
        # Check if there's a running live trading process
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'run_live_trading.py' in cmdline and proc.is_running():
                    return True, f"Live trading active (PID: {proc.info['pid']})"
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return False, "Live trading not detected"
        
    except ImportError:
        # Fallback: check for trading log activity
        import os
        from datetime import datetime, timedelta
        
        # Check if there are recent trading activities
        if os.path.exists('.metaapi'):
            metaapi_files = os.listdir('.metaapi')
            recent_activity = False
            for file in metaapi_files:
                file_path = os.path.join('.metaapi', file)
                if os.path.isfile(file_path):
                    # Check if file was modified in last 10 minutes
                    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if datetime.now() - mtime < timedelta(minutes=10):
                        recent_activity = True
                        break
            
            if recent_activity:
                return True, "Recent trading activity detected"
        
        return False, "No live trading activity"
    
    except Exception as e:
        return False, f"Error checking live trading: {str(e)}"

def check_system_health():
    """Comprehensive system health check"""
    health = {
        'model_loaded': False,
        'data_available': False,
        'connection_status': 'Disconnected',
        'last_signal_time': None,
        'errors': []
    }
    
    try:
        # Check model
        if os.path.exists('models/random_forest_model.pkl') and os.path.exists('models/scaler.pkl'):
            health['model_loaded'] = True
        else:
            health['errors'].append('ML Model files not found')
        
        # Check data
        if os.path.exists('data/xauusd_m15_real.csv'):
            df = pd.read_csv('data/xauusd_m15_real.csv')
            if len(df) > 0:
                health['data_available'] = True
                health['last_data_time'] = df.iloc[-1].get('Date', 'Unknown')
            else:
                health['errors'].append('Data file is empty')
        else:
            health['errors'].append('Real-time data file not found')
        
        # Check trading config
        config = TradingConfig()
        config_errors = config.validate_config()
        if config_errors:
            health['errors'].extend(config_errors)
            health['connection_status'] = 'Configuration Error'
        else:
            health['connection_status'] = 'Configuration OK'
            
    except Exception as e:
        health['errors'].append(f'Health check error: {str(e)}')
    
    return health

async def get_real_price_safe():
    """Safely get real-time price with comprehensive error handling"""
    try:
        from metaapi_trader import MetaAPITrader
        from trading_config import TradingConfig
        
        config = TradingConfig()
        
        # Validate configuration first
        if (config.META_API_TOKEN == "YOUR_METAAPI_TOKEN_HERE" or 
            config.ACCOUNT_ID == "YOUR_ACCOUNT_ID_HERE" or
            not config.META_API_TOKEN or not config.ACCOUNT_ID):
            logger.warning("MetaAPI credentials not configured")
            return None, "MetaAPI credentials not configured"
        
        trader = MetaAPITrader(
            token=config.META_API_TOKEN,
            account_id=config.ACCOUNT_ID,
            symbol="XAUUSD"
        )
        
        # Quick connection attempt with strict timeout
        logger.info("Attempting to get real-time price...")
        
        init_success = await asyncio.wait_for(trader.initialize(), timeout=30)
        if not init_success:
            await trader.cleanup()
            return None, "Failed to initialize MetaAPI connection"
        
        # Get real-time data
        df = await asyncio.wait_for(trader.get_real_time_data(count=1), timeout=15)
        await trader.cleanup()
        
        if not df.empty and 'Close' in df.columns:
            price = float(df['Close'].iloc[-1])
            logger.info(f"Successfully retrieved real-time price: ${price:.2f}")
            return price, "real-time"
        else:
            return None, "No price data returned"
            
    except asyncio.TimeoutError:
        logger.warning("Real-time price request timed out")
        return None, "Connection timeout"
    except Exception as e:
        logger.error(f"Error getting real-time price: {str(e)}")
        return None, f"Error: {str(e)}"

def get_fallback_price():
    """Get fallback price from historical data"""
    try:
        if os.path.exists('data/xauusd_m15_real.csv'):
            df = pd.read_csv('data/xauusd_m15_real.csv')
            if not df.empty and 'Close' in df.columns:
                price = float(df['Close'].iloc[-1])
                logger.info(f"Using fallback price from historical data: ${price:.2f}")
                return price, "historical"
    except Exception as e:
        logger.error(f"Error getting fallback price: {e}")
    
    # Ultimate fallback
    return 2650.0, "fallback"

def get_signal_safely(data_path='data/xauusd_m15_real.csv'):
    """Get trading signal with comprehensive error handling"""
    try:
        if not os.path.exists(data_path):
            return {
                'signal': 'NO_DATA',
                'confidence': 0.0,
                'current_price': 2650.0,
                'timestamp': datetime.now(),
                'error': 'Data file not found',
                'price_source': 'fallback'
            }
        
        # Check if model exists
        if not (os.path.exists('models/random_forest_model.pkl') and 
                os.path.exists('models/scaler.pkl')):
            return {
                'signal': 'NO_MODEL',
                'confidence': 0.0,
                'current_price': 2650.0,
                'timestamp': datetime.now(),
                'error': 'ML model not found',
                'price_source': 'fallback'
            }
        
        # Get signal with error handling
        signal = get_latest_signal(data_path)
        
        # Add price source info
        signal['price_source'] = 'historical'
        signal['error'] = None
        
        return signal
        
    except Exception as e:
        logger.error(f"Error getting signal: {str(e)}")
        fallback_price, price_source = get_fallback_price()
        return {
            'signal': 'ERROR',
            'confidence': 0.0,
            'current_price': fallback_price,
            'timestamp': datetime.now(),
            'error': str(e),
            'price_source': price_source
        }

def load_dashboard_data():
    """Load dashboard data with comprehensive error handling"""
    global dashboard_data
    
    logger.info("Loading dashboard data...")
    
    try:
        # System health check
        dashboard_data['system_health'] = check_system_health()
        
        # Load trading config
        config = TradingConfig()
        
        # Get current signal
        signal = get_signal_safely()
        
        # Try to get real-time price
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            real_price, price_source = loop.run_until_complete(
                asyncio.wait_for(get_real_price_safe(), timeout=25)
            )
            
            if real_price is not None:
                signal['current_price'] = real_price
                signal['price_source'] = price_source
                signal['timestamp'] = datetime.now()
            else:
                logger.warning(f"Real-time price failed: {price_source}")
                # Keep existing price from signal
                
        except Exception as price_error:
            logger.error(f"Real-time price error: {price_error}")
            # Keep existing price from signal
        finally:
            try:
                loop.close()
            except:
                pass
        
        dashboard_data['current_signal'] = signal
        
        # Check live trading status
        live_trading_active, live_trading_message = check_live_trading_status()
        dashboard_data['live_trading_active'] = live_trading_active
        dashboard_data['live_trading_message'] = live_trading_message
        dashboard_data['last_trade_check'] = datetime.now()
        
        # Set status based on live trading and signal quality
        if signal.get('error'):
            dashboard_data['status'] = 'Error'
        elif live_trading_active:
            dashboard_data['status'] = '🚀 LIVE TRADING'
        elif signal['confidence'] >= 0.7:
            dashboard_data['status'] = 'Ready for Trading'
        else:
            dashboard_data['status'] = 'Monitoring'
        
        # Account info
        dashboard_data['balance'] = 5000.0  # Demo balance
        dashboard_data['last_update'] = datetime.now()
        dashboard_data['error_message'] = None
        
        # Trading stats
        dashboard_data['trades_today'] = 0
        dashboard_data['win_rate'] = 0.0
        dashboard_data['daily_pnl'] = 0.0
        
        logger.info("Dashboard data loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading dashboard data: {str(e)}")
        dashboard_data['error_message'] = f"Dashboard error: {str(e)}"
        dashboard_data['last_update'] = datetime.now()

@app.route('/')
def dashboard():
    """Main dashboard page with comprehensive data"""
    try:
        load_dashboard_data()
        
        # Enhanced data for template
        enhanced_data = dashboard_data.copy()
        
        # Add performance metrics
        enhanced_data['performance'] = {
            'daily_return': 0.0,
            'weekly_return': 0.0,
            'monthly_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Add recent signals (mock for now)
        enhanced_data['recent_signals'] = [
            {
                'time': '16:55',
                'signal': 'SELL',
                'confidence': 46.7,
                'price': 3293.78,
                'executed': False
            }
        ]
        
        # Add market hours info
        now = datetime.now()
        enhanced_data['market_hours'] = {
            'is_open': True,  # Markets are generally always open for XAUUSD
            'current_time': now.strftime('%H:%M:%S'),
            'timezone': 'UTC'
        }
        
        return render_template('dashboard.html', 
                             data=enhanced_data,
                             timestamp=now.strftime('%Y-%m-%d %H:%M:%S'))
                             
    except Exception as e:
        logger.error(f"Dashboard route error: {str(e)}")
        error_data = {
            'error_message': f"Dashboard error: {str(e)}",
            'status': 'Error',
            'last_update': datetime.now()
        }
        return render_template('dashboard.html', 
                             data=error_data,
                             timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/api/health')
def api_health():
    """Comprehensive system health API"""
    try:
        health = check_system_health()
        return jsonify({
            'success': True,
            'health': health,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/status')
def api_status():
    """Enhanced API endpoint for current status"""
    try:
        load_dashboard_data()
        return jsonify({
            'success': True,
            'data': dashboard_data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/signal')
def api_signal():
    """Enhanced API endpoint for latest signal"""
    try:
        if os.path.exists('data/xauusd_m15_real.csv'):
            signal = get_signal_safely('data/xauusd_m15_real.csv')
            return jsonify({
                'success': True,
                'signal': signal,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No data available',
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/config')
def api_config():
    """API endpoint for trading configuration"""
    try:
        config = TradingConfig()
        validation_errors = config.validate_config()
        
        return jsonify({
            'success': True,
            'config': {
                'symbol': config.SYMBOL,
                'lot_size': config.LOT_SIZE,
                'confidence_threshold': config.CONFIDENCE_THRESHOLD,
                'stop_loss_pips': config.STOP_LOSS_PIPS,
                'take_profit_pips': config.TAKE_PROFIT_PIPS,
                'max_risk_per_trade': config.MAX_RISK_PER_TRADE,
                'max_daily_trades': config.MAX_DAILY_TRADES,
                'trading_hours': f"{config.TRADING_START_HOUR}:00 - {config.TRADING_END_HOUR}:00"
            },
            'validation_errors': validation_errors,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/logs')
def api_logs():
    """API endpoint for recent system logs"""
    try:
        logs = []
        
        # Add recent log entries (in a real system, you'd read from log files)
        logs.append({
            'timestamp': datetime.now().isoformat(),
            'level': 'INFO',
            'message': 'Dashboard data refreshed successfully'
        })
        
        if dashboard_data.get('current_signal'):
            signal = dashboard_data['current_signal']
            logs.append({
                'timestamp': signal.get('timestamp', datetime.now()).isoformat(),
                'level': 'INFO',
                'message': f"Signal: {signal.get('signal', 'Unknown')} | Confidence: {signal.get('confidence', 0):.1%}"
            })
        
        if dashboard_data.get('system_health', {}).get('errors'):
            for error in dashboard_data['system_health']['errors']:
                logs.append({
                    'timestamp': datetime.now().isoformat(),
                    'level': 'ERROR',
                    'message': error
                })
        
        return jsonify({
            'success': True,
            'logs': logs[-10:],  # Last 10 logs
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/backtest')
def api_backtest():
    """API endpoint for backtest results"""
    try:
        if os.path.exists('backtest_results.png'):
            return jsonify({
                'success': True,
                'backtest_available': True,
                'message': 'Backtest results available',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': True,
                'backtest_available': False,
                'message': 'No backtest results found',
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

if __name__ == '__main__':
    print("🌐 Starting Enhanced Forex Trading Dashboard...")
    print("📊 Dashboard URL: http://0.0.0.0:5000")
    print("🔄 Auto-refresh: 30 seconds")
    print("📋 Health check: /api/health")
    print("📊 System logs: /api/logs")
    
    # Load initial data
    load_dashboard_data()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
