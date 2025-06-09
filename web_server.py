
from flask import Flask
from threading import Thread
import run_live_trading
import asyncio
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def health():
    return {
        'status': 'Trading system is running',
        'message': 'Live trading bot is active in background'
    }

@app.route('/status')
def status():
    return {
        'trading_active': True,
        'message': 'Background trading process running'
    }

def run_trading_background():
    """Run trading in background thread"""
    try:
        asyncio.run(run_live_trading.main())
    except Exception as e:
        logger.error(f"Trading error: {e}")

if __name__ == '__main__':
    # Start trading in background thread
    trading_thread = Thread(target=run_trading_background, daemon=True)
    trading_thread.start()
    logger.info("Started trading in background thread")
    
    # Start web server
    app.run(host='0.0.0.0', port=5000)
