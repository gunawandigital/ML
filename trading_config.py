
"""
Configuration file for MetaAPI automated trading
"""

class TradingConfig:
    """Trading configuration settings"""
    
    # MetaAPI Credentials (Get from https://app.metaapi.cloud/)
    META_API_TOKEN = "your_metaapi_token_here"  # Your MetaAPI token
    ACCOUNT_ID = "your_mt_account_id_here"      # Your MetaTrader account ID
    
    # Trading Parameters
    SYMBOL = "XAUUSD"                           # Trading symbol
    TIMEFRAME = "M15"                           # Timeframe for analysis
    
    # Risk Management
    LOT_SIZE = 0.01                             # Default position size
    MAX_RISK_PER_TRADE = 0.02                   # Maximum 2% risk per trade
    STOP_LOSS_PIPS = 50                         # Stop loss in pips
    TAKE_PROFIT_PIPS = 100                      # Take profit in pips
    
    # ML Model Settings
    CONFIDENCE_THRESHOLD = 0.7                  # Minimum 70% confidence for trades
    MODEL_PATH = "models/"                      # Path to trained model
    
    # Trading Schedule
    CHECK_INTERVAL = 300                        # Check for signals every 5 minutes
    TRADING_HOURS = {
        'start': '00:00',                       # Start trading time (UTC)
        'end': '23:59'                          # End trading time (UTC)
    }
    
    # Advanced Settings
    MAX_POSITIONS = 1                           # Maximum concurrent positions
    ENABLE_TRAILING_STOP = False                # Enable trailing stop loss
    TRAILING_STOP_PIPS = 30                     # Trailing stop distance
    
    # Logging
    LOG_LEVEL = "INFO"                          # Logging level
    LOG_TO_FILE = True                          # Save logs to file
    LOG_FILE = "trading_log.txt"                # Log file name

# Risk Management Rules
class RiskManagement:
    """Risk management rules and calculations"""
    
    @staticmethod
    def calculate_position_size(balance: float, risk_per_trade: float, stop_loss_pips: int, pip_value: float) -> float:
        """Calculate optimal position size"""
        risk_amount = balance * risk_per_trade
        position_size = risk_amount / (stop_loss_pips * pip_value)
        return round(position_size, 2)
    
    @staticmethod
    def validate_trade(signal: dict, config: TradingConfig) -> bool:
        """Validate if trade meets risk criteria"""
        return (
            signal['confidence'] >= config.CONFIDENCE_THRESHOLD and
            signal['signal'] in ['BUY', 'SELL']
        )

# Trading Sessions
TRADING_SESSIONS = {
    'london': {'start': '08:00', 'end': '17:00'},
    'new_york': {'start': '13:00', 'end': '22:00'},
    'tokyo': {'start': '00:00', 'end': '09:00'},
    'sydney': {'start': '22:00', 'end': '07:00'}
}

# Currency pair specifications
SYMBOL_SPECS = {
    'XAUUSD': {
        'pip_value': 0.01,
        'min_lot': 0.01,
        'max_lot': 100.0,
        'spread_typical': 3.0
    },
    'EURUSD': {
        'pip_value': 0.0001,
        'min_lot': 0.01,
        'max_lot': 100.0,
        'spread_typical': 1.5
    },
    'GBPUSD': {
        'pip_value': 0.0001,
        'min_lot': 0.01,
        'max_lot': 100.0,
        'spread_typical': 2.0
    }
}
