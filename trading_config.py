"""
Configuration file for MetaAPI automated trading
"""

class TradingConfig:
    """Trading configuration settings"""

    def __init__(self):
        # MetaAPI Settings (HARUS DIISI!)
        self.META_API_TOKEN = "YOUR_METAAPI_TOKEN_HERE"  # Dapatkan dari metaapi.cloud
        self.ACCOUNT_ID = "YOUR_ACCOUNT_ID_HERE"         # ID akun MetaTrader

        # Trading Settings
        self.SYMBOL = "XAUUSD"              # Gold/USD
        self.LOT_SIZE = 0.01                # 0.01 lot = 1000 units
        self.MAX_RISK_PER_TRADE = 0.02      # 2% risk per trade
        self.STOP_LOSS_PIPS = 100           # 100 pips stop loss
        self.TAKE_PROFIT_PIPS = 200         # 200 pips take profit
        self.CONFIDENCE_THRESHOLD = 0.70    # 70% minimum confidence

        # Risk Management
        self.MAX_DAILY_TRADES = 10          # Max 10 trades per day
        self.MAX_OPEN_POSITIONS = 3         # Max 3 open positions
        self.TRADING_START_HOUR = 6         # Start trading at 6 AM
        self.TRADING_END_HOUR = 22          # Stop trading at 10 PM

        # Timing Settings
        self.SIGNAL_CHECK_INTERVAL = 300    # Check signals every 5 minutes
        self.DATA_REFRESH_INTERVAL = 60     # Refresh data every 1 minute

    def validate_config(self):
        """Validate configuration settings"""
        errors = []

        if self.META_API_TOKEN == "YOUR_METAAPI_TOKEN_HERE":
            errors.append("MetaAPI token not set")

        if self.ACCOUNT_ID == "YOUR_ACCOUNT_ID_HERE":
            errors.append("Account ID not set")

        if self.LOT_SIZE <= 0:
            errors.append("Lot size must be positive")

        if self.CONFIDENCE_THRESHOLD < 0.5 or self.CONFIDENCE_THRESHOLD > 1.0:
            errors.append("Confidence threshold must be between 0.5 and 1.0")

        return errors

    def get_trading_hours(self):
        """Get trading hours range"""
        return self.TRADING_START_HOUR, self.TRADING_END_HOUR

    def is_trading_time(self):
        """Check if current time is within trading hours"""
        from datetime import datetime
        current_hour = datetime.now().hour
        start, end = self.get_trading_hours()
        return start <= current_hour <= end