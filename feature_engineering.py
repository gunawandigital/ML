import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    ema = EMAIndicator(close=data, window=period)
    return ema.ema_indicator()

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    rsi = RSIIndicator(close=data, window=period)
    return rsi.rsi()

def calculate_returns(data, period=1):
    """Calculate price returns"""
    return data.pct_change(period)

def create_features(df):
    """Create all technical indicators and features"""
    # Copy dataframe
    data = df.copy()

    # Technical indicators
    data['EMA_9'] = calculate_ema(data['Close'], 9)
    data['EMA_21'] = calculate_ema(data['Close'], 21)
    data['EMA_50'] = calculate_ema(data['Close'], 50)

    data['RSI_14'] = calculate_rsi(data['Close'], 14)
    data['RSI_21'] = calculate_rsi(data['Close'], 21)

    # Price features
    data['HL_Ratio'] = (data['High'] - data['Low']) / data['Close']
    data['OC_Ratio'] = (data['Close'] - data['Open']) / data['Open']

    # Returns
    data['Return_1'] = calculate_returns(data['Close'], 1)
    data['Return_5'] = calculate_returns(data['Close'], 5)
    data['Return_15'] = calculate_returns(data['Close'], 15)

    # Volatility
    data['Volatility_5'] = data['Return_1'].rolling(5).std()
    data['Volatility_15'] = data['Return_1'].rolling(15).std()

    # Moving average crossovers
    data['EMA_Cross_9_21'] = np.where(data['EMA_9'] > data['EMA_21'], 1, 0)
    data['EMA_Cross_21_50'] = np.where(data['EMA_21'] > data['EMA_50'], 1, 0)

    # Price position relative to EMAs
    data['Price_Above_EMA9'] = np.where(data['Close'] > data['EMA_9'], 1, 0)
    data['Price_Above_EMA21'] = np.where(data['Close'] > data['EMA_21'], 1, 0)
    data['Price_Above_EMA50'] = np.where(data['Close'] > data['EMA_50'], 1, 0)

    # Advanced technical indicators
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(20).mean()
    data['BB_Std'] = data['Close'].rolling(20).std()
    data['BB_Upper'] = data['BB_Middle'] + (2 * data['BB_Std'])
    data['BB_Lower'] = data['BB_Middle'] - (2 * data['BB_Std'])
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

    # MACD
    data['EMA_12'] = calculate_ema(data['Close'], 12)
    data['EMA_26'] = calculate_ema(data['Close'], 26)
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = calculate_ema(data['MACD'], 9)
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

    # Stochastic Oscillator
    data['Lowest_Low'] = data['Low'].rolling(14).min()
    data['Highest_High'] = data['High'].rolling(14).max()
    data['Stoch_K'] = 100 * (data['Close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low'])
    data['Stoch_D'] = data['Stoch_K'].rolling(3).mean()

    # Volume indicators (if volume data available)
    if 'Volume' in data.columns:
        data['Volume_SMA'] = data['Volume'].rolling(20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']

    # Time-based features
    data['Hour'] = data.index.hour
    data['DayOfWeek'] = data.index.dayofweek
    data['IsLondonSession'] = np.where((data['Hour'] >= 8) & (data['Hour'] <= 16), 1, 0)
    data['IsNYSession'] = np.where((data['Hour'] >= 13) & (data['Hour'] <= 21), 1, 0)

    # Momentum indicators
    data['ROC_5'] = ((data['Close'] - data['Close'].shift(5)) / data['Close'].shift(5)) * 100
    data['ROC_10'] = ((data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10)) * 100

    # Smart Money Concept (SMC) Features
    data = add_smart_money_features(data)

    return data

def add_smart_money_features(data):
    """Add Smart Money Concept technical indicators"""

    # 1. Market Structure Analysis
    data = calculate_market_structure(data)

    # 2. Order Blocks Detection
    data = detect_order_blocks(data)

    # 3. Fair Value Gaps (FVG)
    data = detect_fair_value_gaps(data)

    # 4. Liquidity Areas
    data = calculate_liquidity_areas(data)

    # 5. Premium/Discount Analysis
    data = calculate_premium_discount(data)

    # 6. Break of Structure (BOS)
    data = detect_break_of_structure(data)

    return data

def calculate_market_structure(data):
    """Calculate market structure - Higher Highs, Lower Lows, etc."""

    # Find swing highs and lows (lookback period = 5)
    lookback = 5

    # Swing Highs
    data['SwingHigh'] = data['High'].rolling(window=lookback*2+1, center=True).max() == data['High']

    # Swing Lows  
    data['SwingLow'] = data['Low'].rolling(window=lookback*2+1, center=True).min() == data['Low']

    # Higher Highs and Lower Lows
    swing_highs = data[data['SwingHigh']]['High']
    swing_lows = data[data['SwingLow']]['Low']

    # Market structure signals
    data['HigherHigh'] = 0
    data['LowerLow'] = 0
    data['HigherLow'] = 0
    data['LowerHigh'] = 0

    for i in range(1, len(swing_highs)):
        if len(swing_highs) > 1:
            current_high = swing_highs.iloc[i]
            previous_high = swing_highs.iloc[i-1]

            idx = swing_highs.index[i]
            if current_high > previous_high:
                data.loc[idx, 'HigherHigh'] = 1
            elif current_high < previous_high:
                data.loc[idx, 'LowerHigh'] = 1

    for i in range(1, len(swing_lows)):
        if len(swing_lows) > 1:
            current_low = swing_lows.iloc[i]
            previous_low = swing_lows.iloc[i-1]

            idx = swing_lows.index[i]
            if current_low > previous_low:
                data.loc[idx, 'HigherLow'] = 1
            elif current_low < previous_low:
                data.loc[idx, 'LowerLow'] = 1

    # Market trend classification
    hh_count = data['HigherHigh'].rolling(20).sum()
    ll_count = data['LowerLow'].rolling(20).sum()
    hl_count = data['HigherLow'].rolling(20).sum()
    lh_count = data['LowerHigh'].rolling(20).sum()

    # Bullish structure: HH + HL
    # Bearish structure: LL + LH
    data['BullishStructure'] = np.where((hh_count + hl_count) > (ll_count + lh_count), 1, 0)
    data['BearishStructure'] = np.where((ll_count + lh_count) > (hh_count + hl_count), 1, 0)

    return data

def detect_order_blocks(data):
    """Detect institutional order blocks (supply/demand zones)"""

    # Order block detection based on strong moves after consolidation
    # Strong move = price movement > 2 * ATR
    atr_period = 14
    data['ATR'] = data['High'].sub(data['Low']).rolling(atr_period).mean()

    # Calculate price movements
    data['PriceMove'] = abs(data['Close'] - data['Open'])
    data['StrongMove'] = data['PriceMove'] > (2 * data['ATR'])

    # Bullish Order Block: Strong upward move after consolidation
    data['BullishOB'] = 0
    data['BearishOB'] = 0

    for i in range(5, len(data)):
        # Look for strong upward moves
        if (data['StrongMove'].iloc[i] and 
            data['Close'].iloc[i] > data['Open'].iloc[i] and
            data['Close'].iloc[i] > data['High'].iloc[i-1]):

            # Mark the last bearish candle before the move as bullish OB
            for j in range(i-1, max(0, i-5), -1):
                if data['Close'].iloc[j] < data['Open'].iloc[j]:
                    data.iloc[j, data.columns.get_loc('BullishOB')] = 1
                    break

        # Look for strong downward moves
        if (data['StrongMove'].iloc[i] and 
            data['Close'].iloc[i] < data['Open'].iloc[i] and
            data['Close'].iloc[i] < data['Low'].iloc[i-1]):

            # Mark the last bullish candle before the move as bearish OB
            for j in range(i-1, max(0, i-5), -1):
                if data['Close'].iloc[j] > data['Open'].iloc[j]:
                    data.iloc[j, data.columns.get_loc('BearishOB')] = 1
                    break

    # Order block proximity (price near order block)
    data['NearBullishOB'] = 0
    data['NearBearishOB'] = 0

    ob_proximity_pips = 10  # 10 pips proximity for XAUUSD

    for i in range(len(data)):
        current_price = data['Close'].iloc[i]

        # Check last 20 periods for order blocks
        start_idx = max(0, i-20)

        # Bullish OB proximity
        bullish_obs = data.iloc[start_idx:i][data.iloc[start_idx:i]['BullishOB'] == 1]
        if len(bullish_obs) > 0:
            for _, ob in bullish_obs.iterrows():
                if abs(current_price - ob['Low']) <= ob_proximity_pips:
                    data.iloc[i, data.columns.get_loc('NearBullishOB')] = 1
                    break

        # Bearish OB proximity
        bearish_obs = data.iloc[start_idx:i][data.iloc[start_idx:i]['BearishOB'] == 1]
        if len(bearish_obs) > 0:
            for _, ob in bearish_obs.iterrows():
                if abs(current_price - ob['High']) <= ob_proximity_pips:
                    data.iloc[i, data.columns.get_loc('NearBearishOB')] = 1
                    break

    return data

def detect_fair_value_gaps(data):
    """Detect Fair Value Gaps (FVG) - price imbalances"""

    data['BullishFVG'] = 0
    data['BearishFVG'] = 0

    for i in range(2, len(data)):
        # Bullish FVG: Low[i] > High[i-2]
        if data['Low'].iloc[i] > data['High'].iloc[i-2]:
            data.iloc[i, data.columns.get_loc('BullishFVG')] = 1

        # Bearish FVG: High[i] < Low[i-2]  
        if data['High'].iloc[i] < data['Low'].iloc[i-2]:
            data.iloc[i, data.columns.get_loc('BearishFVG')] = 1

    return data

def calculate_liquidity_areas(data):
    """Calculate liquidity sweep areas"""

    # Recent highs and lows (potential liquidity areas)
    lookback = 10

    data['RecentHigh'] = data['High'].rolling(lookback).max()
    data['RecentLow'] = data['Low'].rolling(lookback).min()

    # Liquidity sweep detection
    data['LiquiditySweepHigh'] = 0
    data['LiquiditySweepLow'] = 0

    for i in range(lookback, len(data)):
        recent_high = data['RecentHigh'].iloc[i-1]
        recent_low = data['RecentLow'].iloc[i-1]

        # High sweep: price goes above recent high then reverses
        if (data['High'].iloc[i] > recent_high and 
            data['Close'].iloc[i] < data['Open'].iloc[i]):
            data.iloc[i, data.columns.get_loc('LiquiditySweepHigh')] = 1

        # Low sweep: price goes below recent low then reverses
        if (data['Low'].iloc[i] < recent_low and 
            data['Close'].iloc[i] > data['Open'].iloc[i]):
            data.iloc[i, data.columns.get_loc('LiquiditySweepLow')] = 1

    return data

def calculate_premium_discount(data):
    """Calculate if price is in premium or discount relative to range"""

    # Use recent range (50 periods)
    range_period = 50

    data['RangeHigh'] = data['High'].rolling(range_period).max()
    data['RangeLow'] = data['Low'].rolling(range_period).min()
    data['RangeMid'] = (data['RangeHigh'] + data['RangeLow']) / 2

    # Premium/Discount calculation
    data['Premium'] = np.where(data['Close'] > data['RangeMid'], 1, 0)
    data['Discount'] = np.where(data['Close'] < data['RangeMid'], 1, 0)

    # Equilibrium (near mid-point)
    range_size = data['RangeHigh'] - data['RangeLow']
    equilibrium_zone = range_size * 0.1  # 10% of range around midpoint

    data['Equilibrium'] = np.where(
        abs(data['Close'] - data['RangeMid']) <= equilibrium_zone, 1, 0
    )

    return data

def detect_break_of_structure(data):
    """Detect Break of Structure (BOS) and Change of Character (CHoCH)"""

    data['BOS_Bullish'] = 0
    data['BOS_Bearish'] = 0
    data['CHoCH'] = 0

    # Simple BOS detection based on breaking recent significant levels
    lookback = 20

    for i in range(lookback, len(data)):
        recent_data = data.iloc[i-lookback:i]
        significant_high = recent_data['High'].max()
        significant_low = recent_data['Low'].min()

        current_high = data['High'].iloc[i]
        current_low = data['Low'].iloc[i]
        current_close = data['Close'].iloc[i]

        # Bullish BOS: break above significant high with strong close
        if (current_high > significant_high and 
            current_close > significant_high):
            data.iloc[i, data.columns.get_loc('BOS_Bullish')] = 1

        # Bearish BOS: break below significant low with strong close
        if (current_low < significant_low and 
            current_close < significant_low):
            data.iloc[i, data.columns.get_loc('BOS_Bearish')] = 1

        # Change of Character: opposite structure break
        recent_bos_bull = recent_data['BOS_Bullish'].sum()
        recent_bos_bear = recent_data['BOS_Bearish'].sum()

        if (recent_bos_bull > 0 and data['BOS_Bearish'].iloc[i] == 1) or \
           (recent_bos_bear > 0 and data['BOS_Bullish'].iloc[i] == 1):
            data.iloc[i, data.columns.get_loc('CHoCH')] = 1

    return data

def create_target(df, lookahead=5, threshold=0.001):
    """Create target variable for classification"""
    data = df.copy()

    # Calculate future returns
    data['Future_Return'] = data['Close'].shift(-lookahead) / data['Close'] - 1

    # Create target: 1 for buy, 0 for hold, -1 for sell
    data['Target'] = np.where(data['Future_Return'] > threshold, 1,
                             np.where(data['Future_Return'] < -threshold, -1, 0))

    return data

def prepare_data(file_path):
    """Main function to prepare data with features and targets"""
    print(f"Loading data from: {file_path}")

    # Load data
    df = pd.read_csv(file_path)
    print(f"Raw data shape: {df.shape}")
    print(f"Raw data columns: {df.columns.tolist()}")

    # Check required columns
    required_columns = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns.  Dataframe must contain: {required_columns}")

    # Check if we have enough data
    if len(df) < 100:
        print(f"Warning: Only {len(df)} rows of data available. Need at least 100 rows for proper analysis.")
        # Generate more sample data if needed
        df = generate_sample_data(len(df))

    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('DateTime')

    print(f"Data after datetime processing: {df.shape}")

    # Create features
    df = create_features(df)
    print(f"Data after feature creation: {df.shape}")

    # Create target
    df = create_target(df)
    print(f"Data after target creation: {df.shape}")

    # Remove rows with NaN values but keep some data
    initial_rows = len(df)
    df = df.dropna()
    final_rows = len(df)

    print(f"Removed {initial_rows - final_rows} rows with NaN values")
    print(f"Final data shape: {df.shape}")

    if len(df) == 0:
        raise ValueError("No data remaining after processing. Check your data file.")

    return df

def generate_sample_data(current_rows):
    """Generate additional sample data if needed"""
    print("Generating additional sample data...")

    # Base price around 2070-2080 range
    np.random.seed(42)
    n_additional = max(1000 - current_rows, 500)

    dates = pd.date_range('2024-01-01', periods=n_additional, freq='15min')

    # Generate realistic OHLC data
    base_price = 2075.0
    prices = []
    current_price = base_price

    for i in range(n_additional):
        # Random walk with some trend
        change = np.random.normal(0, 0.5)
        current_price += change

        # Generate OHLC
        open_price = current_price
        high_price = open_price + abs(np.random.normal(0, 0.8))
        low_price = open_price - abs(np.random.normal(0, 0.8))
        close_price = open_price + np.random.normal(0, 0.6)

        # Ensure High >= Low and OHLC relationships are maintained
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        prices.append({
            'Date': dates[i].strftime('%Y-%m-%d'),
            'Time': dates[i].strftime('%H:%M'),
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': np.random.randint(1000, 2000)
        })

        current_price = close_price

    return pd.DataFrame(prices)

def select_features(df):
    """Select relevant features for training/prediction"""
    feature_columns = [
        # Original features
        'EMA_9', 'EMA_21', 'EMA_50',
        'RSI_14', 'RSI_21',
        'HL_Ratio', 'OC_Ratio',
        'Return_1', 'Return_5', 'Return_15',
        'Volatility_5', 'Volatility_15',
        'EMA_Cross_9_21', 'EMA_Cross_21_50',
        'Price_Above_EMA9', 'Price_Above_EMA21', 'Price_Above_EMA50',

        # Advanced features
        'BB_Position', 'MACD', 'MACD_Histogram',
        'Stoch_K', 'Stoch_D',
        'Hour', 'DayOfWeek', 'IsLondonSession', 'IsNYSession',
        'ROC_5', 'ROC_10',

        # Smart Money Concept (SMC) features
        'BullishStructure', 'BearishStructure',
        'BullishOB', 'BearishOB', 'NearBullishOB', 'NearBearishOB',
        'BullishFVG', 'BearishFVG',
        'LiquiditySweepHigh', 'LiquiditySweepLow',
        'Premium', 'Discount', 'Equilibrium',
        'BOS_Bullish', 'BOS_Bearish', 'CHoCH'
    ]

    # Only include volume features if available
    if 'Volume_Ratio' in df.columns:
        feature_columns.append('Volume_Ratio')

    # Filter out columns that don't exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]

    return df[available_features]

if __name__ == "__main__":
    # Test the feature engineering
    try:
        data = prepare_data('data/xauusd_m15.csv')
        print("Features created successfully!")
        print(f"Data shape: {data.shape}")
        print(f"Features: {data.columns.tolist()}")
        print("\nTarget distribution:")
        print(data['Target'].value_counts())
    except Exception as e:
        print(f"An error occurred during feature engineering: {e}")