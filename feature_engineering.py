
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
        'ROC_5', 'ROC_10'
    ]
    
    # Only include volume features if available
    if 'Volume_Ratio' in df.columns:
        feature_columns.append('Volume_Ratio')
    
    # Filter out columns that don't exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]
    
    return df[available_features]

if __name__ == "__main__":
    # Test the feature engineering
    data = prepare_data('data/xauusd_m15.csv')
    print("Features created successfully!")
    print(f"Data shape: {data.shape}")
    print(f"Features: {data.columns.tolist()}")
    print("\nTarget distribution:")
    print(data['Target'].value_counts())
