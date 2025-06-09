
import pandas as pd
import numpy as np
import talib

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return talib.EMA(data, timeperiod=period)

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    return talib.RSI(data, timeperiod=period)

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
    # Load data
    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('DateTime')
    
    # Create features
    df = create_features(df)
    
    # Create target
    df = create_target(df)
    
    # Remove rows with NaN values
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    # Test the feature engineering
    data = prepare_data('data/xauusd_m15.csv')
    print("Features created successfully!")
    print(f"Data shape: {data.shape}")
    print(f"Features: {data.columns.tolist()}")
    print("\nTarget distribution:")
    print(data['Target'].value_counts())
