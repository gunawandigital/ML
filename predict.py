
import pandas as pd
import numpy as np
import joblib
import os
from feature_engineering import prepare_data

def select_features(df):
    """Select relevant features for prediction"""
    feature_columns = [
        'EMA_9', 'EMA_21', 'EMA_50',
        'RSI_14', 'RSI_21',
        'HL_Ratio', 'OC_Ratio',
        'Return_1', 'Return_5', 'Return_15',
        'Volatility_5', 'Volatility_15',
        'EMA_Cross_9_21', 'EMA_Cross_21_50',
        'Price_Above_EMA9', 'Price_Above_EMA21', 'Price_Above_EMA50'
    ]
    
    return df[feature_columns]

def load_model(model_path='models/'):
    """Load trained model and scaler"""
    model_file = os.path.join(model_path, 'random_forest_model.pkl')
    scaler_file = os.path.join(model_path, 'scaler.pkl')
    
    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        raise FileNotFoundError("Model files not found. Please train the model first.")
    
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    
    return model, scaler

def get_latest_signal(data_path='data/xauusd_m15.csv', model_path='models/'):
    """Get the latest trading signal"""
    
    # Load model and scaler
    model, scaler = load_model(model_path)
    
    # Prepare data
    df = prepare_data(data_path)
    
    # Get the latest row (most recent data)
    latest_data = df.iloc[-1:].copy()
    
    # Select features
    features = select_features(latest_data)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    prediction_proba = model.predict_proba(features_scaled)[0]
    
    # Get current price info
    current_price = latest_data['Close'].iloc[0]
    current_time = latest_data.index[0]
    
    # Interpret signal
    signal_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
    signal = signal_map[prediction]
    
    # Get confidence (probability of predicted class)
    confidence = max(prediction_proba)
    
    result = {
        'timestamp': current_time,
        'current_price': current_price,
        'signal': signal,
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': {
            'sell_prob': prediction_proba[0] if len(prediction_proba) > 2 else 0,
            'hold_prob': prediction_proba[1] if len(prediction_proba) > 2 else prediction_proba[0],
            'buy_prob': prediction_proba[2] if len(prediction_proba) > 2 else prediction_proba[1]
        }
    }
    
    return result

def predict_batch(data_path='data/xauusd_m15.csv', model_path='models/', n_latest=10):
    """Get predictions for the latest n data points"""
    
    # Load model and scaler
    model, scaler = load_model(model_path)
    
    # Prepare data
    df = prepare_data(data_path)
    
    # Get the latest n rows
    latest_data = df.iloc[-n_latest:].copy()
    
    # Select features
    features = select_features(latest_data)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make predictions
    predictions = model.predict(features_scaled)
    predictions_proba = model.predict_proba(features_scaled)
    
    # Create results dataframe
    results = latest_data[['Close']].copy()
    results['Prediction'] = predictions
    results['Signal'] = [{'−1': 'SELL', 0: 'HOLD', 1: 'BUY'}[p] for p in predictions]
    results['Confidence'] = [max(proba) for proba in predictions_proba]
    
    return results

if __name__ == "__main__":
    try:
        # Get latest signal
        signal = get_latest_signal()
        
        print("=== LATEST TRADING SIGNAL ===")
        print(f"Timestamp: {signal['timestamp']}")
        print(f"Current Price: {signal['current_price']:.2f}")
        print(f"Signal: {signal['signal']}")
        print(f"Confidence: {signal['confidence']:.4f}")
        print("\nProbabilities:")
        print(f"  SELL: {signal['probabilities']['sell_prob']:.4f}")
        print(f"  HOLD: {signal['probabilities']['hold_prob']:.4f}")
        print(f"  BUY:  {signal['probabilities']['buy_prob']:.4f}")
        
        print("\n=== LATEST 5 PREDICTIONS ===")
        batch_predictions = predict_batch(n_latest=5)
        print(batch_predictions)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train.py first to create the model.")
