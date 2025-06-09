
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import json
import time

# MetaAPI imports (install with: pip install metaapi-cloud-sdk)
try:
    from metaapi_cloud_sdk import MetaApi
except ImportError:
    print("⚠️ MetaAPI SDK not installed. Install with: pip install metaapi-cloud-sdk")

from feature_engineering import prepare_data, select_features
from predict import load_model

class MetaAPITrader:
    """
    Automated trading system using MetaAPI for real-time trading
    """
    
    def __init__(self, 
                 token: str,
                 account_id: str,
                 symbol: str = "XAUUSD",
                 lot_size: float = 0.01,
                 max_risk_per_trade: float = 0.02,
                 stop_loss_pips: int = 50,
                 take_profit_pips: int = 100,
                 confidence_threshold: float = 0.7):
        
        self.token = token
        self.account_id = account_id
        self.symbol = symbol
        self.lot_size = lot_size
        self.max_risk_per_trade = max_risk_per_trade
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.confidence_threshold = confidence_threshold
        
        # Trading state
        self.is_trading = False
        self.current_position = None
        self.last_signal_time = None
        self.trades_log = []
        
        # MetaAPI objects
        self.api = None
        self.account = None
        self.connection = None
        
        # ML Model
        self.model = None
        self.scaler = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize MetaAPI connection and load ML model"""
        try:
            # Initialize MetaAPI
            self.api = MetaApi(self.token)
            self.account = await self.api.metatrader_account_api.get_account(self.account_id)
            
            # Wait for account deployment with timeout
            await asyncio.wait_for(self.account.deploy(), timeout=30)
            await asyncio.wait_for(self.account.wait_connected(), timeout=60)
            
            # Create streaming connection for trading and data
            self.connection = self.account.get_streaming_connection()
            await asyncio.wait_for(self.connection.connect(), timeout=30)
            await asyncio.wait_for(self.connection.wait_synchronized(), timeout=120)
            
            # Load ML model
            self.model, self.scaler = load_model()
            
            self.logger.info("✅ MetaAPI RPC connection established successfully!")
            
            # Get account balance
            balance = await self.get_account_balance()
            self.logger.info(f"✅ Account balance: ${balance:.2f}")
            
            self.logger.info("✅ ML model loaded successfully!")
            
            return True
            
        except asyncio.TimeoutError:
            self.logger.error("❌ Initialization failed: Connection timeout")
            return False
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            return False
    
    async def get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            # Use the connection to get account information
            account_info = await self.connection.get_account_information()
            return account_info.get('balance', 0.0)
        except Exception as e:
            self.logger.warning(f"Could not retrieve balance: {e}")
            return 5000.0  # Return demo balance as fallback
    
    async def get_real_time_data(self, timeframe: str = "15m", count: int = 100) -> pd.DataFrame:
        """Get real-time OHLC data from MetaTrader"""
        try:
            # Get historical data using streaming connection
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=count * 2)
            
            # Use account method for getting recent candles with proper parameters
            candles = await self.account.get_historical_candles(
                self.symbol,
                timeframe,
                start_time,
                count
            )
            
            # Take only the last 'count' candles
            if len(candles) > count:
                candles = candles[-count:]
            
            # Convert to DataFrame
            data = []
            for candle in candles:
                # MetaAPI History API candle format
                if isinstance(candle['time'], str):
                    time_obj = datetime.fromisoformat(candle['time'].replace('Z', '+00:00'))
                else:
                    time_obj = candle['time']
                
                data.append({
                    'Date': time_obj.strftime('%Y-%m-%d'),
                    'Time': time_obj.strftime('%H:%M:%S'),
                    'Open': candle['open'],
                    'High': candle['high'],
                    'Low': candle['low'],
                    'Close': candle['close'],
                    'Volume': candle.get('tickVolume', 0)
                })
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting real-time data: {e}")
            return pd.DataFrame()
    
    async def generate_trading_signal(self) -> Dict:
        """Generate trading signal using ML model"""
        try:
            # Get real-time data
            df = await self.get_real_time_data()
            
            if df.empty:
                # Fallback to sample data if real data fails
                self.logger.warning("No real-time data available, using fallback data")
                return {
                    'signal': 'HOLD', 
                    'confidence': 0.0, 
                    'current_price': 2650.0,  # Default XAUUSD price
                    'timestamp': datetime.now(),
                    'error': 'No real-time data available'
                }
            
            # Prepare features
            try:
                # Save DataFrame to temporary file for processing
                temp_file = 'temp_realtime_data.csv'
                df.to_csv(temp_file, index=False)
                processed_data = prepare_data(temp_file)
                # Clean up temp file
                import os
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as feat_error:
                self.logger.error(f"Feature processing error: {feat_error}")
                current_price = df['Close'].iloc[-1] if len(df) > 0 and 'Close' in df.columns else 2650.0
                return {
                    'signal': 'HOLD', 
                    'confidence': 0.0, 
                    'current_price': current_price,
                    'timestamp': datetime.now(),
                    'error': f'Feature processing failed: {feat_error}'
                }
            
            if processed_data.empty:
                current_price = df['Close'].iloc[-1] if len(df) > 0 and 'Close' in df.columns else 2650.0
                return {
                    'signal': 'HOLD', 
                    'confidence': 0.0, 
                    'current_price': current_price,
                    'timestamp': datetime.now(),
                    'error': 'Feature processing returned empty data'
                }
            
            latest_data = processed_data.iloc[-1:].copy()
            features = select_features(latest_data)
            
            # Scale features and predict
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]
            prediction_proba = self.model.predict_proba(features_scaled)[0]
            
            # Get current price
            current_price = latest_data['Close'].iloc[0]
            current_time = datetime.now()
            
            signal_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
            signal = signal_map[prediction]
            confidence = max(prediction_proba)
            
            return {
                'signal': signal,
                'prediction': prediction,
                'confidence': confidence,
                'current_price': current_price,
                'timestamp': current_time,
                'probabilities': {
                    'sell': prediction_proba[0] if len(prediction_proba) > 2 else 0,
                    'hold': prediction_proba[1] if len(prediction_proba) > 2 else prediction_proba[0],
                    'buy': prediction_proba[2] if len(prediction_proba) > 2 else prediction_proba[1]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return {
                'signal': 'HOLD', 
                'confidence': 0.0, 
                'current_price': 2650.0,  # Default fallback price
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    async def calculate_position_size(self, stop_loss_pips: int) -> float:
        """Calculate position size based on risk management"""
        try:
            balance = await self.get_account_balance()
            risk_amount = balance * self.max_risk_per_trade
            
            # Get symbol specification
            symbol_spec = await self.connection.get_symbol_specification(self.symbol)
            pip_value = symbol_spec.get('pipValue', 1.0)
            
            # Calculate position size
            position_size = risk_amount / (stop_loss_pips * pip_value)
            
            # Round to appropriate lot size
            min_lot = symbol_spec.get('minVolume', 0.01)
            max_lot = symbol_spec.get('maxVolume', 100.0)
            
            position_size = max(min_lot, min(position_size, max_lot))
            position_size = round(position_size / min_lot) * min_lot
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return self.lot_size
    
    async def place_order(self, signal: Dict) -> Optional[str]:
        """Place trading order based on signal"""
        try:
            if signal['confidence'] < self.confidence_threshold:
                self.logger.info(f"🔸 Signal confidence {signal['confidence']:.1%} below threshold {self.confidence_threshold:.1%}")
                return None
            
            order_type = signal['signal']
            if order_type == 'HOLD':
                return None
            
            current_price = signal['current_price']
            position_size = await self.calculate_position_size(self.stop_loss_pips)
            
            # Calculate SL and TP
            pip_size = 0.01  # For XAUUSD
            
            if order_type == 'BUY':
                stop_loss = current_price - (self.stop_loss_pips * pip_size)
                take_profit = current_price + (self.take_profit_pips * pip_size)
                action = 'ORDER_TYPE_BUY'
            else:  # SELL
                stop_loss = current_price + (self.stop_loss_pips * pip_size)
                take_profit = current_price - (self.take_profit_pips * pip_size)
                action = 'ORDER_TYPE_SELL'
            
            # Place order using correct MetaAPI format
            order_request = {
                'actionType': action,
                'symbol': self.symbol,
                'volume': position_size,
                'stopLoss': stop_loss,
                'takeProfit': take_profit,
                'comment': f"ML-Signal-{signal['confidence']:.1%}",
                'clientId': f"ml-trade-{int(datetime.now().timestamp())}"
            }
            
            result = await self.connection.create_market_order(order_request)
            
            # Log trade
            trade_log = {
                'timestamp': signal['timestamp'],
                'signal': order_type,
                'confidence': signal['confidence'],
                'price': current_price,
                'volume': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'order_id': result.get('orderId', 'unknown')
            }
            
            self.trades_log.append(trade_log)
            self.current_position = trade_log
            
            self.logger.info(f"🎯 {order_type} order placed: {position_size} lots at ${current_price:.2f}")
            self.logger.info(f"   SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}")
            self.logger.info(f"   Confidence: {signal['confidence']:.1%}")
            
            return result.get('orderId')
            
        except Exception as e:
            self.logger.error(f"❌ Error placing order: {e}")
            return None
    
    async def check_and_close_positions(self):
        """Check and manage existing positions"""
        try:
            # Use connection to get positions
            positions = []
            if hasattr(self.connection, 'terminal_state') and self.connection.terminal_state:
                terminal_state = self.connection.terminal_state
                positions = terminal_state.positions if hasattr(terminal_state, 'positions') else []
            
            # Alternative: try to get positions directly from connection
            if not positions:
                try:
                    account_info = await self.connection.get_account_information()
                    positions = account_info.get('positions', [])
                except:
                    positions = []
            
            for position in positions:
                if position.get('symbol') == self.symbol:
                    # Generate new signal to check if we should close
                    current_signal = await self.generate_trading_signal()
                    
                    # Close position if signal changed or confidence is low
                    position_type = position.get('type', '').replace('POSITION_TYPE_', '')
                    should_close = (
                        current_signal['signal'] != position_type or
                        current_signal['confidence'] < self.confidence_threshold * 0.8
                    )
                    
                    if should_close:
                        close_request = {
                            'actionType': 'POSITION_CLOSE_ID',
                            'positionId': position.get('id'),
                        }
                        await self.connection.create_market_order(close_request)
                        self.logger.info(f"🔄 Position closed: {position_type} {position.get('volume', 0)} lots")
                        self.current_position = None
            
        except Exception as e:
            self.logger.error(f"Error managing positions: {e}")
    
    async def start_trading(self, check_interval: int = 60):
        """Start automated trading loop"""
        self.is_trading = True
        self.logger.info("🚀 Starting automated trading...")
        
        try:
            while self.is_trading:
                # Generate trading signal
                signal = await self.generate_trading_signal()
                
                if 'error' not in signal:
                    self.logger.info(f"📊 Signal: {signal['signal']} | Confidence: {signal['confidence']:.1%} | Price: ${signal['current_price']:.2f}")
                    
                    # Check existing positions
                    await self.check_and_close_positions()
                    
                    # Place new order if no current position
                    if not self.current_position:
                        order_id = await self.place_order(signal)
                        if order_id:
                            self.last_signal_time = signal['timestamp']
                
                # Wait before next check
                await asyncio.sleep(check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Trading stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Trading loop error: {e}")
        finally:
            self.is_trading = False
    
    def stop_trading(self):
        """Stop automated trading"""
        self.is_trading = False
        self.logger.info("🛑 Stopping automated trading...")
    
    async def cleanup(self):
        """Clean up connections"""
        try:
            if self.connection:
                await self.connection.close()
            if self.account:
                await self.account.undeploy()
            self.logger.info("✅ Connections cleaned up")
        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup warning: {e}")
    
    def get_trading_stats(self) -> Dict:
        """Get trading statistics"""
        if not self.trades_log:
            return {'total_trades': 0, 'message': 'No trades executed yet'}
        
        df = pd.DataFrame(self.trades_log)
        
        stats = {
            'total_trades': len(df),
            'buy_trades': len(df[df['signal'] == 'BUY']),
            'sell_trades': len(df[df['signal'] == 'SELL']),
            'avg_confidence': df['confidence'].mean(),
            'last_trade': df.iloc[-1].to_dict() if len(df) > 0 else None
        }
        
        return stats

# Configuration and usage example
async def main():
    """Main function to run the automated trading system"""
    
    # MetaAPI Configuration (you need to get these from MetaAPI dashboard)
    META_API_TOKEN = "your_metaapi_token_here"
    ACCOUNT_ID = "your_mt_account_id_here"
    
    # Create trader instance
    trader = MetaAPITrader(
        token=META_API_TOKEN,
        account_id=ACCOUNT_ID,
        symbol="XAUUSD",
        lot_size=0.01,
        max_risk_per_trade=0.02,  # 2% risk per trade
        stop_loss_pips=50,
        take_profit_pips=100,
        confidence_threshold=0.7  # 70% minimum confidence
    )
    
    # Initialize connection
    if await trader.initialize():
        # Start automated trading
        await trader.start_trading(check_interval=300)  # Check every 5 minutes
    else:
        print("❌ Failed to initialize trader")

if __name__ == "__main__":
    # Run the trading system
    asyncio.run(main())
