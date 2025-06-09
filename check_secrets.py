
#!/usr/bin/env python3
"""
Check MetaAPI secrets configuration
"""

import os
from trading_config import TradingConfig

def check_secrets():
    """Check if MetaAPI secrets are properly configured"""
    
    print("🔍 Checking MetaAPI Secrets Configuration")
    print("=" * 50)
    
    # Load configuration
    config = TradingConfig()
    
    # Check environment variables directly
    print("\n📋 Environment Variables:")
    meta_api_token = os.getenv("META_API_TOKEN")
    account_id = os.getenv("ACCOUNT_ID")
    
    print(f"META_API_TOKEN: {'✅ Set' if meta_api_token and meta_api_token != 'YOUR_METAAPI_TOKEN_HERE' else '❌ Not set'}")
    if meta_api_token and meta_api_token != 'YOUR_METAAPI_TOKEN_HERE':
        print(f"   Length: {len(meta_api_token)} characters")
        print(f"   Preview: {meta_api_token[:10]}...{meta_api_token[-4:]}")
    
    print(f"ACCOUNT_ID: {'✅ Set' if account_id and account_id != 'YOUR_ACCOUNT_ID_HERE' else '❌ Not set'}")
    if account_id and account_id != 'YOUR_ACCOUNT_ID_HERE':
        print(f"   Value: {account_id}")
    
    # Check configuration validation
    print("\n🔧 Configuration Validation:")
    errors = config.validate_config()
    
    if not errors:
        print("✅ All configuration is valid!")
        print("\n📊 Current Settings:")
        print(f"   Symbol: {config.SYMBOL}")
        print(f"   Lot Size: {config.LOT_SIZE}")
        print(f"   Stop Loss: {config.STOP_LOSS_PIPS} pips")
        print(f"   Take Profit: {config.TAKE_PROFIT_PIPS} pips")
        print(f"   Confidence Threshold: {config.CONFIDENCE_THRESHOLD:.0%}")
        print(f"   Max Risk per Trade: {config.MAX_RISK_PER_TRADE:.1%}")
        
        print("\n🎯 Ready for MetaAPI connection!")
        return True
    else:
        print("❌ Configuration errors found:")
        for error in errors:
            print(f"   - {error}")
        
        print("\n💡 How to fix:")
        print("1. Go to Tools → Secrets in Replit sidebar")
        print("2. Add these secrets:")
        print("   - Key: META_API_TOKEN, Value: Your MetaAPI token")
        print("   - Key: ACCOUNT_ID, Value: Your MetaTrader account ID")
        print("3. Get your tokens from https://metaapi.cloud")
        
        return False

if __name__ == "__main__":
    success = check_secrets()
    
    if success:
        print("\n🚀 You can now run MetaAPI setup or live trading!")
        print("   Use option 8 in main menu for MetaAPI setup")
    else:
        print("\n⚠️ Please fix the configuration issues above")
