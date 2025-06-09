
#!/usr/bin/env python3
"""
Script untuk mengecek status setup MetaAPI dan konfigurasi
"""

import os
from trading_config import TradingConfig

def check_setup_status():
    """Comprehensive check of MetaAPI setup"""
    
    print("🔍 Checking MetaAPI Setup Status")
    print("=" * 60)
    
    # Load configuration
    config = TradingConfig()
    
    # Check 1: Environment Variables
    print("\n📋 1. Environment Variables:")
    meta_api_token = os.getenv("META_API_TOKEN")
    account_id = os.getenv("ACCOUNT_ID")
    
    token_status = "✅ Set" if meta_api_token and meta_api_token != "YOUR_METAAPI_TOKEN_HERE" else "❌ Not set"
    account_status = "✅ Set" if account_id and account_id != "YOUR_ACCOUNT_ID_HERE" else "❌ Not set"
    
    print(f"   META_API_TOKEN: {token_status}")
    if meta_api_token and meta_api_token != "YOUR_METAAPI_TOKEN_HERE":
        print(f"      Length: {len(meta_api_token)} characters")
        print(f"      Preview: {meta_api_token[:8]}...{meta_api_token[-4:]}")
    
    print(f"   ACCOUNT_ID: {account_status}")
    if account_id and account_id != "YOUR_ACCOUNT_ID_HERE":
        print(f"      Value: {account_id}")
    
    # Check 2: Configuration Validation
    print("\n🔧 2. Configuration Validation:")
    errors = config.validate_config()
    
    if not errors:
        print("   ✅ All configuration is valid!")
    else:
        print("   ❌ Configuration errors found:")
        for error in errors:
            print(f"      - {error}")
    
    # Check 3: Dependencies
    print("\n📦 3. Dependencies Check:")
    
    try:
        from metaapi_cloud_sdk import MetaApi
        print("   ✅ MetaAPI SDK: Installed")
    except ImportError:
        print("   ❌ MetaAPI SDK: Not installed")
        print("      Run: pip install metaapi-cloud-sdk")
    
    try:
        import pandas as pd
        print("   ✅ Pandas: Installed")
    except ImportError:
        print("   ❌ Pandas: Not installed")
    
    try:
        import numpy as np
        print("   ✅ NumPy: Installed")
    except ImportError:
        print("   ❌ NumPy: Not installed")
    
    try:
        import joblib
        print("   ✅ Joblib: Installed")
    except ImportError:
        print("   ❌ Joblib: Not installed")
    
    # Check 4: Model Files
    print("\n🤖 4. Model Files:")
    model_files = [
        "models/random_forest_model.pkl",
        "models/scaler.pkl"
    ]
    
    for file_path in model_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"   ✅ {file_path}: {size:.1f} KB")
        else:
            print(f"   ❌ {file_path}: Missing")
    
    # Check 5: Data Files
    print("\n📊 5. Data Files:")
    data_files = [
        "data/xauusd_m15_real.csv",
        "data/xauusd_m15_combined.csv",
        "data/xauusd_m15.csv"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            import pandas as pd
            try:
                df = pd.read_csv(file_path)
                print(f"   ✅ {file_path}: {len(df)} rows")
            except:
                print(f"   ⚠️ {file_path}: Error reading")
        else:
            print(f"   ❌ {file_path}: Missing")
    
    # Check 6: Trading Configuration
    print("\n⚙️ 6. Trading Configuration:")
    print(f"   Symbol: {config.SYMBOL}")
    print(f"   Lot Size: {config.LOT_SIZE}")
    print(f"   Stop Loss: {config.STOP_LOSS_PIPS} pips")
    print(f"   Take Profit: {config.TAKE_PROFIT_PIPS} pips")
    print(f"   Confidence Threshold: {config.CONFIDENCE_THRESHOLD:.0%}")
    print(f"   Max Risk per Trade: {config.MAX_RISK_PER_TRADE:.1%}")
    
    # Final Status
    print("\n" + "=" * 60)
    
    all_good = (
        not errors and 
        meta_api_token and meta_api_token != "YOUR_METAAPI_TOKEN_HERE" and
        account_id and account_id != "YOUR_ACCOUNT_ID_HERE" and
        os.path.exists("models/random_forest_model.pkl") and
        os.path.exists("models/scaler.pkl")
    )
    
    if all_good:
        print("🎉 SETUP STATUS: ✅ READY FOR LIVE TRADING!")
        print("\nNext steps:")
        print("1. Use option 8 in main menu for MetaAPI connection test")
        print("2. Start with demo account for safety")
        print("3. Monitor performance before using real money")
    else:
        print("⚠️ SETUP STATUS: ❌ NEEDS ATTENTION")
        print("\nTo fix:")
        if errors:
            print("1. Fix configuration errors shown above")
        if not (meta_api_token and meta_api_token != "YOUR_METAAPI_TOKEN_HERE"):
            print("2. Add META_API_TOKEN to Replit Secrets")
        if not (account_id and account_id != "YOUR_ACCOUNT_ID_HERE"):
            print("3. Add ACCOUNT_ID to Replit Secrets")
        if not os.path.exists("models/random_forest_model.pkl"):
            print("4. Train model first (option 4 in main menu)")
    
    return all_good

if __name__ == "__main__":
    check_setup_status()
