
#!/usr/bin/env python3
"""
Comprehensive System Health Checker
Checks all components of the trading system
"""

import os
import sys
import pandas as pd
import traceback
from datetime import datetime, timedelta
import logging
import psutil

def check_file_status():
    """Check critical files"""
    print("📁 File System Check:")
    
    critical_files = {
        'Model Files': [
            'models/random_forest_model.pkl',
            'models/scaler.pkl',
            'models/feature_importance.csv'
        ],
        'Data Files': [
            'data/xauusd_m15_real.csv'
        ],
        'Config Files': [
            'trading_config.py',
            'web_dashboard.py'
        ],
        'Templates': [
            'templates/dashboard.html'
        ]
    }
    
    all_good = True
    for category, files in critical_files.items():
        print(f"\n   {category}:")
        for file_path in files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                print(f"   ✅ {file_path} ({size:,} bytes, modified: {mtime.strftime('%Y-%m-%d %H:%M')})")
            else:
                print(f"   ❌ {file_path} - MISSING")
                all_good = False
    
    return all_good

def check_data_quality():
    """Check data file quality"""
    print("\n📊 Data Quality Check:")
    
    data_file = 'data/xauusd_m15_real.csv'
    if not os.path.exists(data_file):
        print("   ❌ No real data file found")
        return False
    
    try:
        df = pd.read_csv(data_file)
        print(f"   ✅ Data loaded: {len(df):,} rows")
        print(f"   ✅ Columns: {list(df.columns)}")
        
        if len(df) > 0:
            latest_row = df.iloc[-1]
            print(f"   ✅ Latest price: ${latest_row.get('Close', 'N/A')}")
            print(f"   ✅ Date range: {df.iloc[0].get('Date', 'N/A')} to {df.iloc[-1].get('Date', 'N/A')}")
            
            # Check for missing values
            missing = df.isnull().sum().sum()
            if missing == 0:
                print("   ✅ No missing values")
            else:
                print(f"   ⚠️  {missing} missing values found")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data error: {e}")
        return False

def check_model_health():
    """Check ML model status"""
    print("\n🤖 Model Health Check:")
    
    try:
        from predict import load_model, get_latest_signal
        
        # Test model loading
        model, scaler = load_model()
        print("   ✅ Model loaded successfully")
        
        # Test prediction
        signal = get_latest_signal('data/xauusd_m15_real.csv')
        print(f"   ✅ Latest signal: {signal.get('signal', 'N/A')}")
        print(f"   ✅ Confidence: {signal.get('confidence', 0):.1%}")
        print(f"   ✅ Current price: ${signal.get('current_price', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model error: {e}")
        return False

def check_configuration():
    """Check trading configuration"""
    print("\n⚙️ Configuration Check:")
    
    try:
        from trading_config import TradingConfig
        
        config = TradingConfig()
        errors = config.validate_config()
        
        print(f"   Symbol: {config.SYMBOL}")
        print(f"   Lot Size: {config.LOT_SIZE}")
        print(f"   Confidence Threshold: {config.CONFIDENCE_THRESHOLD:.0%}")
        print(f"   Stop Loss: {config.STOP_LOSS_PIPS} pips")
        print(f"   Take Profit: {config.TAKE_PROFIT_PIPS} pips")
        
        # Check secrets
        meta_token = os.getenv("META_API_TOKEN", "")
        account_id = os.getenv("ACCOUNT_ID", "")
        
        if meta_token and meta_token != "YOUR_METAAPI_TOKEN_HERE":
            print(f"   ✅ MetaAPI Token: Set ({len(meta_token)} chars)")
        else:
            print("   ❌ MetaAPI Token: Not set")
            
        if account_id and account_id != "YOUR_ACCOUNT_ID_HERE":
            print(f"   ✅ Account ID: {account_id}")
        else:
            print("   ❌ Account ID: Not set")
        
        if not errors:
            print("   ✅ Configuration valid")
            return True
        else:
            print("   ❌ Configuration errors:")
            for error in errors:
                print(f"      - {error}")
            return False
            
    except Exception as e:
        print(f"   ❌ Config error: {e}")
        return False

def check_processes():
    """Check running processes"""
    print("\n🔄 Process Check:")
    
    try:
        current_process = psutil.Process()
        print(f"   ✅ Current process: {current_process.name()} (PID: {current_process.pid})")
        print(f"   ✅ Memory usage: {current_process.memory_info().rss / 1024 / 1024:.1f} MB")
        print(f"   ✅ CPU usage: {current_process.cpu_percent():.1f}%")
        
        # Check for trading processes
        trading_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if any(keyword in cmdline for keyword in ['web_dashboard.py', 'run_live_trading.py', 'main.py']):
                    trading_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': cmdline
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if trading_processes:
            print("   ✅ Trading processes found:")
            for proc in trading_processes:
                print(f"      - PID {proc['pid']}: {proc['name']}")
        else:
            print("   ⚠️  No specific trading processes detected")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Process check error: {e}")
        return False

def check_network_connectivity():
    """Check network and MetaAPI connectivity"""
    print("\n🌐 Network & API Check:")
    
    try:
        import socket
        import asyncio
        from metaapi_trader import MetaAPITrader
        from trading_config import TradingConfig
        
        # Basic network check
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            print("   ✅ Internet connection: OK")
        except OSError:
            print("   ❌ Internet connection: FAILED")
            return False
        
        # MetaAPI check
        config = TradingConfig()
        if (config.META_API_TOKEN != "YOUR_METAAPI_TOKEN_HERE" and 
            config.ACCOUNT_ID != "YOUR_ACCOUNT_ID_HERE"):
            
            async def test_metaapi():
                trader = MetaAPITrader(
                    token=config.META_API_TOKEN,
                    account_id=config.ACCOUNT_ID,
                    symbol="XAUUSD"
                )
                try:
                    success = await asyncio.wait_for(trader.initialize(), timeout=20)
                    if success:
                        balance = await trader.get_account_balance()
                        await trader.cleanup()
                        return True, balance
                    else:
                        await trader.cleanup()
                        return False, 0
                except Exception as e:
                    await trader.cleanup()
                    return False, str(e)
            
            try:
                loop = asyncio.new_event_loop() if not asyncio.get_event_loop().is_running() else asyncio.get_event_loop()
                success, result = loop.run_until_complete(test_metaapi())
                if success:
                    print(f"   ✅ MetaAPI connection: OK (Balance: ${result:.2f})")
                else:
                    print(f"   ⚠️  MetaAPI connection: {result}")
            except Exception as e:
                print(f"   ⚠️  MetaAPI test error: {e}")
        else:
            print("   ⚠️  MetaAPI credentials not configured")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Network check error: {e}")
        return False

def check_dashboard_status():
    """Check web dashboard status"""
    print("\n🌐 Dashboard Status Check:")
    
    try:
        import requests
        from urllib.parse import urljoin
        
        # Try to connect to dashboard
        try:
            response = requests.get("http://0.0.0.0:5000/api/health", timeout=5)
            if response.status_code == 200:
                print("   ✅ Dashboard: Running and accessible")
                data = response.json()
                if data.get('success'):
                    print("   ✅ Dashboard API: Healthy")
                else:
                    print("   ⚠️  Dashboard API: Issues detected")
            else:
                print(f"   ⚠️  Dashboard: HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("   ❌ Dashboard: Not running or not accessible")
        except Exception as e:
            print(f"   ⚠️  Dashboard check error: {e}")
        
        return True
        
    except ImportError:
        print("   ⚠️  requests library not available for dashboard check")
        return True
    except Exception as e:
        print(f"   ❌ Dashboard check error: {e}")
        return False

def generate_system_report():
    """Generate comprehensive system report"""
    print("="*70)
    print("🔍 COMPREHENSIVE SYSTEM HEALTH CHECK")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    checks = [
        ("File System", check_file_status),
        ("Data Quality", check_data_quality),
        ("Model Health", check_model_health),
        ("Configuration", check_configuration),
        ("Processes", check_processes),
        ("Network & API", check_network_connectivity),
        ("Dashboard", check_dashboard_status)
    ]
    
    results = {}
    overall_health = True
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
            if not result:
                overall_health = False
        except Exception as e:
            print(f"\n❌ {check_name} check failed: {e}")
            results[check_name] = False
            overall_health = False
    
    # Summary
    print("\n" + "="*70)
    print("📋 SYSTEM HEALTH SUMMARY")
    print("="*70)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {check_name}: {status}")
    
    print("\n" + "="*70)
    if overall_health:
        print("🎉 OVERALL STATUS: ✅ SYSTEM HEALTHY")
        print("\nSystem is ready for:")
        print("   • Real-time trading signal generation")
        print("   • Web dashboard access")
        print("   • Live trading (if credentials configured)")
    else:
        print("⚠️ OVERALL STATUS: ❌ ISSUES DETECTED")
        print("\nRecommendations:")
        print("   • Fix the failed checks above")
        print("   • Check log files for detailed errors")
        print("   • Restart services if needed")
    
    print("="*70)
    
    return overall_health, results

if __name__ == "__main__":
    try:
        overall_health, results = generate_system_report()
        sys.exit(0 if overall_health else 1)
    except KeyboardInterrupt:
        print("\n🛑 Health check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Health check failed: {e}")
        traceback.print_exc()
        sys.exit(1)
