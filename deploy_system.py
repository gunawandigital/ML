
#!/usr/bin/env python3
"""
Fast Deployment Script for Complete Forex Trading System
Deploys entire trading system with optimized startup
"""

import os
import sys
import subprocess
import asyncio
import time
from datetime import datetime

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  🚀 {title}")
    print("="*60)

def check_system_requirements():
    """Quick system requirements check"""
    print("🔍 Checking system requirements...")
    
    # Check data files
    if not os.path.exists('data/xauusd_m15_real.csv'):
        print("❌ Real-time data missing")
        return False
    
    # Check models
    if not (os.path.exists('models/random_forest_model.pkl') and 
            os.path.exists('models/scaler.pkl')):
        print("❌ ML models missing")
        return False
    
    print("✅ System requirements OK")
    return True

def optimize_for_deployment():
    """Optimize system for fast deployment"""
    print("⚡ Optimizing for deployment...")
    
    # Set environment variables for production
    os.environ['AUTO_TRADING'] = 'true'
    os.environ['DEPLOYMENT_MODE'] = 'true'
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # Minimize logging for performance
    os.environ['LOG_LEVEL'] = 'WARNING'
    
    print("✅ Optimization complete")

async def start_trading_services():
    """Start all trading services in optimal order"""
    print("🎯 Starting trading services...")
    
    services = []
    
    try:
        # Set environment variables for faster startup
        os.environ['DEPLOYMENT_MODE'] = 'true'
        os.environ['PYTHONUNBUFFERED'] = '1'
        
        # 1. Start web dashboard (ultra-lightweight startup)
        print("📊 Starting web dashboard...")
        dashboard_process = subprocess.Popen([
            sys.executable, "web_dashboard.py"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # Suppress output for faster startup
        services.append(('Dashboard', dashboard_process))
        
        # Shorter wait for faster deployment
        await asyncio.sleep(1)
        
        # 2. Start live trading (if enabled) - non-blocking
        auto_trading = os.getenv("AUTO_TRADING", "false").lower() == "true"
        if auto_trading:
            print("🤖 Starting automated trading (background)...")
            trading_process = subprocess.Popen([
                sys.executable, "run_live_trading.py"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            services.append(('Live Trading', trading_process))
        
        print("✅ All services started successfully")
        return services
        
    except Exception as e:
        print(f"❌ Error starting services: {e}")
        return []

def get_deployment_status():
    """Get current deployment status"""
    status = {
        'dashboard_running': False,
        'trading_running': False,
        'data_available': False,
        'models_loaded': False
    }
    
    # Check dashboard
    try:
        import requests
        response = requests.get('http://0.0.0.0:5000/api/health', timeout=2)
        if response.status_code == 200:
            status['dashboard_running'] = True
    except:
        pass
    
    # Check data
    status['data_available'] = os.path.exists('data/xauusd_m15_real.csv')
    
    # Check models
    status['models_loaded'] = (os.path.exists('models/random_forest_model.pkl') and 
                              os.path.exists('models/scaler.pkl'))
    
    # Check trading (via process check)
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'run_live_trading.py' in cmdline and proc.is_running():
                    status['trading_running'] = True
                    break
            except:
                continue
    except ImportError:
        pass
    
    return status

async def main():
    """Main deployment function"""
    print_header("FOREX TRADING SYSTEM - FAST DEPLOYMENT")
    print(f"Deployment started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: System check
    if not check_system_requirements():
        print("\n❌ System requirements not met!")
        print("💡 Run main.py first to set up the system")
        return
    
    # Step 2: Optimize for deployment
    optimize_for_deployment()
    
    # Step 3: Start services
    print_header("STARTING SERVICES")
    services = await start_trading_services()
    
    if not services:
        print("❌ Failed to start services")
        return
    
    # Step 4: Deployment summary
    print_header("DEPLOYMENT SUMMARY")
    status = get_deployment_status()
    
    print("📊 Service Status:")
    print(f"   Dashboard: {'✅ Running' if status['dashboard_running'] else '❌ Not Running'}")
    print(f"   Live Trading: {'✅ Running' if status['trading_running'] else '❌ Not Running'}")
    print(f"   Data Available: {'✅ Yes' if status['data_available'] else '❌ No'}")
    print(f"   Models Loaded: {'✅ Yes' if status['models_loaded'] else '❌ No'}")
    
    # Access URLs
    print("\n🌐 Access URLs:")
    print("   📊 Trading Dashboard: https://{repl-url}")
    print("   📈 API Health Check: https://{repl-url}/api/health")
    print("   🎯 Latest Signal: https://{repl-url}/api/signal")
    
    # Auto-trading status
    auto_trading = os.getenv("AUTO_TRADING", "false").lower() == "true"
    print(f"\n🤖 Auto-Trading: {'✅ Enabled' if auto_trading else '⚠️ Disabled'}")
    
    if auto_trading:
        print("   💰 Live trades will be executed automatically")
        print("   🛑 Use dashboard to stop trading if needed")
    else:
        print("   📊 Manual trading mode - signals only")
    
    print(f"\n🎉 Deployment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🚀 Your trading system is now live!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Deployment interrupted")
    except Exception as e:
        print(f"❌ Deployment error: {e}")
