
#!/usr/bin/env python3
"""
Ultra-Fast Deployment for Forex Trading Dashboard
Optimized for Replit deployments with minimal timeouts
"""

import os
import sys

def main():
    """Ultra-fast deployment"""
    print("🚀 ULTRA-FAST DEPLOYMENT STARTING...")
    
    # Set environment for fastest startup
    os.environ['DEPLOYMENT_MODE'] = 'true'
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['FLASK_ENV'] = 'production'
    
    # Import and run dashboard directly (no subprocess overhead)
    try:
        from web_dashboard import app
        print("📊 Starting dashboard on port 5000...")
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False,
            load_dotenv=False
        )
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        # Fallback to basic Flask server
        from flask import Flask
        fallback_app = Flask(__name__)
        
        @fallback_app.route('/')
        def fallback():
            return """
            <h1>🚀 Forex Trading System</h1>
            <p>Dashboard is starting up...</p>
            <p>Please refresh in a few seconds.</p>
            <script>setTimeout(() => location.reload(), 5000);</script>
            """
        
        fallback_app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()
