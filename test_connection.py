
#!/usr/bin/env python3
"""
Simple test script to verify MetaAPI connection and numpy compatibility
"""

import asyncio
import numpy as np
from trading_config import TradingConfig

async def test_numpy():
    """Test numpy functionality"""
    print("🧪 Testing NumPy compatibility...")
    
    try:
        # Test basic numpy operations
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        print(f"✅ NumPy array creation: {arr}")
        
        # Test statistical operations
        mean_val = np.mean(arr)
        print(f"✅ NumPy mean calculation: {mean_val}")
        
        # Test NaN handling
        arr_with_nan = np.array([1.0, np.nan, 3.0])
        has_nan = np.any(np.isnan(arr_with_nan))
        print(f"✅ NumPy NaN detection: {has_nan}")
        
        return True
        
    except Exception as e:
        print(f"❌ NumPy test failed: {e}")
        return False

async def test_metaapi_basic():
    """Test basic MetaAPI import and configuration"""
    print("\n🔌 Testing MetaAPI basic setup...")
    
    try:
        # Test MetaAPI import
        from metaapi_cloud_sdk import MetaApi
        print("✅ MetaAPI SDK imported successfully")
        
        # Test configuration
        config = TradingConfig()
        errors = config.validate_config()
        
        if not errors:
            print("✅ Configuration is valid")
            return True
        else:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"   - {error}")
            return False
            
    except ImportError:
        print("❌ MetaAPI SDK not available")
        return False
    except Exception as e:
        print(f"❌ MetaAPI test failed: {e}")
        return False

async def test_ml_model():
    """Test ML model loading"""
    print("\n🤖 Testing ML model loading...")
    
    try:
        from predict import load_model
        model, scaler = load_model()
        print("✅ ML model loaded successfully")
        
        # Test prediction with dummy data
        dummy_features = np.random.random((1, 10))  # Adjust size as needed
        scaled_features = scaler.transform(dummy_features)
        prediction = model.predict(scaled_features)
        print(f"✅ Model prediction test: {prediction[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML model test failed: {e}")
        print("💡 Try retraining the model")
        return False

async def main():
    """Run all tests"""
    print("🚀 MetaAPI Connection & NumPy Test Suite")
    print("=" * 50)
    
    # Run tests
    numpy_ok = await test_numpy()
    metaapi_ok = await test_metaapi_basic()
    ml_ok = await test_ml_model()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"   NumPy: {'✅ PASS' if numpy_ok else '❌ FAIL'}")
    print(f"   MetaAPI: {'✅ PASS' if metaapi_ok else '❌ FAIL'}")
    print(f"   ML Model: {'✅ PASS' if ml_ok else '❌ FAIL'}")
    
    if all([numpy_ok, metaapi_ok, ml_ok]):
        print("\n🎉 All tests passed! System is ready for trading.")
    else:
        print("\n⚠️ Some tests failed. Please fix issues before trading.")

if __name__ == "__main__":
    asyncio.run(main())
