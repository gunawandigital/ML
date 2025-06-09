# Added error handling for the libgomp issue in the main block.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import os
import time
from feature_engineering import prepare_data, select_features

def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost model"""
    print("\n🚀 Training XGBoost...")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }

    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1
    )

    # Time series split for financial data
    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=tscv, 
        scoring='accuracy', n_jobs=-1, verbose=1
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'model': best_model,
        'accuracy': accuracy,
        'training_time': training_time,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_
    }

def train_lightgbm(X_train, X_test, y_train, y_test):
    """Train LightGBM model"""
    print("\n⚡ Training LightGBM...")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 70]
    }

    lgb_model = lgb.LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(
        lgb_model, param_grid, cv=tscv,
        scoring='accuracy', n_jobs=-1, verbose=1
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'model': best_model,
        'accuracy': accuracy,
        'training_time': training_time,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_
    }

def train_catboost(X_train, X_test, y_train, y_test):
    """Train CatBoost model"""
    print("\n🐱 Training CatBoost...")

    param_grid = {
        'iterations': [100, 200, 300],
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    cat_model = CatBoostClassifier(
        random_state=42,
        verbose=False
    )

    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(
        cat_model, param_grid, cv=tscv,
        scoring='accuracy', n_jobs=-1, verbose=1
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'model': best_model,
        'accuracy': accuracy,
        'training_time': training_time,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_
    }

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest model"""
    print("\n🌲 Training Random Forest...")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }

    rf_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(
        rf_model, param_grid, cv=tscv,
        scoring='accuracy', n_jobs=-1, verbose=1
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'model': best_model,
        'accuracy': accuracy,
        'training_time': training_time,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_
    }

def create_ensemble_model(models, X_test, y_test):
    """Create ensemble model from multiple models"""
    print("\n🎭 Creating Ensemble Model...")

    predictions = []
    for name, model_info in models.items():
        pred = model_info['model'].predict(X_test)
        predictions.append(pred)

    # Voting ensemble (majority vote)
    ensemble_pred = []
    for i in range(len(X_test)):
        votes = [pred[i] for pred in predictions]
        # Get most common prediction
        ensemble_pred.append(max(set(votes), key=votes.count))

    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

    return {
        'predictions': ensemble_pred,
        'accuracy': ensemble_accuracy
    }

def compare_models(data_path='data/xauusd_m15_real.csv', model_path='models/'):
    """Compare multiple ML models for trading"""

    # Create models directory
    os.makedirs(model_path, exist_ok=True)

    print("📊 Loading and preparing data...")
    df = prepare_data(data_path)

    # Select features and target
    X = select_features(df)
    y = df['Target']

    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    # Split data with time series considerations
    split_idx = int(len(df) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")

    # Train all models
    models = {}

    # 1. Random Forest
    rf_result = train_random_forest(X_train_scaled, X_test_scaled, y_train, y_test)
    models['Random Forest'] = rf_result

    # 2. XGBoost
    xgb_result = train_xgboost(X_train_scaled, X_test_scaled, y_train, y_test)
    models['XGBoost'] = xgb_result

    # 3. LightGBM
    lgb_result = train_lightgbm(X_train_scaled, X_test_scaled, y_train, y_test)
    models['LightGBM'] = lgb_result

    # 4. CatBoost
    cat_result = train_catboost(X_train_scaled, X_test_scaled, y_train, y_test)
    models['CatBoost'] = cat_result

    # 5. Ensemble
    ensemble_result = create_ensemble_model(models, X_test_scaled, y_test)
    models['Ensemble'] = {'accuracy': ensemble_result['accuracy'], 'training_time': 0}

    # Compare results
    print("\n" + "="*80)
    print("📈 MODEL COMPARISON RESULTS")
    print("="*80)

    results_df = pd.DataFrame({
        'Model': [],
        'Accuracy': [],
        'CV_Score': [],
        'Training_Time': []
    })

    for name, result in models.items():
        if name != 'Ensemble':
            results_df = pd.concat([results_df, pd.DataFrame({
                'Model': [name],
                'Accuracy': [f"{result['accuracy']:.4f}"],
                'CV_Score': [f"{result['cv_score']:.4f}"],
                'Training_Time': [f"{result['training_time']:.1f}s"]
            })], ignore_index=True)
        else:
            results_df = pd.concat([results_df, pd.DataFrame({
                'Model': [name],
                'Accuracy': [f"{result['accuracy']:.4f}"],
                'CV_Score': ['-'],
                'Training_Time': ['Combined']
            })], ignore_index=True)

    print(results_df.to_string(index=False))

    # Find best model
    best_model_name = max(
        [k for k in models.keys() if k != 'Ensemble'], 
        key=lambda x: models[x]['accuracy']
    )
    best_model = models[best_model_name]

    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Accuracy: {best_model['accuracy']:.4f}")
    print(f"   CV Score: {best_model['cv_score']:.4f}")
    print(f"   Training Time: {best_model['training_time']:.1f}s")
    print(f"   Best Params: {best_model['best_params']}")

    # Check if ensemble is better
    if models['Ensemble']['accuracy'] > best_model['accuracy']:
        print(f"\n🎭 ENSEMBLE MODEL is even better!")
        print(f"   Ensemble Accuracy: {models['Ensemble']['accuracy']:.4f}")
        print(f"   Improvement: +{models['Ensemble']['accuracy'] - best_model['accuracy']:.4f}")

    # Save best model
    joblib.dump(best_model['model'], os.path.join(model_path, 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(model_path, 'best_scaler.pkl'))

    # Save comparison results
    results_df.to_csv(os.path.join(model_path, 'model_comparison.csv'), index=False)

    print(f"\n💾 Best model saved to {model_path}")

    return models, best_model_name

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import xgboost
        import lightgbm
        import catboost
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Please install with: pip install xgboost lightgbm catboost")
        exit(1)
    except Exception as e:
        if "libgomp" in str(e):
            print(f"Library linking error: {e}")
            print("This is a system configuration issue. Trying alternative approach...")
            print("Please restart the repl after the .replit file update.")
            exit(1)
        else:
            raise e

    models, best_model = compare_models()
    print(f"\n✅ Model comparison complete")