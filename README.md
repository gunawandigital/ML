
# Forex Machine Learning Trading System - XAUUSD

Sistem trading otomatis untuk pasangan XAUUSD (Gold/USD) menggunakan Random Forest dan analisis teknikal.

## 📁 Struktur Project

```
forex-ml-rf-xauusd/
├── data/
│   └── xauusd_m15.csv           # Data historis XAUUSD M15
├── models/                      # Model dan scaler tersimpan (auto-generated)
├── feature_engineering.py       # Perhitungan indikator teknikal
├── train.py                     # Training Random Forest model
├── predict.py                   # Prediksi sinyal trading
├── backtest.py                  # Backtesting dan analisis performa
├── main.py                      # Eksekusi pipeline lengkap
├── requirements.txt             # Dependencies
└── README.md                    # Dokumentasi ini
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Pipeline Lengkap
```bash
python main.py
```

### 3. Atau Jalankan Individual Modules

#### Feature Engineering
```bash
python feature_engineering.py
```

#### Training Model
```bash
python train.py
```

#### Prediksi Terbaru
```bash
python predict.py
```

#### Backtesting
```bash
python backtest.py
```

## 📊 Features

### Technical Indicators
- **EMA (Exponential Moving Average)**: 9, 21, 50 periode
- **RSI (Relative Strength Index)**: 14, 21 periode
- **Price Ratios**: High-Low ratio, Open-Close ratio
- **Returns**: 1, 5, 15 periode
- **Volatility**: 5, 15 periode rolling
- **Moving Average Crossovers**
- **Price Position vs EMAs**

### Model Features
- **Algorithm**: Random Forest Classifier
- **Hyperparameter Tuning**: Grid Search CV
- **Feature Scaling**: StandardScaler
- **Target Classes**: BUY (1), HOLD (0), SELL (-1)

### Backtesting Metrics
- Total Return
- Win Rate
- Average Win/Loss
- Profit Factor
- Maximum Drawdown
- Equity Curve Visualization

## 📈 Trading Signals

### Signal Types
- **BUY (1)**: Signal beli - harga diperkirakan naik
- **HOLD (0)**: Signal tahan - tidak ada aksi
- **SELL (-1)**: Signal jual - harga diperkirakan turun

### Signal Generation
- Prediksi berdasarkan future return > 0.1% (threshold)
- Confidence level dari probabilitas model
- Real-time signal generation

## 🔧 Configuration

### Model Parameters
```python
# Default parameters in train.py
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

### Backtesting Settings
```python
# Default settings in backtest.py
initial_capital = 10000      # Modal awal
position_size = 0.1          # Ukuran posisi (10%)
transaction_cost = 0.0001    # Biaya transaksi (0.01%)
```

## 📊 Output Files

- `models/random_forest_model.pkl`: Model Random Forest tersimpan
- `models/scaler.pkl`: StandardScaler tersimpan
- `models/feature_importance.csv`: Importance features
- `backtest_results.png`: Grafik equity curve dan sinyal

## ⚠️ Disclaimer

Sistem ini dibuat untuk tujuan edukasi dan penelitian. Tidak ada jaminan profit dalam trading forex. Selalu gunakan risk management yang tepat dan jangan menginvestasikan lebih dari yang Anda mampu untuk kehilangan.

## 📚 Dependencies

- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning
- matplotlib: Plotting
- TA-Lib: Technical analysis
- joblib: Model serialization

## 🔮 Future Enhancements

- [ ] Real-time data integration
- [ ] Multiple timeframe analysis
- [ ] Advanced risk management
- [ ] Portfolio optimization
- [ ] Web dashboard
- [ ] Database integration
- [ ] Alert system

---

**Happy Trading! 📈💰**
