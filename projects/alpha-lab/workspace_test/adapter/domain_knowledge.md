# Time Series Domain Knowledge

## Key Concepts
- Walk-forward backtesting prevents lookahead bias
- Sharpe ratio is the primary risk-adjusted return metric
- Max drawdown measures worst-case loss from peak to trough
- Embargo periods prevent data leakage between train/test folds

## Common Pitfalls
- Lookahead bias: using future data in features or training
- Survivorship bias: only analyzing assets that still exist
- Overfitting: tuning hyperparameters on test data
- Non-stationarity: assuming statistical properties don't change over time

## Model Families
- Tree-based: XGBoost, LightGBM (strong baselines, fast)
- Transformer: TFT, PatchTST, Informer (state-of-art on many benchmarks)
- RNN: LSTM, GRU, DeepAR (sequential modeling)
- CNN: TCN, TimesNet (dilated causal convolutions)
- MLP: N-BEATS, N-HiTS, TSMixer (surprisingly competitive)

## Libraries
- neuralforecast: N-BEATS, N-HiTS, PatchTST, TFT, LSTM
- pytorch-forecasting: TFT, DeepAR, N-BEATS
- darts: unified API for many models
- Raw PyTorch: maximum flexibility
