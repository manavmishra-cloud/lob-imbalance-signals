# 📊 lob-imbalance-signals

> Order flow imbalance features and transformer-based models for sub-second price prediction on limit order book data.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://github.com/manavmishra-cloud/lob-imbalance-signals/actions/workflows/tests.yml/badge.svg)
![Status](https://img.shields.io/badge/status-in_development-yellow)

## Research question

Can short-horizon (1s–30s) price direction be predicted from limit order book imbalance features using attention-based neural networks, beating classical OFI baselines from the microstructure literature?

## Approach

This repo implements and benchmarks a hierarchy of models for predicting short-term mid-price moves:

1. **Linear baseline** — classical OFI regression (Cont, Kukanov & Stoikov 2014)
2. **Tree baseline** — XGBoost on engineered LOB features
3. **Sequence baseline** — LSTM on raw L2 book snapshots
4. **Transformer model** — attention-based architecture on time-sequenced book imbalance

All models trained and evaluated under strict walk-forward out-of-sample validation.

## Data

- **[LOBSTER](https://lobsterdata.com/)** academic samples for AAPL, GOOG, MSFT, INTC, AMZN
- (Optional extension) Live Binance L2 order book snapshots for crypto evaluation

## Planned repo structure

```
lob-imbalance-signals/
├── README.md
├── LICENSE
├── requirements.txt
├── data/                    # raw and processed LOB data (gitignored)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   └── 04_transformer_model.ipynb
├── src/
│   ├── data/                # LOB parsing, OFI feature computation
│   ├── models/              # Linear, XGB, LSTM, Transformer
│   ├── training/            # Walk-forward CV, training loops
│   ├── evaluation/          # Metrics, statistical tests
│   └── viz/                 # Plotting utilities
├── results/
│   ├── figures/             # Saved plots
│   └── metrics/             # JSON results per model
└── tests/                   # Unit tests
```

## Key features (engineered)

- Order flow imbalance at depths 1-5 (multi-level OFI)
- Trade flow imbalance over rolling 1s, 5s, 30s windows
- Bid-ask spread normalized by avg
- Book-depth ratio (bid-side vs ask-side cumulative volume)
- Microprice deviation from mid

## Status

**Pipeline running on real LOB data (Binance L2) · LOBSTER samples currently unavailable from upstream.**

The full pipeline now runs end-to-end on live Binance order book captures, in addition to synthetic data for offline pipeline validation.

### First real-data results (BTC/USDT, Binance L2)

Captured 9,001 order-book snapshots of BTC/USDT (5 levels deep, 100 ms cadence) via the public Binance WebSocket stream (`btcusdt@depth5@100ms`). Run via `src/data/binance_capture.py`.

Target: 3-class next-5-events mid-price direction. Class distribution: -1 (down) 34%, 0 (no move) 40%, +1 (up) 26%.

| Model | 3-class Accuracy | Directional Accuracy |
|---|---|---|
| Persistence | 32.0% ± 4.5% | 24.7% |
| **Linear (OFI)** | **52.8% ± 5.7%** | **46.8%** |
| XGBoost | 52.5% ± 4.8% | 46.3% |

Naive "always predict 0" baseline would achieve **39.5%** (the modal class). Linear OFI beats this baseline by **+13 percentage points**, demonstrating that order-flow-imbalance features carry real predictive signal in BTC microstructure. Directional accuracy of 46.8% on non-zero target rows is well above the 33.3% random-prediction floor for a 3-class problem.

XGBoost does not improve over the linear OFI model on this dataset (52.5% vs 52.8%), consistent with the established finding that OFI's relationship to short-term mid-price changes is largely linear (Cont, Kukanov, Stoikov 2014).

**Caveats:** This is a 15-minute capture (~9,000 events). For paper-grade research we would want hours-to-days of capture across multiple symbols. The synthetic message file limitation (no true trade-flow events; only periodic book snapshots) means trade-flow-window features are uninformative in this run; results above come from book-derived OFI alone.

### Synthetic data pipeline check

A built-in synthetic LOB generator (`src/data/synthetic.py`) produces LOBSTER-format files that exercise the full pipeline. Pipeline-validation run on 99k synthetic events at horizon h=5:

| Model | Accuracy | Directional Accuracy | Notes |
|---|---|---|---|
| Persistence | 45.0% | 25.8% | dies to class imbalance |
| Linear (OFI) | 78.5% | 0.05% | learns trivial "predict 0" rule |
| XGBoost | 76.9% | 18.1% | makes non-zero predictions, no real signal in synthetic data |

These results confirm the pipeline runs cleanly. They do **not** represent real research findings — the synthetic generator does not model true OFI → future-return causality. Real LOBSTER data is expected to show learnable signal, consistent with the published literature (Cont/Kukanov/Stoikov 2014, Sirignano/Cont 2019).

### Current results (real data)

| Model | Horizon | Accuracy | Sharpe (paper portfolio) |
|---|---|---|---|
| Linear (OFI) | 1s | pending real data | — |
| XGBoost | 1s | pending real data | — |
| LSTM | 1s | pending real data | — |
| **Transformer** | **1s** | **pending real data** | **—** |

## Quick start — synthetic data pipeline test

```bash
# Set up environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate synthetic LOB data
python3 -m src.data.synthetic --n-events 100000 --levels 5

# Run baselines (persistence, linear, XGBoost) with walk-forward CV
python3 -m src.training.train_baselines --ticker SYNTH --target-horizon 5
```

## References

- Cont, R., Kukanov, A., & Stoikov, S. (2014). *The Price Impact of Order Book Events*. JFE.
- Sirignano, J., & Cont, R. (2019). *Universal features of price formation in financial markets: perspectives from deep learning*. Quantitative Finance.
- Briola, A., Vidler, J., & Aste, T. (2020). *Deep Learning modeling of Limit Order Book*. arXiv:2007.07319.

## Planned arXiv submission

Manuscript in preparation — target submission Q4 2026 to `q-fin.TR` (Trading and Market Microstructure) category.

## License

MIT — see [LICENSE](LICENSE)

## Contact

Manav Mishra · [LinkedIn](https://linkedin.com/in/manav-mishra-23a26b308) · manavmishra260205@gmail.com
