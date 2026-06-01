  # 📊 lob-imbalance-signals

  > Order flow imbalance features and transformer-based models for sub-second price prediction on limit order book data.

  ![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)
  ![License](https://img.shields.io/badge/license-MIT-green)
  ![Status](https://img.shields.io/badge/status-in_development-yellow)

  ## Research question

  Can short-horizon (1s–30s) price direction be predicted from limit order book imbalance features using attention-based neural networks, beating classical OFI baselines from the microstructure
  literature?

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

  ## Current results

  *In development — results table will populate as models complete.*

  | Model | Horizon | Accuracy | Sharpe (paper portfolio) |
  |---|---|---|---|
  | Linear (OFI) | 1s | TBD | TBD |
  | XGBoost | 1s | TBD | TBD |
  | LSTM | 1s | TBD | TBD |
  | **Transformer** | **1s** | **TBD** | **TBD** |

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
