# From Market Data to Risk Insight | Financial Risk & Data Engineering Case Study

End-to-end, reproducible, and quantitatively validated solution for the **Risk Data Engineering & Analytics Expert (FlatexDEGIRO)** case study.

This repository delivers:
1. **Task 1a (`etl_ecb.py`)**: A resilient, idempotent Python ETL pipeline ingesting daily ECB USD/EUR reference rates (`EXR.D.USD.EUR.SP00.A`) via the official SDMX RESTful API into a local analytical **DuckDB** database (`data/market_data.duckdb`).
2. **Task 1b (`dashboard.py` & `src/risk_engine.py`)**: An interactive **Daily FX-Risk Dashboard** in Streamlit for the Management Board, featuring exact non-linear **Full Repricing ($1/S$)** over a 5-business-day horizon, **Through-the-Cycle (TTC)** vs. **Point-in-Time (PiT / EWMA $\lambda=0.94$)** VaR & Expected Shortfall, **Critical Trigger Rate ($S_{\text{VaR}}$)**, **5-Day TWAP Execution**, **5-Day FX Outright Forward Hedge Benchmark**, and **Kupiec POF + Christoffersen Backtesting** across both overlapping and non-overlapping 5-day blocks.
3. **Task 2 (`analysis_indices.py`, `src/index_analysis.py` & `EDA_Risk_and_Index_Analysis.ipynb`)**: A detailed 18-year econometric study across 31 global indices (`2008–2026`), resolving the `US6311011026` (`IXIC` vs. `NDX`) key collision and proving **Correlation Breakdown ($+43.8\%$ surge during $\text{VIX} \ge 30$)**, **Forbes & Rigobon (2002)** asymmetric lower-tail contagion, and the **Asian Timezone Asynchrony ($t+1$) effect**.

---

## 1. Repository Architecture

```text
FlatexDegiro/
├── data/
│   └── market_data.duckdb                           # Local columnar DuckDB database (3,030 business days, Task 1a)
├── figures/                                         # High-resolution (300 DPI) analytical charts (Tasks 1b & 2)
│   ├── 01_correlation_diff_heatmap_en.png
│   ├── 02_regional_correlation_shift_en.png
│   ├── 03_rolling_correlation_sp500_dax_vs_vix.png
│   ├── 04_fat_tails_kurtosis_drawdowns.png
│   ├── 05_timezone_lag_effect_en.png
│   ├── 06_fx_pnl_distribution_en.png
│   └── 07_fx_backtest_kupiec_christoffersen_en.png
├── src/
│   ├── __init__.py
│   ├── config.py                                    # Centralized configuration, SDMX endpoints, and sanity bounds
│   ├── ecb_client.py                                # Resilient HTTP client (exponential backoff & TARGET2 holidays)
│   ├── validator.py                                 # Strict schema validation and economic sanity checks
│   ├── database.py                                  # DuckDB persistence layer with ACID idempotent upserts
│   ├── risk_engine.py                               # FX VaR/ES engine, Trigger Rate, TWAP, Forward & Backtesting
│   └── index_analysis.py                            # Index data curation, VIX regimes, correlations & lead-lag engine
├── tests/
│   ├── test_etl.py                                  # Unit & integration tests for ECB ETL & DuckDB (7 tests)
│   ├── test_risk.py                                 # Quantitative risk & Kupiec/Christoffersen tests (7 tests)
│   └── test_indices.py                              # Index deduplication, returns & correlation matrix tests (5 tests)
├── EDA_Risk_and_Index_Analysis.ipynb                # Interactive Jupyter Notebook (Step-by-step EDA with inline plots)
├── etl_ecb.py                                       # CLI entry point for Task 1a (ECB FX Data Ingestion)
├── dashboard.py                                     # Streamlit Risk Dashboard for Task 1b (Management Board)
├── analysis_indices.py                              # CLI entry point for Task 2 (Global Index Analysis)
├── dim_master.csv                                   # Master metadata for the 31 global indices (Task 2)
├── price_hist.parquet                               # Historical daily prices 2008–2026 (143,211 rows, Task 2)
├── FlatexDEGIRO_Risk_Analytics_Presentation_EN.pptx # Presentation Slide Deck (16:9 PowerPoint with Speaker Notes)
├── presentation_slides_en.md                        # Markdown version of the Presentation Deck & Script
├── tool_use_note.md                                 # Transparency & validation note on modern tools and AI usage
├── requirements.txt                                 # Reproducible Python dependencies
└── README.md                                        # Project documentation and execution guide
```

---

## 2. Quick Start & Installation

* **Python Version**: Python 3.10+ (tested on Python 3.13).
* **Install Dependencies**:

```bash
pip install -r requirements.txt
```

---

## 3. Step-by-Step Execution Guide

### A. Interactive Exploratory Data Analysis (`EDA_Risk_and_Index_Analysis.ipynb`)
Open **`EDA_Risk_and_Index_Analysis.ipynb`** directly on GitHub, VS Code, or Jupyter Lab to inspect the full step-by-step reasoning, mathematical derivations, summary tables, and embedded charts (pre-executed so all outputs are immediately visible without running a cell).

### B. Task 1a: ECB FX Data Ingestion (`etl_ecb.py`)
1. **Initial Historical Backfill (`--mode initial`)**:
   Downloads the official ECB series `EXR.D.USD.EUR.SP00.A` from `2015-01-01` to present (~3,030 trading days), validates schema/economic bounds, and initializes `data/market_data.duckdb`:
   ```bash
   python etl_ecb.py --mode initial
   ```
2. **Incremental Daily Update (`--mode daily`)**:
   Queries `MAX(rate_date)` in DuckDB and fetches only new observations (`startPeriod = max_date + 1 day`). On weekends, **TARGET2** bank holidays, or prior to the 16:00 CET publication window, empty/404 responses are logged cleanly at `INFO` level and exit with code `0`:
   ```bash
   python etl_ecb.py --mode daily
   ```

### C. Task 1b: Daily FX-Risk Management Board Dashboard (`dashboard.py`)
Launch the interactive Streamlit web application:

```bash
python -m streamlit run dashboard.py
```

* Opens automatically at **`http://localhost:8501`**.
* Includes a **live ECB refresh button** in the sidebar that triggers Task 1a's incremental pipeline, updates DuckDB, and invalidates calculation caches on the fly.

### D. Task 2: Global Index Analysis CLI (`analysis_indices.py`)
Reproduce the full statistical report in the console and regenerate all high-resolution figures in `figures/`:

```bash
python analysis_indices.py
```

### E. Automated Test Suite (`pytest`)
Run all **19 unit and integration tests** verifying ETL idempotency, non-linear repricing convexity, TWAP risk reduction, Kupiec/Christoffersen likelihood ratio tests, and correlation matrix properties:

```bash
python -m pytest tests/ -v
```

---

## 4. Key Quantitative Findings & Design Decisions

### 4.1 Task 1a — Data Engineering & TARGET2 Calendar Compliance
* **Embedded Analytical Storage (DuckDB)**: Chosen over flat CSVs for ACID transactions, strict primary-key enforcement (`rate_date DATE PRIMARY KEY`), audit timestamps (`ingested_at`), and zero-configuration portability.
* **Strict TARGET2 Convention (No Synthetic Imputation)**: We intentionally avoid forward-filling weekends or mixing Federal Reserve (NY 16:00 ET) rates with ECB fixings (14:15 CET). Artificial forward-filling injects zero-return days that bias empirical volatility downward.

### 4.2 Task 1b — 5-Day FX Risk Engine & Decision Benchmarks ($100,000\text{ USD}$)
* **Indirect Quotation & Full Repricing ($1/S$)**:
  Because the ECB quotes $S = \text{USD/EUR}$ (USD per 1 EUR), the EUR value of $100,000\text{ USD}$ is $V_{\text{EUR}}(S) = \frac{100,000}{S}$. Losses occur when the USD depreciates ($S$ increases). We evaluate exact non-linear scenarios over empirical 5-business-day log-returns $r_{k,5d} = \ln(S_t / S_{t-5})$ without relying on $\sqrt{5}$ scaling:
  $$L^{(k)}_{5d} = V_0 - \frac{100,000}{S_0 \cdot \exp(r_{k, 5d})}$$
* **Summary of Key Risk & Hedging Metrics**:

| Metric / Benchmark | Value (EUR / %) | Decision Impact / Interpretation |
| :--- | :--- | :--- |
| **Portfolio Mark-to-Market ($V_0$)** | **€89,158.35** | $100,000\text{ USD}$ valued at $S_0 = 1.1216\text{ USD/EUR}$ |
| **99% 5-Day Historical VaR (TTC)** | **€2,513.33 (2.82%)** | Anti-cyclical 11-year historical simulation (3,025 scenarios) |
| **99% 5-Day Expected Shortfall (ES)** | **€2,990.32 (3.35%)** | Average tail loss when the 99% VaR threshold is breached |
| **99% 5-Day EWMA VaR ($\lambda=0.94$, PiT)** | **€1,350.19 (1.51%)** | Reflects calm recent 21d volatility (4.66% ann., **Green Traffic Light**) |
| **Critical Trigger Rate ($S_{\text{VaR}}$)** | **1.1541 USD/EUR** | Exact spot threshold (+2.90%) that exhausts the 99% VaR budget |
| **5-Day TWAP Execution ($20\text{k USD/d}$)** | **€1,627.21 (-35.3% VaR)** | Zero-cost execution strategy saving **€886.12** in 5-day VaR |
| **5-Day Outright FX Forward Hedge** | **€28.38 (<1% of ES)** | Total cost (SOFR/€STR carry + spread) to eliminate €2,990 tail risk |

* **Regulatory Backtesting (Overlapping vs. Non-Overlapping Blocks)**:
  * **Daily Rolling Windows ($N = 2,525$)**: 32 breaches ($1.27\%$ vs. $1.00\%$ expected), passing **Kupiec POF** with **$p = 0.1949 > 0.05$**.
  * **Non-Overlapping 5-Day Blocks (`step = 5`, $N = 505$)**: Neutralizes the mechanical $\text{MA}(4)$ serial correlation of overlapping 5-day windows, reducing consecutive clustered breaches ($n_{11}$) by **94.4% ($18 \rightarrow 1$)** while passing Kupiec POF with **$p = 0.3213$**.

### 4.3 Task 2 — Global Equity Index Analysis (2008–2026)
1. **Data Quality Resolution of `US6311011026` (`IXIC.INDX` vs. `NDX.INDX`)**:
   In `dim_master.csv`, both the *NASDAQ Composite* and *NASDAQ 100* shared `asset_key = US6311011026`, with rows interleaved in `price_hist.parquet`. We deterministically decoupled even rows (`IXIC.INDX`, ~2,609 in Jan 2008) from odd rows (`NDX.INDX`, ~2,049 in Jan 2008), preserving all 31 indices.
2. **Empirical Proof of *Correlation Breakdown***:
   Across all 435 unique equity index pairs, mean pairwise correlation increases from **$0.4247$ in Calm ($\text{VIX} < 20$)** to **$0.6109$ in Distress ($\text{VIX} \ge 30$)** — a **$+43.8\%$ surge** ($\Delta \bar{\rho} = +0.1861$). Regional contagion is strongest between Core Europe and Eastern Europe (**$+63.8\%$**). We also demonstrate genuine **asymmetric lower-tail dependence** beyond the variance conditioning bias of **Forbes & Rigobon (2002)**.
3. **Timezone Asynchrony Correction (Asia vs. US)**:
   Because Tokyo closes at 06:00 UTC before Wall Street opens at 14:30 UTC, same-day correlation between the S&P 500 and Nikkei 225 is artificially low (**$0.153$**). Aligning Asia at day $t+1$ with the US at day $t$ causes correlation to more than triple to **$0.511$ ($+234\%$)**.
