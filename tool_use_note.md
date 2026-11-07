# Tool-Use Note | Methodological Transparency & Output Validation

In accordance with the case study guidelines (*"Modern tools are welcome. Understanding is essential. Name the tools used, what they helped with, and how you assessed their output"*), this note details the software stack and AI coding tools used during development and the deterministic verification protocols applied to audit every result.

---

## 1. Tools Used

1. **Python Quantitative & Data Engineering Stack (Python 3.13)**:
   * **Ingestion & Storage**: `duckdb` (embedded columnar OLAP engine), `requests` + `urllib3` (resilient HTTP client with exponential backoff for ECB SDMX-CSV feeds), `pyarrow`, `pandas`, `numpy`.
   * **Econometric & Statistical Modeling**: `scipy.stats` (Gaussian distributions, empirical quantiles, skewness, excess kurtosis, and $\chi^2$ likelihood-ratio tests for Kupiec POF and Christoffersen Conditional Coverage).
   * **Interactive Visualization & Reporting**: `streamlit` and `plotly` (interactive Management Board FX Risk Dashboard), `matplotlib` and `seaborn` (publication-grade static charts), `nbformat` (automated Jupyter Notebook execution).
   * **Automated Testing**: `pytest` (19 unit and integration tests).
2. **AI Coding Assistant (LLM Pair Programmer)**:
   * Used under direct quantitative supervision for boilerplate generation, Streamlit UI styling, and econometric peer review.

---

## 2. What They Helped With

* **Modular Architecture & Boilerplate Acceleration**:
  Structuring the clean separation of concerns across `src/config.py`, `src/ecb_client.py`, `src/validator.py`, `src/database.py`, `src/risk_engine.py`, and `src/index_analysis.py`, alongside CLI interfaces (`argparse`) and structured logging.
* **Exploratory Data Inspection (Task 2)**:
  Rapid programmatic profiling of `dim_master.csv` and `price_hist.parquet`, which immediately surfaced the $4,674$ duplicate dates caused by the shared `US6311011026` key (`IXIC.INDX` vs. `NDX.INDX`).
* **Dashboard Layout & Visual Polish**:
  Drafting the Streamlit custom CSS for key indicator cards, interactive Plotly charts (5-day P&L distribution with VaR/ES/TWAP cutoffs and rolling backtest charts), and the high-resolution figures in `figures/`.
* **Quantitative Challenger Review**:
  Stress-testing theoretical assumptions, including the mechanical $\text{MA}(4)$ serial dependence introduced by overlapping 5-day returns in Christoffersen's Independence test and the heteroskedasticity conditioning bias of Forbes & Rigobon (2002).

---

## 3. How Output Was Assessed, Validated, and Audited

No code snippet or statistical metric was accepted without deterministic verification and financial cross-checking:

1. **Mathematical Audit of Quotation Convention & Non-Linear Convexity ($1/S$)**:
   * Verified manually that because the ECB series `EXR.D.USD.EUR.SP00.A` quotes USD per 1 EUR ($1\text{ EUR} = S\text{ USD}$), a long $100,000\text{ USD}$ position is worth $V_{\text{EUR}} = 100,000 / S$ and **loses EUR value when $S$ rises** (USD depreciation).
   * Verified algebraically the exact **Trigger Rate** identity:
     $$S_{\text{VaR}} = \frac{100,000}{V_0 - \text{VaR}_{99\%}} = \frac{100,000}{89,158.35 - 2,513.33} = 1.1541\text{ USD/EUR}$$
2. **Integrity Verification of the `US6311011026` Decoupling (`IXIC` vs. `NDX`)**:
   * Rather than applying a naive `drop_duplicates()` (which would have silently deleted the NASDAQ 100), we cross-checked historical market closes on January 2, 2008 ($\text{IXIC} = 2,609.63$ vs. $\text{NDX} = 2,049.71$) and their 2026 levels ($\text{NDX} \approx 28,274 > \text{IXIC} \approx 25,373$) to confirm with 100% certainty that even rows correspond to `IXIC.INDX` and odd rows to `NDX.INDX`.
3. **Edge-Case Resolution in Automated Unit Tests (`pytest`)**:
   * During test construction in `tests/test_indices.py`, we identified that evaluating `(df > -0.30).all()` directly on a pivoted multi-country return matrix returned `False` in `pandas` because `NaN > -0.30` evaluates to `False` under IEEE 754 floating-point rules on local country bank holidays. We refined the test to assert strict bounds over non-null observations (`min()` and `max()`).
4. **Econometric Verification of Kupiec & Christoffersen Likelihood Ratios**:
   * Validated boundary behaviors in `tests/test_risk.py` using synthetic indicator vectors: a vector with exactly $10$ dispersed breaches in $1,000$ days yields $LR_{\text{POF}} \approx 0$ and passes independence, whereas a vector with $10$ consecutive clustered breaches passes unconditional Kupiec coverage but is decisively rejected by Christoffersen's Conditional Coverage test ($p < 0.001$).
