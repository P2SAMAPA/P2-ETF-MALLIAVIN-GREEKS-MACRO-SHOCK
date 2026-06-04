# Malliavin Greeks for Macro Shock Sensitivity

Computes the pathwise sensitivity (Greek) of an ETF's expected log return to a macro shock (e.g., +1% VIX) using Malliavin calculus approximated via finite differences and Monte Carlo simulation.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Uses primary macro variable (e.g., VIX) as shock source
- Estimates drift sensitivity β via regression of returns on macro
- Simulates GBM paths to compute expected log return with/without shock
- Score = (E[return | shock] - E[return | no shock]) / perturbation
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-malliavin-greeks-macro-shock-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (slow due to Monte Carlo; reduce `N_SIMULATIONS` for speed)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- Positive Greek → ETF's expected return increases when the macro variable rises (e.g., VIX up).
- Large absolute Greek → ETF is highly sensitive to macro shocks.

## Requirements

See `requirements.txt`.
