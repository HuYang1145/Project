# Time Series Forecasting of Urban Air Pollution with Machine Learning

This repository contains the code, system prototype, experiment scripts, and thesis workspace for the QMUL/BUPT undergraduate final-year project on urban air-quality forecasting. The project studies Beijing PM2.5 forecasting under real deployment constraints and implements an end-to-end workflow covering data preprocessing, model inference, risk alerting, and interactive visualization.

## Project Scope

The repository supports two closely related goals:

1. Research: compare multiple forecasting paradigms for PM2.5 prediction, including statistical baselines, a decomposition-driven BiLSTM-Hybrid model, and a diffusion-based spatio-temporal probabilistic model.
2. Engineering: deliver a runnable forecasting dashboard with local caching, on-demand model loading, 48-hour alert generation, and simulation-based evaluation.

The final research conclusion of the project is that `BiLSTM-Hybrid` is the most suitable model for accurate point forecasting in the current deployment setting, while `DiffSTG` is retained as the uncertainty-analysis branch.

## Repository Layout

- `demo/`: runnable Streamlit application and online/offline inference workflow
- `notebooks/`: model-specific experiments, training notebooks, and analysis scripts
- `data/`: raw or intermediate datasets used by the forecasting pipeline
- `models/`: trained model artifacts and serialized preprocessing objects
- `results/`: generated metrics, comparison summaries, and figure outputs
- `毕业论文中文/`: Chinese thesis XeLaTeX workspace
- `毕业论文英文/`: English thesis XeLaTeX workspace
- `summarize_results.py`: utility script for aggregating experiment outputs

## Models Included

### ARIMA

Classical univariate statistical baseline used for lightweight short-horizon extrapolation and benchmarking.

### Prophet

Interpretable regression-based baseline using meteorological covariates for trend and attribution analysis.

### BiLSTM-Hybrid

Primary point-forecasting model that combines CEEMDAN/EEMD signal decomposition, bidirectional LSTM sequence modeling, and local error correction (LEC).

### DiffSTG

Spatio-temporal diffusion model used for probabilistic forecasting and uncertainty-aware risk-boundary analysis.

## Data Description

The project uses two main data branches:

- A univariate Beijing PM2.5 time-series dataset for `ARIMA`, `BiLSTM-Hybrid`, and `DiffSTG`
- A multivariate Beijing air-quality plus meteorology dataset for `Prophet`

The deployed demo additionally supports external API-fed real-time mode through locally cached weather and air-quality inputs.

## Environment Setup

Recommended environment:

```bash
conda activate project
pip install -r requirements.txt
```

## Running the Demo

Start the Streamlit dashboard from the `demo/` directory:

```bash
cd demo
streamlit run app.py
```

The dashboard supports:

- real-time mode with cached API-backed inputs
- historical simulation mode for reproducible backtesting
- multiple selectable forecasting models
- PM2.5 forecast visualization with optional interval display
- 48-hour AQI-oriented pollution alert generation

## Reproducing Key Outputs

Generate the aggregated experiment summary:

```bash
python summarize_results.py
```

Run the cache regression check:

```bash
python demo/test_cache.py
```

Inspect the local demo cache when needed:

```bash
cd demo
sqlite3 local_air_cache.db "SELECT COUNT(*) FROM air_quality;"
```

## Thesis Workspaces

The repository also contains the complete XeLaTeX source for the Chinese and English thesis versions:

- [毕业论文中文](D:/文件/大四下/Project/毕业论文中文)
- [毕业论文英文](D:/文件/大四下/Project/毕业论文英文)

These folders include source files, figures, appendices, and generated PDFs. Changes to thesis source files should be followed by a XeLaTeX rebuild.

## Engineering Notes

- `BiLSTM-Hybrid` and `DiffSTG` are both resource-heavy and should not normally be loaded together for routine demo use.
- AQI-related changes should keep `demo/aqi_classifier.py`, `demo/predictor.py`, and the Streamlit UI behavior aligned.
- The system uses on-demand loading and local SQLite caching to reduce cold-start latency and memory pressure.

## Limitations

- Large data files, model artifacts, and local caches may not be suitable for full public distribution.
- Some experiment paths depend on locally prepared datasets or pretrained weights.
- `DiffSTG` currently serves as the uncertainty-analysis branch rather than the main online forecasting backbone due to its latency cost.

## Citation and Context

This repository accompanies the undergraduate project report:

`Time Series Forecasting of Urban Air Pollution with Machine Learning`

If you reuse the code or figures, cite the corresponding thesis source and clearly indicate any modified components.
