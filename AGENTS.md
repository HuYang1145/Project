# Repository Guidelines

## Project Layout

This repository is an undergraduate final-year project on Beijing PM2.5 forecasting. It contains both the runnable demo system and the Chinese/English thesis workspaces.

`demo/` contains the runnable Streamlit application. `app.py` handles the UI, `predictor.py` handles inference, `loader.py` and `loader_simulation.py` handle real-time and simulated data, and `db_utils.py` manages the local SQLite cache. `notebooks/` is split by model into `arima/`, `prophet/`, `lstm/`, and `diffusion/`. `results/` stores evaluation summaries and figures, `data/` stores raw and intermediate data, `models/` stores large model files. `毕业论文中文/` and `毕业论文英文/` are separate XeLaTeX thesis workspaces with `main.tex`, `contents/`, `appendices/`, and thesis figures.

The repository also includes supporting presentation files and a rolling `TODO.md` that records defense-facing wording risks, paper fixes, and scope decisions. Read `README.md` and `TODO.md` before making substantial research-facing edits.

## Development Commands

Activate the environment and install dependencies:

```bash
conda activate project
pip install -r requirements.txt
```

Start the demo app:

```bash
cd demo
streamlit run app.py
```

Generate the results summary:

```bash
python summarize_results.py
```

Run the current cache regression check:

```bash
python demo/test_cache.py
```

Inspect the cache from `demo/` when needed:

```bash
sqlite3 local_air_cache.db "SELECT COUNT(*) FROM air_quality;"
```

## Code Style and Experiment Conventions

Use `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_CASE` for constants. Python code uses four-space indentation. Keep UI logic in `demo/app.py` when possible and move reusable logic into standalone modules. Training scripts should also produce thesis-ready outputs: save charts as high-resolution PNG files when possible (`dpi=300` recommended), and save evaluation metrics as UTF-8 CSV files.

When updating research-facing copy, preserve the current project positioning:

- `BiLSTM-Hybrid` is the main point-forecasting model in the current deployment setting.
- `DiffSTG` is retained as the uncertainty-analysis / probabilistic forecasting branch, not the default online backbone.
- `Prophet` uses a different multivariate input setting and should not be ranked directly against the shared-input ARIMA / BiLSTM-Hybrid / DiffSTG comparison track unless the comparison protocol is explicitly redefined.

## Testing and Validation

The repository does not yet have a complete `pytest` suite. `python demo/test_cache.py` is the current baseline check. For changes that affect prediction logic, caching, or UI behavior, also verify the Streamlit app manually in both real-time and simulation modes. Name any new tests as `test_*.py`.

If you change caching or alert logic, also inspect `demo/TEST_CACHE.md`. The current app behavior includes:

- shared simulation-data caching across ARIMA / BiLSTM-Hybrid / DiffSTG
- Prophet-specific simulation caching because its data path differs
- 48-hour alert caching keyed by data sample rather than the visible model selector
- pausing some heavy-model combinations to control memory pressure

## Model and Performance Notes

The project currently includes ARIMA, Prophet, BiLSTM-Hybrid, and DiffSTG. BiLSTM-Hybrid and DiffSTG are both heavy and should not be loaded together unless necessary. For AQI classification or alerting changes, check that `aqi_classifier.py`, `predictor.py`, and the related UI remain consistent.

Stable project-specific context that should usually be preserved unless the user explicitly changes it:

- The engineering target is an end-to-end Beijing air-quality forecasting dashboard with local caching, on-demand model loading, AQI classification, interval visualization, and 48-hour alerting.
- The SQLite cache is not just a convenience layer; it supports long-history loading for decomposition-based models and reduces repeated API traffic.
- The BiLSTM-Hybrid pipeline relies on decomposition plus local error correction (LEC); if you change alerting or AQI-trigger behavior, verify that the written explanation in the thesis still matches the implemented logic.
- Current thesis-level summary metrics referenced in the repo include BiLSTM-Hybrid short-horizon `R^2=0.9518`, DiffSTG PI90 coverage below nominal 0.90, and system-level alerting claims around `87%` accuracy with `36` hours average lead time. Do not casually rewrite these numbers without tracing the source file and evaluation context.

## Thesis Workspace Rules

The thesis workspace uses XeLaTeX. Do not modify `environments.sty` or `requirements.sty` unless it is clearly necessary. Thesis writing should remain academic in tone and should not paste engineering code directly into the main text.
Any change to thesis source files must be followed by a XeLaTeX rebuild so the generated PDF is refreshed and verified before the task is considered complete.

The thesis currently exists in both Chinese and English directories. Keep cross-version claims aligned when editing both. Common high-risk paper issues already noted in `TODO.md` include:

- use `prediction interval (PI)` terminology consistently for DiffSTG rather than mixing confidence / credible / probability interval wording
- define PI90 explicitly at first mention
- avoid overstating Prophet as causal or physically proving attribution; keep it in the interpretability / explanatory decomposition lane
- define any "fair comparison" protocol concretely when comparing non-Prophet models
- define event-level alert metrics such as recall, lead time, and Lead IQR clearly
- keep dataset date ranges and denominator definitions consistent across abstract, methods, results, PPT, and thesis text

## Collaboration and Submission

Keep commit messages short, direct, and verb-led, for example `Fix: ...`. After important changes, sync `TODO.md` with any new unfinished work or repository conventions. Pull requests should describe scope and validation steps. Include screenshots for `demo/` UI changes, and explicitly note any changes that affect models, data, or cache files.

If a task changes thesis claims, PPT wording, or evaluation numbers, update the relevant notes in `TODO.md` so the defense narrative and repository record stay synchronized.

## Safety and Configuration

Do not commit model weights, raw data, databases, cache files, or other large generated artifacts. The demo code currently contains hard-coded API settings; prefer environment variables or local override configuration for new features so real secrets are not added to the repository.
