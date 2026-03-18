# summarize_results.py
# Summarize all model evaluation results and organize figures

import os
import json
import shutil
from datetime import datetime
import pandas as pd
import nbformat

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures_summary')
SUMMARY_FILE = os.path.join(RESULTS_DIR, 'evaluation_summary.txt')
INDEX_FILE = os.path.join(RESULTS_DIR, 'figures_index.txt')

# Create figures directory
os.makedirs(FIGURES_DIR, exist_ok=True)

def extract_metrics_from_csv():
    """Extract metrics from CSV files in results/"""
    metrics = {}

    # Prophet classification
    prophet_csv = os.path.join(RESULTS_DIR, 'prophet_classification_metrics.csv')
    if os.path.exists(prophet_csv):
        df = pd.read_csv(prophet_csv)
        metrics['Prophet Classification'] = df.to_dict('records')

    # LSTM classification
    for days in [3, 5, 7]:
        lstm_csv = os.path.join(RESULTS_DIR, f'lstm_classification_{days}days_metrics.csv')
        if os.path.exists(lstm_csv):
            df = pd.read_csv(lstm_csv)
            metrics[f'LSTM Classification ({days} days)'] = df.to_dict('records')

    return metrics

def extract_metrics_from_notebook(notebook_path):
    """Extract printed metrics from Jupyter notebook outputs"""
    if not os.path.exists(notebook_path):
        return None

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        metrics_text = []
        for cell in nb.cells:
            if cell.cell_type == 'code' and 'outputs' in cell:
                for output in cell.outputs:
                    if output.output_type == 'stream' and 'text' in output:
                        text = output['text']
                        # Look for metric keywords
                        if any(keyword in text.lower() for keyword in ['mae', 'rmse', 'mape', 'accuracy', 'r2']):
                            metrics_text.append(text)

        return '\n'.join(metrics_text) if metrics_text else None
    except:
        return None

def organize_figures():
    """Copy and rename figures from results/ to figures_summary/"""
    figure_index = []

    # Get all PNG files in results/
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith('.png'):
            src = os.path.join(RESULTS_DIR, filename)
            dst = os.path.join(FIGURES_DIR, filename)
            shutil.copy2(src, dst)

            # Parse filename to create description
            if 'prophet_classification' in filename:
                model = 'Prophet Classification'
                if 'accuracy' in filename:
                    desc = 'Accuracy comparison across prediction horizons'
                elif 'confusion' in filename:
                    desc = 'Confusion matrix'
                elif 'distribution' in filename:
                    desc = 'Class distribution'
                else:
                    desc = 'Evaluation result'
            elif 'lstm_classification' in filename:
                if '3days' in filename:
                    model = 'LSTM Classification (3 days)'
                elif '5days' in filename:
                    model = 'LSTM Classification (5 days)'
                elif '7days' in filename:
                    model = 'LSTM Classification (7 days)'
                else:
                    model = 'LSTM Classification'

                if 'confusion' in filename:
                    desc = 'Confusion matrix'
                elif 'curves' in filename:
                    desc = 'Training curves (loss & accuracy)'
                else:
                    desc = 'Evaluation result'
            else:
                model = 'Unknown'
                desc = filename

            figure_index.append({
                'filename': filename,
                'model': model,
                'description': desc
            })

    return figure_index

def generate_summary():
    """Generate comprehensive summary text file"""
    lines = []
    lines.append("=" * 70)
    lines.append("Air Quality Prediction Models - Evaluation Summary")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    # Extract metrics from CSV
    csv_metrics = extract_metrics_from_csv()

    # Section 1: Classification Models
    lines.append("[1] LONG-TERM TREND CLASSIFICATION MODELS")
    lines.append("-" * 70)
    lines.append("")

    if 'Prophet Classification' in csv_metrics:
        lines.append("1.1 Prophet Classification Model")
        for record in csv_metrics['Prophet Classification']:
            lines.append(f"  - Prediction Horizon: {record.get('n_days', 'N/A')} days")
            lines.append(f"  - Accuracy: {record.get('accuracy', 'N/A')}")
            lines.append(f"  - Macro F1-Score: {record.get('macro_f1', 'N/A')}")
        lines.append("")

    lines.append("1.2 LSTM Classification Model")
    for days in [3, 5, 7]:
        key = f'LSTM Classification ({days} days)'
        if key in csv_metrics:
            for record in csv_metrics[key]:
                lines.append(f"  - {days}-day prediction:")
                lines.append(f"    Accuracy: {record.get('best_accuracy', 'N/A')}")
                lines.append(f"    Final Loss: {record.get('final_loss', 'N/A')}")
    lines.append("")

    # Section 2: Regression Models (from notebooks)
    lines.append("[2] SHORT-TERM REGRESSION MODELS")
    lines.append("-" * 70)
    lines.append("")

    # ARIMA
    lines.append("2.1 ARIMA Model")
    arima_metrics = extract_metrics_from_notebook('notebooks/arima/analysis.ipynb')
    if arima_metrics:
        lines.append(arima_metrics)
    else:
        lines.append("  (Run notebooks/arima/analysis.ipynb to generate metrics)")
    lines.append("")

    # Prophet
    lines.append("2.2 Prophet Model")
    prophet_metrics = extract_metrics_from_notebook('notebooks/prophet/analysis_Prophet.ipynb')
    if prophet_metrics:
        lines.append(prophet_metrics)
    else:
        lines.append("  (Run notebooks/prophet/analysis_Prophet.ipynb to generate metrics)")
    lines.append("")

    # LSTM
    lines.append("2.3 BiLSTM-Hybrid Model")
    lstm_metrics = extract_metrics_from_notebook('notebooks/lstm/analysis-LSTM.ipynb')
    if lstm_metrics:
        lines.append(lstm_metrics)
    else:
        lines.append("  (Run notebooks/lstm/analysis-LSTM.ipynb to generate metrics)")
    lines.append("")

    # Diffusion
    lines.append("2.4 DiffSTG (Diffusion) Model")
    diff_metrics = extract_metrics_from_notebook('notebooks/diffusion/train_diffusion.ipynb')
    if diff_metrics:
        lines.append(diff_metrics)
    else:
        lines.append("  (Run notebooks/diffusion/train_diffusion.ipynb to generate metrics)")
    lines.append("")

    # Transformer
    lines.append("2.5 Transformer Model")
    trans_metrics = extract_metrics_from_notebook('notebooks/transformer/train_transformer.ipynb')
    if trans_metrics:
        lines.append(trans_metrics)
    else:
        lines.append("  (Run notebooks/transformer/train_transformer.ipynb to generate metrics)")
    lines.append("")

    lines.append("=" * 70)
    lines.append("End of Summary")
    lines.append("=" * 70)

    return '\n'.join(lines)

def generate_figure_index(figure_list):
    """Generate figure index file"""
    lines = []
    lines.append("=" * 70)
    lines.append("Figures Index")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    # Group by model
    grouped = {}
    for fig in figure_list:
        model = fig['model']
        if model not in grouped:
            grouped[model] = []
        grouped[model].append(fig)

    for model, figs in sorted(grouped.items()):
        lines.append(f"[{model}]")
        for fig in figs:
            lines.append(f"  - {fig['filename']}")
            lines.append(f"    Description: {fig['description']}")
        lines.append("")

    lines.append("=" * 70)
    lines.append(f"Total figures: {len(figure_list)}")
    lines.append("=" * 70)

    return '\n'.join(lines)

if __name__ == '__main__':
    print("Starting results summarization...")

    # Step 1: Organize figures
    print("\n[1/3] Organizing figures...")
    figure_index = organize_figures()
    print(f"  Copied {len(figure_index)} figures to {FIGURES_DIR}")

    # Step 2: Generate summary
    print("\n[2/3] Generating evaluation summary...")
    summary_text = generate_summary()
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"  Summary saved to {SUMMARY_FILE}")

    # Step 3: Generate figure index
    print("\n[3/3] Generating figure index...")
    index_text = generate_figure_index(figure_index)
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        f.write(index_text)
    print(f"  Index saved to {INDEX_FILE}")

    print("\n✅ Summarization complete!")
    print(f"\nOutput files:")
    print(f"  - {SUMMARY_FILE}")
    print(f"  - {INDEX_FILE}")
    print(f"  - {FIGURES_DIR}/ ({len(figure_index)} figures)")
