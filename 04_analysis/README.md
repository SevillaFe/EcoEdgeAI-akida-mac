# Phase 5: Statistical Analysis

This directory contains analysis and visualization scripts.

## Workflow

### 1. Extract Statistics

```bash
# Mac M1 results
python extract_mac_std.py

# Akida NPU results
python extract_akida_std.py
```

**Outputs:**
- `mac_statistics_summary.csv`
- `akida_statistics_summary.csv`

### 2. Compare Platforms

```bash
python compare_benchmarks.py \
    --mac_dir ../results/benchmark_results_mac \
    --akida_dir ../results/benchmark_results_akida \
    --output_dir ../results/comparison
```

**Outputs:**
- `platform_comparison.csv`
- `energy_reduction_factors.csv`
- `accuracy_degradation.csv`

### 3. Generate Figures

```bash
python generate_all_figures_final.py \
    --results_dir ../results \
    --output_dir ../results/figures
```

**Generates:**
- Energy comparison (Fig. 2)
- Power consumption (Fig. 3)
- Latency comparison (Fig. 4)
- Accuracy degradation (Fig. 5)
- EER trade-off space (Fig. 6)
- Carbon footprint (Fig. 7)

### 4. Statistical Validation

```bash
python statistical_analysis_paper_v2.py \
    --mac_csv ../results/mac_statistics_summary.csv \
    --akida_csv ../results/akida_statistics_summary.csv \
    --output_dir ../results/statistical_analysis
```

**Statistical Tests:**
- Paired t-tests (Bonferroni correction, α=0.0167)
- Cohen's d effect sizes
- Two-way ANOVA (Platform × Architecture)
- Bootstrap confidence intervals (10,000 iterations)

**Outputs:**
- `energy_statistical_comparison.csv`
- `figure_energy_statistical.pdf`
- `figure_eer_statistical.pdf`
- `figure_mse_heatmap_statistical.pdf`
- `table_energy_statistics.tex`

## Key Metrics

### Energy Efficiency
- Energy per sample (mWh)
- Power consumption (W)
- Reduction factor (×)
- Savings percentage (%)

### Accuracy
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Degradation percentage (%)

### Eco-Efficiency
- Energy-Error Rate: `EER = 1 / (MSE × Energy)`
- Higher EER = better eco-efficiency

### Performance
- Throughput (samples/second)
- Latency (milliseconds)
- Standard deviation
- Coefficient of variation (%)

## LaTeX Table Generation

Statistical analysis generates publication-ready LaTeX tables:

```latex
\begin{table}[H]
\centering
\caption{Statistical Validation of Energy Efficiency Gains}
\begin{tabular}{lccccc}
\toprule
\textbf{Architecture} & \textbf{Mac (mWh)} & \textbf{Akida (mWh)} & ...
\midrule
Pilotnet & 35.3 & 4.9 & 7.15× & 273.8 & <0.001*** \\
...
\bottomrule
\end{tabular}
\end{table}
```

## Figure Customization

Edit `generate_all_figures_final.py` to customize:
- Color schemes
- Font sizes
- DPI (default: 300)
- Figure dimensions
- Annotation styles
