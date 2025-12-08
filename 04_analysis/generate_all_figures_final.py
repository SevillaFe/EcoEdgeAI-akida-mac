# generate_all_figures_final.py
"""
Generate all figures in the paper with error bars
Read data directly from the generated CSVs
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from pyparsing import col
from sympy import N

# Matplotlib configuration
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# ============================================================
# FOLDER
# ============================================================
base_dir = Path("./output")
output_dir = base_dir / "figures_final_std"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD CSV data
# ============================================================
print("Loading CSVs...")

# Reading data from Mac M1
mac_csv = base_dir / "benchmark_results"/ "mac_statistics_summary.csv"
akida_csv = base_dir / "benchmark_results_Akida" / "akida_statistics_summary.csv"

# Check if they exist
if not mac_csv.exists():
    raise FileNotFoundError(f"No se encuentra: {mac_csv}")
if not akida_csv.exists():
    raise FileNotFoundError(f"No se encuentra: {akida_csv}")

# Load CSVs
df_mac = pd.read_csv(mac_csv)
df_akida = pd.read_csv(akida_csv)

print(f"  ✓ Mac M1 data: {len(df_mac)} arquitecturas")
print(f"  ✓ Akida data: {len(df_akida)} arquitecturas")

# ============================================================
# ORGANIZE DATA BY ARCHITECTURE
# ============================================================
architectures = ['pilotnet', 'laksnet', 'mininet']
arch_labels = ['PilotNet', 'LaksNet', 'MiniNet']

# Create data dictionary
data = {}

for arch in architectures:
    # Buscar fila correspondiente
    mac_row = df_mac[df_mac['architecture'].str.lower() == arch.lower()]
    akida_row = df_akida[df_akida['architecture'].str.lower() == arch.lower()]
    
    if mac_row.empty or akida_row.empty:
        print(f"   WARNING: No se encontraron datos para {arch}")
        continue
    
    # Extraer valores (primer match)
    mac_row = mac_row.iloc[0]
    akida_row = akida_row.iloc[0]
    
    data[arch] = {
        'mac': {
            'latency_ms': mac_row['mean_latency_ms'],
            'latency_std': mac_row['std_latency_ms'],
            'mse': mac_row['mse'],
            'mae': mac_row['mae'],
            'energy_mwh': mac_row['energy_per_sample_mwh'],
            'power_w': mac_row['avg_inference_power_w'] if pd.notna(mac_row['avg_inference_power_w']) else mac_row['idle_power_w'],
            'throughput_sps': mac_row['throughput_sps'] if pd.notna(mac_row['throughput_sps']) else 0,
            'co2_g_per_sample': mac_row['co2_g_per_sample'] if pd.notna(mac_row['co2_g_per_sample']) else 0,
            
        },
        'akida': {
            'latency_ms': akida_row['mean_latency_ms'],
            'latency_std': akida_row['std_latency_ms'],
            'mse': akida_row['mse'],
            'mae': akida_row['mae'],
            'energy_mwh': akida_row['energy_per_sample_mwh'],
            'power_w': akida_row['avg_inference_power_w'] if pd.notna(akida_row['avg_inference_power_w']) else akida_row['idle_power_w'],
            'throughput_sps': akida_row['throughput_sps'] if pd.notna(akida_row['throughput_sps']) else 0,
            'co2_g_per_sample': akida_row['co2_g_per_sample'] if pd.notna(akida_row['co2_g_per_sample']) else 0,
        }
    }

print(f"\n  Data organized for {len(data)} architectures\n")

# MShow summary
for arch in architectures:
    if arch in data:
        print(f"{arch.upper()}:")
        print(f"  Mac:   Latency={data[arch]['mac']['latency_ms']:.2f}±{data[arch]['mac']['latency_std']:.2f}ms, "
              f"Energy={data[arch]['mac']['energy_mwh']:.2f}mWh, Power={data[arch]['mac']['power_w']:.2f}W")
        print(f"  Akida: Latency={data[arch]['akida']['latency_ms']:.2f}±{data[arch]['akida']['latency_std']:.2f}ms, "
              f"Energy={data[arch]['akida']['energy_mwh']:.2f}mWh, Power={data[arch]['akida']['power_w']:.2f}W")
        
        # Calculate reductions
        energy_reduction = data[arch]['mac']['energy_mwh'] / data[arch]['akida']['energy_mwh']
        power_reduction = data[arch]['mac']['power_w'] / data[arch]['akida']['power_w']
        speedup = data[arch]['mac']['latency_ms'] / data[arch]['akida']['latency_ms']
        print(f"  → Energy: {energy_reduction:.1f}× reduction, Power: {power_reduction:.1f}× reduction, "
              f"Speed: {speedup:.1f}× faster\n")

print("="*70 + "\n")

# ============================================================
# AXIS STYLE OPTIONS
# Choose: "light", "ultra_light", "minimal"
# ============================================================

def apply_axis_style(ax, style="light"):
    """
    Aplica estilos reutilizables a los ejes de Matplotlib.
    style: "light", "ultra_light", "minimal"
    """
    if style == "light":
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(1)

    elif style == "ultra_light":
        for spine in ax.spines.values():
            spine.set_color('#e0e0e0')
            spine.set_linewidth(0.8)

    elif style == "minimal":
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')


# ============================================================
# FIGURE 1: Energy Comparison
# ============================================================
print("Generando Figure 1: Energy Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))
apply_axis_style(ax, style="minimal")

x = np.arange(len(architectures))
width = 0.4

mac_energy = [data[arch]['mac']['energy_mwh'] for arch in architectures if arch in data]
akida_energy = [data[arch]['akida']['energy_mwh'] for arch in architectures if arch in data]

bars1 = ax.bar(x - width/1.8, mac_energy, width,
               label='Mac M1 Pro', capsize=5, alpha=0.7, color='#3498db', 
               edgecolor='black')
bars2 = ax.bar(x + width/1.8, akida_energy, width,
               label='RPi5 + Akida 1.0', capsize=5, alpha=0.7, color='#e74c3c',
               edgecolor='black')

# Reduction annotations
for i, arch in enumerate(architectures):
    if arch not in data:
        continue
    mac_e = data[arch]['mac']['energy_mwh']
    akida_e = data[arch]['akida']['energy_mwh']
    reduction = mac_e / akida_e if akida_e > 0 else 0
    pct = (mac_e - akida_e) / mac_e * 100 if mac_e > 0 else 0
    
    y_pos = max(mac_e, akida_e) * 1.1
    ax.text(i, y_pos, f'{reduction:.1f}×\n({pct:.0f}%↓)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# add value labels on top of bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=12)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=12)


ax.set_ylabel('Energy per Sample (mWh)', fontsize=14, fontweight='bold')
ax.set_yticks([])
#ax.set_xlabel('Architecture', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(arch_labels, fontsize=12)
ax.legend(fontsize=12, loc='upper center', framealpha=0.9, ncol=2 )
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(max(mac_energy), max(akida_energy)) * 1.3)

plt.tight_layout()
output_path = output_dir / 'figure_energy_comparison.pdf'
plt.savefig(output_path)
plt.savefig(output_dir / 'figure_energy_comparison.jpg')
plt.close()
print(f"  ✓ Guardada: {output_path}")

# ============================================================
# FIGURE 2: Latency Comparison (CON ERROR BARS)
# ============================================================
print("Generating Figure 2: Latency Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))
apply_axis_style(ax, style="minimal")

mac_latency = [data[arch]['mac']['latency_ms'] for arch in architectures if arch in data]
mac_latency_std = [data[arch]['mac']['latency_std'] for arch in architectures if arch in data]

akida_latency = [data[arch]['akida']['latency_ms'] for arch in architectures if arch in data]
akida_latency_std = [data[arch]['akida']['latency_std'] for arch in architectures if arch in data]

bars1 = ax.bar(x - width/1.8, mac_latency, width, yerr=mac_latency_std,
               label='Mac M1 Pro', capsize=5, alpha=0.7, color='#3498db',
               edgecolor='black',  error_kw={'linewidth': 1.5})
bars2 = ax.bar(x + width/1.8, akida_latency, width, yerr=akida_latency_std,
               label='RPi5 + Akida 1.0', capsize=5, alpha=0.7, color='#e74c3c',
               edgecolor='black', error_kw={'linewidth': 1.5})

# Speedup annotations
for i, arch in enumerate(architectures):
    if arch not in data:
        continue
    mac_l = data[arch]['mac']['latency_ms']
    akida_l = data[arch]['akida']['latency_ms']
    speedup = mac_l / akida_l if akida_l > 0 else 0
    pct = (mac_l - akida_l) / mac_l * 100 if mac_l > 0 else 0
    
    y_pos = max(mac_l + mac_latency_std[i], akida_l + akida_latency_std[i]) * 1.0
    ax.text(i, y_pos, f'{speedup:.1f}× faster\n({pct:.0f}%↓)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# add value labels on top of bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            height/2,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=12)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            height/2.2,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=12)


ax.set_ylabel('Inference Latency (ms)', fontsize=14, fontweight='bold')
ax.set_yticks([])
#ax.set_xlabel('Architecture', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(arch_labels, fontsize=12)
ax.legend(fontsize=12, loc='upper center', framealpha=0.9, ncol=2 )
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max([m + s for m, s in zip(mac_latency, mac_latency_std)]) * 1.25)

plt.tight_layout()
output_path = output_dir / 'figure_latency_comparison.pdf'
plt.savefig(output_path)
plt.savefig(output_dir / 'figure_latency_comparison.jpg')
plt.close()
print(f"  ✓ Guardada: {output_path}")

# ============================================================
# FIGURE 3: Power Consumption Comparison
# ============================================================
print("Generating Figure 3: Power Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))
apply_axis_style(ax, style="minimal")

mac_power = [data[arch]['mac']['power_w'] for arch in architectures if arch in data]
akida_power = [data[arch]['akida']['power_w'] for arch in architectures if arch in data]

bars1 = ax.bar(x - width/2, mac_power, width,
               label='Mac M1 Pro', capsize=5, alpha=0.7, color='#3498db',
               edgecolor='black')
bars2 = ax.bar(x + width/2, akida_power, width,
               label='RPi5 + Akida 1.0', capsize=5, alpha=0.7, color='#e74c3c',
               edgecolor='black')

# Power reduction annotations
for i, arch in enumerate(architectures):
    if arch not in data:
        continue
    mac_p = data[arch]['mac']['power_w']
    akida_p = data[arch]['akida']['power_w']
    reduction = mac_p / akida_p if akida_p > 0 else 0
    pct = (mac_p - akida_p) / mac_p * 100 if mac_p > 0 else 0
    
    y_pos = max(mac_p, akida_p) * 1.1
    ax.text(i, y_pos, f'{reduction:.1f}×\n({pct:.0f}%↓)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# add value labels on top of bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=12)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=12)


ax.set_ylabel('Average Power (W)', fontsize=14, fontweight='bold')
ax.set_yticks([])
#ax.set_xlabel('Architecture', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(arch_labels, fontsize=12)
ax.legend(fontsize=12, loc='upper center', framealpha=0.9, ncol=2 )
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(mac_power) * 1.25)

plt.tight_layout()
output_path = output_dir / 'figure_power_comparison.pdf'
plt.savefig(output_path)
plt.savefig(output_dir / 'figure_power_comparison.jpg')
plt.close()
print(f"  ✓ Guardada: {output_path}")

# ============================================================
# FIGURE 4: Gramms CO2 emission per sample Comparison
# ============================================================
print("Generating Figure 4: Gramms CO2 emission per sample Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))
apply_axis_style(ax, style="minimal")

# Convertimos de g → mg
mac_co2 = [data[arch]['mac']['co2_g_per_sample'] * 1000 for arch in architectures if arch in data] 
akida_co2 = [data[arch]['akida']['co2_g_per_sample'] * 1000 for arch in architectures if arch in data] 

bars1 = ax.bar(x - width/2, mac_co2, width,
               label='Mac M1 Pro', capsize=5, alpha=0.7, color='#3498db',
               edgecolor='black')
bars2 = ax.bar(x + width/2, akida_co2, width,
               label='RPi5 + Akida 1.0', capsize=5, alpha=0.7, color='#e74c3c',
               edgecolor='black')

# Power reduction annotations
for i, arch in enumerate(architectures):
    if arch not in data:
        continue
    mac_c = data[arch]['mac']['co2_g_per_sample'] * 1000
    akida_c = data[arch]['akida']['co2_g_per_sample'] * 1000
    reduction = mac_c / akida_c if akida_c > 0 else 0
    pct = (mac_c - akida_c) / mac_c * 100 if mac_c > 0 else 0

    y_pos = max(mac_c, akida_c) * 1.1
    ax.text(i, y_pos, f'{reduction:.1f}×\n({pct:.0f}%↓)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# add value labels on top of bars (en mg)
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.4}',
            ha='center', va='bottom', fontsize=12)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=12)

ax.set_ylabel('Average CO2 Emission (mg/sample)', fontsize=14, fontweight='bold')
ax.set_yticks([])
ax.set_xticks(x)
ax.set_xticklabels(arch_labels, fontsize=12)
ax.legend(fontsize=12, loc='upper center', framealpha=0.9, ncol=2)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(max(mac_co2), max(akida_co2)) * 1.25)

plt.tight_layout()
output_path = output_dir / 'figure_co2_comparison.pdf'
plt.savefig(output_path)
plt.savefig(output_dir / 'figure_co2_comparison.jpg')
plt.close()
print(f"  ✓ Guardada: {output_path}")


# ============================================================
# FIGURE 5: Energy-Accuracy Trade-off (Scatter)
# ============================================================
print("Generating Figure 5: Energy-Accuracy Tradeoff...")

fig, ax = plt.subplots(figsize=(10, 8))
apply_axis_style(ax, style="minimal")

colors = {'pilotnet': '#E63946', 'laksnet': '#3498db', 'mininet': "#C5D609"}

for arch, label in zip(architectures, arch_labels):
    if arch not in data:
        continue
    
    # Mac (círculos)
    ax.scatter(data[arch]['mac']['energy_mwh'], 
               data[arch]['mac']['mse'],
               marker='o', s=200, label=f'{label} (Mac)',
               color=colors[arch], alpha=0.7, edgecolors='black')
    
    # Akida (cuadrados)
    ax.scatter(data[arch]['akida']['energy_mwh'],
               data[arch]['akida']['mse'],
               marker='s', s=200, label=f'{label} (Akida)',
               color=colors[arch], alpha=0.7, edgecolors='black')
    
    # Flecha de Mac a Akida
    ax.annotate('', 
                xy=(data[arch]['akida']['energy_mwh'], data[arch]['akida']['mse']),
                xytext=(data[arch]['mac']['energy_mwh'], data[arch]['mac']['mse']),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=colors[arch], alpha=0.5))

ax.set_xlabel('Energy per Sample (mWh)', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean Squared Error', fontsize=14, fontweight='bold')
#ax.set_title('Energy-Accuracy Trade-off Space', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=10, loc='upper center', ncol=3, framealpha=0.9)
#ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = output_dir / 'figure_energy_accuracy_tradeoff.pdf'
plt.savefig(output_path)
plt.savefig(output_dir / 'figure_energy_accuracy_tradeoff.jpg')
plt.close()
print(f"  ✓ Guardada: {output_path}")



# ============================================================
# FIGURE 6: Accuracy Degradation Bar Chart
# ============================================================
print("Generating Figure 6: Accuracy Degradation...")

fig, ax = plt.subplots(figsize=(10, 6))
apply_axis_style(ax, style="minimal")

mac_mse = [data[arch]['mac']['mse'] for arch in architectures if arch in data]
akida_mse = [data[arch]['akida']['mse'] for arch in architectures if arch in data]

bars1 = ax.bar(x - width/1.8, mac_mse, width,
               label='Mac M1 (Float32)', capsize=5, alpha=0.7, color='#3498db',
               edgecolor='black')
bars2 = ax.bar(x + width/1.8, akida_mse, width,
               label='Akida (4-bit PTQ)', capsize=5, alpha=0.7, color='#e74c3c',
               edgecolor='black')

# MSE degradation annotations
for i, arch in enumerate(architectures):
    if arch not in data:
        continue
    mac_m = data[arch]['mac']['mse']
    akida_m = data[arch]['akida']['mse']
    degradation = akida_m / mac_m if mac_m > 0 else 0
    pct = (akida_m - mac_m) / mac_m * 100 if mac_m > 0 else 0
    

    y_pos = max(mac_m, akida_m) * 1.07
    ax.text(i, y_pos, f'{degradation:.1f}×\n({pct:.0f}%↓)',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            color='darkred')

# add value labels on top of bars (en mg)
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=12)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=12)
    

ax.set_ylabel('Mean Squared Error', fontsize=14, fontweight='bold')
ax.set_yticks([])
#ax.set_xlabel('Architecture', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(arch_labels, fontsize=12)
ax.legend(fontsize=12, loc='upper center', framealpha=0.9, ncol=2)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(akida_mse) * 1.25)

plt.tight_layout()
output_path = output_dir / 'figure_accuracy_degradation.pdf'
plt.savefig(output_path)
plt.savefig(output_dir / 'figure_accuracy_degradation.jpg')
plt.close()
print(f"   Guardada: {output_path}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print(" ALL FIGURES SUCCESSFULLY GENERATED")
print("="*70)
print(f"Directory: {output_dir}")
print("\nFigures generated:")
print("  1. figure_energy_comparison.pdf/.jpg")
print("  2. figure_latency_comparison.pdf/.jpg (CON ERROR BARS)")
print("  3. figure_power_comparison.pdf/.jpg")
print("  4. figure_co2_comparison.pdf/.jpg")
print("  5. figure_energy_accuracy_tradeoff.pdf/.jpg")
print("  6. figure_accuracy_degradation.pdf/.jpg")
print("="*70)
