import json
from pathlib import Path
import csv
from datetime import datetime

# Directorio de resultados
results_dir = Path("/Users/fernando/Documents/Doctorado/Udacity_Dataset/paper_5/output/benchmark_results")
results_dir.mkdir(parents=True, exist_ok=True)

models = ['pilotnet', 'laksnet', 'mininet']

# Archivo de salida para texto
output_txt = results_dir / "mac_statistics_summary.txt"
output_csv = results_dir / "mac_statistics_summary.csv"

# Preparar datos
all_stats = []

print("="*80)
print("MAC M1 FLOAT32 - STATISTICS")
print("="*80)
print(f"{'Architecture':<15} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'CV (%)':<10}")
print("-"*80)

with open(output_txt, 'w') as f_txt:
    f_txt.write("="*80 + "\n")
    f_txt.write("MAC M1 FLOAT32 - STATISTICS\n")
    f_txt.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f_txt.write("="*80 + "\n\n")
    f_txt.write(f"{'Architecture':<15} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'CV (%)':<10}\n")
    f_txt.write("-"*80 + "\n")
    
    for model in models:
        #json_file = Path(f"./{model}_float32_unified_benchmark_results.json")
        json_file = Path(f"/Users/fernando/Documents/Doctorado/Udacity_Dataset/paper_5/output/benchmark_results/{model}_float32_unified_benchmark_results.json")
        
        if json_file.exists():
            with open(json_file) as f:
                data = json.load(f)
            
            mean = data['avg_latency_ms']
            std = data['std_latency_ms']
            min_lat = data['min_latency_ms']
            max_lat = data['max_latency_ms']
            cv = (std / mean * 100) if mean > 0 else 0
            num_samples = data.get('num_samples', None)
            inference_co2_kg = data.get('codecarbon_total_co2_kg', None)
            
            # Calcular CO2 por muestra (kg y g)
            co2_kg_per_sample = None
            co2_g_per_sample = None

            if inference_co2_kg is not None and num_samples not in (None, 0):
                co2_kg_per_sample = inference_co2_kg / num_samples
                co2_g_per_sample = co2_kg_per_sample * 1000
            
            # Imprimir en consola
            line = f"{model.capitalize():<15} {mean:<12.4f} {std:<12.4f} {min_lat:<12.4f} {max_lat:<12.4f} {cv:<10.2f}"
            print(line)
            f_txt.write(line + "\n")
            
            # Detalles adicionales
            details = [
                f"  → MSE: {data['mse']:.6f}, MAE: {data['mae']:.6f}",
                f"  → Energy: {data.get('energy_per_sample_mwh', 'N/A')} mWh, Power: {data.get('avg_inference_power_w', 'N/A')} W",
                f"  → Throughput: {data.get('throughput_samples_per_second', 'N/A')} samples/s",
                ""
            ]
            for detail in details:
                print(detail)
                f_txt.write(detail + "\n")
            
            # Guardar para CSV
            all_stats.append({
                'architecture': model.capitalize(),
                'platform': 'Mac M1 Pro',
                'mean_latency_ms': mean,
                'std_latency_ms': std,
                'min_latency_ms': min_lat,
                'max_latency_ms': max_lat,
                'cv_percent': cv,
                'mse': data['mse'],
                'mae': data['mae'],
                'total_energy_wh': data.get('total_energy_wh', None),
                'idle_energy_wh': data.get('idle_energy_wh', None),
                'inference_energy_wh': data.get('inference_energy_wh', None),
                'energy_per_sample_mwh': data.get('energy_per_sample_mwh', None),
                'avg_total_power_w': data.get('avg_total_power_w', None),
                'avg_inference_power_w': data.get('avg_inference_power_w', None),
                'idle_power_w': data.get('idle_power_w', None),
                'inference_co2_kg': data.get('inference_co2_kg' , None),
                'co2_g_per_sample': data.get('co2_g_per_sample', None),
                'throughput_sps': data.get('throughput_samples_per_second', None),
                'codecarbon_total_co2_kg': data.get('codecarbon_total_co2_kg', None), 
            })
            
        else:
            msg = f"{model.capitalize():<15} FILE NOT FOUND: {json_file}"
            print(msg)
            f_txt.write(msg + "\n\n")
    
    f_txt.write("="*80 + "\n")

print("="*80)

# Guardar CSV
with open(output_csv, 'w', newline='') as csvfile:
    if all_stats:
        fieldnames = all_stats[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)

print(f"\n✓ Resultados guardados:")
print(f"  - Texto: {output_txt}")
print(f"  - CSV:   {output_csv}")