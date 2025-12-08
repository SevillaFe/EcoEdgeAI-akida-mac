import json
from pathlib import Path
import csv
from datetime import datetime

# Directorio de resultados
results_dir = Path("/Users/fernando/Documents/Doctorado/Udacity_Dataset/paper_5/output/benchmark_results_Akida")
results_dir.mkdir(parents=True, exist_ok=True)

models = ['pilotnet', 'laksnet', 'mininet']

# Archivos de salida
output_txt = results_dir / "akida_statistics_summary.txt"
output_csv = results_dir / "akida_statistics_summary.csv"

# Preparar datos
all_stats = []

print("="*80)
print("AKIDA PTQ 4-BIT - STATISTICS")
print("="*80)
print(f"{'Architecture':<15} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'CV (%)':<10}")
print("-"*80)

with open(output_txt, 'w') as f_txt:
    f_txt.write("="*80 + "\n")
    f_txt.write("AKIDA PTQ 4-BIT - STATISTICS\n")
    f_txt.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f_txt.write("="*80 + "\n\n")
    f_txt.write(f"{'Architecture':<15} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'CV (%)':<10}\n")
    f_txt.write("-"*80 + "\n")
    
    for model in models:
        # Probar varias rutas posibles
        json_paths = [
            Path(f"/Users/fernando/Documents/Doctorado/Udacity_Dataset/paper_5/output/benchmark_results_Akida/{model}/{model}_best_q4_unified_benchmark_results.json")
            ]
        
        json_file = None
        for path in json_paths:
            if path.exists():
                json_file = path
                break
        
        if json_file:
            with open(json_file) as f:
                data = json.load(f)
            
            mean = data['avg_latency_ms']
            std = data['std_latency_ms']
            min_lat = data['min_latency_ms']
            max_lat = data['max_latency_ms']
            cv = (std / mean * 100) if mean > 0 else 0
            
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
                'platform': 'RPi5 + Akida 1.0',
                'quantization': '4-bit PTQ',
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
                'inference_co2_kg': data.get('inference_co2_kg', None),
                'co2_g_per_sample': data.get('co2_g_per_sample', None),
                'throughput_sps': data.get('throughput_samples_per_second', None),
                'codecarbon_total_co2_kg': data.get('codecarbon_total_co2_kg', None),  
            })
            
        else:
            msg = f"{model.capitalize():<15} FILE NOT FOUND"
            print(msg)
            f_txt.write(msg + "\n")
            f_txt.write(f"  Tried: {[str(p) for p in json_paths]}\n\n")
    
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