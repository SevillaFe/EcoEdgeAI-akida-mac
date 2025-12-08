#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Benchmark for MacBook Pro (M1) - Float32 Models
========================================================
Compatible methodology with RPi5+Akida benchmark for fair comparison.

CRITICAL DIFFERENCES FROM ORIGINAL:
- Uses batch_size=1 to match Akida processing
- Measures idle power baseline to subtract system overhead
- More accurate energy attribution to inference only
- Consistent preprocessing and validation split

Usage:
python unified_benchmark_mac.py \
  --model /Users/fernando/Documents/Doctorado/Udacity_Dataset/paper_5/output/pilotnet/pilotnet_float32.h5 \
  --data_dir /Users/fernando/Documents/Doctorado/Udacity_Dataset/Original_Images \
  --output_dir /Users/fernando/Documents/Doctorado/Udacity_Dataset/paper_5/output/benchmark_results \
  --num_samples 1000 \
  --measure_idle \
  --idle_duration 5
"""

import argparse
import time
import json
from pathlib import Path
import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    print("⚠️ CodeCarbon not available - install with: pip install codecarbon")
    CODECARBON_AVAILABLE = False


# ============================================================================
# PREPROCESSING (IDENTICAL TO AKIDA)
# ============================================================================

def preprocess_image(img_path, grayscale=False):
    """
    Preprocess image - MUST match Akida preprocessing exactly
    
    Steps:
    1. Load BGR image
    2. Convert to RGB
    3. Crop: remove sky (60px top) and hood (25px bottom)
    4. Resize to (200, 66)
    5. Normalize to float32 [0, 1] for Keras
    
    Args:
        img_path: Path to image
        grayscale: Use grayscale (for some models)
    
    Returns:
        Preprocessed image as float32 [0, 1]
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[60:-25, :]
        img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[60:-25, :]
        img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
    
    return img


def load_validation_split(data_dir, num_samples=1000):
    """
    Load validation split - IDENTICAL to Akida script
    
    Args:
        data_dir: Dataset directory
        num_samples: Number of samples to load
    
    Returns:
        img_paths, steerings
    """
    csv_path = Path(data_dir) / "driving_log.csv"
    columns = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]
    df = pd.read_csv(csv_path, names=columns)
    
    # Use same validation split (last 20%)
    val_start = int(0.8 * len(df))
    df_val = df.iloc[val_start:].reset_index(drop=True)
    
    img_paths = []
    steerings = []
    
    for _, row in df_val.iterrows():
        if len(img_paths) >= num_samples:
            break
        center_filename = os.path.basename(str(row["center"]).strip())
        img_path = os.path.join(data_dir, "IMG", center_filename)
        if os.path.exists(img_path):
            img_paths.append(img_path)
            steerings.append(float(row["steering"]))
    
    steerings = np.array(steerings, dtype=np.float32)
    return img_paths, steerings


def preload_images_to_ram(img_paths, grayscale=False):
    """
    Preload images to RAM to exclude I/O from timing
    
    Args:
        img_paths: List of image paths
        grayscale: Use grayscale
    
    Returns:
        numpy array of preprocessed images
    """
    print(f"\n{'='*70}")
    print("PRELOADING IMAGES TO RAM")
    print('='*70)
    
    images = []
    for i, p in enumerate(img_paths):
        images.append(preprocess_image(p, grayscale=grayscale))
        if (i + 1) % 200 == 0:
            print(f"  Preloaded {i+1}/{len(img_paths)} images")
    
    arr = np.array(images, dtype=np.float32)
    print(f"\n✓ Preloaded images shape: {arr.shape}")
    print(f"  Memory: {arr.nbytes/1024/1024:.1f} MB")
    return arr


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_keras_model(model_path):
    """Load Keras model"""
    p = Path(model_path)
    if p.is_dir():
        model = keras.models.load_model(str(p))
    else:
        model = keras.models.load_model(str(p), compile=False)
    return model


# ============================================================================
# IDLE POWER MEASUREMENT
# ============================================================================

def measure_idle_power(duration_seconds=5):
    """
    Measure idle system power consumption
    This helps isolate inference power from system overhead
    
    Args:
        duration_seconds: How long to measure idle
    
    Returns:
        dict with idle energy metrics
    """
    if not CODECARBON_AVAILABLE:
        return {'idle_energy_kwh': 0.0, 'idle_power_w': 0.0}
    
    print(f"\n{'='*70}")
    print("MEASURING IDLE SYSTEM POWER")
    print('='*70)
    print(f"Duration: {duration_seconds} seconds")
    print("Please wait...")
    
    try:
        tracker = EmissionsTracker(
            project_name="idle_measurement",
            output_dir=".",
            save_to_file=False,
            log_level="error"
        )
        tracker.start()
        time.sleep(duration_seconds)
        emissions = tracker.stop()
        
        # Extract energy
        energy_kwh = 0.0
        try:
            fed = getattr(tracker, "final_emissions_data", None)
            if fed:
                energy_kwh = float(getattr(fed, "energy_consumed", 0.0))
            else:
                ed = getattr(tracker, "_emissions_data", None)
                if ed:
                    energy_kwh = float(getattr(ed, "energy_consumed", 0.0))
        except:
            pass
        
        idle_power_w = (energy_kwh * 1000.0 * 3600.0) / duration_seconds if duration_seconds > 0 else 0.0
        
        print(f"\n✓ Idle measurement complete")
        print(f"  Idle power: {idle_power_w:.2f} W")
        print(f"  Idle energy: {energy_kwh*1000:.4f} Wh")
        
        return {
            'idle_energy_kwh': energy_kwh,
            'idle_power_w': idle_power_w,
            'idle_duration_s': duration_seconds
        }
    
    except Exception as e:
        print(f"⚠️ Could not measure idle power: {e}")
        return {'idle_energy_kwh': 0.0, 'idle_power_w': 0.0}


# ============================================================================
# UNIFIED BENCHMARK (MATCHING AKIDA METHODOLOGY)
# ============================================================================

def benchmark_inference_unified(model, images, steerings, warmup_runs=10, 
                               idle_power_w=0.0, country_code='auto'):
    """
    Unified benchmark matching Akida methodology
    
    CRITICAL: Uses batch_size=1 to match Akida processing
    
    Args:
        model: Keras model
        images: Preloaded images (numpy array)
        steerings: Ground truth steering angles
        warmup_runs: Number of warmup runs
        idle_power_w: Idle system power to subtract
        country_code: Country code for emissions
    
    Returns:
        Dictionary with benchmark results
    """
    num_samples = len(steerings)
    predictions = []
    inference_times = []
    
    print(f"\n{'='*70}")
    print("UNIFIED BENCHMARK (AKIDA-COMPATIBLE)")
    print('='*70)
    print(f"Samples: {num_samples}")
    print(f"Batch size: 1 (matching Akida)")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Idle power: {idle_power_w:.2f} W (will be subtracted)")
    print('='*70)
    
    # Warmup
    print("\nWarming up...")
    warmup_img = np.expand_dims(images[0], axis=0)
    for _ in range(warmup_runs):
        _ = model.predict(warmup_img, verbose=0)
    print("✓ Warmup complete")
    
    # Start CodeCarbon tracker
    tracker = None
    codecarbon_emissions = 0.0
    total_energy_kwh = 0.0
    
    if CODECARBON_AVAILABLE:
        try:
            tracker = EmissionsTracker(
                project_name="mac_unified_benchmark",
                output_dir=".",
                save_to_file=False,
                log_level="error"
            )
            tracker.start()
            print("\n🔋 CodeCarbon tracker started")
        except Exception as e:
            print(f"⚠️ CodeCarbon error: {e}")
            tracker = None
    
    # Benchmark with batch_size=1 (matching Akida)
    print("\nRunning inference (batch_size=1)...")
    total_start = time.perf_counter()
    
    for i in range(num_samples):
        # Process ONE image at a time (matching Akida)
        img_batch = np.expand_dims(images[i], axis=0)
        
        t0 = time.perf_counter()
        pred = model.predict(img_batch, verbose=0)
        t1 = time.perf_counter()
        
        dt = t1 - t0
        inference_times.append(dt)
        
        # Extract scalar prediction (matching Akida)
        pred_scalar = float(pred.ravel()[0])
        predictions.append(pred_scalar)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{num_samples} samples")
    
    total_end = time.perf_counter()
    total_time = total_end - total_start
    
    # Stop CodeCarbon
    if tracker:
        try:
            codecarbon_emissions = tracker.stop() or 0.0
            fed = getattr(tracker, "final_emissions_data", None)
            if fed:
                total_energy_kwh = float(getattr(fed, "energy_consumed", 0.0))
            else:
                ed = getattr(tracker, "_emissions_data", None)
                if ed:
                    total_energy_kwh = float(getattr(ed, "energy_consumed", 0.0))
            print(f"🔋 CodeCarbon tracker stopped")
        except Exception as e:
            print(f"⚠️ Error stopping tracker: {e}")
    
    # Subtract idle power consumption
    # CodeCarbon measures total system, we want only inference overhead
    total_energy_wh = total_energy_kwh * 1000.0
    idle_energy_wh = (idle_power_w * total_time) / 3600.0
    inference_energy_wh = max(0.0, total_energy_wh - idle_energy_wh)
    
    # Calculate metrics
    predictions = np.array(predictions, dtype=np.float32)
    inference_times = np.array(inference_times, dtype=np.float64)
    
    throughput = num_samples / total_time
    avg_latency_ms = np.mean(inference_times) * 1000.0
    std_latency_ms = np.std(inference_times) * 1000.0
    
    mse = mean_squared_error(steerings, predictions)
    mae = mean_absolute_error(steerings, predictions)
    
    # Energy per sample
    energy_per_sample_mwh = (inference_energy_wh * 1000.0) / num_samples
    
    # Average inference power (excluding idle)
    avg_inference_power_w = (inference_energy_wh * 3600.0) / total_time if total_time > 0 else 0.0
    
    # Emission factor (from CodeCarbon)
    emission_factor = 0.0
    if total_energy_kwh > 0 and codecarbon_emissions > 0:
        emission_factor = codecarbon_emissions / total_energy_kwh
    
    # CO2 for inference only (subtract idle)
    inference_energy_kwh = inference_energy_wh / 1000.0
    inference_co2_kg = inference_energy_kwh * emission_factor if emission_factor > 0 else 0.0
    co2_g_per_sample = (inference_co2_kg * 1000.0) / num_samples if num_samples > 0 else 0.0
    
    # Results summary
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print('='*70)
    print(f"\nPerformance:")
    print(f"  Total time:      {total_time:.2f} seconds")
    print(f"  Throughput:      {throughput:.2f} samples/second")
    print(f"  Avg latency:     {avg_latency_ms:.2f} ± {std_latency_ms:.2f} ms")
    print(f"\nAccuracy:")
    print(f"  MSE:             {mse:.4f}")
    print(f"  MAE:             {mae:.4f}")
    print(f"\nEnergy (CodeCarbon with idle subtraction):")
    print(f"  Total measured:  {total_energy_wh:.4f} Wh")
    print(f"  Idle baseline:   {idle_energy_wh:.4f} Wh")
    print(f"  Inference only:  {inference_energy_wh:.4f} Wh ← USE THIS")
    print(f"  Avg power:       {avg_inference_power_w:.2f} W (inference only)")
    print(f"  Per sample:      {energy_per_sample_mwh:.6f} mWh")
    print(f"\nEmissions:")
    print(f"  CO2 (inference): {inference_co2_kg:.6f} kgCO2eq")
    print(f"  Per sample:      {co2_g_per_sample:.6f} gCO2")
    print(f"  Emission factor: {emission_factor:.4f} kgCO2/kWh")
    print('='*70)
    
    return {
        'num_samples': int(num_samples),
        'total_time_seconds': float(total_time),
        'throughput_samples_per_second': float(throughput),
        'avg_latency_ms': float(avg_latency_ms),
        'std_latency_ms': float(std_latency_ms),
        'min_latency_ms': float(np.min(inference_times) * 1000.0),
        'max_latency_ms': float(np.max(inference_times) * 1000.0),
        'mse': float(mse),
        'mae': float(mae),
        'predictions': predictions.tolist(),
        'ground_truth': steerings.tolist(),
        'inference_times_ms': (inference_times * 1000.0).tolist(),
        # Energy metrics
        'total_energy_wh': float(total_energy_wh),
        'idle_energy_wh': float(idle_energy_wh),
        'inference_energy_wh': float(inference_energy_wh),
        'energy_per_sample_mwh': float(energy_per_sample_mwh),
        'avg_inference_power_w': float(avg_inference_power_w),
        'idle_power_w': float(idle_power_w),
        # Emissions
        'codecarbon_total_co2_kg': float(codecarbon_emissions),
        'inference_co2_kg': float(inference_co2_kg),
        'co2_g_per_sample': float(co2_g_per_sample),
        'emission_factor_kg_per_kwh': float(emission_factor),
        'country_code': str(country_code),
        'energy_source': 'codecarbon_with_idle_subtraction'
    }


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(output_dir, model_name, results):
    """Save benchmark results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full results JSON
    results_file = output_dir / f"{model_name}_unified_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved: {results_file}")
    
    # Summary CSV (compatible with Akida format)
    summary = {
        'model_name': model_name,
        'hardware': 'MacBook Pro (Apple Silicon)',
        'mapping_success': True,
        'input_mode': 'float32',
        'batch_size': 1,  # Important: document batch size used
        'num_samples': results['num_samples'],
        'throughput_sps': results['throughput_samples_per_second'],
        'avg_latency_ms': results['avg_latency_ms'],
        'mse': results['mse'],
        'mae': results['mae'],
        # Energy (inference only, idle subtracted)
        'total_energy_wh': results['inference_energy_wh'],
        'energy_per_sample_mwh': results['energy_per_sample_mwh'],
        'avg_power_w': results['avg_inference_power_w'],
        'idle_power_w': results['idle_power_w'],
        # Emissions
        'co2_kg': results['inference_co2_kg'],
        'co2_g_per_sample': results['co2_g_per_sample'],
        'emission_factor_kg_per_kwh': results['emission_factor_kg_per_kwh'],
        'country_code': results['country_code'],
        'energy_source': 'codecarbon_idle_subtracted'
    }
    
    summary_file = output_dir / f"{model_name}_unified_benchmark_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_file, index=False)
    print(f"✓ Summary saved: {summary_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Benchmark for Mac (compatible with Akida methodology)"
    )
    parser.add_argument("--model", required=True, help="Path to Keras model")
    parser.add_argument("--data_dir", required=True, help="Dataset directory")
    parser.add_argument("--output_dir", default="./benchmark_results", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--warmup_runs", type=int, default=10, help="Warmup runs")
    parser.add_argument("--measure_idle", action="store_true", 
                       help="Measure idle power before benchmark")
    parser.add_argument("--idle_duration", type=int, default=5,
                       help="Idle measurement duration (seconds)")
    parser.add_argument("--grayscale", action="store_true", help="Use grayscale images")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("UNIFIED MAC BENCHMARK (AKIDA-COMPATIBLE)")
    print('='*70)
    print(f"Model:       {args.model}")
    print(f"Dataset:     {args.data_dir}")
    print(f"Samples:     {args.num_samples}")
    print(f"Batch size:  1 (matching Akida)")
    print('='*70)
    
    # Measure idle power if requested
    idle_power_w = 0.0
    if args.measure_idle:
        idle_results = measure_idle_power(args.idle_duration)
        idle_power_w = idle_results['idle_power_w']
    
    # Load model
    print("\nLoading model...")
    model = load_keras_model(args.model)
    print(f"✓ Model loaded: {model.input_shape} → {model.output_shape}")
    
    # Load validation data
    img_paths, steerings = load_validation_split(args.data_dir, args.num_samples)
    
    # Preload images
    images = preload_images_to_ram(img_paths, grayscale=args.grayscale)
    
    # Run unified benchmark
    results = benchmark_inference_unified(
        model=model,
        images=images,
        steerings=steerings,
        warmup_runs=args.warmup_runs,
        idle_power_w=idle_power_w,
        country_code='auto'
    )
    
    # Save results
    model_name = Path(args.model).stem
    save_results(args.output_dir, model_name, results)
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print('='*70)


if __name__ == "__main__":
    main()