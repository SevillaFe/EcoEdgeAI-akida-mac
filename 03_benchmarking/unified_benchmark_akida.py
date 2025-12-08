#!/usr/bin/env python3
"""
Unified Benchmark for RPi5 + Akida NPU (FIXED TC66)
====================================================
Compatible methodology with Mac benchmark for fair comparison.
Uses TC66C library for proper power measurement.

Dependencies:
    pip install TC66C

Usage:
python unified_benchmark_akida.py \
    --akida_model ./akida_models/pilotnet_akida/pilotnet_best_q4_akida.fbz \
    --data_dir ./Original_Images \
    --output_dir ./akida_models/benchmark_results \
    --num_samples 1000 \
    --tc66_port /dev/ttyACM0 \
    --measure_idle
"""

import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import cv2
import os
import json
import sys
import threading
from collections import deque

try:
    import akida
    from akida import Model
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

from sklearn.metrics import mean_squared_error, mean_absolute_error

# TC66C Library
TC66C_AVAILABLE = False
try:
    from TC66C import TC66C

    TC66C_AVAILABLE = True
except ImportError:
    print("\n⚠️ Warning: TC66C not installed")
    print("   Install with: pip install TC66C")
    print("   Running without power measurement...\n")


# ============================================================================
# TC66C POWER MEASUREMENT
# ============================================================================

class TC66CRecorder:
    """TC66 power meter recorder using TC66C library"""

    def __init__(self, port='/dev/ttyACM0', poll_interval=0.1):
        self.port = port
        self.poll_interval = float(poll_interval)
        self.tc66 = None
        self.recording = False
        self.thread = None
        self.measurements = deque()

    def connect(self):
        """Connect to TC66"""
        if not TC66C_AVAILABLE:
            print("TC66C library not available")
            return False

        try:
            self.tc66 = TC66C(self.port)
            dev = self.tc66.Poll()
            print(f"✓ TC66 connected on {self.port}")
            print(f"  Device: {dev.Name} v{dev.Version}")
            print(f"  Current: {dev.Volt:.3f}V, {dev.Current * 1000:.1f}mA, {dev.Power:.3f}W")
            return True
        except Exception as e:
            print(f"Failed to connect TC66: {e}")
            return False

    def _measurement_loop(self):
        """Background thread to collect measurements"""
        t0 = time.time()
        while self.recording:
            try:
                data = self.tc66.Poll()
                ts = time.time() - t0
                self.measurements.append({
                    "t": ts,
                    "v": data.Volt,
                    "i": data.Current,
                    "p": data.Power,
                    "energy_mwh": data.G0_mWh,
                    "temp": data.Temp
                })
                time.sleep(self.poll_interval)
            except Exception as e:
                print(f"TC66 poll error: {e}")
                break

    def start(self):
        """Start recording"""
        if not self.tc66:
            raise RuntimeError("TC66 not connected")

        self.measurements.clear()
        self.recording = True
        self.thread = threading.Thread(target=self._measurement_loop, daemon=True)
        self.thread.start()
        print(f" TC66 recording started ({int(self.poll_interval * 1000)}ms)")

    def stop(self):
        """Stop recording and return measurements"""
        self.recording = False
        if self.thread:
            self.thread.join(timeout=2.0)

        meas = list(self.measurements)
        print(f" TC66 recording stopped ({len(meas)} samples)")
        return meas

    @staticmethod
    def summarize(measurements):
        """Compute summary statistics from measurements"""
        if not measurements:
            return None

        powers = np.array([m["p"] for m in measurements])
        times = np.array([m["t"] for m in measurements])

        # Energy by trapezoidal integration
        energy_wh = 0.0
        if len(measurements) > 1:
            dt = np.diff(times)
            p_avg = (powers[:-1] + powers[1:]) / 2.0
            energy_wh = float(np.sum(p_avg * dt) / 3600.0)

        # TC66 internal energy counter
        tc66_energy_wh = None
        if measurements[0].get("energy_mwh") is not None:
            delta_mwh = measurements[-1]["energy_mwh"] - measurements[0]["energy_mwh"]
            tc66_energy_wh = float(delta_mwh / 1000.0)

        return {
            "num_samples": len(measurements),
            "duration_sec": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
            "avg_power_w": float(np.mean(powers)),
            "max_power_w": float(np.max(powers)),
            "min_power_w": float(np.min(powers)),
            "integrated_energy_wh": float(energy_wh),
            "tc66_energy_wh": float(tc66_energy_wh) if tc66_energy_wh else None,
        }


def measure_idle_power_tc66(tc66_port, duration_seconds=10):
    """
    Measure idle system power with TC66C

    Args:
        tc66_port: TC66 serial port
        duration_seconds: Measurement duration

    Returns:
        dict with idle power metrics
    """
    if not TC66C_AVAILABLE:
        return {'idle_power_w': 0.0}

    print(f"\n{'=' * 70}")
    print("MEASURING IDLE SYSTEM POWER (TC66)")
    print('=' * 70)
    print(f"Duration: {duration_seconds} seconds")
    print("System idle - no inference running...")

    recorder = TC66CRecorder(tc66_port, poll_interval=0.1)
    if not recorder.connect():
        print("Could not connect to TC66")
        return {'idle_power_w': 0.0}

    time.sleep(0.5)
    recorder.start()
    time.sleep(duration_seconds)
    measurements = recorder.stop()

    if not measurements:
        print("No measurements collected")
        return {'idle_power_w': 0.0}

    summary = TC66CRecorder.summarize(measurements)
    idle_power_w = summary['avg_power_w']

    print(f"\nIdle measurement complete")
    print(f"  Samples:     {len(measurements)}")
    print(f"  Idle power:  {idle_power_w:.2f} W")
    print(f"  Range:       {summary['min_power_w']:.2f} - {summary['max_power_w']:.2f} W")

    return {
        'idle_power_w': float(idle_power_w),
        'idle_samples': len(measurements),
        'idle_duration_s': duration_seconds
    }


# ============================================================================
# PREPROCESSING (IDENTICAL TO MAC)
# ============================================================================

def preprocess_image(img_path):
    """Preprocess image - IDENTICAL to Mac version"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[60:-25, :]
    img = cv2.resize(img, (200, 66))
    img = img.astype(np.uint8)
    return img


def load_validation_data(data_dir, num_samples=1000):
    """Load validation data"""
    print(f"\n{'=' * 70}")
    print("LOADING VALIDATION DATA")
    print('=' * 70)

    csv_path = Path(data_dir) / 'driving_log.csv'
    columns = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]
    df = pd.read_csv(csv_path, names=columns)

    val_start = int(len(df) * 0.8)
    df_val = df.iloc[val_start:].reset_index(drop=True)

    img_paths = []
    steerings = []

    for idx, row in df_val.iterrows():
        if len(img_paths) >= num_samples:
            break

        center_filename = os.path.basename(row['center'].strip())
        center_path = os.path.join(data_dir, 'IMG', center_filename)

        if os.path.exists(center_path):
            img_paths.append(center_path)
            steerings.append(float(row['steering']))

    steerings = np.array(steerings, dtype=np.float32)

    print(f"\nLoaded {len(img_paths)} validation samples")
    print(f"  Steering range: [{steerings.min():.3f}, {steerings.max():.3f}]")

    return img_paths, steerings


def preload_images_to_ram(img_paths):
    """Preload images to RAM"""
    print(f"\n{'=' * 70}")
    print("PRELOADING IMAGES TO RAM")
    print('=' * 70)

    images = []
    for i, img_path in enumerate(img_paths):
        img = preprocess_image(img_path)
        images.append(img)

        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{len(img_paths)} images")

    images = np.array(images, dtype=np.uint8)
    memory_mb = images.nbytes / (1024 * 1024)

    print(f"\n✓ Images preloaded")
    print(f"  Shape: {images.shape}")
    print(f"  Memory: {memory_mb:.1f} MB")

    return images


# ============================================================================
# AKIDA MODEL LOADING
# ============================================================================

def detect_akida_device():
    """Detect Akida device"""
    print(f"\n{'=' * 70}")
    print("DETECTING AKIDA DEVICE")
    print('=' * 70)

    devices = akida.devices()

    if len(devices) == 0:
        print("NO AKIDA DEVICE DETECTED!")
        print("Inference will run in CPU emulation (VERY SLOW)")
        return None

    device = devices[0]
    print(f"✓ Device found: {device}")

    try:
        print(f"  Version: {device.version}")
        print(f"  SoC: {device.soc}")
    except:
        pass

    return device


def load_akida_model(model_path, device):
    """Load Akida model and map to hardware"""
    print(f"\n{'=' * 70}")
    print("LOADING AKIDA MODEL")
    print('=' * 70)
    print(f"Model: {model_path.name}")

    try:
        model = Model(str(model_path))
        print(f"Model loaded")
        print(f"  Input:  {model.input_shape}")
        print(f"  Output: {model.output_shape}")
        print(f"  Layers: {len(model.layers)}")

        mapping_success = False
        if device is not None:
            print(f"\nMapping to hardware...")
            try:
                model.map(device)
                print("✓ Model mapped to Akida NPU")
                mapping_success = True
            except RuntimeError as e:
                print(f"Mapping failed: {e}")
                print("Model will run in CPU emulation")
        else:
            print("\nNo device detected, running in CPU emulation")

        return model, mapping_success

    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)


# ============================================================================
# UNIFIED BENCHMARK
# ============================================================================

def benchmark_inference_unified(model, images, steerings, warmup_runs=10,
                                tc66_port=None, idle_power_w=0.0,
                                country_code='DEU'):
    """
    Unified benchmark with TC66C power measurement

    Args:
        model: Akida model
        images: Preloaded images (uint8)
        steerings: Ground truth
        warmup_runs: Warmup iterations
        tc66_port: TC66 serial port
        idle_power_w: Idle baseline to subtract
        country_code: Country for emissions

    Returns:
        Dictionary with results
    """
    num_samples = len(steerings)

    print(f"\n{'=' * 70}")
    print("UNIFIED BENCHMARK (MAC-COMPATIBLE)")
    print('=' * 70)
    print(f"Samples:     {num_samples}")
    print(f"Batch size:  1 (matching Mac)")
    print(f"Warmup:      {warmup_runs}")
    print(f"Idle power:  {idle_power_w:.2f} W (will be subtracted)")
    print('=' * 70)

    # Warmup
    print("\nWarming up...")
    warmup_img = np.expand_dims(images[0], axis=0)
    for _ in range(warmup_runs):
        _ = model.predict(warmup_img)
    print("✓ Warmup complete")

    # Setup TC66 recorder
    recorder = None
    if tc66_port and TC66C_AVAILABLE:
        recorder = TC66CRecorder(tc66_port, poll_interval=0.1)
        if recorder.connect():
            time.sleep(0.5)
        else:
            recorder = None

    # Start TC66 recording
    if recorder:
        recorder.start()
        time.sleep(0.5)

    # Benchmark
    print("\nRunning inference (batch_size=1)...")
    predictions = []
    inference_times = []

    start_total = time.perf_counter()

    for i in range(num_samples):
        img_batch = np.expand_dims(images[i], axis=0)

        t0 = time.perf_counter()
        pred = model.predict(img_batch)
        t1 = time.perf_counter()

        dt = t1 - t0
        inference_times.append(dt)

        pred_scalar = float(pred.ravel()[0])
        predictions.append(pred_scalar)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{num_samples} samples")

    end_total = time.perf_counter()
    total_time = end_total - start_total

    # Stop TC66
    tc66_measurements = None
    tc66_summary = None
    if recorder:
        time.sleep(0.3)
        tc66_measurements = recorder.stop()
        tc66_summary = TC66CRecorder.summarize(tc66_measurements)

    # Calculate metrics
    predictions = np.array(predictions, dtype=np.float32)
    inference_times = np.array(inference_times, dtype=np.float32)

    throughput = num_samples / total_time
    avg_latency_ms = np.mean(inference_times) * 1000.0
    std_latency_ms = np.std(inference_times) * 1000.0

    mse = mean_squared_error(steerings, predictions)
    mae = mean_absolute_error(steerings, predictions)

    # Process energy measurements
    total_energy_wh = 0.0
    inference_energy_wh = 0.0
    avg_total_power_w = 0.0
    avg_inference_power_w = 0.0

    if tc66_summary:
        # Use TC66's internal energy counter if available
        total_energy_wh = tc66_summary['tc66_energy_wh'] or tc66_summary['integrated_energy_wh']
        avg_total_power_w = tc66_summary['avg_power_w']

        # Subtract idle baseline
        idle_energy_wh = idle_power_w * total_time / 3600.0
        inference_energy_wh = max(0.0, total_energy_wh - idle_energy_wh)
        avg_inference_power_w = max(0.0, avg_total_power_w - idle_power_w)

    energy_per_sample_mwh = (inference_energy_wh * 1000.0) / num_samples if num_samples > 0 else 0.0

    # CO2 emissions
    emission_factors = {
        'DEU': 0.420, 'ESP': 0.275, 'FRA': 0.070, 'USA': 0.429,
        'GBR': 0.233, 'CHN': 0.555, 'IND': 0.708
    }
    emission_factor = emission_factors.get(country_code, 0.420)

    inference_energy_kwh = inference_energy_wh / 1000.0
    inference_co2_kg = inference_energy_kwh * emission_factor
    co2_g_per_sample = (inference_co2_kg * 1000.0) / num_samples if num_samples > 0 else 0.0

    # Print results
    print(f"\n{'=' * 70}")
    print("BENCHMARK RESULTS")
    print('=' * 70)
    print(f"\nPerformance:")
    print(f"  Total time:      {total_time:.2f} seconds")
    print(f"  Throughput:      {throughput:.2f} samples/second")
    print(f"  Avg latency:     {avg_latency_ms:.2f} ± {std_latency_ms:.2f} ms")
    print(f"\nAccuracy:")
    print(f"  MSE:             {mse:.4f}")
    print(f"  MAE:             {mae:.4f}")

    if tc66_summary:
        print(f"\nEnergy (TC66 with idle subtraction):")
        print(f"  Total measured:  {total_energy_wh:.4f} Wh")
        print(f"  Total power:     {avg_total_power_w:.2f} W")
        print(f"  Idle baseline:   {idle_energy_wh:.4f} Wh ({idle_power_w:.2f} W)")
        print(f"  Inference only:  {inference_energy_wh:.4f} Wh ← USE THIS")
        print(f"  Inference power: {avg_inference_power_w:.2f} W")
        print(f"  Per sample:      {energy_per_sample_mwh:.6f} mWh")
        print(f"\nEmissions:")
        print(f"  CO2 (inference): {inference_co2_kg:.6f} kgCO2eq")
        print(f"  Per sample:      {co2_g_per_sample:.6f} gCO2")
        print(f"  Emission factor: {emission_factor:.3f} kgCO2/kWh ({country_code})")
    else:
        print(f"\nNo TC66 measurements available")

    print('=' * 70)

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
        # Energy
        'total_energy_wh': float(total_energy_wh),
        'idle_energy_wh': float(idle_energy_wh),
        'inference_energy_wh': float(inference_energy_wh),
        'energy_per_sample_mwh': float(energy_per_sample_mwh),
        'avg_total_power_w': float(avg_total_power_w),
        'avg_inference_power_w': float(avg_inference_power_w),
        'idle_power_w': float(idle_power_w),
        # Emissions
        'inference_co2_kg': float(inference_co2_kg),
        'co2_g_per_sample': float(co2_g_per_sample),
        'emission_factor_kg_per_kwh': float(emission_factor),
        'country_code': str(country_code),
        'energy_source': 'tc66_with_idle_subtraction'
    }


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(output_dir, model_name, results, mapping_success):
    """Save results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full JSON
    results_file = output_dir / f"{model_name}_unified_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved: {results_file}")

    # Summary CSV
    summary = {
        'model_name': model_name,
        'hardware': 'RPi5 + Akida 1.0',
        'mapped_to_npu': mapping_success,
        'input_mode': 'uint8_255',
        'batch_size': 1,
        'num_samples': results['num_samples'],
        'throughput_sps': results['throughput_samples_per_second'],
        'avg_latency_ms': results['avg_latency_ms'],
        'mse': results['mse'],
        'mae': results['mae'],
        # Energy (inference only)
        'total_energy_wh': results['inference_energy_wh'],
        'energy_per_sample_mwh': results['energy_per_sample_mwh'],
        'avg_power_w': results['avg_inference_power_w'],
        'idle_power_w': results['idle_power_w'],
        # Emissions
        'co2_kg': results['inference_co2_kg'],
        'co2_g_per_sample': results['co2_g_per_sample'],
        'emission_factor_kg_per_kwh': results['emission_factor_kg_per_kwh'],
        'country_code': results['country_code'],
        'energy_source': 'tc66_idle_subtracted'
    }

    summary_file = output_dir / f"{model_name}_unified_benchmark_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_file, index=False)
    print(f"✓ Summary saved: {summary_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified Benchmark for RPi5 + Akida (uses TC66C library)'
    )
    parser.add_argument('--akida_model', required=True, help='Path to .fbz model')
    parser.add_argument('--data_dir', required=True, help='Dataset directory')
    parser.add_argument('--output_dir', default='./benchmark_results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--warmup_runs', type=int, default=10, help='Warmup runs')
    parser.add_argument('--tc66_port', type=str, default='/dev/ttyACM0', help='TC66 port')
    parser.add_argument('--measure_idle', action='store_true', help='Measure idle power')
    parser.add_argument('--idle_duration', type=int, default=10, help='Idle measurement duration')
    parser.add_argument('--country_code', type=str, default='DEU', help='Country code')

    args = parser.parse_args()

    model_path = Path(args.akida_model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print(f"\n{'=' * 70}")
    print("UNIFIED AKIDA BENCHMARK (MAC-COMPATIBLE)")
    print('=' * 70)
    print(f"Model:       {model_path}")
    print(f"Dataset:     {args.data_dir}")
    print(f"Samples:     {args.num_samples}")
    print(f"Batch size:  1 (matching Mac)")
    print(f"TC66 port:   {args.tc66_port}")
    print('=' * 70)

    # Measure idle power
    idle_power_w = 0.0
    if args.measure_idle:
        idle_results = measure_idle_power_tc66(args.tc66_port, args.idle_duration)
        idle_power_w = idle_results['idle_power_w']

    # Detect Akida
    device = detect_akida_device()

    # Load model
    model, mapping_success = load_akida_model(model_path, device)

    # Load data
    img_paths, steerings = load_validation_data(args.data_dir, args.num_samples)
    images = preload_images_to_ram(img_paths)

    # Benchmark
    results = benchmark_inference_unified(
        model=model,
        images=images,
        steerings=steerings,
        warmup_runs=args.warmup_runs,
        tc66_port=args.tc66_port,
        idle_power_w=idle_power_w,
        country_code=args.country_code
    )

    # Save
    model_name = model_path.stem.replace('_akida', '')
    save_results(args.output_dir, model_name, results, mapping_success)

    print(f"\n{'=' * 70}")
    print("BENCHMARK COMPLETE")
    print('=' * 70)


if __name__ == "__main__":
    main()
