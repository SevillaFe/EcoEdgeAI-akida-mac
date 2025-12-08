# Phase 3 & 4: Inference Benchmarking

This directory contains benchmarking scripts for both platforms.

## Akida NPU Benchmarking (RPi5)

### Prerequisites
- TC66 USB-C power meter connected
- Akida NPU detected (`akida.devices()`)

### Usage

```bash
python unified_benchmark_akida.py \
    --akida_model ../models/akida/pilotnet_best_q4_akida.fbz \
    --data_dir ../data/Original_Images \
    --output_dir ../results/benchmark_results_akida \
    --num_samples 1000 \
    --tc66_port /dev/ttyACM0 \
    --measure_idle \
    --idle_duration 10
```

### TC66 Power Meter

**Connection:**
```
Power Supply (5V/5A) → TC66 → RPi5 USB-C
```

**Measurements:**
- Voltage (0.01V resolution)
- Current (0.001A resolution)
- Power (instantaneous)
- Energy accumulator (0.001Wh resolution)

**Polling Rate:** 10 Hz (100ms intervals)

### Idle Baseline Subtraction

1. Measure idle power for 10 seconds (no inference)
2. Run benchmark with continuous power monitoring
3. Subtract idle energy: `E_inference = E_total - E_idle`

## Mac M1 Benchmarking

### Usage

```bash
python unified_benchmark_mac.py \
    --model ../models/float32/pilotnet_float32_best.h5 \
    --data_dir ../data/Original_Images \
    --output_dir ../results/benchmark_results_mac \
    --num_samples 1000 \
    --measure_idle \
    --idle_duration 10
```

### CodeCarbon Energy Estimation

**Method:** TDP-based power modeling
- Mac M1 CPU: 20W TDP
- Mac M1 GPU: 10W TDP
- Power = `P_CPU × util_CPU + P_GPU × util_GPU`

**Carbon Intensity:** 0.420 kgCO₂/kWh (Germany)

### Idle Baseline Subtraction

1. Measure idle power for 10 seconds
2. Run benchmark with CodeCarbon tracking
3. Subtract idle energy from total

## Benchmark Protocol

**Common Settings:**
- Batch size: 1 (sequential processing)
- Warmup: 10 iterations
- Validation set: 1,000 center-camera images
- Latency measurement: `time.perf_counter()` (μs precision)

## Output Format

### JSON Results
```json
{
  "num_samples": 1000,
  "total_time_seconds": 9.62,
  "throughput_samples_per_second": 103.9,
  "avg_latency_ms": 9.59,
  "std_latency_ms": 0.42,
  "mse": 0.2191,
  "mae": 0.352,
  "inference_energy_wh": 0.0049,
  "energy_per_sample_mwh": 0.0049,
  "avg_inference_power_w": 1.20,
  "idle_power_w": 3.24,
  "inference_co2_kg": 2.058e-06,
  "co2_g_per_sample": 2.058e-06,
  "predictions": [...],
  "ground_truth": [...],
  "inference_times_ms": [...]
}
```

## TC66C Library

The `TC66C.py` file provides programmatic access to the TC66 power meter.

### Key Classes

```python
from TC66C import TC66C, TC66CRecorder

# Direct polling
tc66 = TC66C('/dev/ttyACM0')
data = tc66.Poll()
print(f"Power: {data.Power:.3f} W")

# Background recording
recorder = TC66CRecorder('/dev/ttyACM0', poll_interval=0.1)
recorder.connect()
recorder.start()
# ... run inference ...
measurements = recorder.stop()
summary = TC66CRecorder.summarize(measurements)
```

### Troubleshooting

**TC66 not detected:**
```bash
ls /dev/ttyACM*  # Check device path
sudo usermod -a -G dialout $USER  # Add user to dialout group
```

**Akida not detected:**
```bash
akida devices  # Should show AKD1000
# If empty, check PCIe connection and driver installation
```
  
