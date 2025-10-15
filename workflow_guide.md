# Complete Workflow: Conventional vs Neuromorphic Inference Eco-Efficiency

This guide walks you through the complete process of comparing conventional hardware (Mac GPU/CPU) vs neuromorphic hardware (Akida on RPi5) for steering angle prediction.

## 📋 Prerequisites

### Mac (M-series):
```bash
pip install tensorflow tensorflow-model-optimization pandas numpy pillow codecarbon
```

### RPi5 (Linux):
```bash
pip install tensorflow akida cnn2snn pandas numpy pillow
```

### Hardware:
- **Mac**: MacBook with M-series processor
- **RPi5**: Raspberry Pi 5 with Akida neuromorphic board
- **TC66**: USB power meter (for RPi5 measurements)

---

## 🚀 Complete Workflow

### **PHASE 1: Training on Mac** 🖥️

Train all three models (PilotNet, LaksNet, MiniNet):

```bash
# Train PilotNet
python train_steering_models_tf.py \
  --data_dir /Users/fernando/Documents/Doctorado/Udacity_Datset/Original_Images \
  --output_dir ./output \
  --model_name pilotnet \
  --epochs 20 \
  --qat_epochs 3 \
  --batch_size 64

# Train LaksNet
python train_steering_models_tf.py \
  --data_dir /Users/fernando/Documents/Doctorado/Udacity_Datset/Original_Images \
  --output_dir ./output \
  --model_name laksnet \
  --epochs 20 \
  --qat_epochs 3 \
  --batch_size 64

# Train MiniNet
python train_steering_models_tf.py \
  --data_dir /Users/fernando/Documents/Doctorado/Udacity_Datset/Original_Images \
  --output_dir ./output \
  --model_name mininet \
  --epochs 20 \
  --qat_epochs 3 \
  --batch_size 64
```

**Outputs per model:**
- `float_best.keras` - Float (FP32) model
- `float_history.csv` - Training metrics
- `training_emissions.csv` - Training energy consumption
- `{model}_int8.tflite` - Quantized INT8 model
- `{model}_qat_int8.tflite` - QAT fine-tuned INT8 model (optional)
- `{model}_qat_history.csv` - QAT metrics
- `qat_training_emissions.csv` - QAT energy consumption

---

### **PHASE 2: Inference Benchmarking on Mac** 🖥️⚡

Benchmark float vs quantized models with CodeCarbon:

```bash
# Benchmark PilotNet
python benchmark_inference.py \
  --float_model ./output/pilotnet/float_best.keras \
  --tflite_model ./output/pilotnet/pilotnet_int8.tflite \
  --data_dir /Users/fernando/Documents/Doctorado/Udacity_Datset/Original_Images \
  --num_samples 1000

# Benchmark LaksNet
python benchmark_inference.py \
  --float_model ./output/laksnet/float_best.keras \
  --tflite_model ./output/laksnet/laksnet_int8.tflite \
  --data_dir /Users/fernando/Documents/Doctorado/Udacity_Datset/Original_Images \
  --num_samples 1000

# Benchmark MiniNet
python benchmark_inference.py \
  --float_model ./output/mininet/float_best.keras \
  --tflite_model ./output/mininet/mininet_int8.tflite \
  --data_dir /Users/fernando/Documents/Doctorado/Udacity_Datset/Original_Images \
  --num_samples 1000
```

**Outputs per model:**
- `mac_inference_comparison.json` - Float vs TFLite comparison
- `mac_inference_results.csv` - Results summary
- `float_inference_emissions.csv` - Float model energy
- `tflite_inference_emissions.csv` - TFLite model energy

---

### **PHASE 3: Transfer to RPi5** 📦➡️🥧

Transfer the best models to your Raspberry Pi 5:

```bash
# Create directory on RPi5
ssh pi@raspberrypi "mkdir -p ~/models ~/data"

# Transfer float models (for Akida conversion)
scp ./output/pilotnet/float_best.keras pi@raspberrypi:~/models/pilotnet_float.keras
scp ./output/laksnet/float_best.keras pi@raspberrypi:~/models/laksnet_float.keras
scp ./output/mininet/float_best.keras pi@raspberrypi:~/models/mininet_float.keras

# Transfer Mac benchmark results (for comparison)
scp ./output/pilotnet/mac_inference_comparison.json pi@raspberrypi:~/models/
scp ./output/laksnet/mac_inference_comparison.json pi@raspberrypi:~/models/
scp ./output/mininet/mac_inference_comparison.json pi@raspberrypi:~/models/

# Transfer dataset (if not already on RPi5)
# rsync -avz --progress /Users/fernando/Documents/Doctorado/Udacity_Datset/Original_Images/ \
#   pi@raspberrypi:~/data/Original_Images/
```

---

### **PHASE 4: Convert to Akida on RPi5** 🥧🧠

SSH into your RPi5 and convert models to Akida format:

```bash
ssh pi@raspberrypi

# Convert PilotNet
python convert_to_akida.py \
  --model_path ~/models/pilotnet_float.keras \
  --output_dir ~/models/akida

# Convert LaksNet
python convert_to_akida.py \
  --model_path ~/models/laksnet_float.keras \
  --output_dir ~/models/akida

# Convert MiniNet
python convert_to_akida.py \
  --model_path ~/models/mininet_float.keras \
  --output_dir ~/models/akida
```

**Outputs:**
- `pilotnet_float_akida.fbz`
- `laksnet_float_akida.fbz`
- `mininet_float_akida.fbz`

---

### **PHASE 5: Inference Benchmarking on RPi5 with TC66** 🥧⚡🔌

**Setup TC66 USB Meter:**
1. Connect TC66 between USB-C power adapter and RPi5
2. Reset energy counter on TC66
3. Note initial voltage and current readings

**Run benchmarks:**

```bash
# Benchmark PilotNet
python benchmark_inference_rpi5.py \
  --akida_model ~/models/akida/pilotnet_float_akida.fbz \
  --data_dir ~/data/Original_Images \
  --num_samples 1000 \
  --mac_results ~/models/mac_inference_comparison.json

# Benchmark LaksNet
python benchmark_inference_rpi5.py \
  --akida_model ~/models/akida/laksnet_float_akida.fbz \
  --data_dir ~/data/Original_Images \
  --num_samples 1000 \
  --mac_results ~/models/mac_inference_comparison.json

# Benchmark MiniNet
python benchmark_inference_rpi5.py \
  --akida_model ~/models/akida/mininet_float_akida.fbz \
  --data_dir ~/data/Original_Images \
  --num_samples 1000 \
  --mac_results ~/models/mac_inference_comparison.json
```

**During each benchmark:**
1. Script will prompt for **starting power readings** from TC66
2. Inference runs
3. Script prompts for **ending power readings** from TC66
4. Enter TC66 accumulated energy (Wh) if available

**Outputs per model:**
- `akida_inference_results.json` - Akida performance metrics
- `akida_predictions.csv` - Prediction results
- `final_comparison_mac_vs_akida.json` - Complete comparison

---

## 📊 Results Analysis

### Key Metrics Compared:

| Metric | Mac (Conventional) | RPi5 + Akida (Neuromorphic) |
|--------|-------------------|----------------------------|
| **Accuracy** | MSE, MAE, MAPE | MSE, MAE, MAPE |
| **Latency** | ms per sample | ms per sample |
| **Energy** | mWh per sample (CodeCarbon) | mWh per sample (TC66) |
| **Efficiency** | Accuracy/Energy ratio | Accuracy/Energy ratio |

### Expected Results Structure:

```json
{
  "mac_conventional": {
    "model_type": "TFLite INT8",
    "avg_latency_ms": 15.2,
    "energy_per_sample_mwh": 0.05,
    "mse": 0.0023,
    "accuracy_per_energy": 8695.65
  },
  "akida_neuromorphic": {
    "model_type": "Akida Neuromorphic",
    "avg_latency_ms": 8.7,
    "energy_per_sample_mwh": 0.012,
    "mse": 0.0025,
    "accuracy_per_energy": 33333.33
  },
  "improvements": {
    "latency_percent": +42.8,
    "energy_percent": +76.0,
    "efficiency_percent": +283.5,
    "accuracy_change_percent": +8.7
  }
}
```

---

## 🎯 Final Comparison

After running all benchmarks, you'll have:

1. **Training Efficiency**: Energy consumed during training (all models)
2. **Mac Inference**: Float vs INT8 quantized performance
3. **Akida Inference**: Neuromorphic hardware performance
4. **Final Comparison**: Conventional vs Neuromorphic

### Retrieve Results from RPi5:

```bash
# Download all Akida results
scp pi@raspberrypi:~/models/akida/*_comparison_*.json ./results/
scp pi@raspberrypi:~/models/akida/*_results.json ./results/
scp pi@raspberrypi:~/models/akida/*_predictions.csv ./results/
```

---

## 📈 Visualization Script (Optional)

Create a simple visualization script `visualize_results.py`:

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

def plot_comparison(model_name):
    # Load Mac results
    with open(f'./output/{model_name}/mac_inference_comparison.json') as f:
        mac_data = json.load(f)
    
    # Load Akida results
    with open(f'./results/{model_name}_comparison_mac_vs_akida.json') as f:
        akida_data = json.load(f)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{model_name.upper()} - Conventional vs Neuromorphic', fontsize=16)
    
    # Plot 1: Latency
    ax = axes[0, 0]
    latencies = [
        mac_data['tflite_int8']['avg_latency_ms'],
        akida_data['akida_neuromorphic']['avg_latency_ms']
    ]
    ax.bar(['Mac TFLite', 'Akida'], latencies, color=['blue', 'green'])
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Inference Latency')
    
    # Plot 2: Energy per Sample
    ax = axes[0, 1]
    energy = [
        mac_data['tflite_int8']['energy_per_sample_mwh'],
        akida_data['akida_neuromorphic']['energy_per_sample_mwh']
    ]
    ax.bar(['Mac TFLite', 'Akida'], energy, color=['blue', 'green'])
    ax.set_ylabel('Energy (mWh)')
    ax.set_title('Energy per Sample')
    
    # Plot 3: Accuracy (MSE)
    ax = axes[1, 0]
    mse = [
        mac_data['tflite_int8']['mse'],
        akida_data['akida_neuromorphic']['mse']
    ]
    ax.bar(['Mac TFLite', 'Akida'], mse, color=['blue', 'green'])
    ax.set_ylabel('MSE')
    ax.set_title('Prediction Accuracy (MSE, lower is better)')
    
    # Plot 4: Energy Efficiency
    ax = axes[1, 1]
    efficiency = [
        mac_data['tflite_int8']['accuracy_per_energy'],
        akida_data['akida_neuromorphic']['accuracy_per_energy']
    ]
    ax.bar(['Mac TFLite', 'Akida'], efficiency, color=['blue', 'green'])
    ax.set_ylabel('Accuracy / Energy')
    ax.set_title('Energy Efficiency (higher is better)')
    
    plt.tight_layout()
    plt.savefig(f'./results/{model_name}_comparison.png', dpi=300)
    print(f'Saved: {model_name}_comparison.png')

# Generate for all models
for model in ['pilotnet', 'laksnet', 'mininet']:
    plot_comparison(model)
```

---

## 🎓 Summary

This workflow allows you to:

1. ✅ Train models on Mac with energy tracking
2. ✅ Quantize models (TFLite INT8) on Mac
3. ✅ Benchmark Mac inference with CodeCarbon
4. ✅ Convert to Akida format on RPi5
5. ✅ Benchmark Akida inference with TC66 meter
6. ✅ Compare conventional vs neuromorphic hardware

**Key Research Questions Answered:**
- How much energy does neuromorphic computing save?
- What is the accuracy trade-off?
- What is the latency difference?
- Which architecture (PilotNet, LaksNet, MiniNet) is most efficient?

Good luck with your research! 🚀🧠
