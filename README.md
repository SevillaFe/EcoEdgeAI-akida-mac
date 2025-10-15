# EcoEdgeAI-akida-mac
A comprehensive workflow for comparing energy efficiency between conventional hardware (Mac M-series GPU/CPU) and neuromorphic hardware (Akida on Raspberry Pi 5) for autonomous driving steering angle prediction.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://www.tensorflow.org/)


## 🎯 Project Overview

This project provides a complete pipeline to:
- Train deep learning models (PilotNet, LaksNet, MiniNet) for steering angle prediction
- Benchmark inference performance on conventional hardware with **CodeCarbon** energy tracking
- Convert models to neuromorphic format using **Akida**
- Benchmark neuromorphic inference with **TC66 USB power meter**
- Generate comprehensive eco-efficiency comparisons

### Key Research Questions
- How much energy does neuromorphic computing save?
- What is the accuracy trade-off?
- What is the latency difference?
- Which architecture is most efficient for edge deployment?

## 📊 Results Preview

Our experiments show:
- **Energy Efficiency**: Up to 76% reduction in energy consumption per inference
- **Latency**: 40-50% faster inference on neuromorphic hardware
- **Accuracy**: Minimal degradation (<10% MSE increase)
- **Overall Efficiency**: 280%+ improvement in accuracy-per-energy ratio

## 🛠️ Hardware Requirements

### Mac (Training & Benchmarking)
- MacBook with M-series processor (M1/M2/M3)
- 16GB+ RAM recommended
- macOS 12.0+

### Raspberry Pi 5 (Neuromorphic Benchmarking)
- Raspberry Pi 5 (4GB/8GB)
- BrainChip Akida neuromorphic processor board
- TC66/TC66C USB power meter
- 32GB+ microSD card
- Active cooling recommended

## 📦 Installation

### Mac Setup

```bash
# Clone repository
git clone https://github.com/yourusername/neuromorphic-inference-comparison.git
cd neuromorphic-inference-comparison

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements_mac.txt
```

### Raspberry Pi 5 Setup

```bash
# On RPi5
git clone https://github.com/yourusername/neuromorphic-inference-comparison.git
cd neuromorphic-inference-comparison

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (includes Akida SDK)
pip install -r requirements_rpi5.txt
```

## 📁 Project Structure

```
neuromorphic-inference-comparison/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements_mac.txt               # Mac dependencies
├── requirements_rpi5.txt              # RPi5 dependencies
├── workflow_guide.md                  # Detailed workflow guide
│
├── scripts/                           # Main scripts
│   ├── train_steering_models_keras.py # Training script
│   ├── benchmark_inference.py         # Mac benchmarking
│   ├── convert_to_akida.py           # Akida conversion
│   ├── benchmark_inference_rpi5.py   # RPi5 benchmarking
│   └── generate_summary.py           # Results analysis
│
├── models/                            # Model architectures
│   ├── __init__.py
│   ├── pilotnet.py                   # PilotNet architecture
│   ├── laksnet.py                    # LaksNet architecture
│   └── mininet.py                    # MiniNet architecture
│
├── utils/                             # Utility functions
│   ├── __init__.py
│   ├── data_processing.py            # Data loading & preprocessing
│   ├── augmentation.py               # Image augmentation
│   └── metrics.py                    # Custom metrics
│
├── configs/                           # Configuration files
│   ├── training_config.yaml          # Training hyperparameters
│   └── benchmark_config.yaml         # Benchmarking settings
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # Dataset analysis
│   ├── 02_model_comparison.ipynb     # Model performance
│   └── 03_results_visualization.ipynb # Final visualizations
│
├── docs/                              # Documentation
│   ├── setup_guide.md                # Hardware setup
│   ├── tc66_guide.md                 # TC66 meter usage
│   └── troubleshooting.md            # Common issues
│
├── tests/                             # Unit tests
│   ├── test_models.py
│   ├── test_data_processing.py
│   └── test_inference.py
│
├── results/                           # Results directory (gitignored)
│   ├── pilotnet/
│   ├── laksnet/
│   └── mininet/
│
└── data/                              # Dataset directory (gitignored)
    └── README.md                      # Dataset instructions
```

## 🚀 Quick Start

### Step 1: Prepare Dataset

Download the Udacity Self-Driving Car Dataset:

```bash
# Download from Udacity
# https://github.com/udacity/self-driving-car/tree/master/datasets

# Extract to data directory
mkdir -p data/Original_Images
# Place IMG folder and driving_log.csv in data/Original_Images/
```

### Step 2: Train Models on Mac

```bash
# Train all three models
for model in pilotnet laksnet mininet; do
  python scripts/train_steering_models_keras.py \
    --data_dir ./data/Original_Images \
    --output_dir ./output \
    --model_name $model \
    --epochs 20 \
    --batch_size 64
done
```

### Step 3: Benchmark on Mac

```bash
# Benchmark each model
for model in pilotnet laksnet mininet; do
  python scripts/benchmark_inference.py \
    --float_model ./output/$model/model_best.keras \
    --tflite_model ./output/$model/${model}_int8.tflite \
    --data_dir ./data/Original_Images \
    --num_samples 1000
done
```

### Step 4: Transfer to Raspberry Pi 5

```bash
# Transfer models and scripts
scp -r ./output pi@raspberrypi:~/models
scp scripts/convert_to_akida.py pi@raspberrypi:~/
scp scripts/benchmark_inference_rpi5.py pi@raspberrypi:~/
```

### Step 5: Convert to Akida Format

```bash
# On RPi5
ssh pi@raspberrypi

for model in pilotnet laksnet mininet; do
  python convert_to_akida.py \
    --model_path ~/models/$model/model_best.keras \
    --output_dir ~/models/akida
done
```

### Step 6: Benchmark on Akida

```bash
# On RPi5 with TC66 meter connected
for model in pilotnet laksnet mininet; do
  python benchmark_inference_rpi5.py \
    --akida_model ~/models/akida/${model}_akida.fbz \
    --data_dir ~/data/Original_Images \
    --num_samples 1000 \
    --mac_results ~/models/$model/mac_inference_comparison.json
done
```

### Step 7: Generate Summary Report

```bash
# Back on Mac (after copying results from RPi5)
python scripts/generate_summary.py \
  --results_dir ./output \
  --output_dir ./summary
```

## 📖 Detailed Documentation

- **[Workflow Guide](workflow_guide.md)** - Complete step-by-step workflow
- **[Setup Guide](docs/setup_guide.md)** - Hardware setup instructions
- **[TC66 Guide](docs/tc66_guide.md)** - Power meter usage
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

## 🔬 Model Architectures

### PilotNet (NVIDIA)
- 5 convolutional layers
- 4 fully connected layers
- ~1.7M parameters
- ELU activation with dropout

### LaksNet
- 4 convolutional layers
- 2 fully connected layers
- ~450K parameters
- Optimized for embedded systems

### MiniNet
- 3 convolutional layers with pooling
- 1 fully connected layer
- ~85K parameters
- Ultra-lightweight architecture

## 📊 Metrics Tracked

- **Accuracy**: MSE, MAE, MAPE
- **Latency**: ms per sample, throughput
- **Energy**: Total consumption, per-sample energy
- **Efficiency**: Accuracy-per-energy ratio
- **Carbon**: CO2 emissions (CodeCarbon)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@software{neuromorphic_inference_2025,
  author = {Martínez, Fernando S.},
  title = {Eco-Efficient Inference: Conventional vs Neuromorphic Hardware},
  year = {2025},
  url = {https://github.com/SevillaFe/EcoEdgeAI-akida-mac}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Udacity** for the Self-Driving Car Dataset
- **NVIDIA** for the PilotNet architecture
- **BrainChip** for Akida SDK and hardware
- **CodeCarbon** for energy tracking
- **TensorFlow** team for the framework

## 📧 Contact


Project Link: [https://github.com/SevillaFe/neuromorphic-inference-comparison](https://github.com/SevillaFe/EcoEdgeAI-akida-mac)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SevillaFe/EcoEdgeAI-akida-mac&type=Date)](https://star-history.com/#SevillaFe/EcoEdgeAI-akida-mac&Date)

---

**Made with ❤️ for sustainable AI and neuromorphic computing**
