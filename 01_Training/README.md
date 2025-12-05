# Phase 1: Model Training

This directory contains scripts for training float32 CNN models on Mac M1.

## Usage

```bash
python train_float32_mac.py \
    --model all \
    --data_dir ../data/Original_Images \
    --output_dir ../models/float32 \
    --result_dir ../results/training \
    --batch_size 32 \
    --epochs 10 \
    --learning_rate 1e-4
```

## Arguments

- `--model`: Model to train (pilotnet, laksnet, mininet, or all)
- `--data_dir`: Path to Udacity dataset
- `--output_dir`: Output directory for models
- `--result_dir`: Directory for training plots
- `--batch_size`: Training batch size (default: 32)
- `--epochs`: Maximum epochs (default: 10)
- `--learning_rate`: Initial learning rate (default: 1e-4)

## Outputs

- `{model}_float32_best.h5` - Best model checkpoint
- `{model}_float32_history.csv` - Training history
- `training_emissions.csv` - CodeCarbon energy tracking
- `{model}_float32_predictions.png` - Validation predictions plot

## Energy Tracking

CodeCarbon automatically tracks:
- CPU/GPU energy consumption
- CO₂ emissions (Germany: 0.420 kgCO₂/kWh)
- Training duration

## Architecture Verification

All models are verified for Akida v1.0 compatibility before training:
- Conv2D stride=2 uses 3×3 or 1×1 kernels
- BatchNormalization after Conv2D
- ReLU max_value=6.0
- No unsupported layers (GlobalAveragePooling, Lambda, etc.)
