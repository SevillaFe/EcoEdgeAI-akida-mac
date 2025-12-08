#!/usr/bin/env python3
"""
Script 1 (Mac M1): Train Float32 Models with Energy Tracking
==============================================================

Trains PilotNet, LaksNet, and MiniNet with Akida 1.0 compatible architectures.
Uses CodeCarbon to track energy consumption and emissions.

IMPORTANT: This script ONLY trains float32 models.
Quantization and Akida conversion will be done on RPi5.

AKIDA 1.0 ARCHITECTURE REQUIREMENTS:
- Conv2D with stride=2 must use kernel 3x3 or 1x1
- BatchNormalization after Conv2D (will be fused on RPi5)
- ReLU can have max_value=6.0
- NO GlobalAveragePooling2D, Lambda, or LayerNormalization

USAGE:
======
python train_float32_mac.py \
    --model all \
    --data_dir ./Udacity_Dataset/Original_Images \
    --output_dir ./Udacity_Dataset/paper_5/output \
    --result_dir ./Udacity_Dataset/paper_5/output/result \
    --batch_size 64 \
    --epochs 10 \
    --learning_rate 1e-4

ARGUMENTS:
==========
--model                   Model to train: pilotnet, laksnet, mininet, or all (default: all)
--data_dir                Path to data directory containing driving_log.csv and IMG/
--output_dir              Base output directory
--result_dir              Directory for prediction plots and training history
--train_val_split         Train/validation split ratio (default: 0.8)
--steering_correction     Steering correction for left/right cameras (default: 0.2)
--batch_size              Batch size for training (default: 32)
--epochs                  Number of training epochs (default: 10)
--learning_rate           Initial learning rate (default: 1e-4)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from PIL import Image
from keras import models, layers
from keras.optimizers import Adam
from keras.optimizers import legacy
from sklearn.metrics import mean_squared_error, mean_absolute_error
import ntpath
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from codecarbon import EmissionsTracker
import random
import cv2

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
rng = np.random.default_rng(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_W, IMG_H = 200, 66

# ============================================================================
# MODEL ARCHITECTURES (AKIDA 1.0 COMPATIBLE)
# ============================================================================

def create_pilotnet_model():
    """
    PilotNet architecture compatible with Akida 1.0
    
    Key changes from original:
    - All Conv2D with stride=2 use kernel 3x3 (Akida 1.0 requirement)
    - use_bias=False for Conv2D (BatchNorm handles bias)
    - ReLU separated from Dense layers
    """
    return models.Sequential([
        # Conv Block 1: stride=2 → kernel 3x3 (REQUIRED for Akida 1.0)
        layers.Conv2D(24, (3, 3), strides=2, padding='same', 
                      input_shape=(66, 200, 3), use_bias=False, name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.ReLU(max_value=6, name='relu1'),

        # Conv Block 2
        layers.Conv2D(36, (3, 3), strides=2, padding='same', use_bias=False, name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.ReLU(max_value=6, name='relu2'),

        # Conv Block 3
        layers.Conv2D(48, (3, 3), strides=2, padding='same', use_bias=False, name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.ReLU(max_value=6, name='relu3'),

        # Conv Block 4: stride=1 → any kernel OK
        layers.Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False, name='conv4'),
        layers.BatchNormalization(name='bn4'),
        layers.ReLU(max_value=6, name='relu4'),

        # Conv Block 5
        layers.Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False, name='conv5'),
        layers.BatchNormalization(name='bn5'),
        layers.ReLU(max_value=6, name='relu5'),

        layers.Dropout(0.4, name='dropout1'),
        layers.Flatten(name='flatten'),
        
        # Dense layers
        layers.Dense(100, use_bias=True, name='fc1'),
        layers.BatchNormalization(name='bn_fc1'),
        layers.ReLU(max_value=6, name='relu_fc1'),

        layers.Dense(50, use_bias=True, name='fc2'),
        layers.BatchNormalization(name='bn_fc2'),
        layers.ReLU(max_value=6, name='relu_fc2'),

        layers.Dense(10, use_bias=True, name='fc3'),
        layers.BatchNormalization(name='bn_fc3'),
        layers.ReLU(max_value=6, name='relu_fc3'),

        # Output layer (no activation for regression)
        layers.Dense(1, use_bias=True, name='output')
    ], name="PilotNet")


def create_laksnet_model():
    """LaksNet architecture - lighter than PilotNet, Akida 1.0 compatible"""
    return models.Sequential([
        layers.Conv2D(16, (3, 3), strides=2, padding='same', 
                      input_shape=(66, 200, 3), use_bias=False, name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.ReLU(max_value=6, name='relu1'),
        
        layers.Conv2D(32, (3, 3), strides=2, padding='same', use_bias=False, name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.ReLU(max_value=6, name='relu2'),
        
        layers.Conv2D(48, (3, 3), strides=2, padding='same', use_bias=False, name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.ReLU(max_value=6, name='relu3'),
        
        layers.Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False, name='conv4'),
        layers.BatchNormalization(name='bn4'),
        layers.ReLU(max_value=6, name='relu4'),
        
        layers.Dropout(0.3, name='dropout1'),
        layers.Flatten(name='flatten'),
        
        layers.Dense(50, use_bias=True, name='fc1'),
        layers.BatchNormalization(name='bn_fc1'),
        layers.ReLU(max_value=6, name='relu_fc1'),
        
        layers.Dense(10, use_bias=True, name='fc2'),
        layers.BatchNormalization(name='bn_fc2'),
        layers.ReLU(max_value=6, name='relu_fc2'),
        
        layers.Dense(1, use_bias=True, name='output')
    ], name="LaksNet")


def create_mininet_model():
    """MiniNet architecture - minimal network, Akida 1.0 compatible"""
    return models.Sequential([
        layers.Conv2D(16, (3, 3), strides=2, padding='same',
                      input_shape=(66, 200, 3), use_bias=False, name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.ReLU(max_value=6, name='relu1'),
        
        layers.Conv2D(32, (3, 3), strides=2, padding='same', use_bias=False, name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.ReLU(max_value=6, name='relu2'),
        
        layers.Conv2D(32, (3, 3), strides=2, padding='same', use_bias=False, name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.ReLU(max_value=6, name='relu3'),
        
        layers.Dropout(0.2, name='dropout1'),
        layers.Flatten(name='flatten'),
        
        layers.Dense(32, use_bias=True, name='fc1'),
        layers.BatchNormalization(name='bn_fc1'),
        layers.ReLU(max_value=6, name='relu_fc1'),
        
        layers.Dense(1, use_bias=True, name='output')
    ], name="MiniNet")


MODEL_DICT = {
    'pilotnet': create_pilotnet_model,
    'laksnet': create_laksnet_model,
    'mininet': create_mininet_model
}


# ============================================================================
# ENERGY TRACKING CALLBACK
# ============================================================================

class EnergyTrackerCallback(Callback):
    """Track energy consumption during training using CodeCarbon"""
    def __init__(self, output_dir, phase='training'):
        super().__init__()
        self.output_dir = Path(output_dir).absolute()
        self.phase = phase
        self.tracker = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def on_train_begin(self, logs=None):
        output_file_path = str(self.output_dir / f'{self.phase}_emissions.csv')
        self.tracker = EmissionsTracker(
            output_dir=str(self.output_dir),
            output_file=f'{self.phase}_emissions.csv',
            on_csv_write="update",
            log_level='warning'
        )
        self.tracker.start()
        print(f"\n Energy tracking started: {output_file_path}")
        
    def on_train_end(self, logs=None):
        if self.tracker:
            emissions = self.tracker.stop()
            print(f"\n Energy tracking stopped")
            print(f"   Total emissions: {emissions:.6f} kgCO2eq")


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def path_leaf(path):
    """Extract filename from path"""
    head, tail = ntpath.split(path)
    return tail


def load_and_preprocess_image(img_path, steering):
    """
    Load and preprocess image - following your original script
    Returns image as float32 and steering (both potentially augmented)
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Crop: remove sky (60px top) and hood (25px bottom)
    img = img[60:-25, :]
    
    # Resize
    img = cv2.resize(img, (IMG_W, IMG_H))
    
    # Keep as float32 (NOT normalized yet - for augmentation)
    return img.astype(np.float32), steering


def augment_image(image, steering):
    """
    Apply augmentation - following your original script
    Input: float32 image (not normalized)
    Output: augmented float32 image, potentially flipped steering
    """
    # Convert to uint8 for OpenCV operations
    image = image.astype(np.uint8)
    
    # 1. Horizontal flip (50% probability) - CRITICAL: invert steering!
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering = -steering  # ← CRITICAL: invert steering when flipping
    
    # 2. Random brightness adjustment (40% probability)
    if np.random.rand() < 0.4:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        brightness_factor = np.random.uniform(0.6, 1.4)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
        hsv = hsv.astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # 3. Random shadow (30% probability)
    if np.random.rand() < 0.3:
        h, w = image.shape[:2]
        shadow_factor = np.random.uniform(0.3, 0.7)
        
        # Create random shadow region (horizontal band)
        top_y = int(h * np.random.uniform(0.0, 0.5))
        bottom_y = int(h * np.random.uniform(0.5, 1.0))
        
        if bottom_y > top_y:
            image = image.astype(np.float32)
            image[top_y:bottom_y, :, :] *= shadow_factor
            image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Convert back to float32 (still not normalized)
    return image.astype(np.float32), steering


# ============================================================================
# DATA GENERATORS
# ============================================================================

def batch_generator(img_paths, steerings, batch_size, is_training=True):
    """
    Generate batches of images and steering angles
    Following your original script's approach
    
    Args:
        img_paths: List of image paths
        steerings: List of steering angles
        batch_size: Batch size
        is_training: Whether to apply augmentation
    """
    num_samples = len(img_paths)
    
    while True:
        # Shuffle data at the start of each epoch
        if is_training:
            indices = np.random.permutation(num_samples)
            img_paths_shuffled = [img_paths[i] for i in indices]
            steerings_shuffled = steerings[indices]
        else:
            img_paths_shuffled = img_paths
            steerings_shuffled = steerings
        
        for offset in range(0, num_samples, batch_size):
            batch_paths = img_paths_shuffled[offset:offset + batch_size]
            batch_steerings = steerings_shuffled[offset:offset + batch_size]
            
            images = []
            angles = []
            
            for img_path, steering in zip(batch_paths, batch_steerings):
                # Load and preprocess (crop + resize)
                img, steer = load_and_preprocess_image(img_path, steering)
                
                # Augmentation only during training
                if is_training:
                    img, steer = augment_image(img, steer)
                
                # Normalize to [0, 1] AFTER augmentation
                img = img / 255.0
                
                images.append(img)
                angles.append(steer)
            
            X = np.array(images, dtype=np.float32)
            y = np.array(angles, dtype=np.float32)
            
            yield X, y


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(data_dir, train_val_split=0.8, steering_correction=0.2):
    """
    Load and split dataset
    
    Returns:
        train_paths, train_steerings, val_paths, val_steerings
    """
    csv_path = Path(data_dir) / 'driving_log.csv'
    print(f"\nLoading data from: {csv_path}")
    
    # CSV has no headers, need to specify column names
    columns = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]
    df = pd.read_csv(csv_path, names=columns)
    
    print(f"Total samples in CSV: {len(df)}")
    
    # Process all camera images (center, left, right)
    img_paths = []
    steerings = []
    
    for idx, row in df.iterrows():
        # Center camera
        center_filename = os.path.basename(row['center'].strip())
        center_path = os.path.join(data_dir, 'IMG', center_filename)
        if os.path.exists(center_path):
            img_paths.append(center_path)
            steerings.append(float(row['steering']))
        
        # Left camera (add positive correction)
        left_filename = os.path.basename(row['left'].strip())
        left_path = os.path.join(data_dir, 'IMG', left_filename)
        if os.path.exists(left_path):
            img_paths.append(left_path)
            steerings.append(float(row['steering']) + steering_correction)
        
        # Right camera (add negative correction)
        right_filename = os.path.basename(row['right'].strip())
        right_path = os.path.join(data_dir, 'IMG', right_filename)
        if os.path.exists(right_path):
            img_paths.append(right_path)
            steerings.append(float(row['steering']) - steering_correction)
    
    steerings = np.array(steerings, dtype=np.float32)
    
    print(f"\nTotal samples (with all cameras): {len(img_paths)}")
    print(f"Steering angle range: [{steerings.min():.3f}, {steerings.max():.3f}]")
    
    # Split train/validation
    split_idx = int(len(img_paths) * train_val_split)
    
    train_paths = img_paths[:split_idx]
    train_steerings = steerings[:split_idx]
    
    val_paths = img_paths[split_idx:]
    val_steerings = steerings[split_idx:]
    
    print(f"\nTrain samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    return train_paths, train_steerings, val_paths, val_steerings


# ============================================================================
# PREDICTION AND VISUALIZATION
# ============================================================================

def make_predictions_and_plot(model, img_paths, true_steerings, result_dir, 
                              model_name, model_type):
    """
    Make predictions and create visualization plots
    
    Returns:
        mse, mae
    """
    print(f"\nMaking predictions for {model_type}...")
    
    predictions = []
    for img_path in img_paths:
        # Load and preprocess (no augmentation)
        img, _ = load_and_preprocess_image(img_path, 0.0)
        
        # Normalize to [0, 1]
        img = img / 255.0
        
        img_batch = np.expand_dims(img, axis=0)
        pred = model.predict(img_batch, verbose=0)[0][0]
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    mse = mean_squared_error(true_steerings, predictions)
    mae = mean_absolute_error(true_steerings, predictions)
    
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    # Create plots
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Predictions vs Ground Truth
    axes[0].plot(true_steerings, 'b-', label='Ground Truth', alpha=0.6, linewidth=1)
    axes[0].plot(predictions, 'r-', label='Predictions', alpha=0.6, linewidth=1)
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Steering Angle')
    axes[0].set_title(f'{model_name.upper()} - {model_type.upper()}\n'
                     f'MSE: {mse:.4f}, MAE: {mae:.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    axes[1].scatter(true_steerings, predictions, alpha=0.3, s=10)
    axes[1].plot([true_steerings.min(), true_steerings.max()],
                [true_steerings.min(), true_steerings.max()], 
                'r--', label='Perfect prediction')
    axes[1].set_xlabel('Ground Truth')
    axes[1].set_ylabel('Predictions')
    axes[1].set_title('Prediction Scatter Plot')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = result_dir / f'{model_name}_{model_type}_predictions.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"  Plot saved: {plot_path}")
    
    return mse, mae


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model_name, data_dir, base_output_dir, result_dir,
                train_val_split=0.8, steering_correction=0.2,
                batch_size=32, epochs=10, learning_rate=1e-4):
    """
    Train a float32 model with energy tracking
    
    Args:
        model_name: Name of the model architecture
        data_dir: Path to dataset
        base_output_dir: Base directory for outputs
        result_dir: Directory for plots
        train_val_split: Train/validation split ratio
        steering_correction: Steering correction for left/right cameras
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Initial learning rate
    
    Returns:
        Dictionary with training results
    """
    print("\n" + "="*70)
    print(f"TRAINING {model_name.upper()} (FLOAT32)")
    print("="*70)
    
    # Create output directories
    output_dir = Path(base_output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_paths, train_steerings, val_paths, val_steerings = load_data(
        data_dir, train_val_split, steering_correction
    )
    
    # Create data generators
    steps_per_epoch = len(train_paths) // batch_size
    validation_steps = len(val_paths) // batch_size
    
    train_gen = batch_generator(train_paths, train_steerings, batch_size, is_training=True)
    val_gen = batch_generator(val_paths, val_steerings, batch_size, is_training=False)
    
    print(f"\nSteps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    
    # Build model
    print(f"\nBuilding {model_name} model...")
    model = MODEL_DICT[model_name]()
    model.summary()
    
    # Verify Akida 1.0 compatibility
    print("\n" + "-"*70)
    print("VERIFYING AKIDA 1.0 COMPATIBILITY")
    print("-"*70)
    verify_akida_v1_compatibility(model)
    
    # Compile
    model.compile(
        optimizer=legacy.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        str(output_dir / f'{model_name}_float32_best.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    csv_logger = CSVLogger(str(output_dir / f'{model_name}_float32_history.csv'))
    
    energy_tracker = EnergyTrackerCallback(str(output_dir), phase='training')
    
    # Train
    print("\n" + "="*70)
    print(f"TRAINING FLOAT32 MODEL")
    print("="*70)
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[checkpoint, early_stop, reduce_lr, csv_logger, energy_tracker],
        verbose=1
    )
    
    # Load best model
    model = models.load_model(
        str(output_dir / f'{model_name}_float32_best.h5'),
        compile=False
    )
    model.compile(optimizer=legacy.Adam(1e-5), loss='mse', metrics=['mae'])
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    mse, mae = make_predictions_and_plot(
        model, val_paths, val_steerings, result_dir, model_name, "float32"
    )
    
    # Save final model
    final_path = output_dir / f'{model_name}_float32.h5'
    model.save(str(final_path))
    print(f"\n✓ Saved final model: {final_path}")
    
    # Summary
    print("\n" + "="*70)
    print(f" {model_name.upper()} TRAINING COMPLETE")
    print("="*70)
    print(f"\nModel saved: {final_path}")
    print(f"\nValidation Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"\nEnergy tracking saved in: {output_dir}")
    print(f"Prediction plots saved in: {result_dir}")
    print("\n" + "="*70)
    
    return {
        'model_name': model_name,
        'mse': mse,
        'mae': mae,
        'model_path': str(final_path)
    }


def verify_akida_v1_compatibility(model):
    """Verify that model architecture is compatible with Akida 1.0"""
    issues = []
    
    for layer in model.layers:
        # Check Conv2D with stride=2
        if isinstance(layer, layers.Conv2D):
            if layer.strides == (2, 2):
                kernel = layer.kernel_size
                if kernel not in [(3, 3), (1, 1)]:
                    issues.append(
                        f" {layer.name}: Conv2D with stride=2 has kernel={kernel} "
                        f"(must be 3x3 or 1x1 for Akida 1.0)"
                    )
        
        # Check for unsupported layers
        if isinstance(layer, layers.GlobalAveragePooling2D):
            issues.append(
                f" {layer.name}: GlobalAveragePooling2D not supported by Akida 1.0"
            )
        
        if isinstance(layer, layers.Lambda):
            issues.append(
                f" {layer.name}: Lambda layers not supported by Akida 1.0"
            )
        
        if isinstance(layer, layers.LayerNormalization):
            issues.append(
                f" {layer.name}: LayerNormalization not supported by Akida 1.0"
            )
    
    if issues:
        print("\n  COMPATIBILITY ISSUES DETECTED:")
        for issue in issues:
            print(f"  {issue}")
        print("\n Model is NOT compatible with Akida 1.0!")
        print("Please fix architecture before training.")
        return False
    else:
        print(" Architecture is compatible with Akida 1.0")
        print(" Ready for quantization and conversion on RPi5")
        return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train float32 models with energy tracking (Mac M1)'
    )
    
    # Model selection
    parser.add_argument('--model', type=str, default='all',
                       choices=['pilotnet', 'laksnet', 'mininet', 'all'],
                       help='Model to train (default: all)')
    
    # Paths
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Base output directory')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Directory for prediction plots')
    
    # Data settings
    parser.add_argument('--train_val_split', type=float, default=0.8,
                       help='Train/validation split ratio (default: 0.8)')
    parser.add_argument('--steering_correction', type=float, default=0.2,
                       help='Steering correction for left/right cameras (default: 0.2)')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate (default: 1e-4)')
    
    args = parser.parse_args()
    
    # Determine which models to train
    if args.model == 'all':
        models_to_train = ['pilotnet', 'laksnet', 'mininet']
    else:
        models_to_train = [args.model]
    
    print("\n" + "="*70)
    print("FLOAT32 TRAINING WITH ENERGY TRACKING (MAC M1)")
    print("="*70)
    print(f"Models to train: {', '.join(models_to_train)}")
    print(f"\nPaths:")
    print(f"  Data: {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Results: {args.result_dir}")
    print(f"\nSettings:")
    print(f"  Train/val split: {args.train_val_split}")
    print(f"  Steering correction: {args.steering_correction}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print("\n CodeCarbon energy tracking enabled")
    print("="*70)
    
    # Train all models
    all_results = []
    for model_name in models_to_train:
        results = train_model(
            model_name=model_name,
            data_dir=args.data_dir,
            base_output_dir=args.output_dir,
            result_dir=args.result_dir,
            train_val_split=args.train_val_split,
            steering_correction=args.steering_correction,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        all_results.append(results)
    
    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_path = Path(args.result_dir) / 'training_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*70)
    print(" ALL TRAINING COMPLETE")
    print("="*70)
    print(f"\nSummary saved to: {summary_path}")
    print("\n Results:")
    print(summary_df.to_string(index=False))
    print("\n Next Steps:")
    print("  1. Transfer .h5 files to RPi5:")
    for result in all_results:
        print(f"     {result['model_path']}")
    print("\n  2. On RPi5, run:")
    print("     python quantize_and_convert_rpi5.py --model_path <model>.h5")
    print("="*70)


if __name__ == "__main__":
    main()
