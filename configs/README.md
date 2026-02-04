"""
Configuration files for training different model architectures.
"""

# This directory contains YAML configuration files for training
# Example usage:
#   python src/training/trainer.py --config configs/baseline_config.yaml

# File contents (to be created as separate YAML files):

# baseline_config.yaml
baseline_yaml = """
model_type: baseline
input_size: 224
num_classes: 2

batch_size: 32
epochs: 50
learning_rate: 0.001
optimizer: adam
loss_function: binary_crossentropy

dropout_rate: 0.3
augmentation_level: light

early_stopping_patience: 10
early_stopping_monitor: val_accuracy
save_best_model: true
save_dir: models/

validation_split: 0.15
test_split: 0.15
"""

# transfer_config.yaml
transfer_yaml = """
model_type: resnet50
input_size: 224
num_classes: 2

batch_size: 16
epochs: 100
learning_rate: 0.0001
optimizer: adam
loss_function: weighted_bce

dropout_rate: 0.5
augmentation_level: moderate

early_stopping_patience: 20
early_stopping_monitor: val_sensitivity
save_best_model: true
save_dir: models/

validation_split: 0.15
test_split: 0.15
use_class_weights: true
"""

# medical_config.yaml
medical_yaml = """
model_type: medical_cnn
input_size: 224
num_classes: 2

batch_size: 16
epochs: 100
learning_rate: 0.0005
optimizer: adam
loss_function: focal_loss

dropout_rate: 0.4
augmentation_level: moderate

early_stopping_patience: 20
early_stopping_monitor: val_sensitivity
save_best_model: true
save_dir: models/

validation_split: 0.15
test_split: 0.15
use_class_weights: true
"""

# production_config.yaml
production_yaml = """
model_type: resnet50
input_size: 224
num_classes: 2

batch_size: 8
epochs: 200
learning_rate: 0.00005
optimizer: adam
loss_function: focal_loss

dropout_rate: 0.6
augmentation_level: strong

early_stopping_patience: 30
early_stopping_monitor: val_sensitivity
save_best_model: true
save_dir: models/

validation_split: 0.15
test_split: 0.15
use_class_weights: true
"""
