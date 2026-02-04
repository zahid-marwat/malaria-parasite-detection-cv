"""
Training configuration and hyperparameters.
"""

from dataclasses import dataclass
from typing import Literal, Optional
import yaml
from pathlib import Path


@dataclass
class TrainingConfig:
    """Training configuration class."""
    
    # Model configuration
    model_type: str = 'baseline'  # 'baseline', 'medical', 'resnet50', etc.
    input_size: int = 224
    num_classes: int = 2
    
    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    optimizer: str = 'adam'
    loss_function: str = 'weighted_bce'
    
    # Regularization
    dropout_rate: float = 0.5
    l1_reg: float = 0.0
    l2_reg: float = 1e-4
    
    # Early stopping and checkpointing
    early_stopping_patience: int = 15
    early_stopping_monitor: str = 'val_sensitivity'  # Medical metric!
    save_best_model: bool = True
    save_dir: str = 'models/'
    
    # Data augmentation
    augmentation_level: str = 'moderate'  # 'light', 'moderate', 'strong'
    
    # Class weighting (for imbalanced datasets)
    use_class_weights: bool = True
    class_weights: Optional[dict] = None
    
    # Validation
    validation_split: float = 0.15
    test_split: float = 0.15
    random_state: int = 42
    
    # Logging
    log_dir: str = 'tb_logs/'
    save_interval: int = 5  # Save logs every N epochs
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'model_type': self.model_type,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer,
            'loss_function': self.loss_function,
            'dropout_rate': self.dropout_rate,
            'augmentation_level': self.augmentation_level,
            'early_stopping_patience': self.early_stopping_patience,
        }
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f)


# Default configurations for different scenarios

BASELINE_CONFIG = TrainingConfig(
    model_type='baseline',
    batch_size=32,
    epochs=50,
    learning_rate=1e-3,
    augmentation_level='light'
)

MEDICAL_CONFIG = TrainingConfig(
    model_type='medical',
    batch_size=16,
    epochs=100,
    learning_rate=5e-4,
    augmentation_level='moderate',
    early_stopping_patience=20,
    dropout_rate=0.4
)

TRANSFER_CONFIG = TrainingConfig(
    model_type='resnet50',
    batch_size=16,
    epochs=50,
    learning_rate=1e-4,  # Lower for transfer learning
    augmentation_level='moderate',
    early_stopping_patience=15,
    dropout_rate=0.5
)

PRODUCTION_CONFIG = TrainingConfig(
    model_type='resnet50',
    batch_size=8,
    epochs=200,
    learning_rate=5e-5,
    augmentation_level='strong',
    early_stopping_patience=30,
    use_class_weights=True,
    dropout_rate=0.6
)
