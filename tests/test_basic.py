"""
Unit tests for malaria parasite detection system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import tensorflow as tf

# Test imports
from src.data.dataset_loader import ThickBloodSmearsLoader
from src.data.preprocessor import MicroscopyPreprocessor
from src.data.augmentation import MedicalImageAugmenter
from src.data.data_splitter import StratifiedSplitter
from src.evaluation.clinical_metrics import ClinicalMetrics


class TestDataLoading:
    """Tests for data loading functionality."""
    
    def test_loader_initialization(self):
        """Test ThickBloodSmearsLoader initialization."""
        loader = ThickBloodSmearsLoader(data_dir='data/raw/ThickBloodSmears_150')
        assert loader is not None
    
    def test_label_mapping(self):
        """Test label mapping."""
        loader = ThickBloodSmearsLoader(data_dir='data/raw/ThickBloodSmears_150')
        assert 'infected' in loader.label_map
        assert 'uninfected' in loader.label_map


class TestPreprocessing:
    """Tests for image preprocessing."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = MicroscopyPreprocessor(target_size=(224, 224))
        assert preprocessor.target_size == (224, 224)
    
    def test_preprocessing_dummy_image(self):
        """Test preprocessing of dummy image."""
        preprocessor = MicroscopyPreprocessor(target_size=(224, 224))
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Preprocess
        processed = preprocessor.preprocess_image(dummy_image)
        
        # Check output
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.float32


class TestAugmentation:
    """Tests for data augmentation."""
    
    def test_augmenter_initialization(self):
        """Test augmenter initialization."""
        augmenter = MedicalImageAugmenter(augmentation_level='moderate')
        assert augmenter is not None
    
    def test_augmentation_single_image(self):
        """Test augmentation of single image."""
        augmenter = MedicalImageAugmenter(augmentation_level='light')
        
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        augmented = augmenter.apply_single(dummy_image, num_augmentations=2)
        
        assert len(augmented) == 3  # Original + 2 augmentations


class TestDataSplitting:
    """Tests for data splitting."""
    
    def test_stratified_split(self):
        """Test stratified splitting."""
        splitter = StratifiedSplitter(test_size=0.15, val_size=0.15)
        
        # Create dummy data with balanced classes
        images = np.random.randn(100, 224, 224, 3)
        labels = np.array([0] * 50 + [1] * 50)
        
        (train_img, train_lbl), (val_img, val_lbl), (test_img, test_lbl) = splitter.split(
            images, labels
        )
        
        # Check splits
        assert len(train_img) == 70
        assert len(val_img) == 15
        assert len(test_img) == 15


class TestClinicalMetrics:
    """Tests for clinical metrics calculation."""
    
    def test_metrics_calculation(self):
        """Test clinical metrics calculation."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 0])
        
        metrics = ClinicalMetrics()
        results = metrics.calculate_all(y_true, y_pred)
        
        assert 'sensitivity' in results
        assert 'specificity' in results
        assert 'npv' in results
        assert 'auc_roc' not in results  # Need scores for AUC
    
    def test_sensitivity_priority(self):
        """Test that sensitivity is correctly calculated."""
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0])
        
        metrics = ClinicalMetrics()
        results = metrics.calculate_all(y_true, y_pred)
        
        # Sensitivity should be 2/3 (2 TPs out of 3 infected)
        expected_sensitivity = 2 / 3
        assert abs(results['sensitivity'] - expected_sensitivity) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
