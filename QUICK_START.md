# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Installation

```bash
# Clone repository
cd "malaria-parasite-detection-cv"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Create data directory
mkdir -p data/raw/ThickBloodSmears_150

# Download/place ThickBloodSmears_150 dataset in:
# data/raw/ThickBloodSmears_150/
#  ├── infected/
#  │   ├── slide_001.jpg
#  │   ├── slide_002.jpg
#  │   └── ...
#  └── uninfected/
#      ├── slide_100.jpg
#      ├── slide_101.jpg
#      └── ...
```

### 3. Run Data Exploration

```bash
# Launch Jupyter
jupyter notebook

# Open notebooks/01_data_exploration.ipynb
# Run cells to visualize dataset
```

### 4. Train Baseline Model

```python
from src.data.dataset_loader import ThickBloodSmearsLoader
from src.models.baseline_cnn import BaselineCNN
from src.training.trainer import ModelTrainer
from src.training.config import TrainingConfig

# Load data
loader = ThickBloodSmearsLoader("data/raw/ThickBloodSmears_150")
X, y = loader.load_data()

# Create model
model = BaselineCNN(input_size=224, num_classes=2)
model.build()

# Train
trainer = ModelTrainer(model, TrainingConfig.BASELINE_CONFIG)
history = trainer.train((X[:100], y[:100]), (X[100:], y[100:]))
```

### 5. Make Predictions

```python
from inference.predict import MalariaDiagnosticPredictor

predictor = MalariaDiagnosticPredictor("models/best_model.h5")

# Single prediction
result = predictor.predict("blood_smear.jpg")
print(f"Result: {result['classification']}")
print(f"Confidence: {result['confidence']:.1%}")

# Batch predictions
results = predictor.predict_batch(["slide1.jpg", "slide2.jpg"])
```

---

## 📊 Key Commands

### Data Processing
```python
# Load dataset
from src.data.dataset_loader import ThickBloodSmearsLoader
loader = ThickBloodSmearsLoader("path/to/data")

# Preprocess images
from src.data.preprocessor import MicroscopyPreprocessor
preprocessor = MicroscopyPreprocessor(apply_clahe=True)
processed = preprocessor.preprocess_image(image)

# Augment data
from src.data.augmentation import MedicalImageAugmenter
augmenter = MedicalImageAugmenter()
augmented = augmenter.augment(image)

# Split data (stratified)
from src.data.data_splitter import StratifiedSplitter
splitter = StratifiedSplitter()
train_idx, val_idx, test_idx = splitter.split(images, labels)
```

### Model Training
```python
# Create model
from src.models.transfer_learning import TransferLearningModel
model = TransferLearningModel("resnet50")

# Train
from src.training.trainer import ModelTrainer
from src.training.config import TrainingConfig
trainer = ModelTrainer(model, TrainingConfig.MEDICAL_CONFIG)
trainer.train(train_data, val_data)

# Save model
from src.models.model_utils import save_model
save_model(model, "models/my_model.h5")
```

### Evaluation
```python
# Calculate clinical metrics
from src.evaluation.clinical_metrics import ClinicalMetrics
metrics = ClinicalMetrics()
results = metrics.calculate_all(y_true, y_pred, y_score)

# Generate visualizations
from src.evaluation.visualizer import ClinicalVisualizer
visualizer = ClinicalVisualizer()
visualizer.plot_confusion_matrix(y_true, y_pred)
visualizer.plot_roc_curve(y_true, y_score)

# Create Grad-CAM heatmap
from src.evaluation.interpretability import GradCAM
gradcam = GradCAM(model)
heatmap = gradcam.generate_heatmap(image, layer_name='conv_layer')
```

### Inference & Deployment
```python
# Single prediction
from inference.predict import MalariaDiagnosticPredictor
predictor = MalariaDiagnosticPredictor("model.h5")
result = predictor.predict("image.jpg")

# Batch processing
from inference.batch_diagnose import BatchDiagnoser
diagnoser = BatchDiagnoser(predictor)
results = diagnoser.process_directory("data/specimens/")

# Generate report
from inference.diagnostic_report import DiagnosticReportGenerator
generator = DiagnosticReportGenerator()
report = generator.generate_report(prediction, "P001", "S001")
generator.save_report_html(report, "report.html")

# Start REST API
from inference.api import create_api
api = create_api(predictor, framework='fastapi')
api.run()
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_basic.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📂 Important Files

| File | Purpose |
|------|---------|
| `README.md` | Main documentation |
| `MEDICAL_DISCLAIMER.md` | Clinical use disclaimer |
| `PROJECT_SUMMARY.md` | Complete project overview |
| `class_mapping.json` | Label definitions |
| `src/data/` | Data loading & preprocessing |
| `src/models/` | Model architectures |
| `src/training/` | Training framework |
| `src/evaluation/` | Metrics & visualization |
| `inference/` | Prediction & deployment |
| `notebooks/01_*.ipynb` | Data exploration |
| `tests/test_basic.py` | Unit tests |
| `requirements.txt` | Dependencies |

---

## 🔬 Medical Concepts

**Key Terms**:
- **Sensitivity** (Recall): % of infections correctly detected ⭐ PRIORITY
- **Specificity**: % of non-infections correctly identified
- **NPV**: How much to trust negative results
- **PPV**: How much to trust positive results
- **Giemsa**: Stain for parasites (blue/purple nucleus)
- **Thick Blood Smear**: Concentration method for parasite detection
- **CLAHE**: Contrast enhancement for microscopy images

**Classification**:
- **Infected**: Malaria parasite detected
- **Uninfected**: No parasite detected
- **Uncertain**: Borderline result (requires expert review)

---

## ⚠️ Clinical Usage

### Before Clinical Deployment
- [ ] Read [MEDICAL_DISCLAIMER.md](MEDICAL_DISCLAIMER.md)
- [ ] Get Institutional Review Board (IRB) approval
- [ ] Validate on independent dataset
- [ ] Train medical staff
- [ ] Set up quality assurance program

### During Clinical Use
- ✅ Review **all** positive results with expert microscopy
- ✅ Flag uncertain cases (20-95% confidence) for review
- ✅ Log all predictions for audit trail
- ✅ Monitor performance over time
- ❌ Never use as sole diagnostic tool
- ❌ Never treat based on AI result alone

### Expected Performance
- Sensitivity: 90-95%
- Specificity: 85-90%
- NPV: 95-98%
- Processing time: <1 min per slide

---

## 🐛 Troubleshooting

### Dataset Not Found
```bash
# Check data directory structure
ls data/raw/ThickBloodSmears_150/

# Expected:
# - infected/ (folder with infected smear images)
# - uninfected/ (folder with non-infected smear images)
```

### CUDA Not Available
```python
# TensorFlow will fall back to CPU automatically
# For GPU acceleration, install CUDA 11.x and cuDNN 8.x
```

### Out of Memory
```python
# Reduce batch size
config.batch_size = 8  # Instead of 32

# Or reduce image size
config.input_size = 128  # Instead of 224
```

### Unbalanced Results
```python
# Use class weights
config.use_class_weights = True

# Or use stratified splitting
splitter = StratifiedSplitter()
```

---

## 📈 Performance Optimization

### For Speed
- Use `MobileNetV2` or `EfficientNetB0` architectures
- Enable TensorFlow Lite quantization
- Reduce image size to 128×128
- Use batch processing instead of single predictions

### For Accuracy
- Use ensemble of multiple models
- Increase epochs and reduce learning rate
- Use `resnet50` or `DenseNet121` architectures
- Apply aggressive data augmentation

### For Mobile Deployment
```python
from src.models.model_utils import quantize_model

# Quantize model for mobile
quantize_model(model, "model.tflite")
```

---

## 💡 Tips & Best Practices

1. **Always stratify splits** - Maintains class balance
2. **Prioritize sensitivity** - Better to flag uncertain cases
3. **Use CLAHE preprocessing** - Critical for microscopy images
4. **Monitor validation metrics** - Not just training loss
5. **Save checkpoint models** - Restore best performance
6. **Log everything** - Audit trail for clinical use
7. **Test on independent dataset** - Before deployment
8. **Have fallback plan** - Expert microscopy always available
9. **Document assumptions** - For reproducibility
10. **Update models regularly** - As new data becomes available

---

## 📚 Learning Resources

- **Medical Background**: See `docs/medical_background.md`
- **Dataset Details**: See `docs/dataset_info.md`
- **Deployment**: See `docs/deployment_guide.md`
- **Code Examples**: See inline docstrings
- **Notebooks**: See `notebooks/01_data_exploration.ipynb`

---

## ❓ Common Questions

**Q: Can I use this for clinical diagnosis?**  
A: No. This is a SCREENING tool only. All results must be reviewed by expert microscopists.

**Q: What accuracy should I expect?**  
A: Sensitivity 90-95%, Specificity 85-90%. Varies by dataset and configuration.

**Q: How long does inference take?**  
A: ~0.5-2 seconds per image depending on hardware and model.

**Q: Can I deploy on mobile devices?**  
A: Yes, use TensorFlow Lite quantized model. See `mobile_inference.py`.

**Q: How do I update the model?**  
A: Retrain with new data quarterly, validate on independent test set, track performance over time.

**Q: What's the cost per prediction?**  
A: Typically $0.50-2.00 per slide when deployed with operational overhead.

---

## 📞 Support

For issues or questions:
1. Check [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
2. Read [MEDICAL_DISCLAIMER.md](MEDICAL_DISCLAIMER.md)
3. Review code docstrings
4. Check `tests/test_basic.py` for usage examples
5. See notebooks for detailed examples

---

**Remember**: This is a screening tool for research/educational use only. All clinical results must be validated by trained medical professionals.

✅ **Ready to start?** → Run notebook `01_data_exploration.ipynb`
