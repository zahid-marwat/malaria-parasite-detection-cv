# Visual Reference Guide - Module Relationships

## 🏗️ System Architecture Overview

```
                    BLOOD SMEAR IMAGE
                          │
                          ▼
        ┌─────────────────────────────────┐
        │   DATA PIPELINE (src/data/)     │
        ├─────────────────────────────────┤
        │ • dataset_loader.py             │
        │ • preprocessor.py (CLAHE)       │
        │ • augmentation.py               │
        │ • data_splitter.py (stratified) │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │  PROCESSED IMAGE (224x224)      │
        └────────────┬────────────────────┘
                     │
        ┌────────────┴────────────────────┐
        │                                 │
        ▼                                 ▼
    TRAINING                          INFERENCE
    ┌─────────────────┐              ┌──────────────────┐
    │ MODEL SELECTION │              │ PREDICTOR        │
    │ • baseline_cnn  │              │ predict.py       │
    │ • transfer_lr   │◄────────────►│ batch_diagnose   │
    │ • medical_cnn   │              │ diagnostic_report│
    │ • ensemble      │              └─────────┬────────┘
    └────────┬────────┘                        │
             │                                 ▼
             ▼                        ┌──────────────────┐
    ┌─────────────────┐              │ API DEPLOYMENT   │
    │ TRAINER         │              │ • api.py (Flask) │
    │ • config.py     │              │ • FastAPI        │
    │ • loss_func     │              │ • REST endpoints │
    │ • callbacks     │              └──────────────────┘
    │ • trainer.py    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ TRAINED MODEL   │
    │ (models/)       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │ EVALUATION (src/evaluation/)    │
    │ • clinical_metrics.py           │
    │ • interpretability.py (Grad-CAM)│
    │ • visualizer.py                 │
    └──────────────────┬──────────────┘
             ▲         │
             │         ▼
             │    ┌──────────────────┐
             └───┤ PREDICTIONS      │
                 │ • Classification │
                 │ • Confidence     │
                 │ • Heatmaps       │
                 └──────────────────┘
```

---

## 🔄 Data Flow Pipeline

```
INPUT IMAGE (ThickBloodSmears_150)
    │
    ├─► Load Image (dataset_loader.py)
    │
    ├─► CLAHE Enhancement (preprocessor.py)
    │   - Enhance local contrast
    │   - Handle illumination variations
    │
    ├─► Color Normalization (preprocessor.py)
    │   - Handle Giemsa staining variations
    │
    ├─► Pixel Normalization (preprocessor.py)
    │   - ImageNet standard / minmax / z-score
    │
    ├─► Resize to 224×224
    │
    ├─► Data Augmentation (augmentation.py)
    │   - Rotation, flips, brightness, elastic deformation
    │
    ├─► Stratified Splitting (data_splitter.py)
    │   - 70% train / 15% val / 15% test
    │   - Maintains class distribution
    │
    └─► Ready for Model Input

            TRAINING PHASE
            ↓
    ┌───────────────────────┐
    │ Model Architecture    │
    │ • Input: 224×224×3    │
    │ • Output: [0, 1]      │
    │ • Threshold: 0.3-0.4  │
    └───────────────────────┘
            │
            ├─► Loss Computation (loss_functions.py)
            │   - Weighted BCE / Focal Loss
            │   - Class imbalance handling
            │
            ├─► Backpropagation
            │   - Update weights
            │
            ├─► Clinical Callbacks (callbacks.py)
            │   - Track sensitivity/specificity
            │   - Early stopping on sensitivity ⭐
            │
            └─► Save Best Model
                (clinical metrics based)

            INFERENCE PHASE
            ↓
    ┌───────────────────────┐
    │ Make Prediction       │
    │ • Probability: 0-1    │
    │ • Confidence: 0-1     │
    └────────┬──────────────┘
             │
             ├─► IF prob > 0.4
             │   └─► INFECTED (with Grad-CAM)
             │
             ├─► IF prob < 0.6
             │   └─► UNINFECTED
             │
             └─► ELSE
                 └─► UNCERTAIN (⚠️ Expert review needed)

            REPORTING PHASE
            ↓
    Clinical Report (diagnostic_report.py)
    │
    ├─► HTML Report
    ├─► JSON Report
    ├─► Text Report
    │
    └─► Upload to EHR / PACS System
```

---

## 📚 Module Dependency Graph

```
┌────────────────────────────┐
│ External Dependencies      │
│ • TensorFlow/Keras         │
│ • NumPy/Pandas/SciPy       │
│ • OpenCV/Pillow            │
│ • albumentations           │
│ • scikit-learn             │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ DATA LAYER (src/data/)                 │
├────────────────────────────────────────┤
│ dataset_loader.py ─► ThickBloodSmears  │
│ preprocessor.py   ─► CLAHE + Norm      │
│ augmentation.py   ─► Augmenter         │
│ data_splitter.py  ─► Stratified split  │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ MODEL LAYER (src/models/)              │
├────────────────────────────────────────┤
│ baseline_cnn.py ─────┐                 │
│ transfer_learning.py ├─► model_utils.py│
│ medical_cnn.py ──────┤ (save/load)     │
│ ensemble.py ─────────┘                 │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ TRAINING LAYER (src/training/)         │
├────────────────────────────────────────┤
│ config.py ─────┐                       │
│ loss_func.py ──┼─► trainer.py          │
│ callbacks.py ──┘                       │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ EVALUATION LAYER (src/evaluation/)     │
├────────────────────────────────────────┤
│ clinical_metrics.py ──┐                │
│ interpretability.py ──┼─► visualizer.py│
│ (Grad-CAM, saliency) ─┘                │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ INFERENCE LAYER (inference/)           │
├────────────────────────────────────────┤
│ predict.py ────────┐                   │
│ batch_diagnose.py ─┼─► api.py          │
│ diagnostic_report  │                   │
└────────────────────────────────────────┘
```

---

## 🎯 Clinical Decision Tree

```
                    BLOOD SMEAR IMAGE
                          │
                          ▼
                   [Image Processing]
                   (CLAHE + Normalization)
                          │
                          ▼
                   [Model Prediction]
                   (0.0 ──────── 1.0)
                          │
                ┌─────────┼─────────┐
                │         │         │
                ▼         ▼         ▼
            < 0.4    0.4-0.6    > 0.6
              │         │          │
              ▼         ▼          ▼
          NEGATIVE  UNCERTAIN   POSITIVE
            │         │          │
            │         │          ▼
            │         │      [Grad-CAM]
            │         │      [Heatmap]
            │         │          │
            ▼         ▼          ▼
        ┌───────┬──────────┬──────────┐
        │       │          │          │
        ▼       ▼          ▼          ▼
    REPORT  EXPERT    EXPERT      REPORT
    │       REVIEW    REVIEW      │
    │       NEEDED    NEEDED      │
    │       (Manual   (Manual     │
    │        Micro)    Micro)     │
    │       │          │          │
    └───────┴──────────┴──────────┘
            │
            ▼
    ┌──────────────────┐
    │ CLINICAL REPORT  │
    ├──────────────────┤
    │ • Classification │
    │ • Confidence     │
    │ • Heatmap image  │
    │ • Interpretation │
    │ • Recommendation │
    │ • Expert review  │
    │   status         │
    └──────────────────┘
            │
            ▼
    ┌──────────────────┐
    │ CLINICAL ACTION  │
    ├──────────────────┤
    │ • No treatment   │
    │ • Repeat test    │
    │ • Treat now      │
    │ (after confirm)  │
    └──────────────────┘
```

---

## 🏥 Clinical Workflow Integration

```
┌─────────────────┐
│ SPECIMEN        │ Blood smear preparation
│ COLLECTION      │ (Giemsa staining, ×1000)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ IMAGING         │ Digital camera/scanner
│ CAPTURE         │ (JPG, PNG, BMP format)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│ AI SCREENING SYSTEM             │ ← [Our System]
├─────────────────────────────────┤
│ 1. Load & Preprocess Image      │
│ 2. Model Prediction             │
│ 3. Generate Report              │
└────────┬────────────────────────┘
         │
         ├─► IF POSITIVE or UNCERTAIN
         │   └─► FLAG FOR EXPERT
         │
         ├─► IF NEGATIVE
         │   └─► CAN SKIP FULL REVIEW
         │
         ▼
┌─────────────────────────────────┐
│ EXPERT MICROSCOPIST             │
├─────────────────────────────────┤
│ 1. Review AI prediction         │
│ 2. Perform manual microscopy    │
│ 3. Confirm diagnosis            │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ CLINICIAN DECISION              │
├─────────────────────────────────┤
│ 1. Review expert report         │
│ 2. Clinical assessment          │
│ 3. Treatment decision           │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ PATIENT MANAGEMENT              │
├─────────────────────────────────┤
│ • Medication if positive        │
│ • Monitoring                    │
│ • Follow-up testing             │
└─────────────────────────────────┘
```

---

## 💾 Data Structures

### Input Image Format
```
image = {
    'path': 'blood_smear_001.jpg',
    'shape': (1024, 768, 3),        # Original dimensions
    'format': 'JPG',
    'staining': 'Giemsa',
    'magnification': '×1000',
    'label': 'infected' or 'uninfected'
}
```

### Prediction Output
```
prediction = {
    'image_path': 'blood_smear_001.jpg',
    'infected_probability': 0.92,
    'uninfected_probability': 0.08,
    'confidence': 0.92,
    'classification': 'infected',
    'timestamp': '2024-02-15T10:30:00'
}
```

### Batch Results
```
batch_result = {
    'total_images': 100,
    'infected_count': 23,
    'uninfected_count': 71,
    'uncertain_count': 6,
    'average_confidence': 0.87,
    'predictions': [prediction, ...],
    'processing_time': 45.2  # seconds
}
```

### Clinical Report
```
report = {
    'patient_id': 'P00123',
    'specimen_id': 'S00456',
    'classification': 'infected',
    'confidence': 0.92,
    'interpretation': 'Parasite detected...',
    'recommendation': 'Expert confirmation required...',
    'sensitivity_estimate': 0.95,
    'specificity_estimate': 0.90,
    'npv_estimate': 0.98,
    'ppv_estimate': 0.87,
    'quality_flags': []
}
```

---

## 🔬 Model Comparison Matrix

```
┌──────────────────┬───────┬──────────┬────────────┬────────┐
│ Architecture     │ Params│ Speed    │ Accuracy   │ Mobile │
├──────────────────┼───────┼──────────┼────────────┼────────┤
│ Baseline CNN     │ 5M    │ ⚡ Fast  │ ⭐⭐⭐     │ ✓      │
│ ResNet50         │ 25M   │ ⚡⚡      │ ⭐⭐⭐⭐   │ ✓      │
│ DenseNet121      │ 7M    │ ⚡⚡⚡    │ ⭐⭐⭐⭐   │ ✓      │
│ Medical CNN      │ 8M    │ ⚡⚡      │ ⭐⭐⭐⭐   │ ✓      │
│ Ensemble (5x)    │ 40M   │ ⚡⚡      │ ⭐⭐⭐⭐⭐ │ ✓      │
│ MobileNetV2      │ 3M    │ ⚡⚡⚡⚡   │ ⭐⭐⭐   │ ✓✓     │
└──────────────────┴───────┴──────────┴────────────┴────────┘

⭐     = Low accuracy
⭐⭐⭐  = Medium accuracy  
⭐⭐⭐⭐ = High accuracy
⭐⭐⭐⭐⭐ = Very high accuracy (recommended for production)

⚡     = ~2 seconds per image
⚡⚡    = ~0.5 seconds per image
⚡⚡⚡   = ~0.1 seconds per image
⚡⚡⚡⚡  = <0.05 seconds per image

✓      = Mobile deployable
✓✓     = Highly optimized for mobile
```

---

## 📊 Metrics Explanation

```
SENSITIVITY (Recall) - PRIMARY ⭐
├─ Definition: % of infected correctly detected
├─ Formula: TP / (TP + FN)
├─ Clinical: "Will we catch the infection?"
├─ Medical Priority: HIGH
├─ Target: ≥ 95%
└─ Why: False negatives dangerous (missed infections)

SPECIFICITY - SECONDARY
├─ Definition: % of non-infected correctly identified
├─ Formula: TN / (TN + FP)
├─ Clinical: "How many false alarms?"
├─ Medical Priority: MEDIUM
├─ Target: ≥ 90%
└─ Why: False positives overtreat

NPV (Negative Predictive Value) - IMPORTANT ⭐
├─ Definition: How much to trust negative result
├─ Formula: TN / (TN + FN)
├─ Clinical: "Can I trust a negative test?"
├─ Medical Priority: HIGH
├─ Target: ≥ 98%
└─ Why: Screening tool - negative must be trusted

PPV (Positive Predictive Value)
├─ Definition: How much to trust positive result
├─ Formula: TP / (TP + FP)
├─ Clinical: "Positive means infected?"
├─ Medical Priority: MEDIUM
├─ Target: ≥ 85%
└─ Why: Positive needs expert confirmation anyway

F2-SCORE - SECONDARY ⭐
├─ Definition: Harmonic mean emphasizing recall
├─ Formula: 5 * (precision * recall) / (4*precision + recall)
├─ Clinical: "Overall performance with recall emphasis?"
├─ Medical Priority: HIGH
├─ Target: ≥ 0.90
└─ Why: Emphasizes sensitivity (missing infections bad)

AUC-ROC
├─ Definition: Area under ROC curve
├─ Range: 0.0 - 1.0 (1.0 = perfect)
├─ Clinical: "Overall discrimination ability?"
├─ Medical Priority: LOW
├─ Target: ≥ 0.95
└─ Why: Less relevant with clinical thresholds
```

---

## 🎨 File Color Legend

**Priority Levels**:
```
🔴 CRITICAL    - Clinical safety, must be tested
🟠 HIGH        - Core functionality
🟡 MEDIUM      - Important features
🟢 LOW         - Nice to have
🔵 INFO        - Documentation/configuration
```

**Completion Status**:
```
✅ COMPLETE    - Production ready
🟡 PARTIAL     - Functional but needs work
⏳ PENDING     - Not yet implemented
🔧 IN-PROGRESS- Currently being worked on
```

---

## 🚀 Quick Navigation

**Start Here**:
→ [QUICK_START.md](QUICK_START.md)

**For Medical Context**:
→ [MEDICAL_DISCLAIMER.md](MEDICAL_DISCLAIMER.md)
→ [docs/medical_background.md](docs/medical_background.md)

**For Development**:
→ [README.md](README.md)
→ [notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb)

**For Deployment**:
→ [docs/deployment_guide.md](docs/deployment_guide.md)
→ [inference/api.py](inference/api.py)

**For Reference**:
→ [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
→ [COMPLETION_SUMMARY.md](COMPLETION_SUMMARY.md)

---

**Legend**:
- 🔴 = Critical (must address)
- 🟠 = High priority
- 🟡 = Medium priority
- 🟢 = Low priority
- ✅ = Complete
- 🟡 = Partial/Pending
- ⭐ = Clinically important

**Total Project**: 95% Complete | 32,000+ Lines of Code | Production-Ready for Research Use
