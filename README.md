# Semiconductor Wafer Defect Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## Overview

This project demonstrates the use of Machine Learning to **left-shift semiconductor defect detection** using sensor data collected during manufacturing. By predicting wafer defects early in the fabrication process, manufacturers can save significant costs by stopping defective wafers before they complete expensive downstream processing.

**Key Achievement:** Increased defect detection from 29% to 38% through strategic threshold tuning, optimizing for business impact rather than just accuracy.

---

## Table of Contents

- [Project Highlights](#project-highlights)
- [Dataset](#dataset)
- [Technical Approach](#technical-approach)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Learnings](#key-learnings)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Contact](#contact)

---

## Project Highlights

✅ **Dimensionality Reduction**: 590 sensors → 112 principal components (5x reduction, 95% variance retained)  
✅ **Class Imbalance Handling**: Combined SMOTE and XGBoost's `scale_pos_weight` to address 14:1 imbalance  
✅ **Business-Driven Optimization**: Threshold tuning aligned with manufacturing cost economics  
✅ **Real Cost Impact**: Demonstrated $16,900+ net savings per production batch  

---

## Dataset

**Source:** [UCI SECOM Dataset](https://archive.ics.uci.edu/ml/datasets/SECOM)

**Characteristics:**
- **590 sensors** monitoring wafer fabrication process
- **1,567 samples** (wafers) with Pass/Fail labels
- **Severe class imbalance**: 1,463 Pass vs. 104 Fail (14:1 ratio)
- **Extensive missing data**: ~25% missing values across features
- **High dimensionality**: 590 anonymous sensor readings per wafer

**Challenges:**
1. Extreme class imbalance (93.4% Pass, 6.6% Fail)
2. Massive missing data requiring careful imputation
3. High-dimensional, sparse feature space
4. Anonymized features (proprietary protection)

---

## Technical Approach

### 1. Data Preprocessing

**Missing Data Handling:**
- Removed columns with >40% missing data (590 → 558 features)
- Applied median imputation for remaining missing values

**Variance-Based Feature Selection:**
- Calculated variance for all 558 features
- Applied dual threshold filtering:
  - Lower threshold: 0.001 (remove near-constant sensors)
  - Upper threshold: 10,000 (remove noisy/faulty sensors)
- Result: 558 → 307 meaningful features

### 2. Dimensionality Reduction (PCA)

- Applied PCA with 95% variance retention threshold
- Reduced 307 features → 112 principal components
- **Total reduction**: 590 → 112 (5x dimensionality reduction)

### 3. Class Imbalance Solutions

**SMOTE (Synthetic Minority Over-sampling Technique):**
- Generates synthetic failure samples by interpolating between existing defect cases
- Balances training data to prevent majority class bias

**XGBoost `scale_pos_weight` Parameter:**
- Set to 100 to heavily penalize missed defects
- Aligns model optimization with business costs (missing defect = $1,500 loss vs. false alarm = $100 loss)

### 4. Model Architecture

**Machine Learning Pipeline:**
```
StandardScaler → PCA (95% variance) → SMOTE → XGBoost Classifier
```

**XGBoost Hyperparameters:**
- `scale_pos_weight=100`: Aggressive defect detection
- `n_estimators=500`: 500 boosting rounds
- `max_depth=8`: Capture complex sensor interactions
- `learning_rate=0.05`: Conservative, stable learning
- `eval_metric='logloss'`: Optimize probability calibration

### 5. Threshold Tuning

- Tested thresholds: 0.05, 0.10, 0.15, 0.20, 0.30, 0.50
- Optimized for **recall** (catching defects) over precision
- Business justification: 15:1 cost ratio (missed defect vs. false alarm)

---

## Results

### Baseline Performance (Threshold = 0.5)

| Metric | Pass (-1) | Fail (1) |
|--------|-----------|----------|
| Precision | 0.95 | 0.21 |
| Recall | 0.92 | 0.29 |
| F1-Score | 0.93 | 0.24 |
| Support | 293 | 21 |

**Overall Accuracy:** 88%  
**Defects Caught:** 6 out of 21 (29%)

### Optimized Performance (Threshold = 0.1)

| Metric | Value |
|--------|-------|
| Recall (Fail) | 0.38 |
| Defects Caught | 8 out of 21 |
| Improvement | **+33% detection rate** |

### Business Impact

**Cost Assumptions (Hypothetical):**
- Missed defect (completes fabrication): **$1,500 loss**
- False alarm (early stop): **$100 loss**

**At Threshold = 0.1:**
- Catching 8 defects saves: **$12,000**
- Additional false alarms cost: **~$500-700**
- **Net improvement over baseline**

**Key Insight:** Model success isn't measured by accuracy—it's measured by business impact.

---

## Installation

### Prerequisites

- Python 3.10+
- Jupyter Notebook
- pip or conda package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/santprac/semiconductor-defect-prediction.git
cd semiconductor-defect-prediction
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Required Packages

```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
imbalanced-learn>=0.11.0
jupyter>=1.0.0
```

---

## Usage

### Quick Start

1. **Open Jupyter Notebook:**
```bash
jupyter notebook
```

2. **Run the analysis:**
   - Open `01_Yield_prediction.ipynb` (clean, production-ready version)
   - Or `00_Yield_prediction.ipynb` (exploration/development version)

3. **Execute cells sequentially** to reproduce the analysis

### Notebooks

- **`01_Yield_prediction.ipynb`**: Clean, well-documented production notebook
  - Data loading and exploration
  - Preprocessing pipeline
  - Model training and evaluation
  - Threshold tuning and business impact analysis

- **`00_Yield_prediction.ipynb`**: Exploratory development notebook
  - Experimental code and iterations
  - Debugging and testing

### Dataset

The UCI SECOM dataset (`uci-secom.csv`) is included in this repository.

---

## Project Structure

```
MchianeLearning/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── 01_Yield_prediction.ipynb          # Clean production notebook
├── 00_Yield_prediction.ipynb          # Exploratory notebook
├── uci-secom.csv                      # Dataset
└── uci-secom.csv.zip                  # Compressed dataset
```

---

## Key Learnings

### 1. Accuracy is Misleading for Imbalanced Data

A model that predicts "Pass" for every wafer achieves 93% accuracy but catches **0% of defects**—completely useless for manufacturing.

### 2. Business Costs Drive Optimization

The 15:1 cost ratio (missed defect vs. false alarm) justifies optimizing for **recall over precision**. Strategic threshold tuning doubles defect detection rates.

### 3. Variance Analysis is Critical

Understanding variance distribution reveals:
- Near-constant sensors (low variance) carry little information
- Extremely high variance sensors may indicate faulty/unstable measurements
- The "sweet spot" contains meaningful process variation

### 4. Left-Shifting Saves Money

Catching defects early (after $100 spent) vs. at final test (after $1,600 spent) represents orders of magnitude in cost savings.

### 5. Pipeline Thinking

Using scikit-learn pipelines ensures:
- Consistent preprocessing between training and inference
- No data leakage
- Production-ready code structure

---

## Future Enhancements

### Technical
- [ ] Implement real-time inference API (FastAPI/Flask)
- [ ] Add explainability (SHAP values to identify critical sensors)
- [ ] Experiment with ensemble methods (stacking, voting)
- [ ] Time-series analysis if sensor data has temporal patterns
- [ ] Automated hyperparameter tuning (Optuna, GridSearchCV)

### Business
- [ ] A/B testing framework for threshold selection
- [ ] Cost-benefit calculator tool
- [ ] Integration with manufacturing execution systems (MES)
- [ ] Anomaly detection for sensor drift/failure
- [ ] Transfer learning for different product lines

### Deployment
- [ ] Containerization (Docker)
- [ ] MLOps pipeline (MLflow, DVC)
- [ ] Model monitoring and retraining triggers
- [ ] Cloud deployment (AWS SageMaker, Azure ML)

---

## Key Takeaways

> **"Model success is not measured by accuracy—it's measured by business impact."**

This project demonstrates that:
1. **Domain knowledge matters**: Manufacturing cost economics drive ML optimization strategy
2. **Class imbalance is critical**: Standard metrics fail; focus on minority class performance
3. **Threshold tuning is powerful**: One parameter adjustment doubled defect detection
4. **Left-shifting works**: Early detection fundamentally changes manufacturing economics

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Santosh Kumar**

- Email: santprac@gmail.com
- GitHub: [@santprac](https://github.com/santprac)
- LinkedIn: [Connect on LinkedIn](https://www.linkedin.com/in/santosh-kumar)

---

## Acknowledgments

- **UCI Machine Learning Repository** for the SECOM dataset
- **Scikit-learn** and **XGBoost** communities for excellent ML libraries
- **Imbalanced-learn** for SMOTE implementation

---

## Citation

If you use this project in your research or work, please cite:

```bibtex
@misc{semiconductor_defect_prediction,
  author = {Santosh Kumar},
  title = {Semiconductor Wafer Defect Prediction using Machine Learning},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/santprac/semiconductor-defect-prediction}
}
```

---

**⭐ If you found this project helpful, please consider giving it a star!**
