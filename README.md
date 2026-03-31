# 🍽️ NYC Food Safety Inspection Grade Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Models-red?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![University](https://img.shields.io/badge/Pace%20University-CS677-blue)

> A machine learning final project that predicts the **food safety inspection grade (A, B, or C)** assigned to NYC food establishments using multiple classification algorithms — built for CS677: Machine Learning at Pace University (Fall 2025).

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Models & Results](#-models--results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
- [Author](#-author)
- [License](#-license)

---

## 🔬 Overview

This project applies and compares **6 Machine Learning classification algorithms** on the NYC Food Safety Inspections dataset. The goal is to predict whether a food establishment will receive a Grade A, B, or C on their health inspection.

The notebook covers:
- ✅ **Problem Statement** — Multiclass classification (A / B / C)
- ✅ **EDA** — 8 separate visualizations covering distribution, geography, time series, grade by type, grade by county, grade by year, and correlations
- ✅ **Feature Engineering** — 8 new engineered features from raw columns
- ✅ **Train/Validation/Test Split** — Stratified 60/20/20 split
- ✅ **5 ML Models** — Decision Tree, Random Forest, LR, SVM Linear, SVM RBF
- ✅ **SGD Optimizer** — Incremental gradient descent with error tracking over 100 epochs
- ✅ **Hyperparameter Tuning** — GridSearchCV on Decision Tree and Random Forest
- ✅ **Full Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrices, Classification Reports
- ✅ **Model Comparison** — Summary table, bar charts, train vs test analysis

---

## 📊 Dataset

**Source:** NYC Open Data — Food Safety Inspections Current Ratings

| Property | Details |
|---|---|
| Original Records | ~137,000 |
| Sampled Records | 41,075 (30%) |
| Raw Features | 13 |
| Engineered Features | 8 additional |
| Total Features Used | 14 |
| Target | `Inspection Grade` (A / B / C) |
| Class Balance | C: 52.66% / B: 24.32% / A: 23.01% |
| Date Range | 2023 – 2025 |
| Missing Values | Deficiency cols: 23.0%, Georeference: 0.53% |

### Raw Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| `County` | Categorical | County of the establishment |
| `Inspection Grade` | **Target** | A, B, or C |
| `Inspection Date` | Date | Date of the inspection |
| `Establishment Types` | Categorical | Type code of the establishment |
| `Trade Name` | Categorical | Business name |
| `Street` | Categorical | Street address |
| `City` | Categorical | City |
| `Zip Code` | Numerical | ZIP code |
| `Georeference` | Geo | GPS coordinates (POINT format) |

### Engineered Features

| Feature | Description | Importance |
|---|---|---|
| `Failure_Rate` | Historical Grade C rate per establishment | **44.9%** ⭐ |
| `Previous_Grade_Encoded` | Encoded previous inspection grade | **28.0%** ⭐ |
| `Total_Inspections` | Cumulative inspection count | 9.8% |
| `Days_Since_Last` | Days between inspections | 7.9% |
| `Longitude` / `Latitude` | Extracted from Georeference | 3.5% |
| `Has_Multiple_Types` | Length of establishment type string | 1.2% |
| `Is_NYC` | 1 if NYC borough, 0 otherwise | 1.1% |

---

## 📁 Project Structure
```
nyc-food-safety-ml/
│
├── NYC_Food_Safety_Inspection_Grade_Prediction.ipynb  # Main notebook
├── README.md                                           # Documentation
└── requirements.txt                                    # Dependencies
```

---

## 🔬 Methodology

### Step 1 — Exploratory Data Analysis (8 visualizations)
- Target variable distribution (bar + pie chart)
- Establishment types analysis (top 15)
- Geographic distribution (county and city)
- Time series — monthly and yearly inspection volume
- Grade distribution by establishment type
- Grade distribution by county
- Grade distribution by year
- Correlation heatmap of numerical features

### Step 2 — Feature Engineering
8 new features created:
- `Has_Multiple_Types`, `Is_NYC` — domain knowledge flags
- `Longitude`, `Latitude` — extracted from POINT coordinates
- `Previous_Grade` — lag feature per establishment
- `Days_Since_Last` — time since last inspection
- `Total_Inspections` — cumulative count per establishment
- `Failure_Rate` — historical Grade C proportion per establishment

### Step 3 — Data Splitting (Stratified)
```
Total (41,075) → 80% temp + 20% test
temp           → 75% train + 25% val
────────────────────────────────────
Training set   : 24,645 samples (60%)
Validation set :  8,215 samples (20%)
Test set       :  8,215 samples (20%)
```

### Step 4 — Feature Scaling
`StandardScaler` fitted on training data only — preventing data leakage when applied to validation and test sets.

### Step 5 — Model Training
5 models trained with default parameters. Train, Validation, and Test accuracy tracked for all.

### Step 6 — SGD Optimizer
`SGDClassifier` with `partial_fit()` for true incremental gradient descent over 100 epochs with accuracy and error tracking plots.

### Step 7 — Hyperparameter Tuning (GridSearchCV)

**Decision Tree:**
```python
param_grid = {
    'max_depth'        : [5, 10, 15, 20],
    'min_samples_split': [10, 20, 50],
    'min_samples_leaf' : [5, 10]
}
# Best: max_depth=10, min_samples_leaf=10, min_samples_split=50
# Best CV Score: 0.9159
```

**Random Forest:**
```python
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth'   : [10, 20, None]
}
# Best: max_depth=10, n_estimators=200
# Best CV Score: 0.9139
```

### Step 8 — Evaluation
All models evaluated on Accuracy, Precision, Recall, F1-Score with Confusion Matrices and Classification Reports.

---

## 📈 Models & Results

### Default Model Performance

| Model | Train Acc | Val Acc | Test Acc | Status |
|---|---|---|---|---|
| Decision Tree | 0.9999 | 0.8816 | 0.8889 | ⚠️ Overfitting |
| Random Forest | 0.9999 | 0.9108 | 0.9100 | ✅ Good Fit |
| Logistic Regression | 0.8707 | 0.8676 | 0.8684 | ✅ Good Fit |
| SVM RBF | 0.8456 | 0.8329 | 0.8398 | ✅ Good Fit |
| SVM Linear | 0.7919 | 0.7928 | 0.7877 | ✅ Good Fit |

### Cross-Validation Results (5-Fold Stratified)

| Model | CV Mean | CV Std |
|---|---|---|
| Random Forest | 0.9111 | 0.0037 |
| Decision Tree | 0.8881 | 0.0049 |
| Logistic Regression | 0.8702 | 0.0047 |
| SVM RBF | 0.8672 | 0.0160 |
| SVM Linear | 0.7309 | 0.0709 |

### Final Model Performance (After Tuning)

| Rank | Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| 🥇 1 | **Decision Tree (Tuned)** | **0.9110** | **0.9186** | **0.9110** | **0.9101** |
| 🥈 2 | Random Forest (Tuned) | 0.9099 | 0.9168 | 0.9099 | 0.9088 |
| 3 | Logistic Regression | 0.8684 | 0.8700 | 0.8684 | 0.8680 |
| 4 | SGD | 0.8696 | — | — | — |
| 5 | SVM RBF | 0.8398 | — | — | — |
| 6 | SVM Linear | 0.7877 | — | — | — |

### Hyperparameter Tuning Impact

| Model | Before Tuning | After Tuning | Improvement |
|---|---|---|---|
| Decision Tree | 0.8889 | 0.9110 | +2.21% ✅ |
| Random Forest | 0.9100 | 0.9099 | ~0.00% (already optimal) |

### Per-Class Performance (Best Model — Decision Tree Tuned)

| Grade | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Grade A | 0.77 | 0.93 | 0.84 | 1,890 |
| Grade B | 0.95 | 0.74 | 0.83 | 1,998 |
| Grade C | 0.97 | 0.98 | 0.98 | 4,327 |
| **Weighted Avg** | **0.92** | **0.91** | **0.91** | **8,215** |

### SGD Optimizer Results

| Metric | Value |
|---|---|
| Final Train Accuracy | 0.8731 |
| Final Test Accuracy | 0.8696 |
| Epochs | 100 |
| Learning Rate | 0.01 (constant) |

### Top Feature Importances (Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `Failure_Rate` | 44.89% |
| 2 | `Previous_Grade_Encoded` | 27.98% |
| 3 | `Total_Inspections` | 9.81% |
| 4 | `Days_Since_Last` | 7.91% |
| 5 | `Longitude` | 1.88% |
| 6 | `Latitude` | 1.59% |
| 7 | `Has_Multiple_Types` | 1.20% |
| 8 | `Establishment_Encoded` | 1.11% |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Clone & Setup
```bash
# Clone the repository
git clone https://github.com/krishnamaniyar2209/nyc-food-safety-ml.git

# Navigate to the folder
cd nyc-food-safety-ml

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook NYC_Food_Safety_Inspection_Grade_Prediction.ipynb
```

### requirements.txt
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

---

## 🚀 Usage

1. Download the dataset from NYC Open Data and save as:
```
Food_Safety_Inspections_–_Current_Ratings_20251215.csv
```
2. Upload to Google Colab or place in the same folder as the notebook
3. Update the file path in Cell 3 if needed:
```python
df = pd.read_csv("/content/Food_Safety_Inspections_–_Current_Ratings_20251215.csv")
```
4. Run all cells sequentially from top to bottom
5. All EDA plots, metrics, confusion matrices, and comparisons generate automatically

---

## 💡 Key Findings

- **Grade C dominates** at 52.66% — class imbalance is present but handled via stratified splits
- **Failure Rate** (44.9%) and **Previous Grade** (28.0%) are by far the most important predictors — an establishment's history strongly determines its future grade
- **Kings (Brooklyn) and Bronx** have the highest Grade C rates among counties (64%)
- **CDK, ABCH, CD, and B** establishment types have Grade C rates above 70%
- **Decision Tree (Tuned)** achieved the best overall performance with 91.10% accuracy after GridSearchCV eliminated overfitting (from 99.99% → 91.10% train accuracy)
- **Random Forest** had the most stable cross-validation (lowest std: 0.0037) confirming ensemble robustness
- **SVM Linear** had the lowest accuracy (78.77%) — confirms non-linear relationships exist in the data
- **SVM RBF** outperformed Linear SVM — the RBF kernel better captures the non-linearity
- **SGD** converged quickly within ~20 epochs reaching stable 87% accuracy
- **Hyperparameter tuning** improved Decision Tree by +2.21% while Random Forest was already near-optimal
- **Grade C is predicted most accurately** (F1=0.98) due to class majority; Grade A and B are harder to distinguish

---

## 🛠️ Technologies Used

| Tool | Version | Purpose |
|---|---|---|
| [Python](https://python.org) | 3.10+ | Core language |
| [scikit-learn](https://scikit-learn.org/) | Latest | All ML models, metrics, GridSearchCV |
| [pandas](https://pandas.pydata.org/) | Latest | Data manipulation |
| [NumPy](https://numpy.org/) | Latest | Numerical operations |
| [Matplotlib](https://matplotlib.org/) | Latest | Plotting |
| [Seaborn](https://seaborn.pydata.org/) | Latest | Statistical visualization |
| [Jupyter](https://jupyter.org/) | Latest | Development environment |

---

## 👤 Author

**Krishna Maniyar**
- 🎓 Pace University — Seidenberg School of CSIS
- 📘 CS677: Machine Learning (CRN: 72828) | Fall 2025
- 🔗 [GitHub](https://github.com/krishnamaniyar2209)

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">
  Made with ❤️ for CS677 @ Pace University
  <br><br>
  <img src="https://img.shields.io/badge/Pace%20University-Seidenberg%20School%20of%20CSIS-blue" />
</p>
