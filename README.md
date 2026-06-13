# 🍽️ NYC Food Safety Inspection Grade Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Models-red?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![University](https://img.shields.io/badge/Pace%20University-CS677-blue)

> A machine learning project predicting the **food safety inspection grade (A / B / C)** of NYC establishments by comparing 6 classifiers, with rich feature engineering, GridSearchCV tuning, and an honest analysis of overfitting and target leakage — built for CS677: Machine Learning at Pace University (Fall 2025).

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Models & Results](#-models--results)
- [Important Limitation — Target Leakage](#-important-limitation--target-leakage)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
- [Author](#-author)
- [License](#-license)

---

## 🔬 Overview

This project applies and compares **6 machine learning classification algorithms** on the NYC Food Safety Inspections dataset to predict whether an establishment receives Grade A, B, or C — covering EDA, feature engineering, stratified splitting, model comparison, SGD optimization, and hyperparameter tuning.

The notebook covers:
- ✅ **Problem Statement** — multiclass classification (A / B / C)
- ✅ **EDA** — 8 visualizations (distribution, geography, time series, grade by type/county/year, correlations)
- ✅ **Feature Engineering** — 8 engineered features
- ✅ **Stratified 60/20/20 train/validation/test split**
- ✅ **5 models** — Decision Tree, Random Forest, Logistic Regression, SVM (Linear + RBF)
- ✅ **SGD optimizer** — incremental gradient descent over 100 epochs
- ✅ **GridSearchCV tuning** on Decision Tree and Random Forest
- ✅ **Full evaluation** — Accuracy, Precision, Recall, F1, confusion matrices

---

## 📊 Dataset

**Source:** NYC Open Data — Food Safety Inspections (Current Ratings)

| Property | Details |
|---|---|
| Original Records | ~137,000 |
| Sampled Records | 41,075 (30%) |
| Features Used | 14 (6 raw + 8 engineered) |
| Target | `Inspection Grade` (A / B / C) |
| Class Balance | C: 52.66% / B: 24.32% / A: 23.01% |
| Date Range | 2023 – 2025 |

### Engineered Features

| Feature | Description | RF Importance |
|---|---|---|
| `Failure_Rate` | Establishment's historical Grade-C rate | 44.9% ⚠️ *(see Limitations)* |
| `Previous_Grade_Encoded` | Prior inspection grade (lag) | 28.0% |
| `Total_Inspections` | Cumulative inspection count | 9.8% |
| `Days_Since_Last` | Days since previous inspection | 7.9% |
| `Longitude` / `Latitude` | Extracted from Georeference | 3.5% |
| `Has_Multiple_Types` | Establishment type-string length | 1.2% |
| `Is_NYC` | 1 if NYC borough | 1.1% |

---

## 📁 Project Structure
```
nyc-food-inspection-grade-prediction/
│
├── NYC_Food_Safety_Inspection_Grade_Prediction.ipynb  # Main notebook
├── README.md                                           # Documentation
└── requirements.txt                                    # Dependencies
```

---

## 🔬 Methodology

1. **EDA** — 8 visualizations of distribution, geography, time series, and grade breakdowns
2. **Feature Engineering** — 8 features (history, geography, timing, domain flags)
3. **Stratified split** — 60% train (24,645) / 20% val (8,215) / 20% test (8,215)
4. **Scaling** — `StandardScaler` fit on training data only
5. **Model training** — 5 classifiers with default parameters, tracking train/val/test accuracy
6. **SGD** — `partial_fit()` over 100 epochs with error tracking
7. **Tuning** — `GridSearchCV` on Decision Tree and Random Forest
8. **Evaluation** — Accuracy, Precision, Recall, F1, confusion matrices, classification reports

---

## 📈 Models & Results

### Default Model Performance
| Model | Train Acc | Val Acc | Test Acc | Status |
|---|---|---|---|---|
| Decision Tree | 0.9999 | 0.8816 | 0.8889 | ⚠️ Overfitting |
| Random Forest | 0.9999 | 0.9108 | 0.9100 | ✅ Good fit |
| Logistic Regression | 0.8707 | 0.8676 | 0.8684 | ✅ Good fit |
| SVM RBF | 0.8456 | 0.8329 | 0.8398 | ✅ Good fit |
| SVM Linear | 0.7919 | 0.7928 | 0.7877 | ✅ Good fit |

### Cross-Validation (5-Fold Stratified)
| Model | CV Mean | CV Std |
|---|---|---|
| Random Forest | 0.9111 | 0.0037 |
| Decision Tree | 0.8881 | 0.0049 |
| Logistic Regression | 0.8702 | 0.0047 |
| SVM RBF | 0.8672 | 0.0160 |
| SVM Linear | 0.7309 | 0.0709 |

### Final Performance (After Tuning)
| Rank | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 🥇 1 | **Decision Tree (Tuned)** | 0.9110 | 0.9186 | 0.9110 | 0.9101 |
| 🥈 2 | Random Forest (Tuned) | 0.9099 | 0.9183 | 0.9099 | 0.9084 |
| 3 | SGD | 0.8696 | 0.8705 | 0.8696 | 0.8675 |
| 4 | Logistic Regression | 0.8684 | 0.8707 | 0.8684 | 0.8669 |
| 5 | SVM RBF | 0.8398 | 0.8343 | 0.8398 | 0.8365 |
| 6 | SVM Linear | 0.7877 | 0.7979 | 0.7877 | 0.7905 |

> GridSearchCV (`max_depth=10, min_samples_leaf=10, min_samples_split=50`) cut Decision Tree train accuracy from 0.9999 → 0.93 while holding test accuracy at 0.9110 — resolving the overfitting seen in the default tree.

### Top Feature Importances (Random Forest)
| Rank | Feature | Importance |
|---|---|---|
| 1 | `Failure_Rate` | 44.89% ⚠️ |
| 2 | `Previous_Grade_Encoded` | 27.98% |
| 3 | `Total_Inspections` | 9.81% |
| 4 | `Days_Since_Last` | 7.91% |

---

## ⚠️ Important Limitation — Target Leakage

The strongest feature, `Failure_Rate`, is computed as each establishment's share of Grade-C inspections **including the inspection being predicted**:

```python
df.groupby(['Trade Name','Street'])['Inspection Grade'].agg(lambda x: (x == 'C').sum() / len(x))
```

For an establishment with a single inspection this equals **1.0 for a C and 0.0 otherwise** — effectively encoding the target. Combined with a **random row-level train/test split** (the same establishment can appear in both train and test), this leaks label information and **inflates the reported ~91% accuracy**; `Failure_Rate`'s 44.9% importance largely reflects this leak.

**A leakage-free version (in progress) will:**
1. Compute `Failure_Rate` from **prior inspections only** (expanding C-rate shifted by one), so the current grade is excluded.
2. Split by **establishment** (`GroupShuffleSplit`) or by **time**, so no establishment spans train and test.

Honest accuracy is expected to be lower, with `Previous_Grade` (a legitimate backward-looking lag) carrying more of the real signal. *(Leaving this analysis in the README intentionally — identifying leakage is part of the work.)*

---

## ⚙️ Installation

```bash
git clone https://github.com/krishnamaniyar2209/nyc-food-inspection-grade-prediction.git
cd nyc-food-inspection-grade-prediction
pip install -r requirements.txt
jupyter notebook NYC_Food_Safety_Inspection_Grade_Prediction.ipynb
```

---

## 🚀 Usage

1. Download the dataset from NYC Open Data (Food Safety Inspections — Current Ratings) as a `.csv`
2. Open the notebook in Jupyter or Google Colab and update the file path in **Cell 3**
3. Run all cells sequentially — all EDA plots, metrics, and comparisons generate automatically

---

## 💡 Key Findings

- **Grade C dominates** at 52.66% — class imbalance handled via stratified splits
- **Kings (Brooklyn)** and **Bronx** have the highest Grade-C rates (~64%); **Suffolk** and **Monroe** the lowest
- **`Previous_Grade`** (a legitimate lag feature) is a genuinely strong predictor — an establishment's prior grade meaningfully predicts its next
- **Decision Tree (Tuned)** scored 0.9110 test accuracy and GridSearchCV resolved the default tree's overfitting (train 0.9999 → 0.93)
- **Random Forest** had the most stable CV (std 0.0037)
- **SVM RBF > SVM Linear** — confirming non-linear structure in the data
- ⚠️ **The 91% headline is optimistic due to `Failure_Rate` leakage** (see Limitations) — the corrected pipeline is the honest benchmark

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| scikit-learn | Models, metrics, GridSearchCV |
| pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| Jupyter Notebook | Development environment |

---

## 👤 Author

**Krishna Maniyar** — Data Analyst
- 🎓 Pace University — Seidenberg School of CSIS, MS in Data Science
- 📘 CS677: Machine Learning (Fall 2025)
- 📧 krishnamaniyarkm22@gmail.com
- 🔗 [GitHub](https://github.com/krishnamaniyar2209) · [LinkedIn](https://www.linkedin.com/in/krishnamaniyar/) · [Portfolio](https://krishnamaniyar2209.github.io/)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for CS677 @ Pace University
</p>
