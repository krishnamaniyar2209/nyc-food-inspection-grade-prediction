# 🍽️ NYC Food Safety Inspection Grade Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Models-red?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![University](https://img.shields.io/badge/Pace%20University-CS677-blue)

> Predicting NYC food-establishment inspection grades (A / B / C) by comparing 6 classifiers — with feature engineering, GridSearchCV tuning, and a deliberately leakage-free pipeline (prior-only history features + establishment-grouped split). Built for CS677: Machine Learning at Pace University (Fall 2025).

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Models & Results](#-models--results)
- [Data Leakage — Identified & Fixed](#-data-leakage--identified--fixed)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
- [Author](#-author)
- [License](#-license)

---

## 🔬 Overview

This project compares **6 machine-learning classifiers** to predict whether an NYC establishment receives Grade A, B, or C — covering EDA, feature engineering, a **leakage-free establishment-grouped split**, model comparison, SGD optimization, and GridSearchCV tuning.

- ✅ Multiclass classification (A / B / C)
- ✅ 8 visualizations (distribution, geography, time series, grade breakdowns)
- ✅ 8 engineered features (history, geography, timing, domain flags)
- ✅ **Establishment-grouped 60/20/20 split** (no restaurant in two sets)
- ✅ 5 models + SGD optimizer + GridSearchCV tuning
- ✅ Group K-Fold cross-validation + full metric evaluation

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
| `Previous_Grade_Encoded` | Prior inspection grade (backward lag) | 36.8% ⭐ |
| `Failure_Rate` | **Prior-only** historical Grade-C rate | 24.6% |
| `Total_Inspections` | Cumulative inspection count | 11.7% |
| `Days_Since_Last` | Days since previous inspection | 11.1% |
| `Longitude` / `Latitude` | Extracted from Georeference | 6.3% |
| `Has_Multiple_Types`, `Is_NYC` | Domain flags | ~3% |

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

1. **EDA** — 8 visualizations of distribution, geography, time, and grade breakdowns
2. **Feature Engineering** — 8 features; all history features (`Failure_Rate`, `Previous_Grade`, etc.) computed **strictly from prior inspections**
3. **Grouped split** — `GroupShuffleSplit` keyed on `Trade Name + Street`, so no establishment appears in more than one set (60/20/20)
4. **Scaling** — `StandardScaler` fit on training data only
5. **Models** — Decision Tree, Random Forest, Logistic Regression, SVM (Linear + RBF), and an SGD optimizer
6. **Validation** — `GroupKFold` cross-validation (establishment-aware)
7. **Tuning** — `GridSearchCV` on Decision Tree and Random Forest

---

## 📈 Models & Results

### Default Model Performance
| Model | Train Acc | Val Acc | Test Acc |
|---|---|---|---|
| Random Forest | 0.9998 | 0.7908 | 0.7886 |
| Logistic Regression | 0.7732 | 0.7681 | 0.7684 |
| Decision Tree | 0.9998 | 0.7564 | 0.7577 |
| SVM RBF | 0.6109 | 0.6058 | 0.5985 |

### Group K-Fold Cross-Validation
| Model | CV Mean | CV Std |
|---|---|---|
| Random Forest | 0.7890 | 0.0036 |
| Logistic Regression | 0.7728 | 0.0029 |
| Decision Tree | 0.7617 | 0.0071 |

### Final Performance (After Tuning)
| Rank | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 🥇 1 | **Random Forest (Tuned)** | 0.7973 | 0.8178 | 0.7973 | 0.8013 |
| 🥈 2 | Decision Tree (Tuned) | 0.7951 | 0.8263 | 0.7951 | 0.8017 |
| 3 | Logistic Regression | 0.7684 | 0.7800 | 0.7684 | 0.7711 |
| 4 | SGD | 0.7648 | 0.7781 | 0.7648 | 0.7683 |

> Best tuned models: Decision Tree (`max_depth=5`) and Random Forest (`max_depth=10`), both ~0.80 test accuracy with **no overfitting** (train–test gap < 0.04).

### Per-Class (Random Forest Tuned)
| Grade | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| A | 0.61 | 0.74 | 0.67 | 1,923 |
| B | 0.97 | 0.72 | 0.83 | 2,118 |
| C | 0.84 | 0.86 | 0.85 | 4,101 |

### Top Feature Importances
| Rank | Feature | Importance |
|---|---|---|
| 1 | `Previous_Grade_Encoded` | 36.8% |
| 2 | `Failure_Rate` (prior-only) | 24.6% |
| 3 | `Total_Inspections` | 11.7% |
| 4 | `Days_Since_Last` | 11.1% |

---

## 🛡️ Data Leakage — Identified & Fixed

An earlier version of this project reported ~91% accuracy — but that was inflated by **target leakage**. The fix is documented here because finding and removing leakage is part of the work.

**The leak:** `Failure_Rate` originally divided each establishment's *total* Grade-C count by its *total* inspections — **including the inspection being predicted** — while the data was split **randomly by row**, letting the same establishment appear in both train and test.

**The fix (two changes):**
1. **Prior-only history** — `Failure_Rate` is now computed from an establishment's *earlier* inspections only (cumulative count shifted to exclude the current row).
2. **Grouped split** — `GroupShuffleSplit` / `GroupKFold` keyed on the establishment, so no restaurant spans train and test (verified: 0 overlap).

**Result:** honest accuracy settled at **~80%** (down from a leaky 91%), and the top predictor shifted from the leaky `Failure_Rate` to the legitimate `Previous_Grade`. The lower number is the trustworthy one.

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
2. Open the notebook in Jupyter or Google Colab and update the file path in **Cell 2**
3. Run all cells top to bottom — EDA, models, and the leakage check generate automatically

---

## 💡 Key Findings
- **`Previous_Grade`** is the strongest legitimate predictor — an establishment's prior grade meaningfully predicts its next
- **Grade C** (53% of data) is predicted most reliably; **Grade A** is the hardest (precision 0.61)
- **Random Forest (Tuned)** is the best model at **79.7% accuracy / 0.80 F1**, with no overfitting after tuning
- **Geography** (Kings & Bronx ~64% Grade-C) and **establishment type** show strong grade patterns in EDA
- **Removing target leakage dropped accuracy from 91% → 80%** — the realistic, deployable benchmark

---

## 🛠️ Technologies Used
| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| scikit-learn | Models, metrics, GridSearchCV, grouped CV |
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

<p align="center">Made with ❤️ for CS677 @ Pace University</p>
