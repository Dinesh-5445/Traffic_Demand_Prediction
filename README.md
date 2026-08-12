# Traffic Demand Prediction 🚦

This repository contains an end-to-end Machine Learning pipeline to predict urban traffic demand at 15-minute intervals. The project emphasizes robust validation, target leakage prevention, and clean, sequential code architecture over "black-box" over-engineering.

## 🎯 Project Overview
Traffic congestion causes massive economic and environmental inefficiencies. Predicting traffic demand via spatial (geohash) and temporal (timestamp) data enables smart routing and infrastructure optimization. This solution tackles the challenge using a highly robust ensemble of LightGBM and CatBoost.

## 📊 Dataset Description
The dataset consists of:
- **Spatial:** `geohash` (encoded geographic locations)
- **Temporal:** `day`, `timestamp`
- **Contextual:** `RoadType`, `NumberofLanes`, `Landmarks`, `LargeVehicles`
- **Weather:** `Temperature`, `Weather`
- **Target:** `demand` (Continuous Regression)

*Note: Forensics indicate this schema mimics the Grab AI for S.E.A. 2019 dataset. This pipeline ignores external leakage and focuses purely on legitimate ML prediction.*

## 🚀 Pipeline & Feature Engineering
Our feature engineering strategy relies on robust, mathematically sound transformations:
- **Cyclical Time:** Timestamps converted to total minutes and transformed via Sine/Cosine functions to preserve the continuous nature of time across midnight.
- **Hierarchical Spatial:** Extraction of `geohash` prefixes (`prefix_3`, `prefix_4`, `prefix_5`) to capture macro-level geographic density, alongside quantile frequency binning to isolate dense regions.
- **Bayesian Smoothed OOF Target Encoding:** Out-of-fold target mean encodings are regularized via Bayesian smoothing (`C=10`) and applied strictly to spatial locations alongside raw median encodings (avoiding temporal-spatial crosses to eliminate target leakage).

## 🧠 Models & Validation
We employ a dual-validation benchmark:
- **5-Fold CV:** General prediction robustness.
- **GroupKFold (by geohash):** Spatial generalization stress test.

The final ensemble is a deeply regularized blend of **LightGBM** and **CatBoost** (depth constrained, stochastic subsampling). Blend weights are dynamically optimized via OOF grid-search (at 0.01 resolution) to perfectly calibrate the final predictions.

## 📁 Folder Structure
```text
TrafficDemandPrediction/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── submissions/
│   └── submission_final.csv
├── Traffic_Demand_Prediction.ipynb  # Primary sequential notebook
├── approach.txt                     # Detailed narrative of ML strategy
├── experiment_log.md                # Iteration tracking and metrics
└── README.md                        # Project documentation
```

## 🛠 Reproducibility Instructions
To completely reproduce the final submission from scratch:
1. Ensure your Python environment has `numpy`, `pandas`, `matplotlib`, `seaborn`, `lightgbm`, `catboost`, and `scikit-learn` installed.
2. Place the `train.csv` and `test.csv` in the `data/` directory.
3. Open `Traffic_Demand_Prediction.ipynb`.
4. Run all cells from top to bottom (or via CLI: `jupyter nbconvert --execute Traffic_Demand_Prediction.ipynb`).
5. The final output will be saved automatically to `submissions/submission_final.csv`.

## 🔮 Future Improvements
- **Sequence Models:** Framing the problem as a temporal sequence (LSTM / Transformer) rather than a pure tabular task.
- **Graph Neural Networks:** Treating geohashes as nodes and road networks as edges to learn deeper spatial embeddings.
