## Experiment: 2026-05-31 12:11:59.675386
- **Model**: CatBoost Baseline V1
- **Num Features**: 24
- **CV R2 Score**: 0.94830
- **RMSE**: 0.03234
- **MAE**: 0.02177
- **Submission File**: C:/Users/Palav/OneDrive/Desktop/TrafficDemandPrediction/submissions/submission_v1.csv

## Experiment: 2026-05-31 12:19:15.374679
- **Model**: Ensemble V2 (50% CB + 50% LGBM) with OOF Target Encoding
- **Num Features**: 21
- **CV R2 Score**: 0.94184
- **RMSE**: 0.03429
- **MAE**: 0.02246
- **Submission File**: C:/Users/Palav/OneDrive/Desktop/TrafficDemandPrediction/submissions/submission_v2.csv

## Experiment: 2026-06-03 10:10:00 (Exp 1-5 Final Run)
- **Model**: Ensemble V3 (42% LGBM + 58% CB) with Bayesian Smoothed OOF TE
- **Num Features**: 30
- **CV R2 Score**: 0.9418 (LGBM KFold 0.9402, CB KFold 0.9409)
- **GroupKFold R2 Score**: 0.9140
- **RMSE**: 0.0343
- **Submission File**: submissions/submission_final.csv
- **Key Changes**: Added `geohash_prefix_5`, quantile frequency bins, Bayesian Smoothed Mean TE (`C=10`), CatBoost early stopping (`depth=6, iter=1000, lr=0.03`), and 0.01 resolution dynamic ensemble weights optimization.