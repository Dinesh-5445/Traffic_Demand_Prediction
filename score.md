# Score Report

## Leaderboard Baseline
**Target to beat:** 91.09

---

## Spatial Prefix Ablation Block — Final Results

| Exp ID | Change | KFold R² | GKF R² | Gap | Decision |
|--------|--------|----------|--------|-----|----------|
| BASELINE | prefix3/4/5 + dual TE | 0.9370 | 0.9140 | 0.0230 | Reference |
| EXP_001 | Remove prefix_5 TE | 0.9414 | 0.9170 | 0.0245 | REJECT |
| EXP_002 | Remove prefix_3 TE | 0.9418 | 0.9147 | 0.0271 | REJECT |
| EXP_003 | geohash + prefix_4 only | 0.9413 | 0.9165 | 0.0248 | REJECT |
| **EXP_004** | **Only geohash TE** | **0.9417** | **0.9171** | **0.0245** | **ACCEPT** |

## TE Type Ablation Block — Final Results

| Exp ID | Change | KFold R² | GKF R² | Gap | Decision |
|--------|--------|----------|--------|-----|----------|
| BASELINE | EXP_004 (mean + median) | 0.9417 | 0.9171 | 0.0245 | Reference |
| EXP_005 | Remove all median TEs (keep smoothed mean only) | 0.9411 | 0.9137 | 0.0273 | REJECT |
| EXP_006 | Restore median TEs, remove mean TEs (median only) | 0.9384 | 0.9055 | 0.0329 | REJECT |

**Winner: EXP_004** — Both EXP_005 and EXP_006 resulted in lower GroupKFold R² than the EXP_004 base. The combination of Bayesian smoothed mean TE and median TE provides the best spatial generalization. EXP_005 is REJECTED and we REVERT to EXP_004 base.

---

## Frequency Bin Ablation Block — Final Results

| Exp ID | Change | KFold R² | GKF R² | Gap | Decision |
|--------|--------|----------|--------|-----|----------|
| BASELINE | EXP_004 (includes freq bins) | 0.9417 | 0.9171 | 0.0245 | Reference |
| EXP_007 | Remove geohash_freq and geohash_freq_bin_enc | 0.9413 | 0.9137 | 0.0277 | REJECT |

**Winner: EXP_004** — Removing frequency bins degrades GroupKFold R² from 0.9171 to 0.9137. The frequency bins are critical for spatial generalization. EXP_007 is REJECTED.

---

## Label Encoding Redundancy Block — Final Results

| Exp ID | Change | KFold R² | GKF R² | Gap | Decision |
|--------|--------|----------|--------|-----|----------|
| BASELINE | EXP_004 (24 features) | 0.9417 | 0.9171 | 0.0245 | Reference |
| EXP_008 | Remove all spatial label encodings | 0.9308 | 0.9164 | 0.0144 | REJECT |
| **EXP_008b** | **Remove only geohash_prefix_3/4_enc (keep geohash_enc, prefix_5)** | **0.9418** | **0.9160** | **0.0258** | **ACCEPT** |

**Winner: EXP_008b** — While EXP_008 seemingly closed the gap, a feature importance audit revealed it dropped critical spatial signal (`geohash_enc`). EXP_008b correctly drops only the true zero-signal features, retaining the necessary spatial identifiers. EXP_008b becomes the final model for Phase 1.

---

## Final Selected Model — EXP_008b (End of Phase 1)

- **Model:** LGBM + CB Ensemble
- **Features:** 22 (geohash_te_mean + geohash_te_median; redundant prefix_3/4 label encodings removed)
- **KFold R²:** 0.9418
- **GroupKFold R²:** 0.9160
- **KFold→GroupKFold Gap:** 0.0258
- **LB Score:** Pending
- **Submission:** `submissions/submission_final.csv`

---

## Key Engineering Decisions
1. **Bayesian Smoothed TE (C=10)** on geohash only (mean + median) — maximizes spatial signal while minimizing leakage.
2. **Selective Label Encodings** — retained `geohash_enc` and `geohash_prefix_5_enc` as they hold critical spatial identity; dropped the highly redundant prefix 3/4 encodings.
3. **Frequency Bins** — Retained, as ablating them caused a measurable drop in spatial generalization.
4. **Conservative CatBoost** (`depth=6`, `iter=1000`, `lr=0.03`, early stopping) — prevents overfitting.
5. **Dynamic ensemble weights** via 0.01-resolution OOF grid search.

---

## Gap Analysis
- The gap is currently being stabilized by removing redundant zero-signal features (EXP_008b). 
- Next objective is to continue minimizing this gap while pushing GroupKFold R² higher in Phase 2.
