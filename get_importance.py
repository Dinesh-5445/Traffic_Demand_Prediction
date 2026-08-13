"""
Standalone feature importance extraction.
Loads the data, runs LightGBM with EXP_004 config (24 features), prints ALL importances.
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

SEED = 42

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train['is_train'] = 1
test['is_train'] = 0
test['demand'] = np.nan
full_df = pd.concat([train, test], ignore_index=True)

# Feature engineering (same as notebook)
full_df['hour'] = full_df['timestamp'].apply(lambda x: int(x.split(':')[0]))
full_df['minute'] = full_df['timestamp'].apply(lambda x: int(x.split(':')[1]))
full_df['total_minutes'] = full_df['hour'] * 60 + full_df['minute']
full_df['morning_peak'] = ((full_df['hour'] >= 7) & (full_df['hour'] <= 9)).astype(int)
full_df['evening_peak'] = ((full_df['hour'] >= 17) & (full_df['hour'] <= 19)).astype(int)
full_df['rush_hour'] = full_df['morning_peak'] | full_df['evening_peak']
full_df['sin_time'] = np.sin(2 * np.pi * full_df['total_minutes'] / 1440)
full_df['cos_time'] = np.cos(2 * np.pi * full_df['total_minutes'] / 1440)
full_df['is_weekend'] = (full_df['day'] % 7 >= 5).astype(int)

full_df['geohash_prefix_5'] = full_df['geohash'].str[:5]
full_df['geohash_prefix_4'] = full_df['geohash'].str[:4]
full_df['geohash_prefix_3'] = full_df['geohash'].str[:3]
full_df['geohash_freq'] = full_df.groupby('geohash')['geohash'].transform('count')
full_df['geohash_freq_bin'] = pd.qcut(full_df['geohash_freq'].rank(method='first'), q=3, labels=['rare', 'medium', 'common'])

full_df['Landmarks'] = full_df['Landmarks'].map({'Yes': 1, 'No': 0}).fillna(0)
full_df['LargeVehicles'] = full_df['LargeVehicles'].map({'Allowed': 1, 'Not Allowed': 0}).fillna(0)

for col in ['RoadType', 'Weather', 'geohash', 'geohash_prefix_5', 'geohash_prefix_4', 'geohash_prefix_3', 'geohash_freq_bin']:
    full_df[f'{col}_enc'] = full_df[col].astype('category').cat.codes

# Target encoding (EXP_004: geohash only)
train_mask = full_df['is_train'] == 1
kf_enc = KFold(n_splits=5, shuffle=True, random_state=SEED)
encode_cols = ['geohash']
global_mean = full_df.loc[train_mask, 'demand'].mean()
C = 10

for col in encode_cols:
    full_df[f'{col}_te_mean'] = np.nan
    full_df[f'{col}_te_median'] = np.nan

train_idx_arr = full_df[train_mask].index.values
for tr_fold_idx, va_fold_idx in kf_enc.split(train_idx_arr):
    tr_indices = train_idx_arr[tr_fold_idx]
    va_indices = train_idx_arr[va_fold_idx]
    tr_data = full_df.loc[tr_indices]
    for col in encode_cols:
        agg = tr_data.groupby(col)['demand'].agg(['mean', 'median', 'count']).reset_index()
        agg['smoothed_mean'] = (agg['mean'] * agg['count'] + global_mean * C) / (agg['count'] + C)
        va_data = full_df.loc[va_indices, [col]].merge(agg, on=col, how='left')
        full_df.loc[va_indices, f'{col}_te_mean'] = va_data['smoothed_mean'].values
        full_df.loc[va_indices, f'{col}_te_median'] = va_data['median'].values

for col in encode_cols:
    full_df[f'{col}_te_mean'] = full_df[f'{col}_te_mean'].fillna(global_mean)
    full_df[f'{col}_te_median'] = full_df[f'{col}_te_median'].fillna(global_mean)

# Build features
drop_cols = ['is_train', 'demand', 'geohash', 'timestamp',
             'geohash_prefix_3', 'geohash_prefix_4', 'geohash_prefix_5', 'geohash_freq_bin', 'RoadType', 'Weather']
train_df = full_df[full_df['is_train'] == 1].drop(columns=drop_cols)
y = full_df.loc[full_df['is_train'] == 1, 'demand']
X = train_df.drop(columns=['Index'], errors='ignore')

print(f"Features: {X.shape[1]}")

# Train LightGBM
model = LGBMRegressor(
    n_estimators=600, learning_rate=0.04, max_depth=5,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
    random_state=SEED, n_jobs=-1, verbose=-1
)
model.fit(X, y)

fi = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
fi = fi.sort_values('Importance', ascending=False).reset_index(drop=True)

print("\n=== ALL Feature Importances (EXP_004 config, 24 features) ===")
print(fi.to_string(index=True))

print("\n=== TARGET COLUMNS FOR AUDIT ===")
targets = ['geohash_enc', 'geohash_prefix_3_enc', 'geohash_prefix_4_enc', 'geohash_prefix_5_enc']
for t in targets:
    row = fi[fi['Feature'] == t]
    if not row.empty:
        rank = row.index[0] + 1
        imp = row['Importance'].values[0]
        print(f"  {t:30s}  Rank={rank:2d}  Importance={imp}")
    else:
        print(f"  {t:30s}  NOT FOUND")
