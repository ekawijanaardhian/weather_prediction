import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats

from run_rigorous_meteostat_experiment import load_and_clean_dataset

df = load_and_clean_dataset(2023)
df['date'] = pd.to_datetime(df['date'])

station_meta = {
    '96011': {'lat': 5.5167, 'lon': 95.4167, 'elev': 21},
    '96035': {'lat': 3.5667, 'lon': 98.6833, 'elev': 25},
    '96073': {'lat': 1.5500, 'lon': 98.8833, 'elev': 3},
    '96749': {'lat': -6.1167, 'lon': 106.6500, 'elev': 8},
    '96781': {'lat': -6.9000, 'lon': 107.5833, 'elev': 740},
    '96839': {'lat': -6.9833, 'lon': 110.3833, 'elev': 3},
    '96853': {'lat': -7.7833, 'lon': 110.4333, 'elev': 107},
    '97180': {'lat': -5.0667, 'lon': 119.5500, 'elev': 14},
    '97240': {'lat': -8.5333, 'lon': 116.0667, 'elev': 3}
}

df['latitude'] = df['station_id'].map(lambda s: station_meta[s]['lat'])
df['longitude'] = df['station_id'].map(lambda s: station_meta[s]['lon'])
df['elevation'] = df['station_id'].map(lambda s: station_meta[s]['elev'])

features_elev = [
    'heat_index', 'humidity', 'temp_min', 'temp_max', 'temp_range',
    'wind_speed_ms', 'rainfall_ml', 'day_of_year',
    'month_sin', 'month_cos', 'day_sin', 'day_cos', 'season_encoded',
    'latitude', 'longitude', 'elevation'
]

df_sorted = df.sort_values('date').reset_index(drop=True)
dates = np.array(sorted(df_sorted['date'].drop_duplicates().tolist()))
tscv = TimeSeriesSplit(n_splits=5)

rf_tscv_r2, lr_tscv_r2 = [], []
rf_tscv_rmse, lr_tscv_rmse = [], []

for train_idx, test_idx in tscv.split(dates):
    tr_dates = dates[train_idx]
    te_dates = dates[test_idx]
    
    tr = df_sorted[df_sorted['date'].isin(tr_dates)]
    te = df_sorted[df_sorted['date'].isin(te_dates)]
    
    X_tr, y_tr = tr[features_elev], tr['temp_avg']
    X_te, y_te = te[features_elev], te['temp_avg']
    
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)
    p_rf = rf.predict(X_te)
    
    lr = LinearRegression()
    lr.fit(X_tr_sc, y_tr)
    p_lr = lr.predict(X_te_sc)
    
    rf_tscv_r2.append(r2_score(y_te, p_rf))
    rf_tscv_rmse.append(np.sqrt(mean_squared_error(y_te, p_rf)))
    lr_tscv_r2.append(r2_score(y_te, p_lr))
    lr_tscv_rmse.append(np.sqrt(mean_squared_error(y_te, p_lr)))

print(f"TimeSeriesSplit RF (5 Folds): R2 = {np.mean(rf_tscv_r2):.4f} +/- {np.std(rf_tscv_r2):.4f}, RMSE = {np.mean(rf_tscv_rmse):.4f} +/- {np.std(rf_tscv_rmse):.4f}")
print(f"TimeSeriesSplit LR (5 Folds): R2 = {np.mean(lr_tscv_r2):.4f} +/- {np.std(lr_tscv_r2):.4f}, RMSE = {np.mean(lr_tscv_rmse):.4f} +/- {np.std(lr_tscv_rmse):.4f}")

split_date = pd.to_datetime('2023-10-20')
train_df = df_sorted[df_sorted['date'] < split_date].copy()
test_df = df_sorted[df_sorted['date'] >= split_date].copy()

X_train, y_train = train_df[features_elev], train_df['temp_avg']
X_test, y_test = test_df[features_elev], test_df['temp_avg']

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
p_rf = rf.predict(X_test)

lr = LinearRegression()
lr.fit(X_train_sc, y_train)
p_lr = lr.predict(X_test_sc)

e_rf = (test_df['temp_avg'].values - p_rf) ** 2
e_lr = (test_df['temp_avg'].values - p_lr) ** 2
d = e_rf - e_lr

def diebold_mariano_test(d_series, max_lag=7):
    n = len(d_series)
    mean_d = np.mean(d_series)
    gamma_0 = np.var(d_series, ddof=0)
    gamma_sum = 0.0
    for k in range(1, max_lag + 1):
        weight = 1.0 - (k / (max_lag + 1.0))
        gamma_k = np.mean((d_series[k:] - mean_d) * (d_series[:-k] - mean_d))
        gamma_sum += 2.0 * weight * gamma_k
    lr_var = (gamma_0 + gamma_sum) / n
    dm_stat = mean_d / np.sqrt(lr_var)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value

dm_stat, p_val_dm = diebold_mariano_test(d, max_lag=7)
print(f"Diebold-Mariano Test (h=7): DM Stat = {dm_stat:.4f}, p-value = {p_val_dm:.4e}")

df_seq = df.sort_values(['station_id', 'date']).copy()
for col in ['temp_avg', 'temp_min', 'temp_max', 'humidity', 'wind_speed_ms', 'rainfall_ml']:
    for lag in [1, 2, 3]:
        df_seq[f'{col}_lag{lag}'] = df_seq.groupby('station_id')[col].shift(lag)
    df_seq[f'{col}_roll3'] = df_seq.groupby('station_id')[col].transform(lambda s: s.shift(1).rolling(3).mean())
    df_seq[f'{col}_roll7'] = df_seq.groupby('station_id')[col].transform(lambda s: s.shift(1).rolling(7).mean())

df_seq['temp_trend'] = df_seq['temp_avg_lag1'] - df_seq['temp_avg_lag2']
df_seq['delta_target'] = df_seq['temp_avg'] - df_seq['temp_avg_lag1']

lag_rich_features = [c for c in df_seq.columns if '_lag' in c or '_roll' in c or c == 'temp_trend']
meta_features = ['day_of_year', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'season_encoded', 'latitude', 'longitude', 'elevation']
all_lag_features = lag_rich_features + meta_features

df_seq = df_seq.dropna(subset=all_lag_features + ['delta_target', 'temp_avg']).reset_index(drop=True)

tr_seq = df_seq[df_seq['date'] < split_date].copy()
te_seq = df_seq[df_seq['date'] >= split_date].copy()

X_tr_rich, y_tr_seq, y_tr_delta = tr_seq[all_lag_features], tr_seq['temp_avg'], tr_seq['delta_target']
X_te_rich, y_te_seq, y_te_delta = te_seq[all_lag_features], te_seq['temp_avg'], te_seq['delta_target']

sc_rich = StandardScaler()
X_tr_rich_sc = sc_rich.fit_transform(X_tr_rich)
X_te_rich_sc = sc_rich.transform(X_te_rich)

pers_p = te_seq['temp_avg_lag1']
print(f"1. Persistence Baseline     : R2 = {r2_score(y_te_seq, pers_p):.4f}, RMSE = {np.sqrt(mean_squared_error(y_te_seq, pers_p)):.4f} C, MAE = {mean_absolute_error(y_te_seq, pers_p):.4f} C")

lr_rich = LinearRegression()
lr_rich.fit(X_tr_rich_sc, y_tr_seq)
p_lr_rich = lr_rich.predict(X_te_rich_sc)
print(f"2. LR Rich Lags (Direct)    : R2 = {r2_score(y_te_seq, p_lr_rich):.4f}, RMSE = {np.sqrt(mean_squared_error(y_te_seq, p_lr_rich)):.4f} C, MAE = {mean_absolute_error(y_te_seq, p_lr_rich):.4f} C")

ridge = Ridge(alpha=10.0)
ridge.fit(X_tr_rich_sc, y_tr_seq)
p_ridge = ridge.predict(X_te_rich_sc)
print(f"3. Ridge Regression (Direct): R2 = {r2_score(y_te_seq, p_ridge):.4f}, RMSE = {np.sqrt(mean_squared_error(y_te_seq, p_ridge)):.4f} C, MAE = {mean_absolute_error(y_te_seq, p_ridge):.4f} C")

hgb = HistGradientBoostingRegressor(random_state=42)
hgb.fit(X_tr_rich, y_tr_seq)
p_hgb = hgb.predict(X_te_rich)
print(f"4. HistGradientBoosting (Dir): R2 = {r2_score(y_te_seq, p_hgb):.4f}, RMSE = {np.sqrt(mean_squared_error(y_te_seq, p_hgb)):.4f} C, MAE = {mean_absolute_error(y_te_seq, p_hgb):.4f} C")

rf_delta = RandomForestRegressor(n_estimators=100, random_state=42)
rf_delta.fit(X_tr_rich, y_tr_delta)
p_rf_delta = te_seq['temp_avg_lag1'].values + rf_delta.predict(X_te_rich)
print(f"5. RF Delta Model (T_prev+d): R2 = {r2_score(y_te_seq, p_rf_delta):.4f}, RMSE = {np.sqrt(mean_squared_error(y_te_seq, p_rf_delta)):.4f} C, MAE = {mean_absolute_error(y_te_seq, p_rf_delta):.4f} C")

hgb_delta = HistGradientBoostingRegressor(random_state=42)
hgb_delta.fit(X_tr_rich, y_tr_delta)
p_hgb_delta = te_seq['temp_avg_lag1'].values + hgb_delta.predict(X_te_rich)
print(f"6. HGB Delta Model (T_prev+d): R2 = {r2_score(y_te_seq, p_hgb_delta):.4f}, RMSE = {np.sqrt(mean_squared_error(y_te_seq, p_hgb_delta)):.4f} C, MAE = {mean_absolute_error(y_te_seq, p_hgb_delta):.4f} C")
