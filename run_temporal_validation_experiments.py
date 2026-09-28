import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from run_rigorous_meteostat_experiment import load_and_clean_dataset
df = load_and_clean_dataset(2023)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['station_id', 'date']).reset_index(drop=True)
features_full = [
    'heat_index', 'humidity', 'temp_min', 'temp_max', 'temp_range',
    'wind_speed_ms', 'rainfall_ml', 'month', 'quarter', 'day_of_year',
    'month_sin', 'month_cos', 'day_sin', 'day_cos', 'season_encoded',
    'station_encoded', 'region_encoded', 'year', 'day'
]
split_date = pd.to_datetime('2023-10-20')
train_df = df[df['date'] < split_date].copy()
test_df = df[df['date'] >= split_date].copy()
X_train, y_train = train_df[features_full], train_df['temp_avg']
X_test, y_test = test_df[features_full], test_df['temp_avg']
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_r2 = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mae = mean_absolute_error(y_test, lr_pred)
print(f"Chronological Split RF : R2 = {rf_r2:.4f}, RMSE = {rf_rmse:.4f} C, MAE = {rf_mae:.4f} C")
print(f"Chronological Split LR : R2 = {lr_r2:.4f}, RMSE = {lr_rmse:.4f} C, MAE = {lr_mae:.4f} C")
months_test = [7, 8, 9, 10, 11, 12]
rolling_results = []
for m in months_test:
    tr = df[df['month'] < m].copy()
    te = df[df['month'] == m].copy()
    X_tr, y_tr = tr[features_full], tr['temp_avg']
    X_te, y_te = te[features_full], te['temp_avg']
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    rf_m = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_m.fit(X_tr, y_tr)
    p_rf = rf_m.predict(X_te)
    lr_m = LinearRegression()
    lr_m.fit(X_tr_sc, y_tr)
    p_lr = lr_m.predict(X_te_sc)
    rolling_results.append({
        'Test_Month': m,
        'RF_R2': r2_score(y_te, p_rf),
        'RF_RMSE': np.sqrt(mean_squared_error(y_te, p_rf)),
        'LR_R2': r2_score(y_te, p_lr),
        'LR_RMSE': np.sqrt(mean_squared_error(y_te, p_lr))
    })
df_rolling = pd.DataFrame(rolling_results)
print(df_rolling.to_string(index=False))
print(f"Average Rolling RF : R2 = {df_rolling['RF_R2'].mean():.4f}, RMSE = {df_rolling['RF_RMSE'].mean():.4f} C")
print(f"Average Rolling LR : R2 = {df_rolling['LR_R2'].mean():.4f}, RMSE = {df_rolling['LR_RMSE'].mean():.4f} C")
lag_cols = ['temp_avg', 'temp_min', 'temp_max', 'humidity', 'wind_speed_ms', 'rainfall_ml', 'heat_index', 'temp_range']
df_lag = df.copy()
for col in lag_cols:
    df_lag[f'{col}_lag1'] = df_lag.groupby('station_id')[col].shift(1)
df_lag = df_lag.dropna(subset=[f'{c}_lag1' for c in lag_cols]).reset_index(drop=True)
lag_features = [f'{c}_lag1' for c in lag_cols] + [
    'month', 'quarter', 'day_of_year', 'month_sin', 'month_cos', 
    'day_sin', 'day_cos', 'season_encoded', 'station_encoded', 'region_encoded', 'year', 'day'
]
train_lag = df_lag[df_lag['date'] < split_date].copy()
test_lag = df_lag[df_lag['date'] >= split_date].copy()
X_tr_lag, y_tr_lag = train_lag[lag_features], train_lag['temp_avg']
X_te_lag, y_te_lag = test_lag[lag_features], test_lag['temp_avg']
sc_lag = StandardScaler()
X_tr_lag_sc = sc_lag.fit_transform(X_tr_lag)
X_te_lag_sc = sc_lag.transform(X_te_lag)
pers_pred = test_lag['temp_avg_lag1']
pers_r2 = r2_score(y_te_lag, pers_pred)
pers_rmse = np.sqrt(mean_squared_error(y_te_lag, pers_pred))
pers_mae = mean_absolute_error(y_te_lag, pers_pred)
rf_lag = RandomForestRegressor(n_estimators=100, random_state=42)
rf_lag.fit(X_tr_lag, y_tr_lag)
rf_lag_pred = rf_lag.predict(X_te_lag)
rf_lag_r2 = r2_score(y_te_lag, rf_lag_pred)
rf_lag_rmse = np.sqrt(mean_squared_error(y_te_lag, rf_lag_pred))
rf_lag_mae = mean_absolute_error(y_te_lag, rf_lag_pred)
lr_lag = LinearRegression()
lr_lag.fit(X_tr_lag_sc, y_tr_lag)
lr_lag_pred = lr_lag.predict(X_te_lag_sc)
lr_lag_r2 = r2_score(y_te_lag, lr_lag_pred)
lr_lag_rmse = np.sqrt(mean_squared_error(y_te_lag, lr_lag_pred))
lr_lag_mae = mean_absolute_error(y_te_lag, lr_lag_pred)
print(f"Persistence Baseline (t-1) : R2 = {pers_r2:.4f}, RMSE = {pers_rmse:.4f} C, MAE = {pers_mae:.4f} C")
print(f"Linear Regression (t-1)    : R2 = {lr_lag_r2:.4f}, RMSE = {lr_lag_rmse:.4f} C, MAE = {lr_lag_mae:.4f} C")
print(f"Random Forest (t-1)        : R2 = {rf_lag_r2:.4f}, RMSE = {rf_lag_rmse:.4f} C, MAE = {rf_lag_mae:.4f} C")
