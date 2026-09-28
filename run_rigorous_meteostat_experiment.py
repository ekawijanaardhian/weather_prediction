import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy import stats
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
os.makedirs('hasil_riset_bmkg', exist_ok=True)
os.makedirs('hasil_riset_bmkg/figures', exist_ok=True)
os.makedirs('hasil_riset_bmkg/saved_models', exist_ok=True)
STATIONS_CONFIG = {
    '96035': {'name': 'Medan (Kualanamu)', 'region': 'North Sumatra', 'province': 'Sumatera Utara'},
    '96073': {'name': 'Sibolga (Pinangsori)', 'region': 'North Sumatra', 'province': 'Sumatera Utara'},
    '96011': {'name': 'Banda Aceh', 'region': 'Aceh', 'province': 'Aceh'},
    '96781': {'name': 'Bandung (Husein S.)', 'region': 'West Java', 'province': 'Jawa Barat'},
    '96749': {'name': 'Tangerang (Soekarno-Hatta)', 'region': 'Banten', 'province': 'Banten'},
    '96839': {'name': 'Semarang (Ahmad Yani)', 'region': 'Central Java', 'province': 'Jawa Tengah'},
    '96853': {'name': 'Yogyakarta (Adisutjipto)', 'region': 'DI Yogyakarta', 'province': 'DI Yogyakarta'},
    '97240': {'name': 'Mataram / Lombok', 'region': 'NTB', 'province': 'Nusa Tenggara Barat'},
    '97180': {'name': 'Makassar (Sultan Hasanuddin)', 'region': 'South Sulawesi', 'province': 'Sulawesi Selatan'}
}
def calculate_heat_index(temp_c, rh):
    t = temp_c * 1.8 + 32.0
    hi_f = (-42.379 + 
            2.04901523 * t + 
            10.14333127 * rh - 
            0.22475541 * t * rh - 
            6.83783e-3 * (t ** 2) - 
            5.481717e-2 * (rh ** 2) + 
            1.22874e-3 * (t ** 2) * rh + 
            8.5282e-4 * t * (rh ** 2) - 
            1.99e-6 * (t ** 2) * (rh ** 2))
    simple_hi = 0.5 * (t + 61.0 + ((t - 68.0) * 1.2) + (rh * 0.094))
    use_simple = (simple_hi + t) / 2.0 < 80.0
    final_hi_f = np.where(use_simple, simple_hi, hi_f)
    hi_c = (final_hi_f - 32.0) / 1.8
    return hi_c
def load_and_clean_dataset(year=2023):
    csv_source = 'hasil_riset_bmkg/meteostat_indonesia_2010_2023.csv'
    df_all = pd.read_csv(csv_source, low_memory=False)
    df_all['station_id'] = df_all['station_id'].astype(str)
    df_all['date'] = pd.to_datetime(df_all['date'])
    target_ids = list(STATIONS_CONFIG.keys())
    df = df_all[(df_all['station_id'].isin(target_ids)) & (df_all['date'].dt.year == year)].copy()
    df['station_name'] = df['station_id'].map(lambda s: STATIONS_CONFIG[s]['name'])
    df['region'] = df['station_id'].map(lambda s: STATIONS_CONFIG[s]['region'])
    df['province'] = df['station_id'].map(lambda s: STATIONS_CONFIG[s]['province'])
    df['wind_speed_ms'] = df['wind_speed'] / 3.6
    df.loc[df['station_id'] == '96853', 'rainfall'] = np.nan
    for col in ['temp_min', 'temp_max', 'temp_avg', 'humidity', 'wind_speed_ms']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df.groupby('station_id')[col].transform(lambda g: g.interpolate(method='linear').ffill().bfill())
    df['temp_range'] = df['temp_max'] - df['temp_min']
    df['heat_index'] = calculate_heat_index(df['temp_max'].values, df['humidity'].values)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['day'] = df['date'].dt.day
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    def classify_season(m):
        if m in [11, 12, 1, 2, 3]:
            return 'Rainy Season'
        elif m in [6, 7, 8, 9]:
            return 'Dry Season'
        else:
            return 'Transition'
    df['season'] = df['month'].apply(classify_season)
    season_encoder = LabelEncoder()
    df['season_encoded'] = season_encoder.fit_transform(df['season'])
    station_encoder = LabelEncoder()
    df['station_encoded'] = station_encoder.fit_transform(df['station_id'])
    region_encoder = LabelEncoder()
    df['region_encoded'] = region_encoder.fit_transform(df['region'])
    java_stations = ['Semarang (Ahmad Yani)', 'Bandung (Husein S.)', 'Tangerang (Soekarno-Hatta)']
    java_monthly_mean = df[df['station_name'].isin(java_stations)].groupby('month')['rainfall'].mean()
    yog_mask = df['station_name'].str.startswith('Yogya')
    df.loc[yog_mask, 'rainfall'] = df.loc[yog_mask, 'month'].map(java_monthly_mean)
    df['rainfall_ml'] = df['rainfall'].copy()
    df.to_csv('hasil_riset_bmkg/meteostat_paper_9stations_2023.csv', index=False)
    return df
def run_model_experiments_and_ablation(df):
    features_full = [
        'heat_index', 'humidity', 'temp_min', 'temp_max', 'temp_range',
        'wind_speed_ms', 'rainfall_ml', 'month', 'quarter', 'day_of_year',
        'month_sin', 'month_cos', 'day_sin', 'day_cos', 'season_encoded',
        'station_encoded', 'region_encoded', 'year', 'day'
    ]
    features_no_hi = [f for f in features_full if f != 'heat_index']
    X = df[features_full].copy()
    y = df['temp_avg'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler_full = StandardScaler()
    X_train_scaled = scaler_full.fit_transform(X_train)
    X_test_scaled = scaler_full.transform(X_test)
    rf_full = RandomForestRegressor(n_estimators=100, random_state=42)
    t0 = time.time()
    rf_full.fit(X_train, y_train)
    rf_train_time = time.time() - t0
    t0 = time.time()
    rf_preds = rf_full.predict(X_test)
    rf_pred_time = time.time() - t0
    rf_r2 = r2_score(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_mae = mean_absolute_error(y_test, rf_preds)
    lr_full = LinearRegression()
    t0 = time.time()
    lr_full.fit(X_train_scaled, y_train)
    lr_train_time = time.time() - t0
    t0 = time.time()
    lr_preds = lr_full.predict(X_test_scaled)
    lr_pred_time = time.time() - t0
    lr_r2 = r2_score(y_test, lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_mae = mean_absolute_error(y_test, lr_preds)
    X_train_no_hi = X_train[features_no_hi]
    X_test_no_hi = X_test[features_no_hi]
    rf_no_hi = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_no_hi.fit(X_train_no_hi, y_train)
    rf_no_hi_preds = rf_no_hi.predict(X_test_no_hi)
    rf_no_hi_r2 = r2_score(y_test, rf_no_hi_preds)
    rf_no_hi_rmse = np.sqrt(mean_squared_error(y_test, rf_no_hi_preds))
    scaler_no_hi = StandardScaler()
    X_train_scaled_no_hi = scaler_no_hi.fit_transform(X_train_no_hi)
    X_test_scaled_no_hi = scaler_no_hi.transform(X_test_no_hi)
    lr_no_hi = LinearRegression()
    lr_no_hi.fit(X_train_scaled_no_hi, y_train)
    lr_no_hi_preds = lr_no_hi.predict(X_test_scaled_no_hi)
    lr_no_hi_r2 = r2_score(y_test, lr_no_hi_preds)
    lr_no_hi_rmse = np.sqrt(mean_squared_error(y_test, lr_no_hi_preds))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rf_r2 = cross_val_score(rf_full, X, y, cv=kf, scoring='r2')
    cv_rf_rmse = np.sqrt(-cross_val_score(rf_full, X, y, cv=kf, scoring='neg_mean_squared_error'))
    cv_lr_r2 = cross_val_score(lr_full, scaler_full.fit_transform(X), y, cv=kf, scoring='r2')
    cv_lr_rmse = np.sqrt(-cross_val_score(lr_full, scaler_full.fit_transform(X), y, cv=kf, scoring='neg_mean_squared_error'))
    rf_errors = np.abs(rf_preds - y_test)
    lr_errors = np.abs(lr_preds - y_test)
    t_stat, p_val_ttest = stats.ttest_rel(rf_errors, lr_errors)
    w_stat, p_val_wilcoxon = stats.wilcoxon(rf_errors, lr_errors)
    rf_residuals = rf_preds - y_test
    shapiro_stat, p_val_shapiro = stats.shapiro(rf_residuals[:500])
    rf_path = 'hasil_riset_bmkg/saved_models/random_forest_model.joblib'
    lr_path = 'hasil_riset_bmkg/saved_models/linear_regression_model.joblib'
    joblib.dump(rf_full, rf_path)
    joblib.dump(lr_full, lr_path)
    rf_size_mb = os.path.getsize(rf_path) / (1024 * 1024)
    lr_size_mb = os.path.getsize(lr_path) / 1024
    feat_imp = pd.DataFrame({
        'Feature': features_full,
        'Importance': rf_full.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    feat_imp['Cumulative'] = feat_imp['Importance'].cumsum()
    return {
        'rf_full': rf_full,
        'lr_full': lr_full,
        'scaler': scaler_full,
        'X_test': X_test,
        'y_test': y_test,
        'rf_preds': rf_preds,
        'lr_preds': lr_preds,
        'features_full': features_full,
        'rf_metrics': {'r2': rf_r2, 'rmse': rf_rmse, 'mae': rf_mae, 'train_time': rf_train_time, 'pred_time': rf_pred_time},
        'lr_metrics': {'r2': lr_r2, 'rmse': lr_rmse, 'mae': lr_mae, 'train_time': lr_train_time, 'pred_time': lr_pred_time},
        'ablation': {
            'rf_no_hi': {'r2': rf_no_hi_r2, 'rmse': rf_no_hi_rmse},
            'lr_no_hi': {'r2': lr_no_hi_r2, 'rmse': lr_no_hi_rmse}
        },
        'cv': {
            'rf_r2_mean': cv_rf_r2.mean(), 'rf_r2_std': cv_rf_r2.std(),
            'rf_rmse_mean': cv_rf_rmse.mean(), 'rf_rmse_std': cv_rf_rmse.std(),
            'lr_r2_mean': cv_lr_r2.mean(), 'lr_r2_std': cv_lr_r2.std(),
            'lr_rmse_mean': cv_lr_rmse.mean(), 'lr_rmse_std': cv_lr_rmse.std()
        },
        'stats': {
            'ttest_p': p_val_ttest, 'wilcoxon_p': p_val_wilcoxon, 'shapiro_p': p_val_shapiro, 'shapiro_stat': shapiro_stat
        },
        'sizes': {
            'rf_size_mb': rf_size_mb, 'lr_size_kb': lr_size_mb
        },
        'feature_importance': feat_imp
    }
def evaluate_cross_regional_all8(df, features, target='temp_avg'):
    regions = df['region'].unique()
    rf_baseline = RandomForestRegressor(n_estimators=100, random_state=42)
    X_full = df[features]
    y_full = df[target]
    rf_baseline.fit(X_full, y_full)
    baseline_r2 = r2_score(y_full, rf_baseline.predict(X_full))
    results = []
    for test_reg in regions:
        train_df = df[df['region'] != test_reg]
        test_df = df[df['region'] == test_reg]
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(train_df[features], train_df[target])
        preds = rf.predict(test_df[features])
        r2 = r2_score(test_df[target], preds)
        rmse = np.sqrt(mean_squared_error(test_df[target], preds))
        perf_drop = ((r2 - baseline_r2) / baseline_r2) * 100.0
        results.append({
            'Test Region': test_reg,
            'R2 Score': r2,
            'RMSE (C)': rmse,
            'Performance Drop (%)': perf_drop
        })
    res_df = pd.DataFrame(results)
    avg_r2 = res_df['R2 Score'].mean()
    avg_rmse = res_df['RMSE (C)'].mean()
    avg_drop = res_df['Performance Drop (%)'].mean()
    res_formatted = res_df.copy()
    res_formatted['R2 Score'] = res_formatted['R2 Score'].apply(lambda v: f"{v:.4f}")
    res_formatted['RMSE (C)'] = res_formatted['RMSE (C)'].apply(lambda v: f"{v:.4f}")
    res_formatted['Performance Drop (%)'] = res_formatted['Performance Drop (%)'].apply(lambda v: f"{v:+.2f}%")
    avg_row = pd.DataFrame([{
        'Test Region': 'Average (All 8 Regions)',
        'R2 Score': f"{avg_r2:.4f}",
        'RMSE (C)': f"{avg_rmse:.4f}",
        'Performance Drop (%)': f"{avg_drop:+.2f}%"
    }])
    return pd.concat([res_formatted, avg_row], ignore_index=True)
def generate_all_rigorous_tables(df, results, table10_df):
    tables = {}
    t1_data = [
        {'Parameter': 'Rainfall (mm)', 'Minimum': round(df['rainfall'].dropna().min(), 1), 'Maksimum': round(df['rainfall'].dropna().max(), 1), 'Mean': round(df['rainfall'].dropna().mean(), 1), 'Std Dev': round(df['rainfall'].dropna().std(), 1)},
        {'Parameter': 'Temp Max (°C)', 'Minimum': round(df['temp_max'].min(), 1), 'Maksimum': round(df['temp_max'].max(), 1), 'Mean': round(df['temp_max'].mean(), 1), 'Std Dev': round(df['temp_max'].std(), 1)},
        {'Parameter': 'Temp Min (°C)', 'Minimum': round(df['temp_min'].min(), 1), 'Maksimum': round(df['temp_min'].max(), 1), 'Mean': round(df['temp_min'].mean(), 1), 'Std Dev': round(df['temp_min'].std(), 1)},
        {'Parameter': 'Temp Avg (°C)', 'Minimum': round(df['temp_avg'].min(), 1), 'Maksimum': round(df['temp_avg'].max(), 1), 'Mean': round(df['temp_avg'].mean(), 1), 'Std Dev': round(df['temp_avg'].std(), 1)},
        {'Parameter': 'Humidity (%)', 'Minimum': round(df['humidity'].min(), 1), 'Maksimum': round(df['humidity'].max(), 1), 'Mean': round(df['humidity'].mean(), 1), 'Std Dev': round(df['humidity'].std(), 1)},
        {'Parameter': 'Wind Speed (m/s)', 'Minimum': round(df['wind_speed_ms'].min(), 1), 'Maksimum': round(df['wind_speed_ms'].max(), 1), 'Mean': round(df['wind_speed_ms'].mean(), 1), 'Std Dev': round(df['wind_speed_ms'].std(), 1)}
    ]
    tables['Table_1_Dataset_Characteristics'] = pd.DataFrame(t1_data)
    rf_m = results['rf_metrics']
    lr_m = results['lr_metrics']
    cv = results['cv']
    ab = results['ablation']
    tables['Table_2_Model_Evaluation'] = pd.DataFrame([
        {
            'Configuration': 'Random Forest (All 19 Features)',
            'R2 Score (Test)': f"{rf_m['r2']:.4f}",
            'RMSE (°C)': f"{rf_m['rmse']:.4f}",
            '5-Fold CV R2 (Mean ± Std)': f"{cv['rf_r2_mean']:.4f} ± {cv['rf_r2_std']:.4f}",
            'Training Time': f"{rf_m['train_time']:.2f} s",
            'Prediction Time': f"{rf_m['pred_time']:.3f} s"
        },
        {
            'Configuration': 'Linear Regression (All 19 Features)',
            'R2 Score (Test)': f"{lr_m['r2']:.4f}",
            'RMSE (°C)': f"{lr_m['rmse']:.4f}",
            '5-Fold CV R2 (Mean ± Std)': f"{cv['lr_r2_mean']:.4f} ± {cv['lr_r2_std']:.4f}",
            'Training Time': f"{lr_m['train_time']:.2f} s",
            'Prediction Time': f"{lr_m['pred_time']:.3f} s"
        },
        {
            'Configuration': 'Ablation: Random Forest (w/o Heat Index)',
            'R2 Score (Test)': f"{ab['rf_no_hi']['r2']:.4f}",
            'RMSE (°C)': f"{ab['rf_no_hi']['rmse']:.4f}",
            '5-Fold CV R2 (Mean ± Std)': '-',
            'Training Time': '-',
            'Prediction Time': '-'
        },
        {
            'Configuration': 'Ablation: Linear Regression (w/o Heat Index)',
            'R2 Score (Test)': f"{ab['lr_no_hi']['r2']:.4f}",
            'RMSE (°C)': f"{ab['lr_no_hi']['rmse']:.4f}",
            '5-Fold CV R2 (Mean ± Std)': '-',
            'Training Time': '-',
            'Prediction Time': '-'
        }
    ])
    top10_feat = results['feature_importance'].head(10).copy()
    top10_feat['Rank'] = range(1, len(top10_feat) + 1)
    top10_feat['Feature_Clean'] = top10_feat['Feature'].replace({
        'heat_index': 'Heat Index',
        'humidity': 'Relative Humidity',
        'temp_max': 'Maximum Temperature',
        'temp_min': 'Minimum Temperature',
        'temp_range': 'Diurnal Temp Range',
        'wind_speed_ms': 'Wind Speed (m/s)',
        'rainfall_ml': 'Precipitation (mm)',
        'day_sin': 'Day Sin (Cyclical)',
        'day_cos': 'Day Cos (Cyclical)',
        'month_sin': 'Month Sin (Cyclical)',
        'region_encoded': 'Region Encoded',
        'station_encoded': 'Station Encoded'
    })
    top10_feat['Importance_Formatted'] = top10_feat['Importance'].apply(lambda v: f"{v:.4f}")
    top10_feat['Cumulative_Formatted'] = top10_feat['Cumulative'].apply(lambda v: f"{v * 100:.2f}%")
    tables['Table_3_Feature_Importance'] = top10_feat[['Rank', 'Feature_Clean', 'Importance_Formatted', 'Cumulative_Formatted']]
    reg_stats = df.groupby('region').agg(
        Rainfall=('rainfall', lambda s: s.dropna().mean()),
        Temp_Avg=('temp_avg', 'mean'),
        Humidity=('humidity', 'mean'),
        Stations=('station_id', 'nunique')
    ).reset_index()
    reg_stats = reg_stats.sort_values(by='Rainfall', ascending=False).reset_index(drop=True)
    reg_stats['Rainfall (mm/day)'] = reg_stats['Rainfall'].apply(lambda v: f"{v:.1f}")
    reg_stats['Temp Avg (°C)'] = reg_stats['Temp_Avg'].apply(lambda v: f"{v:.1f}")
    reg_stats['Humidity (%)'] = reg_stats['Humidity'].apply(lambda v: f"{v:.1f}")
    tables['Table_4_Regional_Statistics'] = reg_stats[['region', 'Rainfall (mm/day)', 'Temp Avg (°C)', 'Humidity (%)', 'Stations']]
    month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    season_patt = df.groupby('month').agg(
        Monthly_Rainfall=('rainfall', lambda s: s.sum() / df['station_id'].nunique()),
        Daily_Avg_Rainfall=('rainfall', 'mean'),
        Season=('season', 'first')
    ).reset_index()
    season_patt['Month'] = season_patt['month'].map(month_names)
    season_patt['Monthly Rainfall (mm)'] = season_patt['Monthly_Rainfall'].round(1)
    season_patt['Daily Avg (mm)'] = season_patt['Daily_Avg_Rainfall'].round(1)
    tables['Table_5_Seasonal_Rainfall'] = season_patt[['Month', 'Monthly Rainfall (mm)', 'Daily Avg (mm)', 'Season']]
    scenarios = [
        {
            'Scenario': 'Jakarta Rainy Season',
            'Input Conditions': 'Month:1, Humidity:80%, Temp_max:32°C',
            'month': 1, 'humidity': 80.0, 'temp_max': 32.0, 'temp_min': 24.0, 'wind_speed_ms': 2.5, 'rainfall_ml': 12.0,
            'region_encoded': 1, 'station_encoded': 3
        },
        {
            'Scenario': 'Medan Dry Season',
            'Input Conditions': 'Month:7, Humidity:65%, Temp_max:35°C',
            'month': 7, 'humidity': 65.0, 'temp_max': 35.0, 'temp_min': 23.0, 'wind_speed_ms': 3.1, 'rainfall_ml': 0.0,
            'region_encoded': 4, 'station_encoded': 0
        },
        {
            'Scenario': 'Bandung Normal',
            'Input Conditions': 'Month:4, Humidity:70%, Temp_max:28°C',
            'month': 4, 'humidity': 70.0, 'temp_max': 28.0, 'temp_min': 20.0, 'wind_speed_ms': 1.7, 'rainfall_ml': 5.0,
            'region_encoded': 2, 'station_encoded': 2
        }
    ]
    t6_rows = []
    for sc in scenarios:
        row_dict = {}
        for f in results['rf_full'].feature_names_in_:
            row_dict[f] = 0.0
        row_dict['month'] = sc['month']
        row_dict['humidity'] = sc['humidity']
        row_dict['temp_max'] = sc['temp_max']
        row_dict['temp_min'] = sc['temp_min']
        row_dict['temp_range'] = sc['temp_max'] - sc['temp_min']
        row_dict['rainfall_ml'] = sc['rainfall_ml']
        row_dict['wind_speed_ms'] = sc['wind_speed_ms']
        row_dict['heat_index'] = calculate_heat_index(sc['temp_max'], sc['humidity'])
        row_df = pd.DataFrame([row_dict])
        rf_p = results['rf_full'].predict(row_df)[0]
        row_scaled = results['scaler'].transform(row_df)
        lr_p = results['lr_full'].predict(row_scaled)[0]
        t6_rows.append({
            'Scenario': sc['Scenario'],
            'Input Conditions': sc['Input Conditions'],
            'RF Prediction': f"{rf_p:.2f}°C",
            'LR Prediction': f"{lr_p:.2f}°C",
            'Avg Prediction': f"{(rf_p + lr_p)/2.0:.2f}°C"
        })
    tables['Table_6_Scenario_Validation'] = pd.DataFrame(t6_rows)
    tables['Table_7_Comparison_Studies'] = pd.DataFrame([
        {'Study': 'Sari & Wibowo [11]', 'Method': 'Neural Network', 'Metric': 'RMSE', 'Performance': '1.20°C', 'Scope': 'Jakarta Temperature'},
        {'Study': 'Rahman et al. [9]', 'Method': 'SVM', 'Metric': 'Accuracy', 'Performance': '87.30%', 'Scope': 'Java Rainfall'},
        {'Study': 'Kusuma et al. [14]', 'Method': 'K-Means + RF', 'Metric': 'Accuracy', 'Performance': '92.10%', 'Scope': 'Sumatra Monsoon'},
        {'Study': 'This Study', 'Method': 'Linear Regression', 'Metric': 'R² Score / RMSE', 'Performance': f"{lr_m['r2'] * 100:.2f}% / {lr_m['rmse']:.4f}°C", 'Scope': 'Multi-region, National'},
        {'Study': 'This Study', 'Method': 'Random Forest', 'Metric': 'R² Score / RMSE', 'Performance': f"{rf_m['r2'] * 100:.2f}% / {rf_m['rmse']:.4f}°C", 'Scope': 'Multi-region, National'}
    ])
    tables['Table_8_Computational_Performance'] = pd.DataFrame([
        {'Aspect': 'Training Time', 'Random Forest': f"{rf_m['train_time']:.2f} s", 'Linear Regression': f"{lr_m['train_time']:.2f} s"},
        {'Aspect': 'Prediction Time (Batch)', 'Random Forest': f"{rf_m['pred_time']:.3f} s", 'Linear Regression': f"{lr_m['pred_time']:.3f} s"},
        {'Aspect': 'Model File Size on Disk', 'Random Forest': f"{results['sizes']['rf_size_mb']:.2f} MB", 'Linear Regression': f"{results['sizes']['lr_size_kb']:.1f} KB"}
    ])
    rf_errors = results['rf_preds'] - results['y_test']
    lr_errors = results['lr_preds'] - results['y_test']
    tables['Table_9_Error_Distribution'] = pd.DataFrame({
        'Metric': [
            'Mean Error', 'Std Error', 'Max Error', 'Min Error',
            'Error < 0.5°C (%)', 'Error < 1.0°C (%)'
        ],
        'Random Forest': [
            f"{rf_errors.mean():+.4f}°C",
            f"{rf_errors.std():.4f}°C",
            f"{rf_errors.max():+.4f}°C",
            f"{rf_errors.min():+.4f}°C",
            f"{(np.abs(rf_errors) < 0.5).mean() * 100:.1f}%",
            f"{(np.abs(rf_errors) < 1.0).mean() * 100:.1f}%"
        ],
        'Linear Regression': [
            f"{lr_errors.mean():+.4f}°C",
            f"{lr_errors.std():.4f}°C",
            f"{lr_errors.max():+.4f}°C",
            f"{lr_errors.min():+.4f}°C",
            f"{(np.abs(lr_errors) < 0.5).mean() * 100:.1f}%",
            f"{(np.abs(lr_errors) < 1.0).mean() * 100:.1f}%"
        ]
    })
    tables['Table_10_Cross_Regional_Validation'] = table10_df
    for name, tbl in tables.items():
        tbl.to_csv(f"hasil_riset_bmkg/{name}.csv", index=False)
    return tables
def generate_publication_figures(df, results, tables):
    plt.figure(figsize=(9, 5), dpi=300)
    top_feat = results['feature_importance'].head(8).copy()
    top_feat['Feature_Clean'] = top_feat['Feature'].replace({
        'region_encoded': 'Region Encoded',
        'temp_max': 'Maximum Temperature',
        'temp_min': 'Minimum Temperature',
        'humidity': 'Relative Humidity',
        'heat_index': 'Heat Index',
        'day_of_year': 'Day of Year',
        'wind_speed_ms': 'Wind Speed (m/s)',
        'rainfall_ml': 'Precipitation (mm)',
        'temp_range': 'Diurnal Temp Range',
        'day_sin': 'Day Sin (Cyclical)',
        'day_cos': 'Day Cos (Cyclical)',
        'station_encoded': 'Station Encoded'
    })
    sns.barplot(x='Importance', y='Feature_Clean', data=top_feat, hue='Feature_Clean', palette='viridis', legend=False)
    plt.title('Feature Importance Analysis (Random Forest Regressor)', fontsize=13, fontweight='bold')
    plt.xlabel('Relative Importance Score (Mean Decrease in Impurity)', fontsize=11)
    plt.ylabel('Predictor Features', fontsize=11)
    plt.tight_layout()
    plt.savefig('hasil_riset_bmkg/figures/fig_feature_importance.png')
    plt.close()
    plt.figure(figsize=(7, 6), dpi=300)
    y_test = results['y_test']
    rf_preds = results['rf_preds']
    plt.scatter(y_test, rf_preds, alpha=0.5, color='#2b5c8f', edgecolors='none', s=25, label='RF Predictions')
    min_val = min(y_test.min(), rf_preds.min())
    max_val = max(y_test.max(), rf_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit ($y = x$)')
    plt.title(f"Actual vs Predicted Average Temperature\n($R^2 = {results['rf_metrics']['r2']:.4f}$, RMSE = {results['rf_metrics']['rmse']:.4f}°C)", fontsize=12, fontweight='bold')
    plt.xlabel('Actual Temperature (°C)', fontsize=11)
    plt.ylabel('Predicted Temperature (°C)', fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig('hasil_riset_bmkg/figures/fig_actual_vs_predicted.png')
    plt.close()
    plt.figure(figsize=(9, 4.5), dpi=300)
    t5 = tables['Table_5_Seasonal_Rainfall']
    colors = ['#2b5c8f' if c == 'Rainy Season' else ('#e28743' if c == 'Dry Season' else '#218c74') for c in t5['Season']]
    plt.bar(t5['Month'], t5['Monthly Rainfall (mm)'], color=colors, edgecolor='black', alpha=0.85)
    plt.title('Seasonal Rainfall Patterns Across Indonesian Meteorological Stations', fontsize=12, fontweight='bold')
    plt.xlabel('Month', fontsize=11)
    plt.ylabel('Monthly Rainfall (mm)', fontsize=11)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2b5c8f', edgecolor='black', label='Rainy Season (Nov-Mar)'),
        Patch(facecolor='#218c74', edgecolor='black', label='Transition (Apr-May, Oct)'),
        Patch(facecolor='#e28743', edgecolor='black', label='Dry Season (Jun-Sep)')
    ]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig('hasil_riset_bmkg/figures/fig_seasonal_rainfall.png')
    plt.close()
if __name__ == '__main__':
    df = load_and_clean_dataset(year=2023)
    results = run_model_experiments_and_ablation(df)
    table10_df = evaluate_cross_regional_all8(df, results['features_full'])
    tables = generate_all_rigorous_tables(df, results, table10_df)
    generate_publication_figures(df, results, tables)
    print("Experiments completed successfully.")
