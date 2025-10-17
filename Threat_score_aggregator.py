import pandas as pd
import numpy as np
from datetime import timedelta

# === Step 1: Load and preprocess anomaly data ===
df = pd.read_csv('lstm_anomalies.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
df.dropna(subset=['timestamp'], inplace=True)

# === Step 2: Filter recent window ===
lookback_days = 3
now = df['timestamp'].max()
window_start = now - timedelta(days=lookback_days)
df = df[df['timestamp'] >= window_start]

# === Step 3: Per-user Z-score (baseline deviation) ===
user_groups = df.groupby('user_id')['reconstruction_error']
df['error_mean_user'] = user_groups.transform('mean')
df['error_std_user'] = user_groups.transform('std').replace(0, 1e-6)
df['error_zscore'] = (df['reconstruction_error'] - df['error_mean_user']) / df['error_std_user']

# === Step 4: Spike factor (rolling mean) ===
df['rolling_mean_24h'] = df.groupby('user_id')['reconstruction_error'].transform(
    lambda x: x.rolling(window=24, min_periods=1).mean()
)
df['spike_factor'] = (df['reconstruction_error'] - df['rolling_mean_24h']).clip(lower=0)

# === Step 5: Time decay factor ===
decay_half_life_hours = 24
decay_rate = np.log(2) / decay_half_life_hours
delta_hours = (now - df['timestamp']).dt.total_seconds() / 3600
df['decay_weight'] = np.exp(-decay_rate * delta_hours)

# === Step 6: Aggregate per user with weighted averages and sums ===
user_scores = df.groupby('user_id').apply(lambda g: pd.Series({
    'avg_error_z': np.average(g['error_zscore'], weights=g['decay_weight']),
    'max_error': g['reconstruction_error'].max(),
    'spike_sum': np.sum(g['spike_factor'] * g['decay_weight']),
    'anomaly_count': np.sum(g['anomaly']),
    'last_seen': g['timestamp'].max()
})).reset_index()

# === Step 7: Threat scoring formula ===
user_scores['raw_score'] = (
    0.4 * user_scores['avg_error_z'] +
    0.3 * user_scores['max_error'] +
    0.2 * np.log1p(user_scores['spike_sum']) +
    0.1 * np.log1p(user_scores['anomaly_count'])
)

# Normalize scores into 0-100 scale
min_raw = user_scores['raw_score'].min()
max_raw = user_scores['raw_score'].max()
user_scores['threat_score'] = 100 * (user_scores['raw_score'] - min_raw) / (max_raw - min_raw + 1e-6)
user_scores['threat_score'] = user_scores['threat_score'].round(2)

# === Step 8: Risk band categorization ===
def categorize(score):
    if score < 40:
        return "Low"
    elif score < 70:
        return "Medium"
    elif score < 90:
        return "High"
    else:
        return "Critical"

user_scores['risk_level'] = user_scores['threat_score'].apply(categorize)

# === Step 9: Save output ===
output_path = 'enhanced_user_threat_scores.csv'
user_scores.to_csv(output_path, index=False)
print(f"[✔] Threat scores saved to {output_path}")
