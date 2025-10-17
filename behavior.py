import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# STEP 1: Load and preprocess data
# ============================================================
df = pd.read_csv('activity_data.csv', parse_dates=['login_time', 'logout_time'])
df['timestamp'] = df['login_time']

df.rename(columns={
    'failed_login_attempts': 'failed_logins',
    'usb_inserted': 'usb_inserts',
    'remote_access_tool': 'remote_access_tool_usage',
    'data_downloaded': 'data_downloaded_MB',
    'data_uploaded': 'data_uploaded_MB'
}, inplace=True)

df['application_usage_count'] = df['application_usage'].fillna('').str.count(',') + (df['application_usage'].notna()).astype(int)
df['network_usage_MB'] = (df['network_sites'].fillna('').str.count(',') + (df['network_sites'].notna()).astype(int)) * 5

features = [
    'failed_logins', 'usb_inserts', 'files_copied_to_usb', 'bluetooth_usage',
    'clipboard_usage', 'print_usage', 'command_shell_usage', 'files_accessed',
    'files_deleted', 'data_downloaded_MB', 'data_uploaded_MB',
    'remote_access_tool_usage', 'application_usage_count', 'network_usage_MB'
]

df[features] = df[features].fillna(0)
df.sort_values(['user_id', 'timestamp'], inplace=True, ignore_index=True)

# ============================================================
# STEP 2: Scale features efficiently per user
# ============================================================
scalers = {user: StandardScaler().fit(df.loc[df['user_id'] == user, features]) for user in df['user_id'].unique()}
for user, scaler in scalers.items():
    df.loc[df['user_id'] == user, features] = scaler.transform(df.loc[df['user_id'] == user, features])

# ============================================================
# STEP 3: Create sequences using NumPy slicing
# ============================================================
sequence_length = 10

def create_sequences(data, seq_length=10):
    data_array = data[features].to_numpy()
    total_seq = len(data_array) - seq_length + 1
    if total_seq <= 0:
        return np.empty((0, seq_length, len(features)))
    return np.lib.stride_tricks.sliding_window_view(data_array, (seq_length, len(features)))[:, 0, :, :]

user_sequences, user_seq_ids = [], []
for user in df['user_id'].unique():
    user_data = df[df['user_id'] == user].reset_index(drop=True)
    seqs = create_sequences(user_data, sequence_length)
    if seqs.size > 0:
        user_sequences.append(seqs)
        user_seq_ids.extend([(user, idx) for idx in range(len(seqs))])

user_sequences = np.vstack(user_sequences)

# Convert to TensorFlow dataset for parallel loading
dataset = tf.data.Dataset.from_tensor_slices((user_sequences, user_sequences)).shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)

# ============================================================
# STEP 4: Build enhanced BiLSTM Autoencoder with Dropout
# ============================================================
input_dim = len(features)
latent_dim = 32

inputs = Input(shape=(sequence_length, input_dim))
encoded = Bidirectional(LSTM(latent_dim, activation='relu', dropout=0.2))(inputs)
decoded = RepeatVector(sequence_length)(encoded)
decoded = Bidirectional(LSTM(input_dim, activation='relu', return_sequences=True, dropout=0.2))(decoded)
outputs = TimeDistributed(Dense(input_dim))(decoded)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer='adam', loss='mse')

# ============================================================
# STEP 5: Train with early stopping
# ============================================================
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
autoencoder.fit(dataset, epochs=50, shuffle=True, callbacks=[early_stop], verbose=1)

# ============================================================
# STEP 6: Reconstruction errors
# ============================================================
reconstructions = autoencoder.predict(user_sequences, verbose=0)
mse = np.mean(np.square(user_sequences - reconstructions), axis=(1, 2))

# ============================================================
# STEP 7: Isolation Forest for anomaly detection (replaces GMM)
# ============================================================
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
anomaly = (isolation_forest.fit_predict(mse.reshape(-1, 1)) == -1).astype(int)

# ============================================================
# STEP 8: Save results
# ============================================================
results = pd.DataFrame(user_seq_ids, columns=['user_id', 'seq_start_idx'])
results['reconstruction_error'] = mse
results['anomaly'] = anomaly

results['timestamp'] = [df[df['user_id'] == user].iloc[start, df.columns.get_loc('timestamp')] for user, start in user_seq_ids]
results.to_csv('lstm_anomalies.csv', index=False)

print('[✔] Enhanced LSTM anomaly detection finished. Results saved to enhanced_lstm_anomalies.csv')
