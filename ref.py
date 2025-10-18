import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt

# Widen page layout
st.set_page_config(page_title="Advanced Insider Threat Dashboard", layout="wide", initial_sidebar_state="expanded")


def lstm_anomaly_detection(df):
    rename_map = {
        'failed_login_attempts': 'failed_logins',
        'usb_inserted': 'usb_inserts',
        'remote_access_tool': 'remote_access_tool_usage',
        'data_downloaded': 'data_downloaded_MB',
        'data_uploaded': 'data_uploaded_MB'
    }
    df = df.copy()
    df.rename(columns=rename_map, inplace=True)
    df['timestamp'] = df['login_time']

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

    scalers = {user: StandardScaler().fit(df.loc[df['user_id'] == user, features]) for user in df['user_id'].unique()}
    for user, scaler in scalers.items():
        df.loc[df['user_id'] == user, features] = scaler.transform(df.loc[df['user_id'] == user, features])

    sequence_length = 10

    def create_sequences(data, seq_length=sequence_length):
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
    if not user_sequences:
        return pd.DataFrame(columns=['user_id', 'seq_start_idx', 'reconstruction_error', 'anomaly', 'timestamp'])
    user_sequences = np.vstack(user_sequences)

    dataset = tf.data.Dataset.from_tensor_slices((user_sequences, user_sequences)).shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)

    input_dim = len(features)
    latent_dim = 32
    inputs = Input(shape=(sequence_length, input_dim))
    encoded = Bidirectional(LSTM(latent_dim, activation='relu', dropout=0.2))(inputs)
    decoded = RepeatVector(sequence_length)(encoded)
    decoded = Bidirectional(LSTM(input_dim, activation='relu', return_sequences=True, dropout=0.2))(decoded)
    outputs = TimeDistributed(Dense(input_dim))(decoded)

    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    autoencoder.fit(dataset, epochs=50, shuffle=True, callbacks=[early_stop], verbose=0)

    reconstructions = autoencoder.predict(user_sequences, verbose=0)
    mse = np.mean(np.square(user_sequences - reconstructions), axis=(1, 2))

    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    anomaly = (isolation_forest.fit_predict(mse.reshape(-1, 1)) == -1).astype(int)

    results = pd.DataFrame(user_seq_ids, columns=['user_id', 'seq_start_idx'])
    results['reconstruction_error'] = mse
    results['anomaly'] = anomaly
    results['timestamp'] = [df[df['user_id'] == user].iloc[start]['timestamp'] for user, start in user_seq_ids]

    results.to_csv('lstm_anomalies.csv', index=False)
    return results


def compute_threat_scores(anomalies_df):
    anomalies_df = anomalies_df.copy()
    anomalies_df['timestamp'] = pd.to_datetime(anomalies_df['timestamp'], errors='coerce')
    anomalies_df.dropna(subset=['timestamp'], inplace=True)

    lookback_days = 3
    now = anomalies_df['timestamp'].max()
    window_start = now - timedelta(days=lookback_days)
    anomalies_df = anomalies_df[anomalies_df['timestamp'] >= window_start]

    user_groups = anomalies_df.groupby('user_id')['reconstruction_error']
    anomalies_df['error_mean_user'] = user_groups.transform('mean')
    anomalies_df['error_std_user'] = user_groups.transform('std').replace(0, 1e-6)
    anomalies_df['error_zscore'] = (anomalies_df['reconstruction_error'] - anomalies_df['error_mean_user']) / anomalies_df['error_std_user']

    anomalies_df['rolling_mean_24h'] = anomalies_df.groupby('user_id')['reconstruction_error'] \
        .transform(lambda x: x.rolling(window=24, min_periods=1).mean())
    anomalies_df['spike_factor'] = (anomalies_df['reconstruction_error'] - anomalies_df['rolling_mean_24h']).clip(lower=0)

    decay_half_life_hours = 24
    decay_rate = np.log(2) / decay_half_life_hours
    delta_hours = (now - anomalies_df['timestamp']).dt.total_seconds() / 3600
    anomalies_df['decay_weight'] = np.exp(-decay_rate * delta_hours)

    user_scores = anomalies_df.groupby('user_id').apply(lambda g: pd.Series({
        'avg_error_z': np.average(g['error_zscore'], weights=g['decay_weight']),
        'max_error': g['reconstruction_error'].max(),
        'spike_sum': np.sum(g['spike_factor'] * g['decay_weight']),
        'anomaly_count': np.sum(g['anomaly']),
        'last_seen': g['timestamp'].max()
    })).reset_index()

    user_scores['raw_score'] = (
        0.4 * user_scores['avg_error_z'] +
        0.3 * user_scores['max_error'] +
        0.2 * np.log1p(user_scores['spike_sum']) +
        0.1 * np.log1p(user_scores['anomaly_count'])
    )

    min_raw = user_scores['raw_score'].min()
    max_raw = user_scores['raw_score'].max()
    user_scores['threat_score'] = 100 * (user_scores['raw_score'] - min_raw) / (max_raw - min_raw + 1e-6)
    user_scores['threat_score'] = user_scores['threat_score'].round(2)

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
    user_scores.to_csv('enhanced_user_threat_scores.csv', index=False)
    return user_scores

@st.cache_data(show_spinner=False)
def run_processing(df):
    anomalies = lstm_anomaly_detection(df)
    scores = compute_threat_scores(anomalies)
    return anomalies, scores

st.title("🚀 Insider Threat Detection Dashboard")

uploaded_file = st.file_uploader("Upload Activity Dataset CSV", type="csv",
                                 help="Upload the activity dataset to run anomaly detection and threat scoring.")

if uploaded_file is not None:
    activity_df = pd.read_csv(uploaded_file, parse_dates=['login_time', 'logout_time'])
    #st.success(f"Activity dataset loaded with {len(activity_df)} records.")

    anomalies_df, threat_scores = run_processing(activity_df)
    #st.success(f"Anomaly detection completed with {anomalies_df['anomaly'].sum()} anomalies found.")
    #st.success("Threat scoring completed.")
else:
    @st.cache_data(ttl=600)
    def load_dashboard_data():
        try:
            act_df = pd.read_csv("activity_data.csv", parse_dates=["login_time", "logout_time"])
            anom_df = pd.read_csv("lstm_anomalies.csv", parse_dates=["timestamp"])
            thr_df = pd.read_csv("enhanced_user_threat_scores.csv", parse_dates=["last_seen"])
            return act_df, anom_df, thr_df
        except Exception:
            return None, None, None

    activity_df, anomalies_df, threat_scores = load_dashboard_data()
    if activity_df is None or anomalies_df is None or threat_scores is None:
        st.error("No data available yet. Upload an activity dataset to generate anomalies and threat scores.")
        st.stop()

if "user_name" not in activity_df.columns:
    activity_df["user_name"] = activity_df["user_id"].astype(str)
if "department" not in activity_df.columns:
    activity_df["department"] = "Unknown"

employees = activity_df[["user_id", "user_name", "department"]].drop_duplicates()
dashboard_users = employees.merge(threat_scores[['user_id', 'threat_score', 'last_seen', 'risk_level']],
                                 on='user_id', how='left')
dashboard_users['threat_score'] = dashboard_users['threat_score'].fillna(0)
dashboard_users['last_seen'] = pd.to_datetime(dashboard_users['last_seen'], errors='coerce')

# Sidebar controls
recent_days = st.sidebar.slider("Recent window (days)", 1, 30, 7)
threshold_value = st.sidebar.slider("Threat Threshold", 0, 100, 80)
top_n = st.sidebar.slider("Top N Risky Users", 1, 20, 10)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔥 Top Risky Users", "🔍 User Behavior Analysis"])

# Overview tab
with tab1:
    k1, k2, k3, k4 = st.columns(4)
    total_users = dashboard_users["user_id"].nunique()
    becoming_count = int((dashboard_users["threat_score"] >= threshold_value).sum())
    avg_score = dashboard_users["threat_score"].mean().round(2)
    now = pd.Timestamp.now()
    recent_active = 0
    if "last_seen" in dashboard_users.columns and dashboard_users["last_seen"].notna().any():
        recent_active = int(dashboard_users[dashboard_users["last_seen"] >= (now - timedelta(days=recent_days))]["user_id"].nunique())
    k1.metric("👥 Total users", total_users)
    k2.metric("🚨 Above Threshold", becoming_count)
    k3.metric("📈 Avg Threat Score", f"{avg_score:.2f}")
    k4.metric(f"🟢 Active (last {recent_days}d)", recent_active)

    st.markdown("### 🏢 Department Insights")
    col1, col2 = st.columns(2)
    with col1:
        dept_rank = dashboard_users.groupby("department").agg(
            avg_threat_score=("threat_score", "mean"),
            above_threshold=("threat_score", lambda s: (s >= threshold_value).sum())
        ).reset_index().sort_values("avg_threat_score", ascending=True)
        if not dept_rank.empty:
            fig_rank = px.bar(
                dept_rank, x="avg_threat_score", y="department", orientation="h",
                color="avg_threat_score", color_continuous_scale=px.colors.sequential.OrRd,
                text="above_threshold", title="Departments by Avg Threat Score"
            )
            fig_rank.update_traces(texttemplate="%{text} threats", textposition="inside")
            st.plotly_chart(fig_rank, use_container_width=True)

    with col2:
        emp_count = dashboard_users.groupby("department")["user_id"].nunique().reset_index()
        emp_count.columns = ["Department", "Employee Count"]
        if not emp_count.empty:
            fig_pie = px.pie(emp_count, values="Employee Count", names="Department", hole=0.3,
                             color="Department", color_discrete_sequence=px.colors.sequential.OrRd)
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### 📊 Threat Score Distribution")
    fig_hist = px.histogram(dashboard_users, x="threat_score", nbins=20, title="Threat Score Distribution")
    st.plotly_chart(fig_hist, use_container_width=True)

# Top Risky Users tab
with tab2:
    st.markdown(f"### Top {top_n} Risky Users")
    top_risky = dashboard_users.sort_values(by='threat_score', ascending=False).head(top_n)
    top_risky_display = top_risky[["user_id", "user_name", "department", "threat_score", "last_seen"]]
    top_risky_display = top_risky_display.rename(columns={
        "user_id": "User ID",
        "user_name": "User Name",
        "department": "Department",
        "threat_score": "Threat Score",
        "last_seen": "Last Seen"
    }).reset_index(drop=True)
    top_risky_display.insert(0, "Rank", range(1, len(top_risky_display) + 1))
    st.dataframe(top_risky_display.style.background_gradient(subset=["Threat Score"], cmap="Reds"))
    csv = top_risky_display.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Top Risky Users", csv, "top_risky_users.csv", "text/csv")

# User Behavior Analysis tab
with tab3:
    selected_user_id = st.selectbox("Select a User", sorted(dashboard_users['user_id'].unique()))
    if selected_user_id:
        user_info = dashboard_users[dashboard_users['user_id'] == selected_user_id].iloc[0]
        user_name = user_info['user_name']
        dept = user_info['department']
        score = user_info['threat_score']
        last_seen = user_info['last_seen']

        st.subheader(f"👤 {user_name} ")
        st.write(f"**Department:** {dept}")
        st.write(f"**Current Threat Score:** {score}")
        st.write(f"**Last Seen:** {last_seen}")

        user_activity = activity_df[activity_df['user_id'] == selected_user_id]
        user_anomalies = anomalies_df[anomalies_df['user_id'] == selected_user_id] if anomalies_df is not None else pd.DataFrame()

        # Behavior Change Heatmap
        st.markdown("#### User Behavior Change Heatmap")
        features_to_use = [
            'failed_login_attempts', 'usb_inserted', 'files_copied_to_usb',
            'data_downloaded', 'data_uploaded', 'print_usage', 'remote_access_tool'
        ]
        available_features = [f for f in features_to_use if f in user_activity.columns]
        if user_activity.empty or not available_features:
            st.info("No relevant activity data/features available for this user.")
        else:
            user_activity['date'] = user_activity['login_time'].dt.date
            daily_means = user_activity.groupby('date')[available_features].mean().T
            daily_diff = daily_means.diff(axis=1).fillna(0)
            fig_heatmap = px.imshow(
                daily_diff,
                labels={'x': 'Date', 'y': 'Feature', 'color': 'Change'},
                color_continuous_scale='RdBu_r',
                origin='lower',
                aspect="auto",
                title="Daily Change in User Activity Features"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

        # Anomaly Timeline
        st.markdown("#### Anomaly Timeline")
        if user_anomalies.empty or user_anomalies['timestamp'].isna().all():
            st.info("No anomaly data available for this user.")
        else:
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(
                x=user_anomalies['timestamp'],
                y=user_anomalies['reconstruction_error'],
                mode='lines+markers',
                name='Reconstruction Error'
            ))
            anomalies_flagged = user_anomalies[user_anomalies['anomaly'] == 1]
            fig_timeline.add_trace(go.Scatter(
                x=anomalies_flagged['timestamp'],
                y=anomalies_flagged['reconstruction_error'],
                mode='markers',
                marker=dict(color='red', size=8, symbol='x'),
                name='Anomalies'
            ))
            fig_timeline.update_layout(title="Reconstruction Error and Anomalies Over Time",
                                       xaxis_title="Time", yaxis_title="Reconstruction Error")
            st.plotly_chart(fig_timeline, use_container_width=True)

        # Peer Comparison
        st.markdown("#### Peer Comparison by Department Threat Scores")
        peer_scores = dashboard_users[dashboard_users['department'] == dept]
        if peer_scores.empty:
            st.info("No peer data available for this department.")
        else:
            fig_peer = px.histogram(peer_scores, x='threat_score', nbins=20,
                                    title=f"Threat Score Distribution for Department: {dept}")
            fig_peer.add_vline(x=score, line_dash="dash", line_color="red")
            st.plotly_chart(fig_peer, use_container_width=True)
            st.write(f"User Threat Score is shown as a red line: {score:.2f}")

        # Threat Escalation Path Visualization
        st.markdown("#### Threat Escalation Path Visualization")
        events = []
        if not user_activity.empty:
            for _, row in user_activity.iterrows():
                time = row['login_time'] if not pd.isna(row['login_time']) else None
                if time:
                    if "failed_login_attempts" in row and row["failed_login_attempts"] > 0:
                        events.append((time, "Failed Login"))
                    if "usb_inserted" in row and row["usb_inserted"] > 0:
                        events.append((time, "USB Insert"))
                    if "remote_access_tool" in row and row["remote_access_tool"] > 0:
                        events.append((time, "Remote Access"))
                    if "files_copied_to_usb" in row and row["files_copied_to_usb"] > 0:
                        events.append((time, "Files copied to USB"))

        if not user_anomalies.empty and 'timestamp' in user_anomalies.columns:
            for _, row in user_anomalies.iterrows():
                if row['anomaly'] == 1 and pd.notna(row['timestamp']):
                    events.append((row['timestamp'], "Anomaly Detected"))

        if len(events) < 2:
            st.info("Not enough events to build escalation path.")
        else:
            events.sort(key=lambda x: x[0])
            G = nx.DiGraph()
            for i in range(len(events) - 1):
                src, tgt = events[i][1], events[i + 1][1]
                if G.has_edge(src, tgt):
                    G[src][tgt]['weight'] += 1
                else:
                    G.add_edge(src, tgt, weight=1)

            pos = nx.spring_layout(G, seed=42)
            edge_x = []
            edge_y = []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines')

            node_x = []
            node_y = []
            for n in G.nodes():
                x, y = pos[n]
                node_x.append(x)
                node_y.append(y)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=[n for n in G.nodes()],
                textposition="bottom center",
                hoverinfo='text',
                marker=dict(color='#fa8072', size=20, line_width=2))

            fig_graph = go.Figure(data=[edge_trace, node_trace],
                                  layout=go.Layout(
                                      title="Threat Escalation Path",
                                      showlegend=False,
                                      hovermode='closest',
                                      margin=dict(b=20, l=5, r=5, t=40),
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                  )
            st.plotly_chart(fig_graph, use_container_width=True)

        # Threat Explanation
        st.markdown("### 📝 Threat Explanation")
        if score < 50:
            st.success("Not a threat based on current scores. No explanations to provide.")
        else:
            explanation = []
            if not user_activity.empty:
                if "failed_login_attempts" in user_activity.columns and user_activity["failed_login_attempts"].sum() > 5:
                    explanation.append("Multiple failed login attempts detected.")
                if "usb_inserted" in user_activity.columns and user_activity["usb_inserted"].sum() > 0:
                    explanation.append("USB device insertions observed.")
                if "remote_access_tool" in user_activity.columns and user_activity["remote_access_tool"].sum() > 0:
                    explanation.append("Use of remote access tools detected.")
                if "print_usage" in user_activity.columns and user_activity["print_usage"].sum() > 0:
                    explanation.append("Suspicious printing activity detected.")
                if "data_downloaded" in user_activity.columns and user_activity["data_downloaded"].mean() > 1000:
                    explanation.append("Unusually large data downloads.")
                if "files_copied_to_usb" in user_activity.columns and user_activity["files_copied_to_usb"].sum() > 0:
                    explanation.append("Sensitive files copied to USB.")
            if not user_anomalies.empty and user_anomalies["anomaly"].sum() > 0:
                explanation.append("Recent anomalies flagged by the model.")

            if explanation:
                for line in explanation:
                    st.write("- " + line)
            else:
                st.info("User flagged as threat but no specific indicators found in activity logs.")




