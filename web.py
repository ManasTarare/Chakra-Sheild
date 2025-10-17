import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import timedelta

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Advanced Insider Threat Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Utility Functions with caching
# =========================
@st.cache_data(ttl=3600)
def safe_read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None


@st.cache_data(ttl=900)
def load_all_data():
    activity_df = safe_read_csv("activity_data.csv")
    threat_scores = safe_read_csv("enhanced_user_threat_scores.csv")
    anomalies_df = safe_read_csv("lstm_anomalies.csv")

    if activity_df is not None:
        for col in ["login_time", "logout_time"]:
            if col in activity_df.columns:
                activity_df[col] = pd.to_datetime(activity_df[col], errors="coerce")
        if "user_name" not in activity_df.columns and "user_id" in activity_df.columns:
            activity_df["user_name"] = activity_df["user_id"].astype(str)
        activity_df["department"] = activity_df.get("department", "Unknown")
        for col in ["application_usage", "network_sites"]:
            if col in activity_df.columns:
                activity_df[f"{col}_count"] = activity_df[col].fillna("").str.split(",").apply(lambda x: len([p for p in x if p.strip()]))

    if threat_scores is not None:
        if "last_seen" in threat_scores.columns:
            threat_scores["last_seen"] = pd.to_datetime(threat_scores["last_seen"], errors="coerce")
            if "timestamp" not in threat_scores.columns:
                threat_scores["timestamp"] = threat_scores["last_seen"]
        if "becoming_threat" in threat_scores.columns:
            threat_scores["becoming_threat"] = threat_scores["becoming_threat"].astype(bool)
        else:
            threat_scores["becoming_threat"] = False
        if "threat_score" in threat_scores.columns:
            threat_scores["threat_score"] = pd.to_numeric(threat_scores["threat_score"], errors="coerce").fillna(0)
        else:
            threat_scores["threat_score"] = 0
        if "threat_level" not in threat_scores.columns:
            threat_scores["threat_level"] = np.where(
                threat_scores["becoming_threat"], "Becoming Threat", "Normal"
            )

    if anomalies_df is not None:
        if "timestamp" in anomalies_df.columns:
            anomalies_df["timestamp"] = pd.to_datetime(anomalies_df["timestamp"], errors="coerce")
        else:
            anomalies_df["timestamp"] = pd.NaT

        anomalies_df["reconstruction_error"] = pd.to_numeric(anomalies_df.get("reconstruction_error", 0), errors="coerce").fillna(0)
        if "anomaly" in anomalies_df.columns:
            anomalies_df["anomaly"] = anomalies_df["anomaly"].astype(int)

    return activity_df, threat_scores, anomalies_df


activity_df, threat_scores, anomalies_df = load_all_data()

if activity_df is None:
    st.error("❌ Couldn't find `activity_data.csv`. Please put it in the same folder and re-run.")
    st.stop()

employees = activity_df[["user_id", "user_name", "department"]].drop_duplicates(subset=["user_id"]).set_index("user_id")
if threat_scores is None:
    threat_scores = pd.DataFrame({
        "user_id": employees.index,
        "threat_score": 0,
        "becoming_threat": False,
        "threat_level": "Normal",
        "last_seen": pd.NaT
    }).reset_index(drop=False)

@st.cache_data
def prepare_dashboard_users():
    merged = employees.reset_index().merge(threat_scores, on="user_id", how="left")
    merged["threat_score"] = pd.to_numeric(merged["threat_score"].fillna(0))
    merged["department"] = merged["department"].fillna("Unknown")
    return merged.drop_duplicates(subset=["user_id"], keep="last")

dashboard_users = prepare_dashboard_users()

# Sidebar filters
st.sidebar.header("⚙️ Controls")
recent_days = st.sidebar.slider("Recent window (days)", 1, 30, 7)
threshold_value = st.sidebar.slider("Threat Threshold", 0, 100, 80)
top_n = st.sidebar.slider("Top N Risky Users", 1, 20, 10)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔥 Top Risky Users", "🔍 User Analysis"])

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

# Top risk users tab
with tab2:
    st.markdown(f"### 🔥 Top {top_n} Risky Users")
    top_risky = dashboard_users.sort_values(by="threat_score", ascending=False).head(top_n)
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
    csv = top_risky_display.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Top Risky Users", csv, "top_risky_users.csv", "text/csv")

# Enhanced User Analysis with the reversed heatmap colors
def tab3_enhanced(
    dashboard_users,
    activity_df,
    threat_scores,
    anomalies_df,
    threshold_value,
    selected_user_id
):
    st.markdown("### 🔍 Deep User Analysis")
    if not selected_user_id:
        st.info("Select a user to analyze.")
        return

    user_info = dashboard_users[dashboard_users["user_id"] == selected_user_id]
    if user_info.empty:
        st.warning("User not found.")
        return

    user_name = user_info["user_name"].iloc[0]
    dept = user_info["department"].iloc[0]
    score = user_info["threat_score"].iloc[0]
    last_seen = user_info["last_seen"].iloc[0]

    st.subheader(f"👤 {user_name} ({selected_user_id})")
    st.write(f"**Department:** {dept}")
    st.write(f"**Current Threat Score:** {score}")
    st.write(f"**Last Seen:** {last_seen}")

    user_activity = activity_df[activity_df["user_id"] == selected_user_id]
    user_anomalies = anomalies_df[anomalies_df["user_id"] == selected_user_id] if anomalies_df is not None else pd.DataFrame()

    # 1. User behavior change detection heatmap with reversed color scale
    st.markdown("#### User Behavior Change Heatmap")
    if user_activity.empty:
        st.info("No activity data available for this user.")
    else:
        features_to_use = [
            "failed_login_attempts", "usb_inserted", "files_copied_to_usb",
            "data_downloaded", "data_uploaded", "print_usage", "remote_access_tool"
        ]
        available_features = [f for f in features_to_use if f in user_activity.columns]
        if available_features:
            user_activity['date'] = user_activity['login_time'].dt.date
            daily_means = user_activity.groupby('date')[available_features].mean().T
            daily_diff = daily_means.diff(axis=1).fillna(0)
            fig_heatmap = px.imshow(
                daily_diff,
                labels={'x': 'Date', 'y': 'Feature', 'color': 'Change'},
                color_continuous_scale='RdBu_r',  # reversed color scale here
                origin='lower',
                aspect="auto",
                title="Daily Change in User Activity Features"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("No relevant activity features available.")

    # 3. Anomaly Timeline
    st.markdown("#### Anomaly Timeline")
    if user_anomalies.empty or user_anomalies['timestamp'].isna().all():
        st.info("No anomaly data available for this user.")
    else:
        anomaly_times = user_anomalies['timestamp']
        recon_errors = user_anomalies['reconstruction_error']
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=anomaly_times,
            y=recon_errors,
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

    # 4. Peer Comparison
    st.markdown("#### Peer Comparison by Department Threat Scores")
    peer_scores = dashboard_users[dashboard_users["department"] == dept]
    if peer_scores.empty:
        st.info("No peer data available for this department.")
    else:
        fig_peer = px.histogram(
            peer_scores,
            x="threat_score",
            nbins=20,
            title=f"Threat Score Distribution for Department: {dept}"
        )
        fig_peer.add_vline(x=score, line_dash="dash", line_color="red")
        st.plotly_chart(fig_peer, use_container_width=True)
        st.write(f"User Threat Score is shown as a red line: {score:.2f}")

    # 7. Threat escalation path visualization
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
        for i in range(len(events)-1):
            src, tgt = events[i][1], events[i+1][1]
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
                                 margin=dict(b=20,l=5,r=5,t=40),
                                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                             )
        st.plotly_chart(fig_graph, use_container_width=True)

    # Explainable AI for threat explanation
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

# Call third tab with enhancement
with tab3:
    user_list = sorted(dashboard_users["user_id"].dropna().unique())
    selected_user_id = st.selectbox("Select a User", user_list)
    tab3_enhanced(
        dashboard_users,
        activity_df,
        threat_scores,
        anomalies_df,
        threshold_value,
        selected_user_id
    )
