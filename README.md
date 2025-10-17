# Chakra-Sheild
AI powered Insider threat detection system

Insider Threat Detection Dashboard
Project Overview
This project presents an advanced Insider Threat Detection system designed to identify and visualize potential risky behaviors among users in organizational environments. Leveraging a combination of advanced machine learning techniques, including an LSTM autoencoder for anomaly detection, and explainable AI, the system provides security teams with actionable insights to proactively mitigate insider threats.

The project consists of a data processing pipeline, an anomaly detection model, and an interactive dashboard built with Streamlit and Plotly, enabling dynamic threat monitoring and detailed user behavior analysis.

Features
LSTM Autoencoder Anomaly Detection: Detect subtle anomalies in sequential user activity data.

User Behavior Change Heatmaps: Visualize shifts in user activity metrics over time.

Anomaly Timeline: Time-series visualization combining reconstruction error and flagged anomalies.

Peer Comparison: Compare individual user threat scores against departmental peers.

Threat Escalation Visualization: Graph-based flow of suspicious event sequences.

Explainable AI Threat Explanation: Natural language explanations clarifying the rationale behind flagged threats.

Interactive Dashboard: User-friendly controls for filtering, risk thresholding, and drill-down analysis.

Installation & Setup
Prerequisites:

Python 3.8 or higher

Packages listed in requirements.txt including tensorflow, scikit-learn, pandas, streamlit, plotly, networkx

Create and activate a virtual environment:

bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies:

bash
pip install -r requirements.txt
Prepare your data:

Place your activity data (activity_data.csv), anomaly scores (lstm_anomalies.csv), and threat scores (enhanced_user_threat_scores.csv) in the project folder.

Run the dashboard:

bash
streamlit run web.py

Usage
Use sidebar filters to adjust recent threat thresholds, and number of top risky users displayed.

Navigate through tabs to access overall metrics, top risky users, and detailed user analysis with rich visualizations.

Select an individual user to see behavioral heatmaps, anomaly timelines, peer comparisons, and threat explanations.

Highlights
Designed for scalability with efficient sequence processing and caching.

Combines unsupervised ML and domain heuristics for robust risk scoring.

Integrates explainability for transparency and trust in alerts.

Customizable UI with modern design enhancements.

License
This project is open-source under the MIT License.
