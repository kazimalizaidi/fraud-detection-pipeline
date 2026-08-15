import time

import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

# ---- Page Config ----
st.set_page_config(
    page_title="Fraud & Anomaly Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# ---- Custom CSS for Professional Dashboard Styling ----
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1A1D3A 0%, #23264A 100%);
        border: 1px solid #2E3158;
        border-radius: 14px;
        padding: 20px 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetricLabel"] {
        color: #8B8FB8;
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        color: #E4E6F5;
        font-size: 32px;
        font-weight: 700;
    }
    h2, h3 {
        color: #E4E6F5 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #2E3158;
    }
    section[data-testid="stSidebar"] {
        background-color: #14172E;
        border-right: 1px solid #2E3158;
    }
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }
    hr {
        border-color: #2E3158 !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        background-color: #6C5DD3 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: background-color 0.2s ease;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background-color: #5A4CC0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---- Cached Resource Loading ----
@st.cache_resource
def load_artifacts():
    """
    Loads the trained model, scaler, and SHAP explainer from disk.
    Wrapped in try/except so a missing or corrupted artifact file shows
    a clean error message instead of a raw traceback to anyone viewing
    the dashboard (important once this is deployed publicly).
    """
    try:
        model = joblib.load("artifacts/champion_model.pkl")
        scaler = joblib.load("artifacts/scaler.pkl")
        explainer = joblib.load("artifacts/shap_explainer.pkl")
        return model, scaler, explainer
    except FileNotFoundError as e:
        st.error(f"Missing artifact file: {e}. Ensure all files are in the artifacts/ folder.")
        st.stop()


model, _scaler, explainer = load_artifacts()
# Note: scaler is loaded for completeness (e.g. future manual-entry input)
# but not currently used since the streamed test data is already scaled.

# ---- Header ----
st.markdown("""
<div style="background: linear-gradient(135deg, #1A1D3A 0%, #23264A 100%);
            border: 1px solid #2E3158; border-radius: 16px;
            padding: 28px 32px; margin-bottom: 24px;">
    <div style="display:flex; align-items:center; gap:14px;">
        <span style="font-size:36px;">🛡️</span>
        <div>
            <div style="font-size:28px; font-weight:700; color:#E4E6F5;">
                Automated Financial Fraud & Anomaly Detection Dashboard
            </div>
            <div style="font-size:14px; color:#8B8FB8; margin-top:4px;">
                Real-time transaction risk scoring powered by XGBoost, with SHAP-based explainability for audit-ready decisions.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---- Config Constants ----
RISK_THRESHOLD = 0.5
STREAM_INTERLEAVE_GAP = 150  # inserts one fraud case roughly every N legit transactions


# ---- Load Test Data for Streaming Simulation ----
@st.cache_data
def load_stream_data():
    """
    Builds a streaming order that interleaves fraud cases evenly throughout
    the stream, rather than relying on pure random shuffling — which could
    (and did, statistically ~31% of the time) push all fraud cases past
    the first several hundred transactions. This keeps the stream realistic
    (fraud is still sparse) while guaranteeing a reliable demo.
    """
    X_test = joblib.load('artifacts/X_test.pkl')
    y_test = joblib.load('artifacts/y_test.pkl')

    fraud_idx = y_test[y_test == 1].index.tolist()
    legit_idx = y_test[y_test == 0].index.tolist()

    rng = np.random.default_rng(seed=42)
    rng.shuffle(fraud_idx)
    rng.shuffle(legit_idx)

    interleaved_idx = []
    fraud_pointer = 0

    for i, idx in enumerate(legit_idx):
        interleaved_idx.append(idx)
        if (i + 1) % STREAM_INTERLEAVE_GAP == 0 and fraud_pointer < len(fraud_idx):
            interleaved_idx.append(fraud_idx[fraud_pointer])
            fraud_pointer += 1

    interleaved_idx.extend(fraud_idx[fraud_pointer:])

    return X_test.loc[interleaved_idx], y_test.loc[interleaved_idx]


def badge(val):
    """Renders a colored pill badge for a binary fraud/legit prediction."""
    if val == 1:
        return ('<span style="background-color:#3A1A2E;color:#FF5C7A;padding:4px 12px;'
                'border-radius:20px;font-size:12px;font-weight:600;">● FRAUD</span>')
    return ('<span style="background-color:#1A3A2E;color:#4ADE80;padding:4px 12px;'
            'border-radius:20px;font-size:12px;font-weight:600;">● Legit</span>')


X_stream, y_stream = load_stream_data()

# ---- Session State ----
if "stream_position" not in st.session_state:
    st.session_state.stream_position = 0
if "transaction_log" not in st.session_state:
    st.session_state.transaction_log = pd.DataFrame(
        columns=["index", "fraud_probability", "prediction", "actual", "amount"]
    )
if "streaming" not in st.session_state:
    st.session_state.streaming = False

# ---- Sidebar Controls ----
st.sidebar.header("⚙️ Stream Controls")
speed = st.sidebar.slider("Transactions per batch", 1, 20, 5)
interval = st.sidebar.slider("Refresh interval (seconds)", 1, 5, 2)

col1, col2 = st.sidebar.columns(2)
start_btn = col1.button("▶ Start Stream")
stop_btn = col2.button("⏸ Stop Stream")

if start_btn:
    st.session_state.streaming = True
if stop_btn:
    st.session_state.streaming = False

reset_btn = st.sidebar.button("🔄 Reset Stream")
if reset_btn:
    st.session_state.stream_position = 0
    st.session_state.transaction_log = st.session_state.transaction_log.iloc[0:0]
    st.session_state.streaming = False


def process_batch(batch_size):
    """
    Pulls the next `batch_size` rows from the test set, scores them
    with the champion model, and appends results to the running log.
    """
    start = st.session_state.stream_position
    end = min(start + batch_size, len(X_stream))

    if start >= len(X_stream):
        st.session_state.streaming = False
        return None

    batch_X = X_stream.iloc[start:end]
    batch_y = y_stream.iloc[start:end]

    probs = model.predict_proba(batch_X)[:, 1]
    preds = model.predict(batch_X)

    new_rows = pd.DataFrame(
        {
            "index": batch_X.index,
            "fraud_probability": probs,
            "prediction": preds,
            "actual": batch_y.values,
            "amount": batch_X["scaled_amount"].values,
        }
    )

    st.session_state.transaction_log = pd.concat(
        [st.session_state.transaction_log, new_rows], ignore_index=True
    )
    st.session_state.stream_position = end
    return new_rows


# ---- Trigger one batch per script run when streaming is active ----
if st.session_state.streaming:
    process_batch(speed)

# ---- Top-Level Live Metrics ----
log = st.session_state.transaction_log

m1, m2, m3, m4 = st.columns(4)
m1.metric("Transactions Processed", len(log))
m2.metric("Flagged as Fraud", int(log["prediction"].sum()) if len(log) else 0)
m3.metric(
    "Live Fraud Rate",
    f"{(log['prediction'].mean() * 100):.2f}%" if len(log) else "0.00%",
)
m4.metric(
    "Detection Recall (so far)",
    f"{(log[log['actual'] == 1]['prediction'].mean() * 100):.1f}%"
    if len(log[log["actual"] == 1])
    else "N/A",
)

st.divider()

# ---- Live Transaction Feed ----
st.subheader("📡 Live Transaction Feed")

if len(log) > 0:
    display_log = log.tail(15).iloc[::-1].copy()
    display_log["fraud_probability"] = display_log["fraud_probability"].apply(
        lambda x: f"{x:.2%}"
    )
    display_log["prediction"] = display_log["prediction"].apply(badge)
    display_log["actual"] = display_log["actual"].apply(badge)

    table_html = display_log[["index", "fraud_probability", "prediction", "actual"]].to_html(
        escape=False, index=False
    )
    st.markdown(
        f'''
        <div style="overflow-x:auto;overflow-y:auto;max-height:480px;
                    border-radius:12px;border:1px solid #2E3158;">
            <style>
                .styled-table {{ width:100%; border-collapse:collapse; table-layout:fixed; }}
                .styled-table thead th {{
                    position:sticky; top:0; background-color:#1A1D3A;
                    color:#8B8FB8; text-align:left; padding:10px 14px;
                    border-bottom:1px solid #2E3158; z-index:1;
                }}
                .styled-table tbody td {{
                    padding:10px 14px; border-bottom:1px solid #23264A;
                    color:#E4E6F5;
                }}
                .styled-table th:nth-child(1), .styled-table td:nth-child(1) {{ width:20%; }}
                .styled-table th:nth-child(2), .styled-table td:nth-child(2) {{ width:20%; }}
                .styled-table th:nth-child(3), .styled-table td:nth-child(3) {{ width:20%; }}
                .styled-table th:nth-child(4), .styled-table td:nth-child(4) {{ width:20%; }}
            </style>
            {table_html.replace('<table', '<table class="styled-table"')}
        </div>
        ''',
        unsafe_allow_html=True,
    )
else:
    st.info("Press ▶ Start Stream in the sidebar to begin simulating live transactions.")

# ---- Rolling Fraud Rate Over Time ----
st.subheader("📈 Rolling Fraud Detection Rate")

if len(log) >= 5:
    log_indexed = log.reset_index(drop=True)
    log_indexed["transaction_number"] = log_indexed.index + 1
    window = min(20, max(5, len(log_indexed) // 5))
    log_indexed["rolling_fraud_rate"] = (
        log_indexed["prediction"].rolling(window=window, min_periods=1).mean() * 100
    )

    fig = px.line(
        log_indexed,
        x="transaction_number",
        y="rolling_fraud_rate",
        title=f"Rolling Fraud Rate (window={window} transactions)",
        labels={"rolling_fraud_rate": "Fraud Rate (%)", "transaction_number": "Transaction #"},
    )
    fig.update_traces(line_color="#FF4B4B")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("Chart will appear once at least 5 transactions have streamed.")

# ---- SHAP Explanation for Most Recent High-Risk Transaction ----
st.subheader("🔍 Why Was This Flagged? (SHAP Explanation)")

if len(log) > 0:
    recent_fraud = log[log["fraud_probability"] >= RISK_THRESHOLD]

    if len(recent_fraud) > 0:
        latest_flagged = recent_fraud.iloc[-1]
        txn_idx = latest_flagged["index"]

        st.warning(
            f"Transaction `{txn_idx}` flagged with "
            f"{latest_flagged['fraud_probability']:.2%} fraud probability."
        )

        txn_data = X_stream.loc[[txn_idx]]
        shap_vals = explainer.shap_values(txn_data)

        explanation = shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=txn_data.iloc[0].values,
            feature_names=txn_data.columns.tolist(),
        )

        fig = plt.figure(figsize=(10, 5))
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig, bbox_inches="tight")
        plt.close(fig)

        contrib = pd.DataFrame(
            {
                "feature": txn_data.columns,
                "shap_value": shap_vals[0],
            }
        ).sort_values("shap_value", key=lambda s: s.abs(), ascending=False).head(5)

        st.markdown("**Top contributing factors:**")
        for _, row in contrib.iterrows():
            direction = "increased" if row["shap_value"] > 0 else "decreased"
            st.markdown(
                f"- `{row['feature']}` {direction} fraud risk "
                f"(SHAP: {row['shap_value']:.3f})"
            )
    else:
        st.caption("No high-risk transactions detected yet in this stream.")
else:
    st.caption("Start the stream to see live risk explanations.")

# ---- Auto-refresh while streaming ----
if st.session_state.streaming:
    time.sleep(interval)
    st.rerun()