import streamlit as st
import pandas as pd
import joblib
import json
import os

MODEL_PATH = "models/cpu_model.pkl"
METRICS_PATH = "metrics/metrics.json"
PLOTS_DIR = "plots"

st.set_page_config(page_title="CPU Usage Predictor", layout="wide")

def safe_fmt(val):
    try:
        return f"{float(val):.4f}"
    except:
        return "N/A"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
    return model

def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

st.title("⚙️ CPU Usage Prediction Dashboard")
st.markdown("---")

st.subheader("📊 Model Performance Metrics")
metrics = load_metrics()

if metrics:
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", safe_fmt(metrics.get("mae")))
    col2.metric("RMSE", safe_fmt(metrics.get("rmse")))
    col3.metric("R² Score", safe_fmt(metrics.get("r2")))
else:
    st.warning("⚠️ metrics.json not found or empty. Train the model again.")

st.markdown("---")

st.subheader("🧮 Make a Prediction")
model = load_model()

colA, colB = st.columns(2)

with colA:
    cpu_request = st.number_input("CPU Request", min_value=0.0, value=0.5)
    mem_request = st.number_input("Memory Request", min_value=0.0, value=0.5)
    cpu_limit = st.number_input("CPU Limit", min_value=0.0, value=1.0)

with colB:
    mem_limit = st.number_input("Memory Limit", min_value=0.0, value=1.0)
    runtime_minutes = st.number_input("Runtime (Minutes)", min_value=0, value=50)
    controller_kind = st.selectbox("Controller Kind", ["Deployment", "StatefulSet", "CronJob"])

if st.button("Predict CPU Usage"):
    input_df = pd.DataFrame([{
        "cpu_request": cpu_request,
        "mem_request": mem_request,
        "cpu_limit": cpu_limit,
        "mem_limit": mem_limit,
        "runtime_minutes": runtime_minutes,
        "controller_kind": controller_kind
    }])

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🔮 Predicted CPU Usage: {prediction:.4f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")

st.subheader("📉 Model Visualizations")

plot_files = {
    "Predicted vs Actual": "pred_vs_actual.png",
    "Residual Distribution": "residuals.png",
    "Feature Importance": "feature_importance.png"
}

cols = st.columns(3)

for (title, file), col in zip(plot_files.items(), cols):
    plot_path = os.path.join(PLOTS_DIR, file)

    with col:
        st.markdown(f"### {title}")

        if os.path.exists(plot_path):
            st.image(plot_path, width=300)
        else:
            st.warning(f"Plot not found: {file}")
