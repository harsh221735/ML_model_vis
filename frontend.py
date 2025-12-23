import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_regression

BACKEND_URL = "http://127.0.0.1:5001"

st.set_page_config(page_title="Gradient Descent Visualizer", layout="centered")
st.title("📉 Gradient Descent Visualizer")

# -------------------------------
# RESET BUTTON
# -------------------------------
if st.button("🔄 Reset"):
    st.session_state.clear()
    st.experimental_rerun()

# -------------------------------
# DATASET SECTION
# -------------------------------
st.header("1️⃣ Dataset")

dataset_mode = st.radio("Dataset source", ["Default", "Upload CSV"])

uploaded_file = None
if dataset_mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV (2 columns only)", type=["csv"])

if st.button("Load Dataset"):
    if dataset_mode == "Default":
        requests.post(f"{BACKEND_URL}/dataset", data={"mode": "default"})

        X, y = make_regression(
            n_samples=100,
            n_features=1,
            noise=20,
            bias=50,
            random_state=42
        )

        st.session_state.df = pd.DataFrame({
            "input": X.flatten(),
            "output": y
        })

    else:
        if uploaded_file is None:
            st.error("Upload a CSV file")
            st.stop()

        requests.post(
            f"{BACKEND_URL}/dataset",
            data={"mode": "custom"},
            files={"file": uploaded_file}
        )

        st.session_state.df = pd.read_csv(uploaded_file)

    st.success("Dataset loaded")
    st.dataframe(st.session_state.df.head())

# -------------------------------
# TRAINING SECTION
# -------------------------------
if "df" in st.session_state:

    st.header("2️⃣ Training Configuration")

    mode = st.selectbox(
        "Gradient Descent Type",
        ["batch", "mini-batch", "stochastic"]
    )

    lr = st.number_input("Learning Rate", value=0.01, step=0.01)
    epochs = st.number_input("Epochs", value=20, step=5)

    batch_size = None
    if mode == "mini-batch":
        batch_size = st.number_input("Batch Size", value=5, step=1)

    if st.button("Train Model"):
        payload = {
            "mode": mode,
            "lr": lr,
            "epochs": epochs
        }

        if batch_size:
            payload["batch_size"] = batch_size

        res = requests.post(f"{BACKEND_URL}/train", json=payload)

        if res.status_code == 200:
            st.session_state.slopes = res.json()["slopes_array"]
            st.session_state.epochs = epochs
            st.session_state.mode = mode
            st.success("Training complete")

# -------------------------------
# VISUALIZATION SECTION
# -------------------------------
if "slopes" in st.session_state:

    st.header("3️⃣ Gradient Descent Animation")

    df = st.session_state.df
    slopes = st.session_state.slopes

    X = df.iloc[:, 0].values
    Y = df.iloc[:, 1].values

    # 🔥 SPEED CONTROL
    speed = st.slider(
        "Animation Speed (seconds per step)",
        min_value=0.01,
        max_value=0.5,
        value=0.05,
        step=0.01
    )

    plot_area = st.empty()
    status = st.empty()

    iterations_per_epoch = len(slopes) // st.session_state.epochs

    for idx, B in enumerate(slopes):
        epoch = (idx // iterations_per_epoch) + 1
        iteration = idx + 1

        fig, ax = plt.subplots()
        ax.scatter(X, Y, label="Data")

        b0, b1 = B
        y_pred = b1 * X + b0
        ax.plot(X, y_pred, color="red", linewidth=2, label="Model")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()

        plot_area.pyplot(fig)
        status.markdown(
            f"**Epoch:** {epoch} &nbsp;&nbsp; | &nbsp;&nbsp; **Iteration:** {iteration}"
        )

        time.sleep(speed)   # ⬅⬅⬅ SPEED CONTROL IS HERE
