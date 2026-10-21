"""
Interactive Streamlit Web Application for GPR Reflection Enhancer.
Run with: streamlit run app.py
"""

import os
import sys
import tempfile
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.detector_2d import GPRReflectionDetector2D
from demo import load_bscan_file

# Streamlit Page Setup
st.set_page_config(
    page_title="GPR Reflection Enhancer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📡 GPR Low-SNR Reflection Enhancer")
st.markdown(
    """
    **Physics-guided instantaneous phase manifold & 2D spatial deep feature extraction** 
    for low-SNR legacy Ground Penetrating Radar (GPR) B-scans.
    """
)

# Sidebar controls
st.sidebar.header("⚙️ Configuration")
use_svd = st.sidebar.checkbox("Enable SVD Clutter Filter", value=False, help="Zeros out primary singular component to eliminate horizontal direct waves.")
colormap = st.sidebar.selectbox("Color Palette", ["seismic", "coolwarm", "magma", "viridis"], index=0)

@st.cache_resource
def get_detector():
    return GPRReflectionDetector2D()

detector = get_detector()
if use_svd:
    detector.processor.use_svd_clutter_filter = True
else:
    detector.processor.use_svd_clutter_filter = False

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload GPR B-scan (.sgy, .npy, or image)",
    type=["sgy", "segy", "npy", "png", "jpg"],
)

# Default to bundled sample if no upload
sample_choice = st.sidebar.selectbox(
    "Or select a sample:",
    ["Sample Profile (.png)", "Sample Profile (.npy)", "Core Patch 512x600 (.npy)"],
)

data = None
filename = "sample"

if uploaded_file is not None:
    filename = uploaded_file.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        data = load_bscan_file(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if sample_choice == "Sample Profile (.png)":
        s_path = os.path.join(base_dir, "data", "sample", "sample_bscan.png")
    elif sample_choice == "Sample Profile (.npy)":
        s_path = os.path.join(base_dir, "data", "sample", "sample_bscan.npy")
    else:
        s_path = os.path.join(base_dir, "data", "sample", "sample_patch_512x600.npy")
    filename = os.path.basename(s_path)
    data = load_bscan_file(s_path)

if data is not None:
    st.info(f"Loaded: `{filename}` — Dimensions: `{data.shape[0]}` samples (depth) × `{data.shape[1]}` traces (distance)")

    t0 = time.perf_counter()
    res = detector.predict(data)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    st.success(f"⚡ Processing finished in **{infer_ms:.1f} ms**")

    # Layout results
    col1, col2, col3 = st.columns(3)

    v_raw = np.percentile(np.abs(data), 98) or 1.0

    with col1:
        st.subheader("1. Raw B-scan Input")
        fig1, ax1 = plt.subplots(figsize=(5, 6))
        im1 = ax1.imshow(data, cmap="gray", aspect="auto", vmin=-v_raw, vmax=v_raw)
        ax1.axis("off")
        fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.subheader("2. S1: 1D Wavefront")
        fig2, ax2 = plt.subplots(figsize=(5, 6))
        im2 = ax2.imshow(res["norm_input"], cmap="seismic", aspect="auto", vmin=-1.0, vmax=1.0)
        ax2.axis("off")
        fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        st.pyplot(fig2)
        plt.close(fig2)

    with col3:
        st.subheader("3. S2: Reflection Skeleton")
        fig3, ax3 = plt.subplots(figsize=(5, 6))
        im3 = ax3.imshow(res["signed_skeleton"], cmap=colormap, aspect="auto", vmin=-0.8, vmax=0.8)
        ax3.axis("off")
        fig3.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        st.pyplot(fig3)
        plt.close(fig3)

    st.divider()
    st.markdown("### 🔍 Polarity Decoupled Analysis")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.caption("Positive Phase Probability (Void Ceiling / Soil Layers)")
        fig_p, ax_p = plt.subplots(figsize=(6, 4))
        ax_p.imshow(res["pos_prob"], cmap="magma", aspect="auto", vmin=0, vmax=1)
        ax_p.axis("off")
        st.pyplot(fig_p)
        plt.close(fig_p)

    with pcol2:
        st.caption("Negative Phase Probability (Metallic Utilities / Water Saturated Voids)")
        fig_n, ax_n = plt.subplots(figsize=(6, 4))
        ax_n.imshow(res["neg_prob"], cmap="magma", aspect="auto", vmin=0, vmax=1)
        ax_n.axis("off")
        st.pyplot(fig_n)
        plt.close(fig_n)
