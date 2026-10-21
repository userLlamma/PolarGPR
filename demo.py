#!/usr/bin/env python3
"""
Quickstart Demo for GPR Reflection Enhancer.
Run 1D Physics Preprocessing + 2D Deep Spatial Reflection Extraction in < 1 second.

Usage:
    python demo.py --input data/sample/sample_bscan.npy
    python demo.py --input data/sample/sample_bscan.sgy
"""

import argparse
import os
import sys

# Configure clean matplotlib cache directory
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import time
import numpy as np
import matplotlib.pyplot as plt

# Ensure local core package can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.detector_2d import GPRReflectionDetector2D


def load_bscan_file(file_path: str) -> np.ndarray:
    """
    Loads GPR B-scan from .npy, .sgy/.segy, or common image formats.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".npy":
        data = np.load(file_path)
    elif ext in [".sgy", ".segy"]:
        try:
            import segyio
        except ImportError:
            raise ImportError("segyio is required to read .sgy files. Run `pip install segyio`.")

        f = None
        for endian in ["little", "big"]:
            try:
                f = segyio.open(file_path, ignore_geometry=True, endian=endian)
                break
            except Exception:
                continue

        if f is None:
            raise ValueError(f"Failed to open {file_path} with both little and big endian.")

        try:
            data = segyio.tools.collect(f.trace[:]).T  # shape: (Samples/Time, Traces)
        finally:
            f.close()
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
        from PIL import Image
        img = Image.open(file_path).convert("L")
        data = np.array(img, dtype=np.float32)
        # normalize to [-1, 1]
        data = (data / 127.5) - 1.0
    else:
        raise ValueError(f"Unsupported input format: {ext}. Supported: .npy, .sgy, .segy, images.")

    return np.nan_to_num(data.astype(np.float32))


def run_demo(input_path: str, output_dir: str = "results", model_path: str = None, use_svd: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_path))[0]

    print(f"📡 Loading GPR data from: {input_path}")
    raw_data = load_bscan_file(input_path)
    H, W = raw_data.shape
    print(f"📊 B-scan Shape: {H} samples (depth) x {W} traces (distance)")

    print("⚡ Initializing 2D Reflection Detector (ONNX)...")
    detector = GPRReflectionDetector2D(model_path=model_path)
    if use_svd:
        detector.processor.use_svd_clutter_filter = True

    print("🚀 Running Inference...")
    t0 = time.perf_counter()
    res = detector.predict(raw_data)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"✨ Inference completed in {elapsed_ms:.1f} ms!")

    # Save quantitative matrix
    npy_out = os.path.join(output_dir, f"{basename}_skeleton.npy")
    np.save(npy_out, res["signed_skeleton"])
    print(f"💾 Reflection skeleton saved to: {npy_out}")

    # Generate 3-panel comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    # 1. Raw Input
    v_raw = np.percentile(np.abs(raw_data), 98) or 1.0
    im0 = axes[0].imshow(raw_data, cmap="gray", aspect="auto", vmin=-v_raw, vmax=v_raw)
    axes[0].set_title("Input (Raw GPR B-scan)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Two-Way Travel Time / Depth (Samples)")
    axes[0].set_xlabel("Trace / Survey Distance")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. 1D Physics Wavefront
    im1 = axes[1].imshow(res["norm_input"], cmap="seismic", aspect="auto", vmin=-1.0, vmax=1.0)
    axes[1].set_title("S1 (1D Physics Wavefront)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Trace / Survey Distance")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. 2D Signed Reflection Skeleton
    im2 = axes[2].imshow(res["signed_skeleton"], cmap="seismic", aspect="auto", vmin=-0.8, vmax=0.8)
    axes[2].set_title("S2 (2D Signed Prob Skeleton)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Trace / Survey Distance")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"GPR Reflection Enhancement Pipeline [{basename}] ({elapsed_ms:.1f} ms)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    fig_out = os.path.join(output_dir, f"{basename}_comparison.png")
    plt.savefig(fig_out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️ Visualization figure saved to: {fig_out}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPR Reflection Layer Enhancer Quickstart Demo")
    parser.add_argument(
        "--input",
        type=str,
        default="data/sample/sample_bscan.png",
        help="Path to input B-scan file (.png, .npy, or .sgy)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save output figures and matrices",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional custom path to .onnx model weights",
    )
    parser.add_argument(
        "--svd",
        action="store_true",
        help="Enable SVD first-component clutter rejection for severe direct wave",
    )
    args = parser.parse_args()

    run_demo(
        input_path=args.input,
        output_dir=args.output_dir,
        model_path=args.model,
        use_svd=args.svd,
    )
