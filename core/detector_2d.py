"""
2D Spatial Feature Reasoning & Reflection Skeleton Probability Detector.

Powered by a lightweight ONNX runtime engine (~3 MB) trained with:
- Phase manifold representation of electromagnetic wavefields
- Two-stage dilated convolution with global context bottleneck
- Signed polarity preservation (positive vs. negative reflection interfaces)
"""

import os
import numpy as np
import onnxruntime as ort
from .filter_1d import GPRSignalProcessor1D


class GPRReflectionDetector2D:
    """
    ONNX Runtime Inference Engine for 2D GPR Reflection Layer Probability Extraction.
    """

    def __init__(self, model_path: str = None, providers=None):
        if model_path is None:
            # Default to bundled weight
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "weights", "gpr_core_v110.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found at: {model_path}")

        if providers is None:
            available = ort.get_available_providers()
            # Prefer CUDA/CoreML if present, fallback to CPU
            preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in preferred if p in available] or ["CPUExecutionProvider"]

        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.processor = GPRSignalProcessor1D()

    def predict(self, raw_data: np.ndarray):
        """
        Runs full 1D + 2D pipeline on a 2D numpy array (Time/Depth x Distance/Traces).

        Args:
            raw_data: 2D numpy array [Height, Width] of raw GPR B-scan.

        Returns:
            dict containing:
                - 'norm_input': 1D-preprocessed & normalized B-scan
                - 'pos_prob': Probability map of positive phase reflections [0, 1]
                - 'neg_prob': Probability map of negative phase reflections [0, 1]
                - 'signed_skeleton': Signed probability map [-1, 1] (pos - neg)
        """
        # 1. 1D physics preprocessing & Hilbert analytic signals
        norm_input, z_real, z_imag, pad_info = self.processor.prepare_analytic_signal(raw_data)
        orig_H, orig_W, _, _ = pad_info

        # 2. ONNX forward pass
        inputs = {
            "input_real": z_real,
            "input_imag": z_imag
        }
        outputs = self.session.run(None, inputs)
        padded_logits = outputs[0]  # shape: [1, 2, H_pad, W_pad]

        # 3. Crop back to original dimensions
        real_logits = padded_logits[0, :, :orig_H, :orig_W]

        # 4. Numerically stable Sigmoid
        probs = 1.0 / (1.0 + np.exp(-real_logits))
        pos_prob = probs[0, :, :]
        neg_prob = probs[1, :, :]
        signed_skeleton = pos_prob - neg_prob

        return {
            "norm_input": norm_input,
            "pos_prob": pos_prob,
            "neg_prob": neg_prob,
            "signed_skeleton": signed_skeleton,
        }
