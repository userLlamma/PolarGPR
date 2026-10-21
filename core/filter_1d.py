"""
1D Signal Processing & Physics-Prior Feature Extraction for GPR B-scan.

Combines classical geophysical signal processing:
- Background mean-trace subtraction (direct-wave & horizontal ringing suppression)
- SVD low-rank filtering (antenna coupling reduction)
- 98th-percentile dynamic range gain normalization (AGC / amplitude scaling)
- Hilbert transform along time/depth axis to construct analytic signal: z(t) = s(t) + i*H[s(t)]
"""

import numpy as np
from scipy.signal import hilbert


class GPRSignalProcessor1D:
    """
    1D Geophysical Signal Processing Pipeline for GPR profiles.
    """

    def __init__(self, use_svd_clutter_filter: bool = False, pad_multiple: int = 32):
        self.use_svd_clutter_filter = use_svd_clutter_filter
        self.pad_multiple = pad_multiple

    def remove_background(self, data: np.ndarray) -> np.ndarray:
        """
        Subtract average trace across all profiles to eliminate direct coupling and constant horizontal banding.
        """
        mean_trace = np.mean(data, axis=1, keepdims=True)
        return data - mean_trace

    def svd_clutter_filter(self, data: np.ndarray, num_components_to_remove: int = 1) -> np.ndarray:
        """
        Optional SVD clutter filter: zeros out first dominant singular components representing horizontal direct waves.
        """
        try:
            U, S, Vh = np.linalg.svd(data, full_matrices=False)
            S[:num_components_to_remove] = 0.0
            return U @ np.diag(S) @ Vh
        except Exception:
            return data

    def robust_scale(self, data: np.ndarray, percentile: float = 98.0) -> np.ndarray:
        """
        Robust percentile scaling clipping to [-1, 1], preserving weak wavefronts.
        """
        vm = np.percentile(np.abs(data), percentile)
        if vm < 1e-6:
            vm = 1.0
        return np.clip(data / vm, -1.0, 1.0)

    def prepare_analytic_signal(self, data: np.ndarray):
        """
        Pads data to multiple of pad_multiple and computes Hilbert analytic signal along depth axis (axis=0).
        
        Returns:
            clean_normalized: (H, W) float32
            z_real_padded: (1, 1, H_pad, W_pad) float32
            z_imag_padded: (1, 1, H_pad, W_pad) float32
            pad_info: tuple of (orig_H, orig_W, pad_h, pad_w)
        """
        # 1. Clean NaN / Inf
        cleaned = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Background direct wave elimination
        s1_data = self.remove_background(cleaned)

        # 3. Optional SVD filtering
        if self.use_svd_clutter_filter:
            s1_data = self.svd_clutter_filter(s1_data)

        # 4. Robust energy scaling
        norm_data = self.robust_scale(s1_data)

        orig_H, orig_W = norm_data.shape

        # 5. Spatial boundary padding (multiple of 32 for U-Net / Dilated architectures)
        pad_h = (self.pad_multiple - orig_H % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - orig_W % self.pad_multiple) % self.pad_multiple
        padded_data = np.pad(norm_data, ((0, pad_h), (0, pad_w)), mode="edge")

        # 6. Hilbert transform along depth / fast-time axis (axis=0)
        analytic = hilbert(padded_data, axis=0)

        z_real = analytic.real.astype(np.float32)[None, None, :, :]
        z_imag = analytic.imag.astype(np.float32)[None, None, :, :]

        pad_info = (orig_H, orig_W, pad_h, pad_w)
        return norm_data, z_real, z_imag, pad_info
