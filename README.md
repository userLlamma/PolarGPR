<div align="center">

# 📡 GPR Low-SNR Reflection Enhancer
### Physics-Guided Phase Manifold & 2D Spatial Deep Feature Extraction for Legacy Ground Penetrating Radar Profiles

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Model Size: ~3MB](https://img.shields.io/badge/Model_Size-~3_MB-green.svg)]()
[![Inference: <400ms (CPU)](https://img.shields.io/badge/CPU_Inference-%3C400ms-brightgreen.svg)]()
[![ONNX Runtime](https://img.shields.io/badge/Runtime-ONNX_1.15%2B-orange.svg)](https://onnxruntime.ai/)
[![Reddit Discussion](https://img.shields.io/badge/Reddit-r%2Fgeophysics-FF4500?logo=reddit&logoColor=white)](https://www.reddit.com/r/geophysics/comments/1r3f83m/showcase_automating_100mhz_gpr_signal_recovery_nn/)

[English](README.md) | [中文说明](#-中文技术概要与项目故事)

</div>

---

## 📖 Background & Motivation

> *"An exploratory attempt by a signal processing developer to reconstruct low-SNR radar reflections, open-sourced for community reference and feedback."*

### 1. Project Origins & The Intuition Behind the Simulation
I am a developer with a background in audio, array signal processing, and neural network application engineering—**not a trained geophysicist**. A few months ago, a contact shared several sets of legacy, low-SNR Ground Penetrating Radar (GPR) profiles to explore potential collaboration. Intrigued by the challenge, I approached GPR B-scans from a wavefield signal perspective: combining classical 1D analytic signal physics (Hilbert transform, dynamic range AGC, common-mode removal) with a 2D dilated convolutional architecture to recover weak, continuous reflection horizons.

During the exploratory phase, I initially tried using standard FDTD numerical tools like **gprMax**. However, full Maxwell-equation solvers proved far too computationally expensive—generating even a modest training set would take days or weeks. 

While visually inspecting B-scan radargrams, an intuition emerged: the macroscopic hyperbolic events, diffraction tails, and interference patterns strongly resembled **ripples propagating and interfering on a water surface**. Guided by this acoustic analogy, I pivoted from brute-force grid differencing to a simplified, vectorized **Huygens principle propagator** with analytical travel-time superposition and wavelet convolution. This reduced synthesis time from hours to milliseconds, enabling real-time on-the-fly streaming directly into PyTorch.

While the preliminary results successfully recovered continuous reflection boundaries, the exploratory collaboration did not continue. Rather than shelving the code, I am open-sourcing the entire pipeline to share these observations with the community.

### 2. Community Feedback on Reddit
I previously shared some early processed profiles on Reddit ([r/geophysics showcase thread](https://www.reddit.com/r/geophysics/comments/1r3f83m/showcase_automating_100mhz_gpr_signal_recovery_nn/)). The post sparked interesting discussions with domain practitioners. Notably, a community contact tested the inference pipeline against a challenging, coarsely sampled field dataset (`100 MHz`, `dt = 2 ns`, `dx = 1 m`).

The test highlighted several subtle field wavefield characteristics (such as discontinuous horizontal ringing bands, trace-to-trace amplitude variations, and trailing-edge frequency shifts) that pure data-driven processing without in-situ survey context cannot easily explain.

### 3. Why Open Source?
Because I lack formal geological and field radar survey training, I cannot authoritatively interpret whether certain subtle wavefield features correspond to genuine stratigraphic interfaces, local soil moisture variations, or antenna coupling artifacts. 

I am unsure whether this preliminary pipeline holds lasting practical value for real-world geophysics workflows. I am open-sourcing the complete code, lightweight ONNX weights (~3 MB), and interactive tools in hopes that:
- It may serve as a useful reference or baseline for researchers exploring low-SNR GPR enhancement.
- Practicing geophysicists and radar engineers can evaluate the results critically and provide constructive feedback.
- It encourages open dialogue on bridging traditional signal processing with modern learning-based architectures in near-surface geophysics.

---

## 🎯 Problem Statement

Legacy and deep GPR B-scan data typically suffer from three major degradation factors:

1. **Antenna Direct Coupling & Horizontal Clutter**: Dominant horizontal stripes in shallow time-slices saturate dynamic range and obscure top-layer interfaces.
2. **Exponential Path Attenuation**: Reflections from deep soil strata, utilities, and cavities drop below the background speckle noise level.
3. **Ambiguity in Boundary Polarity**: Conventional intensity thresholding loses critical phase orientation (e.g., distinguishing positive reflection at void ceilings from negative reflection at metallic pipes or water interfaces).

```
[Raw B-scan Profile]
        │
        ▼ (1D Physics Prior)
[Background Removal + SVD Clutter Filter + 98% Dynamic AGC + Hilbert Analytic Signal]
        │
        ▼ (2D Deep Feature Extraction)
[Phase Manifold & Dilated Spatial Receptive Field Network]
        │
        ▼
[Continuous Signed Probability Reflection Skeletons: P(+) - P(-)]
```

---

## 🖼️ Visual Comparison

Below is the end-to-end transformation on a real-world B-scan field profile:

![GPR Reflection Enhancement Pipeline](assets/comparison_triptych.png)

* **Input (Raw GPR B-scan)**: Target horizons and hyperbolic events are heavily masked by horizontal striping and high-frequency scattering noise.
* **S1 (1D Physics Wavefront)**: Baseline DC drift, horizontal common-mode clutter, and dynamic range discrepancies are normalized while strictly preserving wavelet phase polarities.
* **S2 (2D Signed Prob Skeleton)**: Non-coherent noise is completely suppressed; subtle structural interfaces and diffraction apexes glow as continuous, high-confidence skeletons with crystal-clear polarity ($P_{pos} - P_{neg}$).

---

## 🔬 Core Algorithm & Architecture

The architecture bridges the gap between **analytical wavefield physics** and **spatial deep learning**:

### 1. 1D Physics-Informed Preprocessing (`core/filter_1d.py`)
* **Background Trace Subtraction**: Eliminates direct coupling waves via cross-profile mean-trace deduction: $\tilde{s}(t, x) = s(t, x) - \frac{1}{N} \sum_{i=1}^N s(t, x_i)$.
* **Optional SVD Rank-1 Clutter Filter**: Separates stationary horizontal noise from dipping stratigraphic horizons.
* **Robust Percentile AGC**: 98th-percentile envelope clipping to avoid saturation from near-surface anomalies.
* **Hilbert Analytic Transform**: Derives complex analytic wavefields $z(t) = s(t) + i\mathcal{H}[s(t)]$ to extract instantaneous amplitude $A(t) = |z(t)|$ and instantaneous phase $\phi(t) = \text{atan2}(z_i, z_r)$.

### 2. 2D Spatial Dilated Convolution Engine (`core/detector_2d.py`)
* **Dual-Channel Polarity Separation**: The network separately predicts probabilities for:
  * **Positive Phase ($+$)**: Transitions from high to low dielectric constant (e.g., void ceilings, dry bedrock boundaries).
  * **Negative Phase ($-$)**: Transitions from low to high dielectric constant (e.g., metal utilities, water-filled anomalies, wet clay beds).
* **Global Context Bottleneck**: Employs dilated convolutions with varying dilation rates ($d=1, 2, 4$) to capture wide hyperbolic tails without downsampling resolution.
* **Signed Skeleton Synthesis**: $P_{signed} = P_{pos} - P_{neg} \in [-1, 1]$.

### 3. Engineering Philosophy: Agile Targeted Adaptation over Blind Generality
* **From gprMax to Huygens Ripples**: Conventional full-wave solvers (like gprMax) are indispensable for rigorous forward modeling, but their computational overhead creates a bottleneck for training data generation. Adopting an analytical, vectorized Huygens formulation trades off fine-grained scattering physics for extreme synthesis speed (~milliseconds per slice).
* **On-the-Fly Physics Streaming**: Training samples are synthesized dynamically in memory during data loading. There is zero disk I/O bottleneck, and the network never encounters the exact same profile twice.
* **Rapid Convergence on Consumer Hardware**: Tested on a standard **Mac mini (M2, 24GB)**, the entire two-stage network **converges within just a few hours**.
* **Parameter Alignment**: The synthetic propagator should be configured to approximate target field parameters (notably sampling interval `dt`, center frequency, and trace spacing `dx`). While the model has reasonable tolerance to variations, a severe scale mismatch (e.g., an order-of-magnitude difference in wavelet period or hyperbolic curvature) can impair reconstruction. The true practical value lies in **fast, customized turnaround and lightweight deployment**, rather than an all-in-one universal model.

---

## ⚡ Performance Benchmarks

Engineered for resource-constrained field hardware, edge devices, and modest cloud VPS instances:

| Metric | Measured Value | Remarks |
| :--- | :--- | :--- |
| **Model Footprint** | **~3.0 MB** (`.onnx`) | Extremely lightweight |
| **Mac (Apple Silicon CPU)** | **~340 ms** | 512 × 400 B-scan slice |
| **1-Core Cloud VPS (ARM CPU)** | **~380 ms** | 100% CPU inference without GPU |
| **Peak Memory Consumption** | **< 60 MB** | Zero GPU / VRAM requirements |

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install the lightweight dependencies:

```bash
git clone https://github.com/your-username/gpr-reflection-enhancer.git
cd gpr-reflection-enhancer
pip install -r requirements.txt
```

### 2. Run Single-File Demo
Process an included high-contrast profile or core patch:

```bash
# Fast run on high-contrast sample profile (image or npy)
python demo.py --input data/sample/sample_bscan.png --output_dir results

# Or run on the 512x600 core patch (~300ms)
python demo.py --input data/sample/sample_patch_512x600.npy --output_dir results
```
Outputs:
* `results/<name>_comparison.png`: 3-panel comparative diagnosis plot.
* `results/<name>_skeleton.npy`: Full-resolution signed probability matrix for downstream CAD / GIS export.

### 3. Launch Interactive Streamlit Web UI
Experience the browser-based visualization tool (identical to the VPS demo tested by industry peers):

```bash
streamlit run app.py
```
Open `http://localhost:8501` to drag-and-drop `.png`, `.npy`, or `.sgy` profiles, toggle SVD filters, and inspect decoupled polarities in real time.

### 4. Train or Fine-Tune (On-the-Fly Physics in Hours)
Adapt the pipeline to custom antenna frequencies and sampling intervals without needing pre-recorded datasets:

```bash
python train.py \
  --epochs 30 \
  --dt 2.0e-9 \
  --freq 100e6 \
  --dx 0.05 \
  --batch_size 8 \
  --workers 4
```
* **Infinite Variety**: Synthesizes undulating layers and Ricker wavelets on the fly.
* **Rapid Turnaround**: Fully converges within a few hours on modern laptops or mini-PCs (e.g. Mac mini M2).

---

## 📁 Repository Structure

```text
gpr-reflection-enhancer/
├── assets/                  # High-resolution comparison figures and visual benchmarks
├── configs/                 # YAML configuration parameters for processing and visualization
├── core/
│   ├── filter_1d.py         # 1D background suppression, AGC & Hilbert analytic transform
│   ├── detector_2d.py       # ONNX Runtime 2D inference engine & signed polarity synthesis
│   └── model_torch.py       # PyTorch reference implementation of two-stage architecture
├── data/
│   └── sample/              # Sanitized benchmark profile & core patches (.png, .npy)
├── weights/
│   └── gpr_core_v110.onnx   # Production pre-trained ONNX model weight (~3 MB)
├── app.py                   # Streamlit web application for interactive analysis
├── demo.py                  # One-click CLI enhancement tool
├── train.py                 # Agile on-the-fly physics simulation & training script
├── requirements.txt         # Minimal dependency manifest
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

---

## 💬 Technical Observations & Discussion

From a pure signal processing standpoint, working with real-world low-frequency (e.g. 100 MHz) radargrams revealed several interesting phenomena worth discussing with experienced domain experts:

1. **Ringing Discontinuities**: Subtle jumps or shifts in horizontal ringing patterns across specific trace intervals (possibly related to surface contact variations, antenna decoupling, or elevation changes).
2. **Horizon Coda Frequency Shifts**: Noticeable apparent frequency drops in the trailing edges of certain deep reflective horizons.
3. **Trace-to-Trace Heterogeneity**: Intermittent single traces exhibiting significantly elevated ringing compared to their neighbors.
4. **Validation Without Ground Truth**: Practical strategies for validating weak reflection horizons when complementary borehole or core data are unavailable.
5. **Alternative High-Throughput Simulators**: While our "water-wave" Huygens simplification is orders of magnitude faster than full-wave FDTD solvers like gprMax, it omits multiple scattering and complex medium dispersion. We warmly invite suggestions on better-balanced, physics-faithful fast simulators!

We warmly welcome geophysicists and radar practitioners to share their domain insights via GitHub Issues or Discussions.

---

## 🇨🇳 中文说明与项目背景

### 1. 跨界背景与核心设计哲学
本项目作者具备**音频与阵列信号处理、神经网络应用研发**背景，并非地球物理专业科班出身。最初因有人提供了几组低信噪比（Low SNR）遗留 GPR 数据探讨潜在合作可能，作者出于技术探索兴趣，尝试运用信号处理与神经网络经验进行概念验证（PoC）。

在前期摸索阶段，作者曾尝试使用业界标准的数值模拟工具 **gprMax**（基于 FDTD 时域有限差分），但很快发现全波数值求解计算开销巨大，单张复杂剖面耗时漫长，难以支撑神经网络训练所需的庞大数据吞吐。

后来通过人工观察大量实测 B-scan 剖面，从波动信号的直觉出发，作者感到地下反射双曲线与波前干涉在几何形态上非常类似于**“水面波纹的相干叠加”**。受此启发，作者抛弃了沉重的网格差分全波求解，转而采用基于**惠更斯原理（Huygens Principle）走时双曲线叠加与子波褶积的向量化简化仿真**。这种近似将单张剖面的生成时间从数十分钟缩短至**几毫秒**，从而奠定了“边生成边训练、数小时收敛”的基石。

**本项目的核心工程价值不在于追求虚幻的“通用超大模型”，而在于实现极度敏捷的“轻量化快速定制、训练与边缘部署闭环”**：
* **向量化惠更斯在线仿真（On-the-Fly Physics）**：彻底摆脱了预先离线生成几十 GB 庞大数据集的负担。在内存中通过多进程实时合成无限变化的地下波场流，边仿真边训练，网络永不重复学习死数据。
* **低成本快速收敛**：在普通的消费级设备（如 **Mac mini M2, 24GB**）上，利用统一内存与 GPU/MPS 加速，**仅需几个小时即可完成训练收敛**。
* **参数逼近原则**：仿真参数（时间步长 `dt`、中心频率 `freq`、道间距 `dx`）需要尽量逼近实际测区工况。模型对参数具备合理的宽容度，但若物理尺度严重失配（如波长或双曲线曲率相差一个数量级），可能导致重构异常。针对具体勘探频段快速微调并轻量化部署，远比纸上谈兵的通用性更具工程落地价值。
* **坦诚求教**：这种“水波叠加”的惠更斯简化毕竟属于几何物理近似，难免忽略了高阶多次波以及复杂介质的频散特性。非常欢迎各位地球物理专家批评指正，探讨是否有兼顾极速吞吐与更高物理保真度的更好方案！

### 2. 社区交流与实测观察
作者曾将部分初步效果分享至 Reddit 社区（参见 [r/geophysics 讨论](https://www.reddit.com/r/geophysics/comments/1r3f83m/showcase_automating_100mhz_gpr_signal_recovery_nn/)），引来部分业内同行的关注与测试交流。在一组粗采样（`dt=2ns, dx=1m`）的高噪声 100MHz 实测数据测试中，作者观察到实测波形中存在许多复杂的物理现象（如局部水平振铃跳跃、反射层尾波频散、道间不均匀响应等），这些现象很难仅凭通用的图像或信号先验去完整判读。

### 3. 开源目的与交流期待
作者深知自己**缺少实际的一线野外地质勘探与雷达工程应用经验**，对于实测数据中某些微弱特征究竟是真实地层界面、局部介质变化还是天线耦合干扰，缺乏专业的地质学解读能力，因此也不确定这些初步探索对专业社区究竟有多大参考价值。

决定将整套算法流程、轻量 ONNX 权重（~3MB）和交互式 Demo 整理开源，主要是希望：
* 梳理并沉淀这次针对未知物理领域的跨界实践经验；
* 抛砖引玉，供相关方向的研究人员和工程师参考、检验或二次开发；
* 真诚期待懂地质、懂雷达一线实践的专家和同行交流指正，共同探讨传统信号处理与深度学习在浅地表地球物理中的合理结合。

---

## 🤝 Open Collaboration & Contact

This project is an open-source engineering exploration. We welcome feedback, critique, and collaborative dialogue:
* **Interpretation & Earth Sciences**: Helping understand subtle wavefield artifacts and validating against real-world geological contexts.
* **Algorithm & Architecture**: Exploring improved physical loss constraints, velocity estimation, and migration methods.
* **Embedded & Edge Deployment**: Testing lightweight, real-time inference on portable radar controllers or field tablets.

Feel free to open an **Issue**, submit a **Pull Request**, or reach out for technical exchange!

---

<div align="center">
<i>Crafted with passion for signal processing and reproducible science.</i>
</div>
