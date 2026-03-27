<div align="center">

# ✂️ V-CAST: Video Curvature-Aware Spatio-Temporal Pruning for Efficient Video Large Language Models

### [Xinying Lin](https://github.com/xinyouu)<sup>1,2</sup> · [Xuyang Liu](https://github.com/xuyang-liu16)<sup>3,&dagger;</sup> · [Yiyu Wang](https://github.com/lern-to-write)<sup>4</sup> · [Teng Ma](https://github.com/MaTengSYSU)<sup>1</sup> · [Wenqi Ren](https://rwenqi.github.io)<sup>1,2,✉</sup>

<sup>1</sup> Shenzhen Campus of Sun Yat-sen University  
<sup>2</sup> Shenzhen Loop Area Institute  
<sup>3</sup> Sichuan University  
<sup>4</sup> EPIC Lab, Shanghai Jiao Tong University

Contact: `xinyinglin@slai.edu.cn`

[![Code](https://img.shields.io/badge/Code-GitHub-111111?style=for-the-badge&logo=github)](https://github.com/xinyouu/V-CAST)
[![Homepage Branch](https://img.shields.io/badge/Homepage-branch-4c6ef5?style=for-the-badge)](https://github.com/xinyouu/V-CAST/tree/homepage)
[![License](https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge)](./LICENSE)

</div>

<div align="center">

[News](#-news) · [Highlights](#-highlights) · [✨ Overview](#-overview) · [🛠 Preparation](#-preparation) · [🚀 Performance-Evaluation](#-performance-evaluation) · [📌 Citation](#-citation) · [👍 Acknowledgment](#-acknowledgment) · [📩 Contact](#-contact)

</div>

---

## 🔥 News

- **`2026.03.28`** Added a dedicated `homepage` branch for the project page.
- **`2026.03.28`** Refined the public README and homepage to match the current paper presentation.
- **`2026.03.27`** Released the public **Qwen3-VL + V-CAST + lmms_eval** evaluation path.
- **`Soon`** Support for **LLaVA** and **Omni** will be released in follow-up updates.

---

## 🎯 Highlights

- **Curvature-aware spatio-temporal pruning:** V-CAST allocates temporal budget according to video curvature and performs coordinate-preserving spatial pruning.
- **Training-free and plug-and-play:** V-CAST can be integrated into VideoLLMs without retraining.
- **Coverage-oriented compression:** V-CAST explicitly addresses discontinuous coverage and token-merging-induced position drift.
- **Strong accuracy-efficiency trade-off:** V-CAST preserves **98.6%** of original performance, outperforms the second-best baseline by **+1.1%** on average, and reduces peak memory and total latency to **86.7%** and **86.4%** of vanilla Qwen3-VL-8B-Instruct.

---

## ✨ Overview

<p align="center">
  <img src="images/teaser.png" width="980" alt="V-CAST teaser">
</p>

V-CAST is a training-free and plug-and-play curvature-aware spatio-temporal pruning framework for efficient long-context video inference. It revisits token compression from the perspective of **spatio-temporal information coverage**, and combines:

- **Curvature-guided temporal allocation** to route more budget to semantic turns and event boundaries.
- **Coordinate-preserving spatial pruning** to retain informative tokens without breaking the original `(t, h, w)` grid.
- **Compatibility with VideoLLMs** through a clean pruning-based design that avoids token merging drift.

<p align="center">
  <img src="images/overview.png" width="920" alt="V-CAST overview">
</p>

---

## 🛠 Preparation

1. Clone the repository:

```bash
git clone https://github.com/xinyouu/V-CAST.git
cd V-CAST
```

2. Create the environment:

```bash
conda create -n vcast python=3.10 -y
conda activate vcast
pip install --upgrade pip
pip install -e .
```

3. Set optional environment variables if needed:

```bash
export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="your_hf_token"
```

---

## 🚀 Performance Evaluation

The current public release focuses on the **Qwen3-VL** evaluation path with **V-CAST** enabled by default.

### Quick Start

```bash
bash examples/v_cast/inference_qwen3vl_v_cast_64.sh
```

### Current Public Evaluation Targets

- `mlvu_dev`
- `mvbench`
- `videomme`
- `egoschema`
- `longvideobench_val_v`

### Core Public Paths

| Component | Path |
| --- | --- |
| V-CAST wrapper | [`compressor/v_cast/main.py`](./compressor/v_cast/main.py) |
| V-CAST core implementation | [`compressor/v_cast/modeling_qwen3_vl_v_cast.py`](./compressor/v_cast/modeling_qwen3_vl_v_cast.py) |
| Qwen3-VL evaluation wrapper | [`lmms_eval/models/simple/qwen3_vl.py`](./lmms_eval/models/simple/qwen3_vl.py) |
| Example script | [`examples/v_cast/inference_qwen3vl_v_cast_64.sh`](./examples/v_cast/inference_qwen3vl_v_cast_64.sh) |

### Default Public Configuration

- model: `Qwen/Qwen3-VL-8B-Instruct`
- max input frames: `64`
- retain ratio: `0.25`
- temporal budgeting: `curvature -> softmax(temp=0.7)`
- spatial score: `hybrid`

---

## 📌 Citation

If you find this repository useful, please cite:

```bibtex
@misc{lin2026vcast,
  title={V-CAST: Video Curvature-Aware Spatio-Temporal Pruning for Efficient Video Large Language Models},
  author={Xinying Lin and Xuyang Liu and Yiyu Wang and Teng Ma and Wenqi Ren},
  year={2026},
  howpublished={\url{https://github.com/xinyouu/V-CAST}}
}
```

---

## 👍 Acknowledgment

This project builds on and benefits from the open-source efforts of:

- Qwen3-VL
- LLaVA
- lmms-eval

---

## 📩 Contact

For questions about the project, please contact:

- `xinyinglin@slai.edu.cn`
