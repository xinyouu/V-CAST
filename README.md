# ✂️ V-CAST: Video Curvature-Aware Spatio-Temporal Pruning for Efficient Video Large Language Models

### [Xinying Lin](https://github.com/xinyouu)1,2, [Xuyang Liu](https://github.com/xuyang-liu16)3,†, [Yiyu Wang](https://github.com/lern-to-write)4, [Teng Ma](https://github.com/MaTengSYSU)1, [Wenqi Ren](https://rwenqi.github.io)1,2,✉

1 Shenzhen Campus of Sun Yat-sen University    2 Shenzhen Loop Area Institute

3 Sichuan University    4 EPIC Lab, Shanghai Jiao Tong University

*⚡ A training-free and plug-and-play **Curvature-Aware** Spatio-Temporal pruning framework for efficient long-context video inference.*

---

## 🔥 News

- `**2026.03.26`** We opened the V-CAST repository.

---

## 🎯 Highlights

- **Curvature-aware spatio-temporal pruning:** V-CAST allocates temporal budget according to video curvature and performs coordinate-preserving spatial pruning.
- **Training-free and plug-and-play:** V-CAST can be integrated into VideoLLMs without retraining.
- **Coverage-oriented compression:** V-CAST explicitly addresses discontinuous coverage and token-merging-induced position drift.
- **Strong accuracy-efficiency trade-off:** V-CAST preserves **98.6%** of original performance, outperforms the second-best baseline by **+1.1%** on average, and reduces peak memory and total latency to **86.7%** and **86.4%** of vanilla Qwen3-VL-8B-Instruct.

---

## ✨ Overview

V-CAST is a training-free and plug-and-play curvature-aware spatio-temporal pruning framework for efficient long-context video inference. It revisits token compression from the perspective of **spatio-temporal information coverage**, and combines:

- **Curvature-guided temporal allocation** to route more budget to semantic turns and event boundaries.
- **Coordinate-preserving spatial pruning** to retain informative tokens without breaking the original `(t, h, w)` grid.
- **Compatibility with VideoLLMs** through a clean pruning-based design that avoids token merging drift.

---

## 🛠 Preparation

1. Clone the repository:

```bash
git clone https://github.com/xinyouu/V-CAST.git
cd V-CAST
```

1. Create the environment:

```bash
conda create -n vcast python=3.10 -y
conda activate vcast
pip install --upgrade pip
pip install -e .
```

---

## 🚀 Performance Evaluation

The current public release focuses on the **Qwen3-VL** evaluation path with **V-CAST** enabled by default.

We support the following model families in our internal codebase. The current public release focuses on **Qwen3-VL**, and additional V-CAST code releases are being organized below.


| Model Base      | Status         | Code Path                                                                                          |
| --------------- | -------------- | -------------------------------------------------------------------------------------------------- |
| Qwen3-VL        | ✅ Released     | `[compressor/v_cast/modeling_qwen3_vl_v_cast.py](./compressor/v_cast/modeling_qwen3_vl_v_cast.py)` |
| Qwen2.5-Omni    | 🚧 Coming Soon | -                                                                                                  |
| Qwen3-VL Omni   | 🚧 Coming Soon | -                                                                                                  |
| LLaVA-OneVision | 🚧 Coming Soon | -                                                                                                  |
| LLaVA-Video     | 🚧 Coming Soon | -                                                                                                  |


### Quick Start

```bash
bash examples/v_cast/inference_qwen3vl_v_cast_64.sh
```

### Core Public Paths


| Component                   | Path                                                                                                 |
| --------------------------- | ---------------------------------------------------------------------------------------------------- |
| V-CAST wrapper              | `[compressor/v_cast/main.py](./compressor/v_cast/main.py)`                                           |
| V-CAST core implementation  | `[compressor/v_cast/modeling_qwen3_vl_v_cast.py](./compressor/v_cast/modeling_qwen3_vl_v_cast.py)`   |
| Qwen3-VL evaluation wrapper | `[lmms_eval/models/simple/qwen3_vl.py](./lmms_eval/models/simple/qwen3_vl.py)`                       |
| Example script              | `[examples/v_cast/inference_qwen3vl_v_cast_64.sh](./examples/v_cast/inference_qwen3vl_v_cast_64.sh)` |


---

## 📌 Citation

If you find this repository useful, please cite:

```bibtex
@article{lin2026v,
  title={V-CAST: Video Curvature-Aware Spatio-Temporal Pruning for Efficient Video Large Language Models},
  author={Lin, Xinying and Liu, Xuyang and Wang, Yiyu and Ma, Teng and Ren, Wenqi},
  journal={arXiv preprint arXiv:2603.27650},
  year={2026}
}
```

---

## 👍 Acknowledgment

We extend our gratitude to the open-source efforts of [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT)and [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL).

---

## 📩 Contact

For any question about our paper or code, please email `xinyinglin@slai.edu.cn`.
`<div align="center">`