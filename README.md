<div align="center">

<h1>
  ✂️ V-CAST: Video Curvature-Aware Spatio-Temporal Pruning for Efficient Video Large Language Models
</h1>

<h3>
  <a href="https://github.com/xinyouu">Xinying Lin</a><sup>1,2</sup>,
  <a href="https://github.com/xuyang-liu16">Xuyang Liu</a><sup>3,&dagger;</sup>,
  <a href="https://github.com/lern-to-write">Yiyu Wang</a><sup>4</sup>,
  <a href="https://github.com/MaTengSYSU">Teng Ma</a><sup>1</sup>,
  <a href="https://rwenqi.github.io">Wenqi Ren</a><sup>1,2,✉</sup>
</h3>

<p>
  <sup>1</sup> Shenzhen Campus of Sun Yat-sen University
  &nbsp;&nbsp;
  <sup>2</sup> Shenzhen Loop Area Institute
</p>
<p>
  <sup>3</sup> Sichuan University
  &nbsp;&nbsp;
  <sup>4</sup> EPIC Lab, Shanghai Jiao Tong University
</p>


<p>
  <i>⚡ A training-free and plug-and-play <b>Curvature-Aware</b> Spatio-Temporal pruning framework for efficient long-context video inference.</i>
</p>

</div>

<div align="center">
</div>

---

## 🔥 News

- **`2026.03.26`** We opened the V-CAST repository.
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


<table>
  <thead>
    <tr>
      <th align="left">Model Base</th>
      <th align="left" width="22%">Status</th>
      <th align="left">Code Path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Qwen3-VL</td>
      <td>✅ Released</td>
      <td><a href="./compressor/v_cast/modeling_qwen3_vl_v_cast.py"><code>compressor/v_cast/modeling_qwen3_vl_v_cast.py</code></a></td>
    </tr>
    <tr>
      <td>LLaVA-OneVision / LLaVA-Video</td>
      <td>🚧 Planned</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni / Qwen3-Omni</td>
      <td>🚧 Planned</td>
      <td>-</td>
    </tr>
  </tbody>
</table>


The current public release focuses on the **Qwen3-VL** evaluation path with **V-CAST** enabled by default.

### Quick Start

```bash
bash examples/v_cast/inference_qwen3vl_v_cast_64.sh
```

### Core Public Paths


<table>
  <thead>
    <tr>
      <th align="left">Component</th>
      <th align="left">Path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>V-CAST wrapper</td>
      <td><a href="./compressor/v_cast/main.py"><code>compressor/v_cast/main.py</code></a></td>
    </tr>
    <tr>
      <td>V-CAST core implementation</td>
      <td><a href="./compressor/v_cast/modeling_qwen3_vl_v_cast.py"><code>compressor/v_cast/modeling_qwen3_vl_v_cast.py</code></a></td>
    </tr>
    <tr>
      <td>Qwen3-VL evaluation wrapper</td>
      <td><a href="./lmms_eval/models/simple/qwen3_vl.py"><code>lmms_eval/models/simple/qwen3_vl.py</code></a></td>
    </tr>
    <tr>
      <td>Example script</td>
      <td><a href="./examples/v_cast/inference_qwen3vl_v_cast_64.sh"><code>examples/v_cast/inference_qwen3vl_v_cast_64.sh</code></a></td>
    </tr>
  </tbody>
</table>


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

We extend our gratitude to the open-source efforts of [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT) and [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL).

---

## 📩 Contact

For any question about our paper or code, please email `xinyinglin@slai.edu.cn`.
