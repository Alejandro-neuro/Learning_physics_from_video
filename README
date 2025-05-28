# 🌌 Learning Physics from Video: Unsupervised Physical Parameter Estimation for Continuous Dynamical Systems

<p align="center">
  <img src="assets/preview.gif" width="600" alt="Method overview animation" />
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2410.01376"><img src="https://img.shields.io/badge/arXiv-2410.01376-b31b1b.svg" alt="arXiv"/></a>
  <img src="https://img.shields.io/badge/Conference-CVPR_2025-blueviolet.svg" alt="CVPR 2025"/>
  <img src="https://img.shields.io/github/license/Alejandro-neuro/Learning_physics_from_video" alt="License"/>
</p>

---

## ✨ Overview

This repository contains the **official implementation** of the CVPR 2025 paper:

> **Learning Physics From Video: Unsupervised Physical Parameter Estimation for Continuous Dynamical Systems**  
> *Alejandro Castañeda Garcia, Jan Warchocki, Jan van Gemert, Daan Brinks, Nergis Tömen*  
> [📄 Read on arXiv](https://arxiv.org/abs/2410.01376)

We present a **decoder-free, unsupervised** framework that estimates physical parameters directly from videos governed by continuous dynamical equations — **without frame reconstruction** or manual labels.

---

## 🔬 Key Contributions

- **Unsupervised** estimation of physical parameters using known ODEs
- **Decoder-free architecture** working directly in latent space
- **Robust training** with KL-divergence to avoid trivial solutions
- Introduced **Delfys75**, a diverse real-world benchmark dataset
- Achieves **state-of-the-art results** on both synthetic and real data

---

## 🧠 Paper Abstract

Extracting physical parameters from video is key in science. Existing unsupervised methods rely on expensive frame reconstructions and only support motion-based systems. We propose a novel model that operates in latent space, bypassing the need for decoders. Our method is efficient, stable across initializations, and applicable to diverse dynamical systems (motion, brightness, scaling). We also introduce **Delfys75**, a new dataset with real-world recordings and ground truth annotations. Our model consistently outperforms baselines and generalizes well to unseen conditions.

---

## 📁 Dataset

We introduce **[Delfys75](https://www.kaggle.com/datasets/jaswar/physical-parameter-prediction)** — the first real-world dataset for unsupervised physical parameter estimation:

- 75 real videos across 5 physical systems: Pendulum, Torricelli flow, Sliding block, LED decay, Free fall scale
- Includes frame-wise **object masks** and **parameter ground truth**
- Resolution: 1920×1080 @ 60fps
- Suitable for testing generalization beyond synthetic setups

---

## 🧪 Data Processing

The code expects the videos in **PyTorch tensor** format:

```python
# Expected shape: [batch_size, num_frames, channels, height, width]
video_tensor = torch.tensor(...)
