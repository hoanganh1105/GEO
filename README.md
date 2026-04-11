# GEO: Graph Energy-based Out-of-Distribution Detection

[![Status](https://img.shields.io/badge/Status-Accepted-blue)]()  
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)]()  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)]()

This repository provides the official PyTorch implementation of our paper:

> **"Your Full Paper Title Here"**  
> *Accepted at [Conference/Journal Name]*

---

## 📌 Overview

**GEO** is a novel framework for detecting **Out-of-Distribution (OOD)** nodes in graph-structured data.  
It integrates:

- Graph Neural Networks (GNNs)  
- Energy-based scoring  
- One-class classification constraints  

to achieve robust and consistent OOD detection performance across multiple scenarios.

---

## 🧩 Implementation & Reproducibility

To ensure a **fair and controlled comparison** with existing baselines, GEO is implemented as a **structural extension** of a strong backbone model.

In this codebase:

- GEO variants are executed using:
  ```
  --method gnnsafe
  ```

- Combined with our proposed components:
  - `--use_reg` → Energy Regularization  
  - `--use_occ` → One-Class Constraint  
  - `--use_prop` → Energy Belief Propagation  

This design ensures:
- Shared components (data loading, message passing, architecture) remain identical  
- Performance improvements come **purely from our proposed contributions**

---

## ⚙️ Environment Setup

Tested with:
- Python ≥ 3.8  
- PyTorch ≥ 2.0  

We recommend using **Conda**:

```
# Create environment
conda create -n geo_env python=3.9
conda activate geo_env

# Install PyTorch (adjust CUDA if needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyG
pip install torch_geometric

# Other dependencies
pip install ogb pandas scikit-learn numpy scipy
```

---

## 📂 Datasets

We evaluate GEO on four benchmark datasets under multiple OOD settings:

- **Cora**  
- **Amazon-Photo**  
- **Coauthor-CS**  
- **ogbn-arxiv**

### OOD Protocols:
- Structure perturbation  
- Feature interpolation  
- Label leave-out  
- Temporal shift  

> 📌 Datasets are automatically downloaded via PyTorch Geometric and OGB APIs on first run.

---

## 🚀 Usage

### 1. Run Individual Experiments

Example: Full GEO (Energy + OCC + Propagation) on Cora (Feature OOD)

```
python main.py \
  --dataset cora \
  --ood_type feature \
  --method gnnsafe \
  --use_bn \
  --use_prop \
  --use_reg \
  --use_occ \
  --beta 0.1
```

---

## 🔑 Key Arguments

| Argument | Description |
|----------|------------|
| `--method gnnsafe` | Backbone framework |
| `--use_reg` | Enable Energy Regularization loss |
| `--use_occ` | Enable One-Class Classification loss |
| `--use_prop` | Enable Energy Belief Propagation |
| `--beta` | Weight for one-class loss |

---

## 📖 Citation

If you find this work useful, please cite:

```
@article{your_lastname2024geo,
  title={Your Full Paper Title Here},
  author={First Author and Second Author and Third Author},
  journal={Name of the Journal or Conference},
  year={2024}
}
```

> 📌 *Note: This citation will be updated upon official publication.*

---

## 🙏 Acknowledgements

This work builds upon the evaluation framework of **GNNSafe**.  
We sincerely thank the original authors for making their code publicly available.

---

## ⚠️ Notes

Before releasing your code:

- Remove unused or experimental directories (e.g., `GKDE&GPN`)  
- Clean personal/debug files  
- Verify reproducibility scripts  
- Update citation once the paper is officially published  

---

## 📬 Contact

For questions or collaborations, please open an issue or contact the authors directly.

---
