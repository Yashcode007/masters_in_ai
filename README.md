# 🎓 Graduate Coursework & Research Portfolio: Machine Learning & Deep Learning
### The University of Texas at Austin (UT Austin)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Lab%20%2F%20Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![LaTeX](https://img.shields.io/badge/LaTeX-Typesetting-008080?style=for-the-badge&logo=latex&logoColor=white)](https://www.latex-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 📖 Overview

This repository contains graduate-level coursework, mathematical derivations, theoretical problem sets, and from-scratch algorithmic implementations completed as part of the **Master's program at The University of Texas at Austin**. 

The codebase spans foundational statistical learning, modern deep neural network architectures, and advanced frontier topics in deep generative modeling (VAEs, Diffusion Models, Normalizing Flows, and Autoregressive Transformer Language Models).

---

## 🗂️ Curriculum & Repository Structure

```
masters/
├── 📂 advances_in_deep_generative_modelling/   # Advanced Generative AI & Probabilistic Models
│   ├── 📁 homework_1/                          # Density estimation, Likelihood-based models
│   ├── 📁 homework_2/                          # Variational Autoencoders (VAEs) & Latent Variable Models
│   ├── 📁 homework_3/                          # Normalizing Flows & Diffusion / Score-based Models
│   ├── 📁 homework_4/                          # nanoGPT: From-scratch Autoregressive Transformer
│   └── 📁 notes/                               # Lecture notes, theoretical derivations & summaries
│
├── 📂 deep_learning/                           # Deep Neural Network Architectures & Optimization
│   ├── 📁 homework/
│   │   ├── 📁 homework-1/                      # Feedforward networks, backprop & custom autograd
│   │   ├── 📁 homework2/                       # CNNs, Residual Connections & Vision modeling
│   │   ├── 📁 homework3/                       # Sequence models (RNNs, LSTMs) & Attention mechanisms
│   │   └── 📁 homework_4/                      # Modern Transformer blocks & Representation learning
│   └── 📁 lecture_pics/                        # Visualized architectures & conceptual reference diagrams
│
└── 📂 machine_learning/                        # Statistical Learning Theory & Classical ML
    ├── 📁 submission_hw_1_programming/         # Linear & Logistic Regression, Gradient Descent variants
    ├── 📁 hw_2/                                # Support Vector Machines (SVMs) & Kernel methods
    ├── 📁 submission_hw_5_programming/         # Unsupervised Learning, PCA, K-Means & GMM/EM
    ├── 📁 submission_hw_6_programming/         # Ensemble Methods, Boosting & Decision Trees
    ├── 📄 hw1_humanized.tex                    # Formal mathematical derivations in LaTeX
    └── 📁 solutions_from_institution/          # Benchmark solutions & theoretical validations
```

---

## 🔬 Core Courses & Key Technical Focus

### 1. 🌌 Advances in Deep Generative Modeling
Explores exact and approximate likelihood estimation, latent variable modeling, implicit generative models, and autoregressive architectures.
- **Autoregressive Transformers (`nanoGPT`)**: End-to-end implementation of a decoder-only transformer with causal multi-head self-attention, Pre-LayerNorm residuals, weight tying, and temperature/top-$k$ sampling on character-level datasets.
- **Variational Autoencoders (VAEs)**: Evidence Lower Bound (ELBO) derivation and optimization, amortized variational inference, reparameterization trick, and latent space regularization ($\mathcal{D}_{\text{KL}}$).
- **Normalizing Flows**: Invertible neural architectures (RealNVP, Planar flows), Jacobian determinant computation, and exact likelihood computation via change-of-variables.
- **Diffusion & Score-Based Models**: Denoising Diffusion Probabilistic Models (DDPM), score matching, reverse-time SDEs/ODEs, and iterative sample generation.

### 2. 🧠 Deep Learning
Covers foundational and state-of-the-art deep learning architectures, loss surfaces, and gradient-based optimization mechanics.
- **Optimization & Mechanics**: Custom backpropagation engines, adaptive optimizers (SGD with Momentum, RMSprop, AdamW), learning rate schedules, and weight initialization strategies (Xavier, Kaiming).
- **Computer Vision**: Convolutional Neural Networks (CNNs), spatial feature hierarchies, batch normalization, and deep residual architectures (ResNets).
- **Sequence Modeling & Attention**: Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) cells, Gated Recurrent Units (GRUs), and Scaled Dot-Product Attention.
- **Regularization & Generalization**: Dropout, weight decay ($L_2$ regularization), LayerNorm, and data augmentation pipelines.

### 3. 📊 Machine Learning (Statistical & Classical)
Rigorous treatment of empirical risk minimization, convex optimization, probabilistic modeling, and statistical learning theory.
- **Supervised Learning**: Ridge/Lasso Regularized Linear Regression, Logistic Regression, Support Vector Machines (dual formulation, Mercer kernels), and Decision Trees / Random Forests / AdaBoost.
- **Unsupervised Learning**: Principal Component Analysis (PCA / SVD), Kernel PCA, $K$-Means clustering, and Gaussian Mixture Models (GMM) with Expectation-Maximization (EM).
- **Theoretical Analysis**: Bias-variance tradeoff, PAC learnability, VC dimension, and Lagrangian duality.

---

## 🛠️ Tech Stack & Tooling

- **Programming Language**: Python 3.10+
- **Deep Learning Frameworks**: PyTorch, Torchvision, Torchaudio
- **Scientific Computing & Data**: NumPy, SciPy, Pandas, Scikit-Learn
- **Visualization**: Matplotlib, Seaborn
- **Development & Typesetting**: Jupyter Notebook / Lab, VS Code, LaTeX ($\TeX$)

---

## 🚀 Getting Started

### Clone the Repository
```bash
git clone https://github.com/<your-username>/masters.git
cd masters
```

### Environment Setup
Create and activate a virtual environment:
```bash
# Using conda
conda create -n masters-env python=3.10 -y
conda activate masters-env

# Or using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Dependencies
```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib seaborn jupyterlab
```

---

## ⚖️ Academic Integrity & Disclaimer

The materials and code provided in this repository are for educational, portfolio, and archival reference purposes only. If you are a current student at The University of Texas at Austin or any other institution taking a related course, please adhere strictly to your university's **Academic Integrity Policy** and **Honor Code**. Do not copy code directly for submitted assignments.

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
