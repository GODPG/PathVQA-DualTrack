# 🔬 PathVQA-DualTrack

Official PyTorch implementation for the paper: **"A Parameter-Efficient Fine-Tuning and Dual-Track Decoding Framework for Pathology Visual Question Answering"**.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 📖 Introduction
Medical Visual Question Answering (Med-VQA) in pathology faces unique challenges, including ultra-high resolutions and severe hallucination issues in closed-ended tasks. This repository implements a framework built upon **MedGemma-1.5-4B**, featuring:
- 🚀 **QLoRA with Loss-on-Answer-Only:** Efficient tuning while freezing the vision encoder.
- 🛤️ **Dual-Track Decoding Engine:** A specialized log-prob scoring mechanism for closed-ended questions and optimized autoregressive generation for open-ended questions.
- 📊 **Multi-dimensional Semantic Evaluation:** Comprehensive metrics (WBSS, CBSS, Exact/Relaxed Match) for rigorous pathology VQA benchmarking.

## 📥 Model Weights
The base model used in this project is **MedGemma-1.5-4B-IT**. 

- **Hugging Face Hub:** [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
- **Official Repository:** [Google-Health/medgemma](https://github.com/Google-Health/medgemma)

> **Note:** Please ensure you have accepted the *Health AI Developer Foundations License* on the Hugging Face model page to access the weights.

## 📂 Repository Structure
```text
PathVQA-DualTrack/
├── README.md               
└── src/                    # Core source code
    ├── train_qlora.py      # QLoRA instruction tuning
    ├── metrics.py          # WBSS, CBSS, and accuracy metrics
    ├── run_baselines.py    # Inference scripts for baselines
    └── evaluate.py         # Evaluation and scoring pipeline
