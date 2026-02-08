# Wave Time Series Classification using Atmospheric Data for Sea State Detection

This repository contains the experimental codebase and model implementations for a comprehensive study on **sea state and storm severity classification** using **multivariate atmospheric and oceanographic time-series data**.

The project formulates sea state detection as a **time-series classification problem**, rather than a conventional regression task, with the goal of producing **operationally meaningful storm severity categories** for maritime safety applications.

> **Note**  
> The accompanying research paper is currently under submission.  
> For this reason, the manuscript itself is **not publicly included** in this repository at this stage.

---

## Project Overview

Accurate and timely assessment of sea state conditions is critical for maritime operations, particularly for small and medium-sized vessels. While most existing studies focus on **predicting continuous wave parameters** (e.g., significant wave height), real-world decision-making typically relies on **discrete severity categories** such as calm, moderate, or severe conditions.

This project reformulates sea state analysis as a **multi-class time-series classification problem**, explicitly modeling the **temporal evolution and persistence** of wave conditions. Instead of point-wise predictions, the approach analyzes fixed-length temporal windows to capture storm development, intensity, and duration.

The study uses long-term atmospheric and marine data from the **Eastern Mediterranean (Antalya region)** as a representative coastal case study.

---

## Key Contributions

- **Time-series classification formulation**  
  Sea state detection is modeled using sliding temporal windows instead of instantaneous labels, enabling the capture of storm buildup and persistence.

- **Comprehensive model benchmarking**  
  A wide range of models are evaluated under a unified experimental setup, including classical machine learning, kernel-based time-series methods, and deep learning architectures.

- **Robust labeling strategy**  
  Storm severity labels are derived using rolling-window statistics combined with percentile-based thresholds, producing site-aware and temporally consistent class definitions.

- **Feature engineering insights**  
  Statistical features, trend-based representations, and temperature-related variables consistently improve classification performance across multiple model families.

- **State-of-the-art performance**  
  Transformer-based sequence models—particularly **PatchTST**—significantly outperform all classical baselines and alternative deep learning approaches.

---

## Main Findings

- Modeling **temporal evolution** is essential for accurate sea state classification.
- **Nonlinear classifiers** are required to capture complex atmospheric–oceanic interactions.
- **Class imbalance handling** is critical due to the rarity of extreme storm events.
- **PatchTST-based architectures** provide the best balance between expressiveness and generalization, achieving near-perfect classification performance in this setting.

---

## Models Implemented

### Classical Machine Learning
- Linear Support Vector Machine (SVM)
- Random Forest
- XGBoost

### Kernel-Based Time-Series Methods
- MiniROCKET with linear classifier
- MiniROCKET with Random Forest classifier

### Deep Learning
- 1D Convolutional Neural Network (CNN)
- Recurrent models (LSTM variants)
- State-space models (Mamba)
- Transformer-based models
  - PatchTST (baseline)
  - PatchTST variants with architectural and normalization modifications

---

## Experimental Setup

- Fixed-length sliding windows (36 hours)
- Time-based train / validation / test split to prevent temporal leakage
- Feature normalization using statistics computed only on the training set
- Evaluation focused on **Macro-F1**, precision, recall, and class-wise performance

All experiments are reproducible using the provided notebooks.

---

## Current Status

- Core experiments completed
- Models and evaluation pipelines finalized
- Paper currently under submission

Once the paper is accepted, a citation and preprint link will be added to this repository.

---

## Contact

For questions, discussion, or collaboration:

**Yusuf Kemahlı**  
Computer Engineering  
Middle East Technical University (METU)

---

## License

This repository is intended for **academic and research purposes**.  
License details will be finalized after publication.
