# Wave Time Series Classification using Atmospheric Data for Sea State Detection

This repository contains the experimental codebase and model implementations for a comprehensive study on **sea state and storm severity classification** using **multivariate atmospheric and oceanographic time-series data**.

The project formulates sea state detection as a **time-series classification problem**, rather than a conventional regression task, with the goal of producing **operationally meaningful storm severity categories** for maritime safety applications.

> **Note:**  
> The accompanying research paper is currently under submission.  
> For this reason, the manuscript itself is **not publicly included** in this repository at this stage.

---

## 📌 Project Overview

Accurate and timely assessment of sea state conditions is critical for maritime operations, particularly for small and medium-sized vessels. While most existing studies focus on **predicting continuous wave parameters** (e.g., significant wave height), real-world decision-making often relies on **discrete severity categories** such as calm, moderate, or severe conditions.

This project:
- Reformulates sea state analysis as a **multi-class time-series classification problem**
- Models **temporal evolution and persistence** of wave conditions
- Systematically compares **classical machine learning**, **kernel-based**, and **deep learning** approaches
- Benchmarks **state-of-the-art transformer-based architectures** against strong baselines

The study uses long-term atmospheric and marine data from the **Eastern Mediterranean (Antalya region)** as a representative coastal case study.

---

## 🧠 Key Contributions

- **Time-series formulation:**  
  Sea state is classified using sliding temporal windows instead of point-wise labels, capturing storm buildup and persistence.

- **Comprehensive model comparison:**  
  Classical ML (SVM, Random Forest, XGBoost), MiniROCKET-based models, CNNs, recurrent models, state-space models, and transformer-based architectures are evaluated under a unified experimental setup.

- **Advanced labeling strategy:**  
  Storm severity labels are derived using **rolling-window statistics** and **percentile-based thresholds**, enabling robust and site-aware severity categorization.

- **Feature engineering insights:**  
  Statistical features, trend-based representations, and temperature-related variables consistently improve classification performance.

- **State-of-the-art performance:**  
  Transformer-based sequence models—particularly **PatchTST**—achieve the highest performance, significantly outperforming all classical and alternative deep learning baselines.

---

## 📊 Main Findings

- Modeling **temporal evolution** is essential for accurate sea state classification.
- **Nonlinear models** are required to capture complex atmospheric–oceanic interactions.
- **Class imbalance handling** is critical due to the rarity of extreme storm events.
- **PatchTST-based architectures** provide the best trade-off between expressiveness and generalization, achieving near-perfect classification performance in this setting.

---

## 📁 Repository Structure

