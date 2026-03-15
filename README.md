# An Evaluation of Motion Representations for Real-World Fall Detection Under Data Scarcity

## Abstract
Falls are a major health concern for older adults, and wearable sensors have been widely explored as a means of detecting falls and enabling timely intervention. However, real-world falls are extremely rare and difficult to capture in practice. It has been estimated that collecting 100 real-world falls may require approximately 100,000 days of monitoring (around 300 person-years), resulting in extremely limited labeled data for training machine learning models. As a consequence, many existing approaches rely heavily on simulated fall datasets, often reporting high accuracy in laboratory settings but struggling to generalize to real-world scenarios. In this work, we investigate how different motion representations influence fall detection performance under data scarcity. Using wearable accelerometer signals, we compare several representation families, including interval-based, kernel-based, symbolic, and foundation-model-based approaches. We also introduce FallLM, a language-inspired symbolic representation that converts short segments of motion signals into discrete tokens augmented with an impact-level descriptor. We evaluate these representations using FallAllD, a simulated falls dataset, and FARSEEING, a dataset of clinically verified real-world falls. Through cross-validation, controlled data scarcity experiments, and cross-dataset transfer from simulated to real-world data, we analyze how representation choices affect robustness under limited fall data. Our results reveal substantial differences in how representations generalize across datasets, highlighting the importance of representation design when learning from scarce real-world fall data and providing practical insights for developing deployable wearable fall detection systems.

---

## Experiments

This repository contains the code used for the experiments in the paper. It includes the streaming evaluation pipeline, and the experimental setup used to compare motion representations for wearable fall detection. The `experiments.ipynb` notebook reproduces the experiments reported in the paper:

1. **Subject-wise cross-validation** on the FARSEEING real-world falls dataset.  
2. **Data scarcity analysis**, where the number of training fall events is progressively reduced.  
3. **Cross-dataset transfer**, where models trained on simulated falls (FallAllD) are evaluated on real-world falls (FARSEEING).  
4. **Symbolic motion motif analysis** for interpreting FallLM representations.

---

## Installation

Install dependencies using:
```bash
pip install -r requirements.txt
```

The experiments were run using **Python 3.10**.

---

## Running the Experiments

All experiments are provided in reproducible form in the notebook [experiments.ipynb](experiments.ipynb).

The preprocessed version of the `FallAllD` dataset used in this study can be downloaded [here](https://drive.google.com/file/d/1mLmq34paps-jz4XQ_ReSpJTEEbieMoIb/view?usp=sharing).

The FARSEEING dataset is available upon request from its maintainers.