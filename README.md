# Beyond Simulated Benchmarks: Evaluating Motion Representations for Fall Detection Under Real-World Data Scarcity

## Abstract
Falls are a major health concern for older adults, and wearable sensors have been widely explored for detecting falls and enabling timely intervention. However, real-world falls are extremely rare. Collecting 100 real-world falls requires an estimated 100,000 days of monitoring, resulting in severely limited labeled data for training machine learning models. Consequently, many approaches rely on simulated datasets, often reporting high laboratory performance but exhibiting limited real-world generalization. In this work, we present a systematic evaluation of motion representations for wearable fall detection under real-world data scarcity. Using accelerometer signals, we compare interval-based, kernel-based, symbolic, and foundation model representations. As an interpretable baseline, we additionally investigate a lightweight symbolic representation that converts short motion segments into symbolic sentences augmented with physically-grounded impact descriptors. Experiments use FallAllD, a large simulated fall dataset, and FARSEEING, a clinically verified real-world fall dataset. Through cross-validation, controlled data scarcity experiments, and cross-dataset transfer, we examine how representation choices affect robustness under realistic deployment conditions. Our results reveal that highly parameterized kernel and foundation models excel on simulated data but suffer severe degradation under both data scarcity and domain shift. Although interval-based representations achieve the strongest absolute real-world performance, augmenting symbolic representations with physically-grounded impact descriptors yields the smallest degradation under domain shift and retains detection sensitivity under extreme data scarcity, albeit at lower precision. These findings highlight the importance of evaluating models beyond simulated benchmarks and show that representation choice is critical for deployable fall detection given the scarcity of real-world data.

---

## Experiments

This repository contains the code used for the experiments in the paper. It includes the streaming evaluation pipeline, and the experimental setup used to compare motion representations for wearable fall detection. The [experiments.ipynb](experiments.ipynb) notebook reproduces the experiments reported in the paper:

1. **Subject-wise cross-validation** on the FARSEEING real-world falls dataset.  
2. **Data scarcity analysis**, where the number of training fall events is progressively reduced.  
3. **Cross-dataset transfer**, where models trained on simulated falls (FallAllD) are evaluated on real-world falls (FARSEEING).  
4. **Symbolic motion motif analysis** for interpreting FallLM representations.

The experiments were run using **Python 3.10**. Dependencies can be installed using:

```bash
pip install -r requirements.txt
```

The preprocessed version of the `FallAllD` dataset used in this study can be downloaded [here](https://drive.google.com/file/d/1mLmq34paps-jz4XQ_ReSpJTEEbieMoIb/view?usp=sharing). The FARSEEING dataset is available upon request from its maintainers.