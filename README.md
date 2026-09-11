# Beyond Simulated Benchmarks: Evaluating Motion Representations for Fall Detection Under Real-World Data Scarcity

## Abstract
Falls are a major health concern for older adults, and wearable sensors have been widely explored for detecting falls and enabling timely intervention. However, real-world falls are extremely rare: collecting 100 of them requires an estimated 100,000 days of monitoring, resulting in severely limited labelled data for training machine learning models. Consequently, many approaches rely on simulated datasets, often reporting high laboratory performance but limited real-world generalisation. We present a systematic evaluation of motion representations for wearable fall detection under real-world data scarcity. Using accelerometer signals, we compare interval-based, kernel-based, symbolic, and foundation model representations. As an interpretable baseline, we additionally investigate a lightweight symbolic representation that converts short motion segments into symbolic sentences augmented with physically-grounded impact descriptors. Experiments use FallAllD, a simulated falls dataset, and FARSEEING, a clinically verified real-world falls dataset. Through cross-validation, controlled data scarcity, and cross-dataset transfer, we examine how representation choices affect robustness under realistic deployment. Our results reveal that highly parameterised kernel and foundation models excel on simulated data but degrade severely under both data scarcity and domain shift. Although the interval-based representation achieves the strongest absolute real-world performance, augmenting a symbolic representation with physically-grounded impact descriptors yields the smallest degradation under domain shift and retains detection sensitivity under extreme scarcity, albeit at lower precision. These findings highlight the importance of evaluating beyond simulated benchmarks and show that representation choice is critical for deployable fall detection given the scarcity of real-world data.

---

## Experiments

This repository contains the code used for the experiments in the [paper](https://arxiv.org/abs/2608.13197). It includes the streaming evaluation pipeline, and the experimental setup used to compare motion representations for wearable fall detection. The [experiments.ipynb](experiments.ipynb) notebook reproduces the experiments reported in the paper:

1. **Subject-wise cross-validation** on the FARSEEING real-world falls dataset.  
2. **Data scarcity analysis**, where the number of training fall events is progressively reduced.  
3. **Cross-dataset transfer**, where models trained on simulated falls (FallAllD) are evaluated on real-world falls (FARSEEING).  
4. **Symbolic motion motif analysis** for interpreting FallLM representations.

The experiments were run using **Python 3.10**. Dependencies can be installed using:

```bash
pip install -r requirements.txt
```

The preprocessed version of the `FallAllD` dataset used in this study can be downloaded [here](https://drive.google.com/file/d/1mLmq34paps-jz4XQ_ReSpJTEEbieMoIb/view?usp=sharing). The FARSEEING dataset is available upon request from its maintainers.

---

## Full Paper and Citation

The full paper PDF is available [here](https://arxiv.org/abs/2608.13197). If this work helps your research, please cite:

``` bibtex
@article{aderinola2026motionreps,
      title={Beyond Simulated Benchmarks: Evaluating Motion Representations for Fall Detection Under Real-World Data Scarcity}, 
      author={Timilehin B. Aderinola and Ilaria D'Ascanio and Luca Palmerini and Lorenzo Chiari and Jochen Klenk and Clemens Becker and Brian Caulfield and Georgiana Ifrim},
      year={2026},
      journal={arXiv preprint arXiv:2608.13197},
      url={https://arxiv.org/abs/2608.13197}, 
}
```