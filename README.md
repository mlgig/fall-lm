# An Evaluation of Motion Representations for Real-World Fall Detection Under Data Scarcity

This repository contains the code used for the experiments in the paper. It includes the streaming evaluation pipeline, and the experimental setup used to compare motion representations for wearable fall detection.

---

## Experiments

The `experiments.ipynb` notebook reproduces the experiments reported in the paper:

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

All experiments can be reproduced using the notebook:
`experiments.ipynb`


