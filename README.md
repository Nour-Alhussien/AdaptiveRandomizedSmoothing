# AdaptiveRandomizedSmoothing(ARS)

This project implements and evaluates **Adaptive Randomized Smoothing (ARS)** for tabular data. The framework trains neural networks on three datasets—**UNSW-NB15**, **URL**, and **CICIDS2018**—generates adversarial examples using the **Adversarial Robustness Toolbox (ART)**, and evaluates certified robustness using two schemes:

- **Original Randomized Smoothing (ORS)** — baseline certification method  
- **AdaptiveSmoothEntropy (ARS)** — proposed method that:
  - injects **entropy-guided noise during training**, and  
  - uses a **margin–confidence proxy** to adapt the number of samples during certification  

The code is provided as a **Colab-friendly notebook**, but can also run locally in a standard Python environment.

---

## Model Training

- **Architecture:** 4-layer MLP for tabular data (`TabularNN`)
- **Loss:** Cross-entropy
- **Optimizers:** Adam / AdamW
- **Scheduler:** Optional StepLR (halves learning rate every 10 epochs)

---

## Adversarial Attacks (ART)

We evaluate robustness under the following attacks:

- **Gradient-based:** FGSM, PGD  
- **Optimization-based:** Carlini–Wagner (CW-L2), ZOO  
- **Decision-based:** HopSkipJump (HSJ)  
- **Geometric:** DeepFool  
- **Optional:** JSMA  

### Tabular-specific attack:
- **LowProFool (LPF):**
  - Weighted-norm perturbations using feature importance  
  - Designed to generate **low-profile, realistic attacks**  

---

## Certification Methods

- **ORS (Smooth):**
  - Standard randomized smoothing  
  - Fixed Gaussian noise  
  - Two-stage sampling \((n_0, n)\)  
  - Clopper–Pearson confidence bounds  

- **ARS (AdaptiveSmoothEntropy):**
  - Entropy-guided **per-sample noise scaling** \(\sigma(x)\)  
  - Noise floor and ramp-based training schedule  
  - **Adaptive sampling** using margin–confidence proxy  

---

## Evaluation Metrics

- **Clean accuracy**
- **Adversarial accuracy** (per attack)
- **Certified accuracy**  
  *(fraction of inputs where the certified prediction matches the true label)*
- **Abstention rate**
- **Average certified radius**
- **Certification time (ms/sample)**

Evaluation is performed using **5-fold partitioning of the test set indices** for stability.

---

## Environment & Setup

### Dependencies:
- `pandas`, `numpy`, `scikit-learn`
- `torch`, `torchvision` (GPU optional)
- `adversarial-robustness-toolbox`
- `scipy`, `statsmodels`
- `matplotlib`

### Install:
```bash
pip install torch pandas numpy scikit-learn adversarial-robustness-toolbox scipy statsmodels matplotlib
```bash
### Citation
If you use this code, please cite our work:
```bash

