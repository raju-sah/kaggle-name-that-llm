# Cookbook 05: OOF Blending & Prior-Shift Calibration

## Purpose
Blends diverse out-of-fold probability matrices (Linear, GBDT, DeBERTa, Sibling Specialists) and calibrates class multipliers to optimize worst-case Balanced Accuracy.

## Probability Blending
```python
import numpy as np

# Log-linear blending with model weights w
log_prob = w[0] * np.log(np.clip(prob_linear, 1e-7, 1.0)) + \
           w[1] * np.log(np.clip(prob_gbdt, 1e-7, 1.0)) + \
           w[2] * np.log(np.clip(prob_deberta, 1e-7, 1.0))

prob_ensemble = np.exp(log_prob)
prob_ensemble /= prob_ensemble.sum(axis=1, keepdims=True)
```

## Prior-Shift Multiplier Optimization
```python
# Optimize multipliers alpha_k to maximize Balanced Accuracy across Dirichlet simulated draws:
# pred = argmax_k (alpha_k * prob_ensemble[:, k])
# Subject to minimum class frequency floor: count_k >= tau_min
```
