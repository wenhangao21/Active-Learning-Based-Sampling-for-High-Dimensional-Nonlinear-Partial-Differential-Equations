# Active Sampling for High-Dimensional Nonlinear PDEs

This repository contains the PyTorch implementation of the paper "**[Active Learning Based Sampling for High-Dimensional Nonlinear Partial Differential Equations](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=te4HWo0AAAAJ&citation_for_view=te4HWo0AAAAJ:u5HHmVD_uO8C)**", **W.Gao**, C.Wang.， Journal of Computational Physics (JCP), 2023.


<p align="center">
  <img src="https://wenhangao21.github.io/images/active_learning_JCP.png" alt="Figure" width="400"/>
</p>

## BibTeX
```bash
@article{DBLP:journals/jcphy/GaoW23,
  author={Wenhan Gao and Chunmei Wang},
  title={Active learning based sampling for high-dimensional nonlinear partial differential equations},
  year={2023},
  month={February},
  cdate={1675209600000},
  journal={J. Comput. Phys.},
  volume={475},
  pages={111848},
  url={https://doi.org/10.1016/j.jcp.2022.111848}
}
```

## Introduction 
In this paper, we present an active learning-based sampling approach for efficient PINNs in high-dimensional spaces. 
We propose a parallelizable alternative to the sequential MCMC sampling method, enabling simulation of draws from an unnormalized distribution based on residuals. 
This sampling approach is compatible with all four mainstream methods as of 2022.


## Structure


```
├── utilities.py           # Contains general utility functions for setting up the PDE, point generation, and error calculation.
├── sampling.py          # Implements proposed sampling for interior and boundary points.
├── network_setting.py     # Defines neural network architectures (FNNs as in Vanilla PINN).
├── train.py               # Main script to train and evaluate the model.
└── unclean_original_code/   # Contains unrefined (original) code for the Sec. 5.1 20D example.
```

## Running Experiments
The code is directly runnable.
```python
python train.py
```


## Running Requirements
This implementation uses only standard libraries, including PyTorch, NumPy, and SciPy.

No training data is required, as the Physics-Informed Neural Network (PINN) is semi-supervised, with the governing equations providing the supervision signal.
