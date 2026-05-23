# MLFD1
RBF surrogate model for eVTOL wing download force from sparse CFD data (9 points). Gaussian, Matérn, and compact C4 kernels with hyperparameter tuning via 3‑fold cross‑validation and Ridge regression. Includes grid search, condition number monitoring, and contour map visualisation.

# RBF Regression for eVTOL Wing Download Force

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains a **Radial Basis Function (RBF) regression** model to predict the wing download force (N) of an eVTOL aircraft in hover. The model is trained on **only 9 CFD simulations** that sample the design space defined by horizontal (`d/D`) and vertical (`z/D`) distances between the actuator disk and the wing.

The code automatically:
- Tests **three RBF kernels**: Gaussian, Matérn C4, and compact C4 (Wendland).
- Performs a **grid search** over RBF width (`σ`) and Ridge regularisation (`α`) using **3‑fold cross‑validation**.
- Selects the best hyperparameters based on the lowest average RMSE.
- Produces a **continuous contour map** of the predicted force over the design space.

**Methodological traceability** – all design choices (removal of polynomial terms, condition number cutoff, hyperparameter ranges) are documented in the code.

## Features

- ✅ Vectorised kernel computations (`scipy.spatial.distance.cdist`)
- ✅ Condition number monitoring to skip ill‑conditioned systems
- ✅ Min‑Max scaling of inputs
- ✅ Ridge regression (L2 penalty) for numerical stability
- ✅ 3‑fold cross‑validation with fixed random seed
- ✅ Easy to extend to other datasets or kernels
- ✅ Publication‑ready plots (matplotlib, LaTeX‑ready, high DPI)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rbf-evtol-force-surrogate.git
   cd rbf-evtol-force-surrogate
