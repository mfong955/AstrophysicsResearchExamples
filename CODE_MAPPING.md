# Code Mapping: Original Research → Data Science Portfolio

This document maps the original research code to the refactored data science portfolio examples, showing which techniques were extracted from each source file.

---

## Quick Reference

| Portfolio Project | Original Source Files | Key Techniques |
|-------------------|----------------------|----------------|
| [bayesian-inference/](./bayesian-inference/) | `biasProfile.py`, `emcee_*.py` | MCMC, Bayesian inference, parallel processing |
| [gaussian-process-regression/](./gaussian-process-regression/) | `z0_bin2D_plot_maps.py` | GPR, sklearn, kernel engineering |
| [statistical-modeling/](./statistical-modeling/) | `fit_DeltaSigma.py`, `biasTools_z.py` | Covariance fitting, curve fitting |
| [large-scale-data/](./large-scale-data/) | `binProfs_*.py`, `plotExamples.py` | Parallel processing, HDF5, vectorization |

---

## Detailed Mapping

### 1. Bayesian Inference Project

**Portfolio Location**: [`bayesian-inference/`](./bayesian-inference/)

| Original File | Techniques Extracted | Industry Application |
|---------------|---------------------|---------------------|
| `research-archive/Astro machine learning tools/biasProfile.py` | MCMC sampling with emcee, parallel processing with multiprocessing.Pool, Bayesian log-likelihood/log-prior functions, 7-parameter simultaneous optimization | A/B testing, risk modeling, hyperparameter tuning |
| `research-archive/Astro measurement examples/code/emcee_WLMassFitExample.py` | MCMC fitting workflow, posterior analysis, corner plot visualization | Parameter estimation, uncertainty quantification |
| `research-archive/Astro measurement examples/code/emcee_biasAllTogether.py` | Multi-parameter MCMC analysis, convergence diagnostics | Model calibration, Bayesian optimization |
| `research-archive/Astro measurement examples/code/emcee_plotPosteriorsRxDeltax.py` | Posterior visualization, credible interval computation | Decision support, risk assessment |

**Key Code Patterns**:
```python
# From biasProfile.py - MCMC with parallel processing
from multiprocessing import Pool
with Pool(processes=n_cpus) as pool:
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool=pool)
    sampler.run_mcmc(initial_positions, n_steps)
```

---

### 2. Gaussian Process Regression Project

**Portfolio Location**: [`gaussian-process-regression/`](./gaussian-process-regression/)

| Original File | Techniques Extracted | Industry Application |
|---------------|---------------------|---------------------|
| `research-archive/Astro theory examples/code/z0_bin2D_plot_maps.py` | sklearn GaussianProcessRegressor, Matern/RBF/RationalQuadratic kernels, multi-dimensional interpolation, uncertainty visualization | Bayesian optimization, spatial interpolation, surrogate modeling |
| `research-archive/Astro theory examples/code/binProfs_save_profilesVsX1X2.py` | Multi-parameter data organization, grid-based predictions | Feature engineering, heatmap generation |

**Key Code Patterns**:
```python
# From z0_bin2D_plot_maps.py - GPR with custom kernels
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

kernel = Matern(length_scale=1.0, nu=0.5)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gpr.fit(X_train, y_train)
y_pred, y_std = gpr.predict(X_test, return_std=True)
```

---

### 3. Statistical Modeling Project

**Portfolio Location**: [`statistical-modeling/`](./statistical-modeling/)

| Original File | Techniques Extracted | Industry Application |
|---------------|---------------------|---------------------|
| `research-archive/Astro measurement examples/code/fit_DeltaSigma.py` | scipy.optimize.curve_fit with full covariance matrices, positive-definiteness validation, chi-squared computation | Sensor calibration, portfolio optimization, clinical trials |
| `research-archive/Astro machine learning tools/biasTools_z.py` | Curve fitting utilities, derivative calculations, statistical binning, interpolation | Feature engineering, data preprocessing |
| `research-archive/Astro measurement examples/code/bin2d_resultplots.py` | 2D statistical analysis, visualization of binned data | Data aggregation, heatmap analysis |

**Key Code Patterns**:
```python
# From fit_DeltaSigma.py - Covariance-weighted fitting
from scipy.optimize import curve_fit

popt, pcov = curve_fit(
    model_func, x, y,
    sigma=covariance_matrix,  # Full covariance, not just errors
    absolute_sigma=True,
    maxfev=int(1e6)
)
```

---

### 4. Large-Scale Data Processing Project

**Portfolio Location**: [`large-scale-data/`](./large-scale-data/)

| Original File | Techniques Extracted | Industry Application |
|---------------|---------------------|---------------------|
| `research-archive/Astro theory examples/code/binProfs_save_SelectedHalos_haloPar_z.py` | Large-scale data selection, efficient filtering, memory management | ETL pipelines, data warehousing |
| `research-archive/Astro theory examples/code/binProfs_save_SelectedHalos_haloProf_z.py` | Batch processing, HDF5 data handling, chunked operations | Big data analytics, streaming processing |
| `research-archive/Data replication Code Ocean/code/plotExamples.py` | HDF5 data loading with h5py, efficient array operations | Data engineering, scientific computing |
| `research-archive/Astro machine learning tools/biasProfile.py` | multiprocessing.Pool for parallel computation | Distributed computing, batch processing |

**Key Code Patterns**:
```python
# From plotExamples.py - HDF5 data handling
import h5py

with h5py.File('data.hdf5', 'r') as f:
    data = f['dataset_name'][:]  # Load entire dataset
    # Or load in chunks for large data:
    for i in range(0, n_rows, chunk_size):
        chunk = f['dataset_name'][i:i+chunk_size]
```

---

## Technique Inventory

### Machine Learning Techniques

| Technique | Original Location | Portfolio Location |
|-----------|-------------------|-------------------|
| Gaussian Process Regression | `z0_bin2D_plot_maps.py` | `gaussian-process-regression/gpr_interpolation.py` |
| MCMC Sampling | `biasProfile.py`, `emcee_*.py` | `bayesian-inference/mcmc_parameter_estimation.py` |
| Bayesian Inference | `biasProfile.py` | `bayesian-inference/mcmc_parameter_estimation.py` |
| Multi-parameter Optimization | `biasProfile.py` | `bayesian-inference/mcmc_parameter_estimation.py` |

### Statistical Methods

| Technique | Original Location | Portfolio Location |
|-----------|-------------------|-------------------|
| Covariance Matrix Fitting | `fit_DeltaSigma.py` | `statistical-modeling/covariance_fitting.py` |
| Chi-squared Analysis | `fit_DeltaSigma.py`, `biasProfile.py` | `statistical-modeling/covariance_fitting.py` |
| Uncertainty Quantification | `emcee_*.py` | `bayesian-inference/mcmc_parameter_estimation.py` |
| Statistical Binning | `biasTools_z.py`, `binProfs_*.py` | `large-scale-data/parallel_data_processing.py` |

### Data Engineering Techniques

| Technique | Original Location | Portfolio Location |
|-----------|-------------------|-------------------|
| Parallel Processing | `biasProfile.py` | `large-scale-data/parallel_data_processing.py` |
| HDF5 Data Handling | `plotExamples.py` | `large-scale-data/parallel_data_processing.py` |
| NumPy Vectorization | All files | `large-scale-data/parallel_data_processing.py` |
| Numerical Integration | `biasProfile.py` | `bayesian-inference/mcmc_parameter_estimation.py` |

---

## Skills Summary

| Skill Category | Demonstrated In | Years Experience |
|----------------|-----------------|------------------|
| **Python** | All projects | 7+ |
| **Bayesian Methods** | bayesian-inference/ | 5+ |
| **Machine Learning** | gaussian-process-regression/ | 5+ |
| **Statistical Analysis** | statistical-modeling/ | 7+ |
| **Big Data Processing** | large-scale-data/ | 5+ |
| **Data Visualization** | visualizations/ | 7+ |

---

*This mapping demonstrates how academic research code translates directly to industry-relevant data science skills.*
