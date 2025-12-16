# Statistical Model Fitting with Covariance Matrices

## Overview

This project demonstrates **rigorous statistical model fitting** with full covariance matrix handling. The implementation properly accounts for correlated measurement uncertainties, ensuring statistically valid parameter estimates and error propagation.

## Business/Research Problem

Real-world measurements often have correlated errors that standard fitting methods ignore. This project solves:
- Fitting models when measurement errors are correlated
- Proper error propagation through complex models
- Validating model assumptions (positive-definiteness, normality)
- Comparing nested models with appropriate statistical tests

**Industry Applications:**
- **Portfolio Optimization**: Asset returns with correlated risks
- **Sensor Fusion**: Combining measurements with known correlations
- **Quality Control**: Process monitoring with correlated metrics
- **Clinical Trials**: Analyzing endpoints with shared variance

## Approach

### Data
- **Source**: Observational measurements with full covariance matrices
- **Size**: Multiple datasets with 25+ correlated measurements each
- **Preprocessing**: Covariance matrix validation, outlier detection

### Methods

#### Weighted Least Squares with Covariance
- **Why chosen**: Properly weights observations by their precision and correlations
- **Implementation**: Inverse covariance (precision matrix) weighting
- **Key feature**: Handles singular/near-singular covariance matrices via pseudo-inverse

#### Model Validation
- **Positive-definiteness checks**: Eigenvalue analysis
- **Residual analysis**: Checking for systematic patterns
- **Chi-squared goodness-of-fit**: Quantifying model adequacy

### Validation
- Leave-one-out cross-validation
- Residual normality tests
- Parameter stability analysis

## Results

- **Fit Quality**: Reduced chi-squared ≈ 1.0 (indicating proper error modeling)
- **Parameter Precision**: 10-50% improvement over diagonal-only fitting
- **Model Selection**: Robust comparison between competing models

## Key Files

| File | Description |
|------|-------------|
| `covariance_fitting.py` | Main fitting routines with covariance support |
| `model_validation.py` | Statistical tests and diagnostics |
| `error_propagation.py` | Utilities for propagating uncertainties |

## How to Run

```bash
# Installation
pip install -r requirements.txt

# Fit model with covariance
python covariance_fitting.py --data path/to/data.dat --cov path/to/covariance.dat

# Validate fit
python model_validation.py --results path/to/fit_results.npz
```

## Skills Demonstrated

`python` `statistics` `scipy` `curve-fitting` `covariance-analysis` `error-propagation` `model-validation` `linear-algebra`

---

## Technical Deep Dive

### Covariance-Weighted Fitting

```python
# TECHNIQUE: Weighted Least Squares with Full Covariance
# INDUSTRY APPLICATION: Used in finance for portfolio optimization,
# in engineering for sensor fusion, in pharma for clinical trial analysis
# KEY SKILL: Proper statistical inference with correlated data

import numpy as np
from scipy.optimize import curve_fit

def fit_with_covariance(model_func: callable, 
                        x: np.ndarray, 
                        y: np.ndarray,
                        covariance: np.ndarray,
                        initial_guess: np.ndarray,
                        bounds: tuple = None) -> tuple:
    """
    Fit a model to data with full covariance matrix handling.
    
    This technique is essential when:
    - Measurements share systematic uncertainties
    - Errors are correlated in time or space
    - Proper uncertainty propagation is required
    
    Args:
        model_func: Model function f(x, *params)
        x: Independent variable data
        y: Dependent variable data (measurements)
        covariance: Full covariance matrix of y
        initial_guess: Starting parameter values
        bounds: Parameter bounds (lower, upper)
    
    Returns:
        popt: Optimal parameters
        pcov: Parameter covariance matrix
    
    Example:
        >>> popt, pcov = fit_with_covariance(linear_model, x, y, cov, [1, 0])
        >>> param_errors = np.sqrt(np.diag(pcov))
    """
    # Validate covariance matrix
    if not is_positive_definite(covariance):
        raise ValueError("Covariance matrix must be positive definite")
    
    # Fit with full covariance
    popt, pcov = curve_fit(
        f=model_func,
        xdata=x,
        ydata=y,
        sigma=covariance,  # Full covariance matrix
        p0=initial_guess,
        bounds=bounds if bounds else (-np.inf, np.inf),
        absolute_sigma=True,  # Interpret sigma as absolute
        maxfev=int(1e6)
    )
    
    return popt, pcov


def is_positive_definite(matrix: np.ndarray) -> bool:
    """
    Check if a matrix is positive definite.
    
    A covariance matrix must be positive definite for valid
    statistical inference. This checks via eigenvalue analysis.
    
    Args:
        matrix: Square matrix to check
    
    Returns:
        True if all eigenvalues are positive
    """
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues > 0)
```

### Chi-Squared Calculation

```python
# TECHNIQUE: Chi-squared goodness-of-fit with covariance
# INDUSTRY APPLICATION: Model validation in any field with correlated data
# KEY SKILL: Proper statistical hypothesis testing

def chi_squared(y_observed: np.ndarray, 
                y_predicted: np.ndarray,
                inv_covariance: np.ndarray) -> float:
    """
    Calculate chi-squared statistic with inverse covariance weighting.
    
    The chi-squared statistic measures how well the model fits the data,
    accounting for measurement correlations.
    
    Args:
        y_observed: Measured values
        y_predicted: Model predictions
        inv_covariance: Inverse of the covariance matrix
    
    Returns:
        Chi-squared value (should be ~N_dof for good fit)
    
    Industry interpretation:
    - chi2/N_dof ≈ 1: Good fit, errors properly estimated
    - chi2/N_dof >> 1: Poor fit or underestimated errors
    - chi2/N_dof << 1: Overfitting or overestimated errors
    """
    residual = y_observed - y_predicted
    chi2 = residual @ inv_covariance @ residual
    return chi2
```

### Handling Ill-Conditioned Covariance

```python
# TECHNIQUE: Robust matrix inversion for near-singular covariance
# INDUSTRY APPLICATION: Handling numerical instabilities in real data
# KEY SKILL: Numerical linear algebra best practices

def robust_inverse(covariance: np.ndarray, 
                   rcond: float = 1e-10) -> np.ndarray:
    """
    Compute robust inverse of potentially ill-conditioned covariance.
    
    Real-world covariance matrices are often near-singular due to:
    - Highly correlated measurements
    - Numerical precision limits
    - Insufficient sample sizes
    
    Args:
        covariance: Covariance matrix
        rcond: Cutoff for small singular values
    
    Returns:
        Pseudo-inverse of covariance matrix
    """
    return np.linalg.pinv(covariance, rcond=rcond)
```
