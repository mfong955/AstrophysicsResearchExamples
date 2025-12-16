#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Statistical Model Fitting with Covariance Matrices
Purpose: Robust curve fitting with full covariance handling and uncertainty propagation
Author: Matthew Fong
Skills Demonstrated: Statistical inference, covariance analysis, scipy optimization

Industry Applications:
- Sensor calibration with correlated measurements
- Financial risk modeling with asset correlations
- Quality control with measurement uncertainty
- Clinical trial analysis with correlated endpoints
"""

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.linalg import cholesky, cho_solve
from typing import Tuple, Dict, List, Optional, Callable, Union
import warnings


class CovarianceFitter:
    """
    TECHNIQUE: Weighted Least Squares with Full Covariance
    INDUSTRY APPLICATION: Proper statistical inference when data points are correlated
    KEY SKILL: Handling measurement uncertainty and correlations
    
    A class for fitting models to data with full covariance matrix handling,
    including validation, regularization, and uncertainty propagation.
    
    Attributes:
        model_func: Model function to fit
        param_names: Names of model parameters
        param_bounds: Bounds for parameters
        
    Example:
        >>> fitter = CovarianceFitter(model_func, ['a', 'b', 'c'])
        >>> results = fitter.fit(x, y, covariance)
        >>> y_pred, y_err = fitter.predict(x_new, return_uncertainty=True)
    """
    
    def __init__(
        self,
        model_func: Callable,
        param_names: List[str],
        param_bounds: Optional[List[Tuple[float, float]]] = None
    ):
        """
        Initialize the covariance fitter.
        
        Args:
            model_func: Model function f(x, *params)
            param_names: List of parameter names
            param_bounds: Optional bounds for each parameter
        """
        self.model_func = model_func
        self.param_names = param_names
        self.n_params = len(param_names)
        self.param_bounds = param_bounds
        
        # Results storage
        self.popt = None
        self.pcov = None
        self.chi2 = None
        self.dof = None
        self._is_fitted = False
        
    @staticmethod
    def validate_covariance(
        cov: np.ndarray,
        regularize: bool = True,
        reg_factor: float = 1e-10
    ) -> Tuple[np.ndarray, bool]:
        """
        Validate and optionally regularize a covariance matrix.
        
        TECHNIQUE: Covariance matrix validation and regularization
        INDUSTRY APPLICATION: Ensuring numerical stability in statistical computations
        
        Args:
            cov: Covariance matrix to validate
            regularize: Whether to regularize if not positive definite
            reg_factor: Regularization factor (added to diagonal)
            
        Returns:
            Tuple of (validated covariance, is_valid flag)
        """
        # Check symmetry
        if not np.allclose(cov, cov.T):
            warnings.warn("Covariance matrix is not symmetric, symmetrizing...")
            cov = (cov + cov.T) / 2
        
        # Check positive definiteness
        try:
            np.linalg.cholesky(cov)
            return cov, True
        except np.linalg.LinAlgError:
            if regularize:
                # Add small value to diagonal
                cov_reg = cov + reg_factor * np.eye(len(cov))
                try:
                    np.linalg.cholesky(cov_reg)
                    warnings.warn(f"Covariance regularized with factor {reg_factor}")
                    return cov_reg, True
                except np.linalg.LinAlgError:
                    # Try larger regularization
                    eigenvalues = np.linalg.eigvalsh(cov)
                    min_eig = eigenvalues.min()
                    if min_eig < 0:
                        cov_reg = cov + (-min_eig + reg_factor) * np.eye(len(cov))
                        return cov_reg, True
            return cov, False
    
    @staticmethod
    def compute_chi2(
        residuals: np.ndarray,
        inv_cov: np.ndarray
    ) -> float:
        """
        Compute chi-squared statistic.
        
        TECHNIQUE: Chi-squared goodness of fit
        INDUSTRY APPLICATION: Model validation and comparison
        
        Args:
            residuals: Data - model residuals
            inv_cov: Inverse covariance matrix
            
        Returns:
            Chi-squared value
        """
        return float(residuals @ inv_cov @ residuals)
    
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        covariance: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        method: str = 'curve_fit',
        maxfev: int = 100000
    ) -> Dict:
        """
        Fit model to data with full covariance handling.
        
        TECHNIQUE: Weighted least squares with covariance
        INDUSTRY APPLICATION: Proper parameter estimation with correlated data
        
        Args:
            x: Independent variable data
            y: Dependent variable data
            covariance: Covariance matrix of y
            initial_guess: Initial parameter values
            method: Fitting method ('curve_fit' or 'minimize')
            maxfev: Maximum function evaluations
            
        Returns:
            Dictionary with fit results
        """
        # Validate covariance
        cov_valid, is_valid = self.validate_covariance(covariance)
        if not is_valid:
            raise ValueError("Covariance matrix is not positive definite")
        
        # Compute inverse covariance
        inv_cov = np.linalg.pinv(cov_valid)
        
        # Set up bounds
        if self.param_bounds is not None:
            bounds = (
                [b[0] for b in self.param_bounds],
                [b[1] for b in self.param_bounds]
            )
        else:
            bounds = (-np.inf, np.inf)
        
        # Initial guess
        if initial_guess is None:
            if self.param_bounds is not None:
                initial_guess = np.array([
                    (b[0] + b[1]) / 2 for b in self.param_bounds
                ])
            else:
                initial_guess = np.ones(self.n_params)
        
        if method == 'curve_fit':
            # Use scipy curve_fit
            self.popt, self.pcov = curve_fit(
                self.model_func,
                x, y,
                p0=initial_guess,
                sigma=cov_valid,
                absolute_sigma=True,
                bounds=bounds,
                maxfev=maxfev
            )
        else:
            # Use minimize with chi-squared objective
            def objective(params):
                model = self.model_func(x, *params)
                residuals = y - model
                return self.compute_chi2(residuals, inv_cov)
            
            result = minimize(
                objective,
                initial_guess,
                method='L-BFGS-B',
                bounds=self.param_bounds,
                options={'maxiter': maxfev}
            )
            self.popt = result.x
            
            # Estimate parameter covariance from Hessian
            try:
                from scipy.optimize import approx_fprime
                hess = np.zeros((self.n_params, self.n_params))
                eps = 1e-8
                for i in range(self.n_params):
                    def grad_i(p):
                        return approx_fprime(p, objective, eps)[i]
                    hess[i] = approx_fprime(self.popt, grad_i, eps)
                self.pcov = np.linalg.pinv(hess)
            except Exception:
                self.pcov = np.eye(self.n_params) * np.inf
        
        # Compute fit statistics
        model_pred = self.model_func(x, *self.popt)
        residuals = y - model_pred
        self.chi2 = self.compute_chi2(residuals, inv_cov)
        self.dof = len(y) - self.n_params
        
        self._is_fitted = True
        
        return self.get_results()
    
    def get_results(self) -> Dict:
        """
        Get fit results as a dictionary.
        
        Returns:
            Dictionary with parameters, uncertainties, and statistics
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Parameter uncertainties from covariance diagonal
        param_errors = np.sqrt(np.diag(self.pcov))
        
        results = {
            'parameters': dict(zip(self.param_names, self.popt)),
            'uncertainties': dict(zip(self.param_names, param_errors)),
            'covariance': self.pcov,
            'chi2': self.chi2,
            'dof': self.dof,
            'reduced_chi2': self.chi2 / self.dof if self.dof > 0 else np.inf
        }
        
        return results
    
    def predict(
        self,
        x: np.ndarray,
        return_uncertainty: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions with optional uncertainty propagation.
        
        TECHNIQUE: Error propagation through model
        INDUSTRY APPLICATION: Uncertainty quantification in predictions
        
        Args:
            x: Points at which to predict
            return_uncertainty: Whether to return prediction uncertainty
            
        Returns:
            Predictions, and optionally uncertainties
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        y_pred = self.model_func(x, *self.popt)
        
        if return_uncertainty:
            # Propagate parameter uncertainty through model
            # Using numerical differentiation
            eps = 1e-8
            jacobian = np.zeros((len(x), self.n_params))
            
            for i in range(self.n_params):
                params_plus = self.popt.copy()
                params_plus[i] += eps
                params_minus = self.popt.copy()
                params_minus[i] -= eps
                
                jacobian[:, i] = (
                    self.model_func(x, *params_plus) -
                    self.model_func(x, *params_minus)
                ) / (2 * eps)
            
            # Prediction variance
            pred_var = np.diag(jacobian @ self.pcov @ jacobian.T)
            pred_std = np.sqrt(np.maximum(pred_var, 0))
            
            return y_pred, pred_std
        
        return y_pred
    
    def plot_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        covariance: np.ndarray,
        x_pred: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """
        Create visualization of fit results.
        
        Args:
            x: Data x values
            y: Data y values
            covariance: Data covariance
            x_pred: Points for prediction curve
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])
        
        # Data errors from covariance diagonal
        y_err = np.sqrt(np.diag(covariance))
        
        # Main plot
        axes[0].errorbar(x, y, yerr=y_err, fmt='o', label='Data', alpha=0.7)
        
        if x_pred is None:
            x_pred = np.linspace(x.min(), x.max(), 200)
        
        y_pred, y_pred_err = self.predict(x_pred, return_uncertainty=True)
        
        axes[0].plot(x_pred, y_pred, 'r-', label='Best fit', linewidth=2)
        axes[0].fill_between(
            x_pred,
            y_pred - y_pred_err,
            y_pred + y_pred_err,
            alpha=0.3,
            color='red',
            label='1σ uncertainty'
        )
        
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].legend()
        axes[0].set_title(f'Model Fit (χ²/dof = {self.chi2/self.dof:.2f})')
        
        # Residual plot
        y_model = self.model_func(x, *self.popt)
        residuals = y - y_model
        normalized_residuals = residuals / y_err
        
        axes[1].errorbar(x, normalized_residuals, yerr=1, fmt='o', alpha=0.7)
        axes[1].axhline(0, color='r', linestyle='--')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('Normalized Residuals')
        axes[1].set_ylim(-5, 5)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


class MultiModelComparison:
    """
    TECHNIQUE: Model Selection using Information Criteria
    INDUSTRY APPLICATION: Choosing between competing models
    KEY SKILL: AIC, BIC, and likelihood ratio tests
    
    Compare multiple models using statistical criteria.
    """
    
    def __init__(self):
        """Initialize model comparison."""
        self.models = {}
        self.results = {}
        
    def add_model(
        self,
        name: str,
        fitter: CovarianceFitter,
        x: np.ndarray,
        y: np.ndarray,
        covariance: np.ndarray
    ):
        """
        Add a fitted model for comparison.
        
        Args:
            name: Model name
            fitter: Fitted CovarianceFitter instance
            x: Data x values
            y: Data y values
            covariance: Data covariance
        """
        if not fitter._is_fitted:
            raise ValueError("Fitter must be fitted before adding")
        
        n = len(y)
        k = fitter.n_params
        chi2 = fitter.chi2
        
        # Compute information criteria
        # AIC = chi2 + 2k (for Gaussian likelihood)
        aic = chi2 + 2 * k
        
        # BIC = chi2 + k * ln(n)
        bic = chi2 + k * np.log(n)
        
        # Corrected AIC for small samples
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n > k + 1 else np.inf
        
        self.models[name] = fitter
        self.results[name] = {
            'n_params': k,
            'chi2': chi2,
            'dof': n - k,
            'reduced_chi2': chi2 / (n - k),
            'aic': aic,
            'bic': bic,
            'aicc': aicc
        }
    
    def get_comparison_table(self) -> Dict:
        """
        Get comparison table of all models.
        
        Returns:
            Dictionary with comparison results
        """
        if not self.results:
            raise ValueError("No models added for comparison")
        
        # Find best model by each criterion
        best_aic = min(self.results.keys(), key=lambda x: self.results[x]['aic'])
        best_bic = min(self.results.keys(), key=lambda x: self.results[x]['bic'])
        
        # Compute delta AIC/BIC
        min_aic = self.results[best_aic]['aic']
        min_bic = self.results[best_bic]['bic']
        
        for name in self.results:
            self.results[name]['delta_aic'] = self.results[name]['aic'] - min_aic
            self.results[name]['delta_bic'] = self.results[name]['bic'] - min_bic
        
        return {
            'models': self.results,
            'best_aic': best_aic,
            'best_bic': best_bic
        }
    
    def print_comparison(self):
        """Print formatted comparison table."""
        comparison = self.get_comparison_table()
        
        print("\nModel Comparison Results")
        print("=" * 80)
        print(f"{'Model':<20} {'k':<5} {'χ²/dof':<10} {'AIC':<12} {'ΔAIC':<10} {'BIC':<12} {'ΔBIC':<10}")
        print("-" * 80)
        
        for name, res in comparison['models'].items():
            print(f"{name:<20} {res['n_params']:<5} {res['reduced_chi2']:<10.3f} "
                  f"{res['aic']:<12.2f} {res['delta_aic']:<10.2f} "
                  f"{res['bic']:<12.2f} {res['delta_bic']:<10.2f}")
        
        print("-" * 80)
        print(f"Best by AIC: {comparison['best_aic']}")
        print(f"Best by BIC: {comparison['best_bic']}")


if __name__ == "__main__":
    # Demonstration of covariance fitting
    np.random.seed(42)
    
    # Generate synthetic data with correlated errors
    n_points = 30
    x = np.linspace(0, 10, n_points)
    
    # True model: exponential decay
    def exp_model(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    true_params = [5.0, 0.5, 1.0]
    y_true = exp_model(x, *true_params)
    
    # Generate correlated noise
    noise_level = 0.3
    correlation_length = 2.0
    
    # Build covariance matrix with exponential correlation
    cov = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            cov[i, j] = noise_level**2 * np.exp(-abs(x[i] - x[j]) / correlation_length)
    
    # Generate correlated noise
    L = np.linalg.cholesky(cov)
    noise = L @ np.random.randn(n_points)
    y = y_true + noise
    
    # Fit model
    print("Fitting model with covariance...")
    fitter = CovarianceFitter(
        exp_model,
        param_names=['amplitude', 'decay_rate', 'offset'],
        param_bounds=[(0, 20), (0, 5), (-5, 10)]
    )
    
    results = fitter.fit(x, y, cov)
    
    print("\nFit Results:")
    print("-" * 40)
    for name, value in results['parameters'].items():
        err = results['uncertainties'][name]
        print(f"{name}: {value:.3f} ± {err:.3f}")
    
    print(f"\nχ²/dof: {results['reduced_chi2']:.3f}")
    print(f"True parameters: {true_params}")
    
    # Compare with simpler model
    def linear_model(x, a, b):
        return a * x + b
    
    fitter_linear = CovarianceFitter(
        linear_model,
        param_names=['slope', 'intercept']
    )
    fitter_linear.fit(x, y, cov)
    
    # Model comparison
    print("\n" + "=" * 40)
    comparison = MultiModelComparison()
    comparison.add_model('Exponential', fitter, x, y, cov)
    comparison.add_model('Linear', fitter_linear, x, y, cov)
    comparison.print_comparison()
