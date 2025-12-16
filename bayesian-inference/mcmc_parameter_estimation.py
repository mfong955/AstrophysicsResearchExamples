#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: MCMC Parameter Estimation
Purpose: Bayesian parameter estimation using Markov Chain Monte Carlo sampling
         with parallel processing and uncertainty quantification.
Author: Matthew Fong
Skills Demonstrated: Bayesian inference, MCMC, parallel processing, optimization

Industry Applications:
- A/B testing with uncertainty quantification
- Risk modeling and financial parameter estimation
- Sensor calibration with measurement uncertainty
- Hyperparameter tuning with confidence bounds
"""

import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import collections
from typing import Tuple, Dict, List, Optional, Callable
import time
import os


class MCMCParameterEstimator:
    """
    TECHNIQUE: Markov Chain Monte Carlo Parameter Estimation
    INDUSTRY APPLICATION: Used in finance for risk modeling,
    in tech for A/B test analysis, in healthcare for clinical trial analysis
    KEY SKILL: Bayesian parameter estimation with uncertainty quantification
    
    A class for performing Bayesian parameter estimation using MCMC sampling.
    Supports multi-parameter optimization with parallel processing and
    full covariance matrix handling.
    
    Attributes:
        n_params: Number of parameters to estimate
        param_names: Names of parameters for reporting
        param_bounds: Bounds for flat priors [lower, upper] for each parameter
        n_walkers: Number of MCMC walkers
        n_steps: Number of MCMC steps
        n_cpus: Number of CPUs for parallel processing
    
    Example:
        >>> estimator = MCMCParameterEstimator(
        ...     n_params=7,
        ...     param_names=['a', 'b', 'c', 'd', 'e', 'f', 'g'],
        ...     param_bounds=[(0, 10), (0, 5), ...],
        ...     n_walkers=72,
        ...     n_steps=50000
        ... )
        >>> results = estimator.fit(x_data, y_data, covariance, model_func)
    """
    
    def __init__(
        self,
        n_params: int,
        param_names: List[str],
        param_bounds: List[Tuple[float, float]],
        n_walkers: int = 72,
        n_steps: int = 50000,
        n_cpus: Optional[int] = None,
        burn_in_fraction: float = 0.02
    ):
        """
        Initialize the MCMC parameter estimator.
        
        Args:
            n_params: Number of parameters to estimate
            param_names: List of parameter names
            param_bounds: List of (lower, upper) bounds for each parameter
            n_walkers: Number of MCMC walkers (should be >= 2 * n_params)
            n_steps: Number of MCMC steps
            n_cpus: Number of CPUs for parallel processing (default: all available)
            burn_in_fraction: Fraction of steps to discard as burn-in
        """
        self.n_params = n_params
        self.param_names = param_names
        self.param_bounds = np.array(param_bounds)
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.n_cpus = n_cpus or os.cpu_count()
        self.burn_in_steps = int(n_steps * burn_in_fraction)
        
        # Results storage
        self.sampler = None
        self.flat_samples = None
        self.best_fit_params = None
        self.param_uncertainties = None
        
    def _log_prior(self, theta: np.ndarray) -> float:
        """
        Compute log-prior probability for flat priors within bounds.
        
        TECHNIQUE: Flat (uniform) prior distribution
        INDUSTRY APPLICATION: Non-informative priors when no prior knowledge exists
        
        Args:
            theta: Parameter vector
            
        Returns:
            0 if within bounds, -inf otherwise
        """
        for i, (param, (lower, upper)) in enumerate(zip(theta, self.param_bounds)):
            if not (lower <= param <= upper):
                return -np.inf
        return 0.0
    
    def _log_likelihood(
        self,
        theta: np.ndarray,
        model_func: Callable,
        x: np.ndarray,
        y: np.ndarray,
        inv_cov: np.ndarray
    ) -> float:
        """
        Compute log-likelihood using chi-squared with inverse covariance.
        
        TECHNIQUE: Chi-squared likelihood with covariance weighting
        INDUSTRY APPLICATION: Proper statistical inference with correlated data
        
        Args:
            theta: Parameter vector
            model_func: Model function f(x, *theta)
            x: Independent variable data
            y: Dependent variable data
            inv_cov: Inverse covariance matrix
            
        Returns:
            Log-likelihood value
        """
        model = model_func(x, *theta)
        
        # Check for invalid model outputs
        if np.any(np.isnan(model)) or np.any(np.isinf(model)):
            return -np.inf
            
        residual = y - model
        chi2 = residual @ inv_cov @ residual
        
        return -0.5 * chi2
    
    def _log_probability(
        self,
        theta: np.ndarray,
        model_func: Callable,
        x: np.ndarray,
        y: np.ndarray,
        inv_cov: np.ndarray
    ) -> float:
        """
        Compute log-probability (prior + likelihood).
        
        Args:
            theta: Parameter vector
            model_func: Model function
            x: Independent variable data
            y: Dependent variable data
            inv_cov: Inverse covariance matrix
            
        Returns:
            Log-probability value
        """
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
            
        ll = self._log_likelihood(theta, model_func, x, y, inv_cov)
        
        result = lp + ll
        if np.isnan(result):
            return -np.inf
            
        return result
    
    def _initialize_walkers(
        self,
        x: np.ndarray,
        y: np.ndarray,
        covariance: np.ndarray,
        model_func: Callable,
        random_init: bool = False
    ) -> np.ndarray:
        """
        Initialize walker positions for MCMC.
        
        TECHNIQUE: Informed initialization from optimization
        INDUSTRY APPLICATION: Faster convergence in production systems
        
        Args:
            x: Independent variable data
            y: Dependent variable data
            covariance: Covariance matrix
            model_func: Model function
            random_init: If True, use random initialization within bounds
            
        Returns:
            Initial positions array of shape (n_walkers, n_params)
        """
        if random_init:
            # Random initialization within bounds
            positions = np.zeros((self.n_walkers, self.n_params))
            for i in range(self.n_params):
                lower, upper = self.param_bounds[i]
                positions[:, i] = np.random.uniform(lower, upper, self.n_walkers)
        else:
            # Initialize from curve_fit optimization
            initial_guess = np.mean(self.param_bounds, axis=1)
            bounds_tuple = (self.param_bounds[:, 0], self.param_bounds[:, 1])
            
            try:
                popt, _ = curve_fit(
                    model_func, x, y,
                    sigma=covariance,
                    p0=initial_guess,
                    bounds=bounds_tuple,
                    maxfev=int(1e6)
                )
                # Scatter walkers around optimum
                positions = popt + 1e-2 * np.random.randn(self.n_walkers, self.n_params)
                
                # Ensure within bounds
                for i in range(self.n_params):
                    lower, upper = self.param_bounds[i]
                    positions[:, i] = np.clip(positions[:, i], lower, upper)
                    
            except Exception:
                # Fall back to random initialization
                return self._initialize_walkers(x, y, covariance, model_func, random_init=True)
                
        return positions
    
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        covariance: np.ndarray,
        model_func: Callable,
        random_init: bool = False,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Fit model parameters using MCMC sampling.
        
        TECHNIQUE: Ensemble MCMC with parallel processing
        INDUSTRY APPLICATION: Production-scale Bayesian inference
        
        Args:
            x: Independent variable data
            y: Dependent variable data
            covariance: Covariance matrix of y
            model_func: Model function f(x, *params)
            random_init: Use random walker initialization
            save_path: Path to save MCMC chains (HDF5 format)
            
        Returns:
            Dictionary with fit results including:
            - 'params': Best-fit parameter values
            - 'uncertainties': Parameter uncertainties (1-sigma)
            - 'samples': Flat MCMC samples
            - 'acceptance_fraction': MCMC acceptance rate
        """
        import emcee
        from multiprocessing import Pool
        
        print(f"Running MCMC with {self.n_walkers} walkers, {self.n_steps} steps...")
        tic = time.time()
        
        # Compute inverse covariance
        inv_cov = np.linalg.pinv(covariance)
        
        # Initialize walkers
        positions = self._initialize_walkers(x, y, covariance, model_func, random_init)
        
        # Set up backend for saving
        backend = None
        if save_path:
            backend = emcee.backends.HDFBackend(save_path)
            backend.reset(self.n_walkers, self.n_params)
        
        # Run MCMC with parallel processing
        with Pool(processes=self.n_cpus) as pool:
            self.sampler = emcee.EnsembleSampler(
                self.n_walkers,
                self.n_params,
                self._log_probability,
                args=(model_func, x, y, inv_cov),
                pool=pool,
                backend=backend
            )
            self.sampler.run_mcmc(positions, self.n_steps, progress=True)
        
        toc = time.time()
        print(f"MCMC completed in {(toc - tic) / 60:.2f} minutes")
        
        # Extract results
        self.flat_samples = self.sampler.get_chain(
            discard=self.burn_in_steps,
            flat=True
        )
        
        # Compute parameter estimates and uncertainties
        self.best_fit_params = {}
        self.param_uncertainties = {}
        
        for i, name in enumerate(self.param_names):
            samples = self.flat_samples[:, i]
            median = np.percentile(samples, 50)
            lower = np.percentile(samples, 16)
            upper = np.percentile(samples, 84)
            
            self.best_fit_params[name] = median
            self.param_uncertainties[name] = {
                'lower': median - lower,
                'upper': upper - median
            }
        
        return {
            'params': self.best_fit_params,
            'uncertainties': self.param_uncertainties,
            'samples': self.flat_samples,
            'acceptance_fraction': np.mean(self.sampler.acceptance_fraction)
        }
    
    def plot_corner(self, save_path: Optional[str] = None, truths: Optional[List] = None):
        """
        Create corner plot of posterior distributions.
        
        TECHNIQUE: Corner plot visualization
        INDUSTRY APPLICATION: Communicating parameter correlations to stakeholders
        
        Args:
            save_path: Path to save figure
            truths: True parameter values (if known) for comparison
        """
        import corner
        import matplotlib.pyplot as plt
        
        if self.flat_samples is None:
            raise ValueError("Must run fit() before plotting")
        
        fig = corner.corner(
            self.flat_samples,
            labels=self.param_names,
            truths=truths,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True
        )
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    
    def plot_chains(self, save_path: Optional[str] = None):
        """
        Plot MCMC chain traces for convergence diagnostics.
        
        Args:
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        
        if self.sampler is None:
            raise ValueError("Must run fit() before plotting")
        
        samples = self.sampler.get_chain()
        
        fig, axes = plt.subplots(self.n_params, figsize=(10, 2 * self.n_params), sharex=True)
        
        for i, (ax, name) in enumerate(zip(axes, self.param_names)):
            ax.plot(samples[:, :, i], alpha=0.3)
            ax.set_ylabel(name)
            ax.axvline(self.burn_in_steps, color='r', linestyle='--', label='Burn-in')
        
        axes[-1].set_xlabel("Step")
        axes[0].legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def example_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Example model function for demonstration.
    
    Args:
        x: Independent variable
        a, b, c: Model parameters
        
    Returns:
        Model predictions
    """
    return a * np.exp(-b * x) + c


if __name__ == "__main__":
    # Demonstration of MCMC parameter estimation
    np.random.seed(42)
    
    # Generate synthetic data
    x_true = np.linspace(0, 5, 50)
    true_params = [2.5, 1.3, 0.5]
    y_true = example_model(x_true, *true_params)
    
    # Add noise with known covariance
    noise_level = 0.1
    y_observed = y_true + noise_level * np.random.randn(len(x_true))
    covariance = noise_level**2 * np.eye(len(x_true))
    
    # Set up estimator
    estimator = MCMCParameterEstimator(
        n_params=3,
        param_names=['amplitude', 'decay_rate', 'offset'],
        param_bounds=[(0, 10), (0, 5), (-1, 2)],
        n_walkers=32,
        n_steps=5000,
        n_cpus=4
    )
    
    # Fit model
    results = estimator.fit(
        x_true, y_observed, covariance, example_model
    )
    
    print("\nFit Results:")
    print("-" * 40)
    for name, value in results['params'].items():
        unc = results['uncertainties'][name]
        print(f"{name}: {value:.3f} (+{unc['upper']:.3f}, -{unc['lower']:.3f})")
    
    print(f"\nAcceptance fraction: {results['acceptance_fraction']:.3f}")
    print(f"True parameters: {true_params}")
