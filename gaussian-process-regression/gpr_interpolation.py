#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Gaussian Process Regression for Multi-dimensional Interpolation
Purpose: Non-linear interpolation with uncertainty quantification using GPR
Author: Matthew Fong
Skills Demonstrated: GPR, sklearn, kernel engineering, uncertainty quantification

Industry Applications:
- Bayesian optimization for hyperparameter tuning
- Spatial interpolation for sensor networks
- Time series forecasting with confidence intervals
- Surrogate modeling for expensive simulations
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, RBF, RationalQuadratic, WhiteKernel, ConstantKernel
)
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional, Union
import warnings


class GPRInterpolator:
    """
    TECHNIQUE: Gaussian Process Regression with Custom Kernels
    INDUSTRY APPLICATION: Bayesian optimization, spatial interpolation,
    surrogate modeling for expensive simulations
    KEY SKILL: Non-parametric regression with uncertainty quantification
    
    A class for performing Gaussian Process Regression on multi-dimensional data
    with automatic kernel selection and hyperparameter optimization.
    
    Attributes:
        kernel_type: Type of kernel to use ('matern', 'rbf', 'rational_quadratic')
        n_restarts: Number of optimizer restarts for kernel hyperparameters
        normalize: Whether to normalize input features
        
    Example:
        >>> gpr = GPRInterpolator(kernel_type='matern', n_restarts=10)
        >>> gpr.fit(X_train, y_train)
        >>> y_pred, y_std = gpr.predict(X_test, return_std=True)
    """
    
    def __init__(
        self,
        kernel_type: str = 'matern',
        n_restarts: int = 10,
        normalize: bool = True,
        alpha: float = 1e-10,
        nu: float = 2.5
    ):
        """
        Initialize the GPR interpolator.
        
        Args:
            kernel_type: Kernel type ('matern', 'rbf', 'rational_quadratic', 'auto')
            n_restarts: Number of optimizer restarts
            normalize: Whether to standardize input features
            alpha: Noise level (regularization parameter)
            nu: Smoothness parameter for Matern kernel (0.5, 1.5, 2.5)
        """
        self.kernel_type = kernel_type
        self.n_restarts = n_restarts
        self.normalize = normalize
        self.alpha = alpha
        self.nu = nu
        
        self.scaler_X = StandardScaler() if normalize else None
        self.scaler_y = StandardScaler() if normalize else None
        self.gpr = None
        self.kernel = None
        self._is_fitted = False
        
    def _build_kernel(self, n_features: int):
        """
        Build the GPR kernel based on specified type.
        
        TECHNIQUE: Kernel engineering for GPR
        INDUSTRY APPLICATION: Encoding domain knowledge into model structure
        
        Args:
            n_features: Number of input features
            
        Returns:
            sklearn kernel object
        """
        # Length scale bounds for each dimension
        length_scale = np.ones(n_features)
        length_scale_bounds = [(1e-3, 1e3)] * n_features
        
        # Constant kernel for amplitude
        constant = ConstantKernel(1.0, (1e-3, 1e3))
        
        if self.kernel_type == 'matern':
            # Matern kernel - good for physical processes
            base_kernel = Matern(
                length_scale=length_scale,
                length_scale_bounds=length_scale_bounds,
                nu=self.nu
            )
        elif self.kernel_type == 'rbf':
            # RBF kernel - infinitely differentiable
            base_kernel = RBF(
                length_scale=length_scale,
                length_scale_bounds=length_scale_bounds
            )
        elif self.kernel_type == 'rational_quadratic':
            # Rational Quadratic - mixture of RBFs
            base_kernel = RationalQuadratic(
                length_scale=1.0,
                alpha=1.0,
                length_scale_bounds=(1e-3, 1e3),
                alpha_bounds=(1e-3, 1e3)
            )
        elif self.kernel_type == 'auto':
            # Composite kernel for automatic selection
            base_kernel = (
                Matern(length_scale=length_scale, nu=2.5) +
                RBF(length_scale=length_scale)
            )
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        # Add white noise kernel for numerical stability
        noise_kernel = WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
        
        self.kernel = constant * base_kernel + noise_kernel
        return self.kernel
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'GPRInterpolator':
        """
        Fit the GPR model to training data.
        
        TECHNIQUE: Maximum likelihood kernel hyperparameter optimization
        INDUSTRY APPLICATION: Automatic model complexity tuning
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
            sample_weight: Optional sample weights
            
        Returns:
            self
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        
        # Normalize data
        if self.normalize:
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            X_scaled = X
            y_scaled = y
        
        # Build kernel
        kernel = self._build_kernel(X.shape[1])
        
        # Create and fit GPR
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            alpha=self.alpha,
            normalize_y=False  # We handle normalization ourselves
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gpr.fit(X_scaled, y_scaled)
        
        self._is_fitted = True
        return self
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
        return_cov: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions with optional uncertainty estimates.
        
        TECHNIQUE: Posterior predictive distribution
        INDUSTRY APPLICATION: Decision making under uncertainty
        
        Args:
            X: Test features of shape (n_samples, n_features)
            return_std: Return standard deviation of predictions
            return_cov: Return full covariance matrix
            
        Returns:
            Predictions, and optionally uncertainty estimates
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.atleast_2d(X)
        
        # Normalize input
        if self.normalize:
            X_scaled = self.scaler_X.transform(X)
        else:
            X_scaled = X
        
        # Get predictions
        if return_cov:
            y_pred, cov = self.gpr.predict(X_scaled, return_cov=True)
        elif return_std:
            y_pred, std = self.gpr.predict(X_scaled, return_std=True)
        else:
            y_pred = self.gpr.predict(X_scaled)
        
        # Inverse transform predictions
        if self.normalize:
            y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            if return_std:
                std = std * self.scaler_y.scale_[0]
            if return_cov:
                cov = cov * (self.scaler_y.scale_[0] ** 2)
        
        if return_cov:
            return y_pred, cov
        elif return_std:
            return y_pred, std
        else:
            return y_pred
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score on test data.
        
        Args:
            X: Test features
            y: True targets
            
        Returns:
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_kernel_params(self) -> Dict:
        """
        Get optimized kernel hyperparameters.
        
        Returns:
            Dictionary of kernel parameters
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        return {
            'kernel': str(self.gpr.kernel_),
            'log_marginal_likelihood': self.gpr.log_marginal_likelihood_value_
        }


class GPRGridInterpolator:
    """
    TECHNIQUE: GPR for 2D Grid Interpolation
    INDUSTRY APPLICATION: Spatial data analysis, image interpolation,
    geographic information systems
    KEY SKILL: Efficient interpolation on regular grids
    
    Specialized GPR interpolator for 2D grid data with efficient
    prediction on regular grids.
    """
    
    def __init__(
        self,
        kernel_type: str = 'matern',
        n_restarts: int = 5
    ):
        """
        Initialize the grid interpolator.
        
        Args:
            kernel_type: Type of kernel to use
            n_restarts: Number of optimizer restarts
        """
        self.gpr = GPRInterpolator(
            kernel_type=kernel_type,
            n_restarts=n_restarts
        )
        self.x1_range = None
        self.x2_range = None
        
    def fit(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        values: np.ndarray
    ) -> 'GPRGridInterpolator':
        """
        Fit GPR to 2D scattered data.
        
        Args:
            x1: First coordinate values
            x2: Second coordinate values
            values: Target values at (x1, x2) points
            
        Returns:
            self
        """
        X = np.column_stack([x1, x2])
        self.gpr.fit(X, values)
        
        self.x1_range = (x1.min(), x1.max())
        self.x2_range = (x2.min(), x2.max())
        
        return self
    
    def predict_grid(
        self,
        x1_grid: np.ndarray,
        x2_grid: np.ndarray,
        return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict on a regular 2D grid.
        
        TECHNIQUE: Efficient grid prediction
        INDUSTRY APPLICATION: Heatmap generation, spatial visualization
        
        Args:
            x1_grid: 1D array of x1 coordinates
            x2_grid: 1D array of x2 coordinates
            return_std: Return uncertainty grid
            
        Returns:
            2D array of predictions (and optionally uncertainties)
        """
        # Create meshgrid
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        X_flat = np.column_stack([X1.ravel(), X2.ravel()])
        
        # Predict
        if return_std:
            y_pred, y_std = self.gpr.predict(X_flat, return_std=True)
            return y_pred.reshape(X1.shape), y_std.reshape(X1.shape)
        else:
            y_pred = self.gpr.predict(X_flat)
            return y_pred.reshape(X1.shape)
    
    def plot_interpolation(
        self,
        x1_grid: np.ndarray,
        x2_grid: np.ndarray,
        x1_train: Optional[np.ndarray] = None,
        x2_train: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """
        Create visualization of GPR interpolation.
        
        Args:
            x1_grid: Grid x1 coordinates
            x2_grid: Grid x2 coordinates
            x1_train: Training x1 coordinates (optional)
            x2_train: Training x2 coordinates (optional)
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        
        # Get predictions with uncertainty
        z_pred, z_std = self.predict_grid(x1_grid, x2_grid, return_std=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Prediction plot
        im1 = axes[0].contourf(x1_grid, x2_grid, z_pred, levels=50, cmap='viridis')
        axes[0].set_title('GPR Prediction')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        plt.colorbar(im1, ax=axes[0])
        
        if x1_train is not None and x2_train is not None:
            axes[0].scatter(x1_train, x2_train, c='red', s=20, alpha=0.5, label='Training data')
            axes[0].legend()
        
        # Uncertainty plot
        im2 = axes[1].contourf(x1_grid, x2_grid, z_std, levels=50, cmap='plasma')
        axes[1].set_title('Prediction Uncertainty (1σ)')
        axes[1].set_xlabel('Feature 1')
        axes[1].set_ylabel('Feature 2')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    # Demonstration of GPR interpolation
    np.random.seed(42)
    
    # Generate synthetic 2D data
    n_train = 100
    x1_train = np.random.uniform(0, 10, n_train)
    x2_train = np.random.uniform(0, 10, n_train)
    
    # True function: 2D sinusoidal
    def true_function(x1, x2):
        return np.sin(x1) * np.cos(x2) + 0.5 * x1 - 0.3 * x2
    
    y_train = true_function(x1_train, x2_train) + 0.1 * np.random.randn(n_train)
    
    # Fit GPR
    print("Fitting GPR model...")
    gpr = GPRGridInterpolator(kernel_type='matern', n_restarts=5)
    gpr.fit(x1_train, x2_train, y_train)
    
    # Create prediction grid
    x1_grid = np.linspace(0, 10, 50)
    x2_grid = np.linspace(0, 10, 50)
    
    # Predict
    z_pred, z_std = gpr.predict_grid(x1_grid, x2_grid, return_std=True)
    
    print(f"\nKernel parameters: {gpr.gpr.get_kernel_params()}")
    print(f"Prediction range: [{z_pred.min():.3f}, {z_pred.max():.3f}]")
    print(f"Mean uncertainty: {z_std.mean():.3f}")
    
    # Compute test score
    X1_test, X2_test = np.meshgrid(x1_grid, x2_grid)
    y_true = true_function(X1_test, X2_test)
    
    mse = np.mean((z_pred - y_true) ** 2)
    print(f"Test MSE: {mse:.4f}")
