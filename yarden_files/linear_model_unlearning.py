import numpy as np
from scipy.special import expit
import time
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import os
import urllib.request


class BaseLinearRegression:
    def __init__(self, X, y):
        self.beta, self.XTX_inv = self.compute_exact_solution(X, y)

    def compute_exact_solution(self, X, y):
        """Compute linear regression solution using normal equations."""
        start = time.perf_counter()
        XTX = X.T @ X
        
        # Add regularization to prevent singular matrix issues
        # Check condition number to detect numerical instability
        try:
            cond_num = np.linalg.cond(XTX)
            if cond_num > 1e10 or np.isnan(cond_num) or np.isinf(cond_num):
                # Matrix is ill-conditioned, add small regularization
                ridge_param = 1e-6 * np.trace(XTX) / XTX.shape[0]
                XTX = XTX + ridge_param * np.eye(XTX.shape[0])
        except:
            # If condition number computation fails, add regularization
            ridge_param = 1e-6 * np.trace(XTX) / XTX.shape[0]
            XTX = XTX + ridge_param * np.eye(XTX.shape[0])
        
        # Suppress warnings for divide by zero (we've handled it above)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            try:
                XTX_inv = np.linalg.inv(XTX)
                beta_hat = XTX_inv @ X.T @ y
                
                # Check for NaN or Inf in results
                if np.any(np.isnan(beta_hat)) or np.any(np.isinf(beta_hat)):
                    # Fall back to pseudo-inverse
                    XTX_inv = np.linalg.pinv(XTX)
                    beta_hat = XTX_inv @ X.T @ y
            except np.linalg.LinAlgError:
                # If inversion fails, use pseudo-inverse
                XTX_inv = np.linalg.pinv(XTX)
                beta_hat = XTX_inv @ X.T @ y
        
        return beta_hat, XTX_inv

    def woodbury_update(self, X_remove):
        """
        Update the inverse matrix after removing multiple rows using the Woodbury matrix identity.
        X_remove: The rows to be removed.
        XTX_inv: The precomputed inverse of X^T X before removal.
        """
        woodbury_matrix = np.linalg.inv(np.eye(X_remove.shape[0]) - X_remove @ self.XTX_inv @ X_remove.T)
        XTX_inv_updated = self.XTX_inv + self.XTX_inv @ X_remove.T @ woodbury_matrix @ X_remove @ self.XTX_inv
        return XTX_inv_updated

    def neumann_update(self, X_remove):
        """
        Update the inverse matrix after removing multiple rows using the truncated Neumann series.
        X_remove: The rows to be removed.
        XTX_inv: The precomputed inverse of X^T X before removal.
        """
        XTX_inv_updated = self.XTX_inv + self.XTX_inv @ X_remove.T @ X_remove @ self.XTX_inv
        return XTX_inv_updated


class BaseLogisticRegression:
    def __init__(self, max_iter=100, tol=1e-4, C=1.0, fit_intercept=False):
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.theta = None
        self.converged = False
        self.n_iter = 0
    
    def sigmoid(self, z):
        return expit(z)
    
    def _compute_gradient(self, X, y, theta):
        m = len(y)
        h = self.sigmoid(X @ theta)
        
        gradient = X.T @ (h - y) / m
        if self.fit_intercept:
            reg_term = np.zeros_like(theta)
            reg_term[1:] = theta[1:] / self.C
            return gradient + reg_term / m
        else:
            return gradient + theta / (self.C * m)
    
    def _compute_hessian(self, X, theta):
        m = len(X)
        h = self.sigmoid(X @ theta)
        S = np.diag(h * (1 - h))
        
        hessian = X.T @ S @ X / m
        
        if self.fit_intercept:
            reg_matrix = np.eye(X.shape[1]) / self.C
            reg_matrix[0, 0] = 0
            return hessian + reg_matrix / m
        else:
            return hessian + np.eye(X.shape[1]) / (self.C * m)
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        m, n = X.shape
        self.theta = np.zeros(n)
        self.converged = False
        self.n_iter = 0
        
        for iteration in range(self.max_iter):
            gradient = self._compute_gradient(X, y, self.theta)
            hessian = self._compute_hessian(X, self.theta)
            
            try:
                delta = np.linalg.solve(hessian, gradient)
                self.theta -= delta
                self.n_iter = iteration + 1
                
                if np.linalg.norm(delta) < self.tol:
                    self.converged = True
                    break
            except np.linalg.LinAlgError:
                print(f"Warning: Singular Hessian encountered at iteration {iteration}.")
                break
        
        if self.fit_intercept:
            self.intercept_ = self.theta[0]
            self.coef_ = self.theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.theta
        
        return self
    
    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        probabilities = self.sigmoid(X @ self.theta)
        return (probabilities >= 0.5).astype(int)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


class FastUnlearningLogisticRegression(BaseLogisticRegression):
    def __init__(self, max_iter: int = 100, tol: float = 1e-4, lambda_reg: float = 0.1):
        super().__init__(max_iter, tol, lambda_reg)
        self.cached_hessian: Optional[np.ndarray] = None

    def fit_incremental(self, X: np.ndarray, y: np.ndarray, X_keep: np.ndarray, y_keep: np.ndarray) -> 'FastUnlearningLogisticRegression':
        if self.theta is None:
            raise ValueError("Model must be initialized with `fit` before using incremental updates.")

        self.cached_hessian = self._compute_hessian(X_keep, self.theta)
        self.converged = False
        self.n_iter = 0

        for iteration in range(self.max_iter):
            gradient = self._compute_gradient(X_keep, y_keep, self.theta)
            try:
                delta = np.linalg.solve(self.cached_hessian, gradient)
                self.theta -= delta
                self.n_iter = iteration + 1

                if np.linalg.norm(delta) < self.tol:
                    self.converged = True
                    break
            except np.linalg.LinAlgError:
                print(f"Warning: Singular Hessian encountered at iteration {iteration}.")
                break

        return self

    def fit_influence_function(self, X: np.ndarray, y: np.ndarray, X_remove: np.ndarray, y_remove: np.ndarray) -> 'FastUnlearningLogisticRegression':
        if self.theta is None:
            raise ValueError("Model must be initialized with `fit` before using influence functions.")

        try:
            m = len(X)
            hessian = self._compute_hessian(X, self.theta)
            hessian_inv = np.linalg.inv(hessian)

            # First-order update
            total_influence = np.zeros_like(self.theta)
            for x_remove, y_remove_i in zip(X_remove, y_remove):
                x_remove = x_remove.reshape(1, -1)
                h_remove = self.sigmoid(x_remove @ self.theta)
                grad_remove = (x_remove.T * (h_remove - y_remove_i) + self.lambda_reg * self.theta.reshape(-1, 1)) / m
                influence = hessian_inv @ grad_remove
                total_influence += influence.flatten()

            # Second-order correction
            theta_interim = self.theta - total_influence
            hessian_removed = self._compute_hessian(X[~np.isin(np.arange(len(X)), range(len(X_remove)))], theta_interim)
            correction = hessian_inv @ (hessian - hessian_removed) @ total_influence

            self.theta = theta_interim - correction
            self.converged = True
            self.n_iter = 1

        except np.linalg.LinAlgError:
            print("Warning: Singular Hessian encountered in influence function calculation.")
            self.converged = False
            self.n_iter = 0

        return self


class CertifiableUnlearningLogisticRegression(BaseLogisticRegression):
    def __init__(self, max_iter: int = 100, tol: float = 1e-4, lambda_reg: float = 0.1):
        super().__init__(max_iter, tol, lambda_reg)
        self.cached_hessian: Optional[np.ndarray] = None

    def fit_incremental_mini_batch(self, X_keep: np.ndarray, y_keep: np.ndarray, X_remove: np.ndarray, y_remove: np.ndarray, sigma: float, batch_size: int) -> np.ndarray:
        if self.theta is None:
            raise ValueError("Model must be initialized with `fit` before using incremental updates.")
        num_batches = int(np.ceil(len(X_remove) / batch_size))
        X_train = np.concatenate((X_remove,X_keep))
        y_train = np.concatenate((y_remove,y_keep))

        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(X_remove))
            X_keep_batch = X_train[batch_end:]
            y_keep_batch = y_train[batch_end:]

            # Compute gradient and Hessian for the current batch
            gradient = self._compute_gradient(X_keep_batch, y_keep_batch, self.theta)
            self.cached_hessian = self._compute_hessian(X_keep_batch, self.theta)
            # Newton correction
            try:
                delta = np.linalg.solve(self.cached_hessian, gradient)
                self.theta -= delta

                # Noise injection
                if sigma > 0:
                    eigvals, eigvecs = np.linalg.eigh(self.cached_hessian)
                    eigvals_inv_4 = np.diag(eigvals ** -0.25)
                    hessian_inv_4 = eigvecs @ eigvals_inv_4 @ eigvecs.T
                    b = np.random.normal(0, 1, size=self.theta.shape)
                    # noise = np.linalg.solve(self.cached_hessian, b)
                    noise = hessian_inv_4 @ b
                    self.theta += sigma * noise
            except np.linalg.LinAlgError:
                print(f"Warning: Singular Hessian encountered in batch {i}. Skipping noise injection for this batch.")
                continue

        return self


class SegmentTestLogisticRegression(BaseLogisticRegression):
    def __init__(self, max_iter: int = 100, tol: float = 1e-4, lambda_reg: float = 0.1):
        super().__init__(max_iter, tol, lambda_reg)
        self.lower_bounds = None
        self.upper_bounds = None
        
    def compute_delta_s(self, X: np.ndarray, y: np.ndarray, X_remove: np.ndarray, y_remove: np.ndarray) -> np.ndarray:
        gradients_removed = []
        
        for i in range(len(X_remove)):
            x_i = X_remove[i].reshape(1, -1)
            y_i = np.array([y_remove[i]])
            gradients_removed.append(self._compute_gradient(x_i, y_i, self.theta))
        
        delta_s = np.zeros_like(self.theta)
        for grad in gradients_removed:
            delta_s -= grad
            
        nS = len(y_remove)
        if nS > 0:
            delta_s /= nS
            
        return delta_s
        
    def compute_delta_L(self, X: np.ndarray, y: np.ndarray, X_remove: np.ndarray, y_remove: np.ndarray) -> np.ndarray:
        gradients_removed = []
        
        for i in range(len(X_remove)):
            x_i = X_remove[i].reshape(1, -1)
            y_i = np.array([y_remove[i]])
            gradients_removed.append(self._compute_gradient(x_i, y_i, self.theta))
        
        delta_L = np.zeros_like(self.theta)
        for grad in gradients_removed:
            delta_L += grad
            
        return delta_L
        
    def estimate_bounds(self, X: np.ndarray, y: np.ndarray, X_remove: np.ndarray, y_remove: np.ndarray) -> tuple:
        if self.theta is None:
            raise ValueError("Model must be initialized with `fit` before estimating bounds")
        
        n0 = len(X)
        nS = len(X_remove)
        n1 = n0 - nS
        
        delta_s = self.compute_delta_s(X, y, X_remove, y_remove)
        delta_L = self.compute_delta_L(X, y, X_remove, y_remove)
        
        q = (n0 + n1) / (2 * n1) * self.theta - (nS) / (2 * self.lambda_reg * n1) * delta_s
        r_vector = (-nS) / (2 * n1) * self.theta + (nS) / (2 * self.lambda_reg * n1) * delta_s
        r = np.linalg.norm(r_vector)
        
        n_vector = (-(nS) / n1 * delta_s + (1 / n1) * delta_L)
        n_norm = np.linalg.norm(n_vector)
        if n_norm > 0:
            n_vector = n_vector / n_norm
        
        c = np.dot(n_vector, (n0 / n1) * self.theta - (nS) / (self.lambda_reg * n1) * delta_s)
        
        vector_in_numerator = r_vector
        vector_norm = np.linalg.norm(vector_in_numerator)
        if vector_norm > 0:
            psi = np.dot(n_vector, vector_in_numerator) / vector_norm
        else:
            psi = 0
        
        lower_bounds = np.zeros_like(self.theta)
        upper_bounds = np.zeros_like(self.theta)
        
        for j in range(len(self.theta)):
            e_j = np.zeros_like(self.theta)
            e_j[j] = 1.0
            
            t = np.dot(n_vector, e_j)
            
            common_term = (n0 + n1) / (2 * n1) * self.theta[j] - (nS) / (2 * self.lambda_reg * n1) * delta_s[j]
            
            if t > psi:
                lower_bounds[j] = common_term - r
            else:
                if abs(1 - psi**2) < 1e-10 or abs(1 - t**2) < 1e-10:
                    lower_bounds[j] = common_term - r
                else:
                    lower_bounds[j] = common_term - psi * r * t - r * np.sqrt(1 - psi**2) * np.sqrt(1 - t**2)
            
            if t < -psi:
                upper_bounds[j] = common_term + r
            else:
                if abs(1 - psi**2) < 1e-10 or abs(1 - t**2) < 1e-10:
                    upper_bounds[j] = common_term + r
                else:
                    upper_bounds[j] = common_term - psi * r * t + r * np.sqrt(1 - psi**2) * np.sqrt(1 - t**2)
        
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        
        return lower_bounds, upper_bounds
    
    def fit_segment_test(self, X: np.ndarray, y: np.ndarray, X_remove: np.ndarray, y_remove: np.ndarray) -> 'SegmentTestLogisticRegression':
        if self.theta is None:
            raise ValueError("Model must be initialized with `fit` before unlearning")
        
        self.estimate_bounds(X, y, X_remove, y_remove)
        self.theta = (self.lower_bounds + self.upper_bounds) / 2
        self.converged = True
        self.n_iter = 1
        
        return self
    
    def get_bounds_tightness(self) -> float:
        if self.lower_bounds is None or self.upper_bounds is None:
            raise ValueError("Bounds have not been estimated yet")
        
        return np.mean(self.upper_bounds - self.lower_bounds)