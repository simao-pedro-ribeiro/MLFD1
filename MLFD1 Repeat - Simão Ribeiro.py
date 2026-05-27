"""
Created on May 26 2026

@author: Simão Ribeiro
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from typing import Tuple, List

#%% 1. LOAD CFD DATA

def load_cfd_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the CFD simulation data for eVTOL wing download force

    Returns
    -------
    dD : np.ndarray, shape (n_samples,)
        Nondimensional horizontal distances d/D
    zD : np.ndarray, shape (n_samples,)
        Nondimensional vertical distances z/D
    force : np.ndarray, shape (n_samples,)
        Wing download force [N] from CFD
    """
    dD = np.array([0.25, 0.25, 0.25, 0.375, 0.375, 0.375, 0.50, 0.50, 0.50])
    zD = np.array([0.50, 0.75, 1.00, 0.50, 0.75, 1.00, 0.50, 0.75, 1.00])
    force = np.array([-9.5, -9.8, -11.0, -2.4, -3.7, -4.7, -1.3, -1.5, -1.5])
    return dD, zD, force

#%% 2. BUILD DESIGN MATRIX

def gaussian_kernel(r: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian RBF kernel

    Parameters
    ----------
    r : np.ndarray, shape (n_samples, n_centers)
        Pairwise Euclidean distances
    sigma : float
        Length scale parameter

    Returns
    -------
    np.ndarray
        Kernel values: exp(-r^2 / σ^2), shape (n_samples, n_centers)
    """
    return np.exp(-(r ** 2) / (sigma ** 2))
    
def matern_c4_kernel(r: np.ndarray, sigma: float) -> np.ndarray:
    """
    Matérn C4 RBF kernel

    Parameters
    ----------
    r : np.ndarray, shape (n_samples, n_centers)
        Pairwise Euclidean distances
    sigma : float
        Length scale parameter

    Returns
    -------
    np.ndarray
        Kernel values: exp(-r/σ) * ( (r/σ)^2 + 3(r/σ) + 3 ), shape (n_samples, n_centers)
    """
    return np.exp(-r / sigma) * ((r / sigma) ** 2 + 3 * (r / sigma) + 3)
    
def compact_c4_kernel(r: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compact C4 RBF kernel

    Parameters
    ----------
    r : np.ndarray, shape (n_samples, n_centers)
        Pairwise Euclidean distances
    sigma : float
        Length scale parameter

    Returns
    -------
    np.ndarray
        Kernel values: (1 + r/σ)^5 * (1 - r/σ)^5 for r ≤ σ, and 0 for r > σ, shape (n_samples, n_centers)
    """
    return np.where(
        r <= sigma,
        ((1 + r / sigma) ** 5) * ((1 - r / sigma) ** 5),0.0)

_KERNEL_FUNCTIONS = {
    "Gaussian": gaussian_kernel,
    "Matérn": matern_c4_kernel,
    "Compact": compact_c4_kernel
}

def build_design_matrix(X: np.ndarray, centers: np.ndarray, sigma: float, kernel: str) -> np.ndarray:
    """
    Build the design matrix Phi = [phi_1, phi_2, ..., phi_n_centers]

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 2)
        Scaled input coordinates
    centers : np.ndarray, shape (n_centers, 2)
        Scaled RBF center positions
    sigma : float
        RBF length scale
    kernel : {'Gaussian', 'Matérn', 'Compact'}
        Type of RBF kernel

    Returns
    -------
    Phi : np.ndarray, shape (n_samples, n_centers)
        Design matrix where Phi[i, j] = phi_j(X[i])

    Note
    ----
    cdist used to compute pairwise Euclidean distances, where r[i, j] = ||X[i] - centers[j]||

    Raises
    ------
    ValueError
        If sigma is negative
        If kernel type is unknown
    """
      
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    if kernel not in _KERNEL_FUNCTIONS:
        valid = ", ".join(_KERNEL_FUNCTIONS.keys())
        raise ValueError(f"Unknown kernel '{kernel}'. Valid kernels: {valid}")
    
    # Compute pairwise Euclidean distances
    r = cdist(X, centers, metric='euclidean')   # shape (n_samples, n_centers)
    
    kernel_function = _KERNEL_FUNCTIONS[kernel]
    Phi = kernel_function(r, sigma)
    
    return Phi
      
#%% 3. HYPERPARAMETER TUNING USING K-FOLD CV

def grid_search_rbf(
    X: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    kernels: List[str],
    sigmas: np.ndarray,
    alphas: np.ndarray,
    n_folds: int = 3,
    random_seed: int = 1
    ) -> Tuple[float, float, float, str, Ridge, float]:
    """
    Perform grid search over kernels, sigma, alpha using K-fold cross-validation

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 2)
        Scaled input coordinates
    y : np.ndarray, shape (n_samples,)
        Target values (force)
    centers : np.ndarray, shape (n_centers, 2)
        Scaled RBF center positions
    kernels : List[str]
        List of kernel names to test
    sigmas : np.ndarray
        Array of sigma values to test
    alphas : np.ndarray
        Array of alpha values to test
    n_folds : int, default=3
        Number of folds for cross-validation
    random_seed : int, default=1
        Seed for KFold shuffling

    Returns
    -------
    best_rmse : float
        Lowest average RMSE across folds
    best_sigma : float
        Corresponding sigma
    best_alpha : float
        Corresponding alpha
    best_kernel : str
        Corresponding kernel name
    best_model : Ridge
        Trained Ridge model on all data with best hyperparameters
    best_condition : float
        Condition number of Hessian matrix for the best configuration

    Note
    -----
    Condition number of the regularised Hessian (H_reg) computed to skip
    numerically unstable configurations (cond > 1e6). Even with Ridge, extreme
    condition numbers can lead to unreliable weight estimates.
    """
    # INITIALIZE CROSS-VALIDATOR
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    # INITIALIZE BEST MODEL TRACKERS
    best_rmse = np.inf
    best_sigma = None
    best_alpha = None
    best_kernel = None
    best_model = None
    best_condition = np.inf

    # LOOP OVER ALL HYPERPARAMETER COMBINATIONS
    for kernel in kernels:
        for sigma in sigmas:
            for alpha in alphas:
                
                # INITIALIZE FOLD METRICS
                fold_errors = []
                fold_conditions = []
                invalid_config = False

                for train_idx, test_idx in kfold.split(X):
                    
                    # Split data into training and test folds
                    X_train = X[train_idx]
                    X_test = X[test_idx]
                    y_train = y[train_idx]
                    y_test = y[test_idx]

                    # Build design matrices for this fold
                    Phi_train = build_design_matrix(X_train, centers, sigma, kernel)
                    Phi_test = build_design_matrix(X_test, centers, sigma, kernel)

                    # Regularized Hessian and condition number
                    H_reg = Phi_train.T @ Phi_train + alpha * np.eye(Phi_train.shape[1])
                    cond_H = np.linalg.cond(H_reg)

                    # Skip if numerically unstable
                    if np.isnan(cond_H) or np.isinf(cond_H) or cond_H > 1e6:
                        invalid_config = True
                        break

                    fold_conditions.append(cond_H)

                    # Train Ridge model and evaluate
                    model = Ridge(alpha=alpha, fit_intercept=False)
                    model.fit(Phi_train, y_train)
                    y_pred = model.predict(Phi_test)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    fold_errors.append(rmse)

                # SKIP UNSTABLE CONFIGURATIONS
                if invalid_config or len(fold_errors) != n_folds:
                    continue

                # AVERAGE METRICS OVER FOLDS
                avg_rmse = np.mean(fold_errors)
                avg_cond = np.mean(fold_conditions)

                # UPDATE BEST MODEL (RMSE first, then condition number)
                if avg_rmse < best_rmse or (np.isclose(avg_rmse, best_rmse, rtol=1e-3) and avg_cond < best_condition):
                    best_rmse = avg_rmse
                    best_sigma = sigma
                    best_alpha = alpha
                    best_kernel = kernel
                    best_condition = avg_cond

    # FINAL TRAINING ON FULL DATASET
    if best_kernel is None:
        raise RuntimeError("No valid hyperparameter configuration found.")

    Phi_full = build_design_matrix(X, centers, best_sigma, best_kernel)
    best_model = Ridge(alpha=best_alpha, fit_intercept=False)
    best_model.fit(Phi_full, y)

    return (best_rmse, best_sigma, best_alpha, best_kernel, best_model, best_condition)

#%% 4. PREDICTION ON GRID AND VISUALISATION

def create_prediction_grid(
    d_range: Tuple[float, float] = (0.25, 0.50),
    z_range: Tuple[float, float] = (0.50, 1.00),
    n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a regular 2D grid of points for prediction

    Parameters
    ----------
    d_range : tuple (d_min, d_max)
        Horizontal distance range
    z_range : tuple (z_min, z_max)
        Vertical distance range
    n_points : int
        Number of points along each dimension

    Returns
    -------
    D_grid : np.ndarray, shape (n_points, n_points)
        2D grid of horizontal distances
    Z_grid : np.ndarray, shape (n_points, n_points)
        2D grid of vertical distances
    X_plot : np.ndarray, shape (n_points^2, 2)
        Flattened grid points (unscaled) for prediction
    """
    # GENERATE 1D COORDINATES
    d_vals = np.linspace(d_range[0], d_range[1], n_points)
    z_vals = np.linspace(z_range[0], z_range[1], n_points)
    
    # CREATE 2D MESHGRIDS
    D_grid, Z_grid = np.meshgrid(d_vals, z_vals)
    
    # FLATTEN FOR PREDICTION INPUT
    X_plot = np.column_stack((D_grid.ravel(), Z_grid.ravel()))
    
    return D_grid, Z_grid, X_plot

def plot_force_map(
    D_grid: np.ndarray,
    Z_grid: np.ndarray,
    force_map: np.ndarray,
    dD_train: np.ndarray,
    zD_train: np.ndarray,
    force_train: np.ndarray,
    kernel_name: str,
    save_path: str = 'RBF_Regression_V2.png'
    ) -> None:
    """
    Plot the predicted force contour map over the design space.

    Parameters
    ----------
    D_grid, Z_grid : np.ndarray, shape (n_points, n_points)
        Grid coordinates
    force_map : np.ndarray, shape (n_points, n_points)
        Predicted force values on the grid
    dD_train, zD_train : np.ndarray, shape (n_samples,)
        Training data coordinates
    force_train : np.ndarray, shape (n_samples,)
        Training force values
    kernel_name : str
        Name of the best kernel
    save_path : str
        File path to save the figure
    """
    plt.rcParams.update({
        'font.size': 20,
        'legend.fontsize': 18,
        'axes.labelsize': 20,
        'axes.titlesize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'text.usetex': False
    })

    plt.figure(figsize=(10, 8))
    plt.title(f'Best RBF Regression ({kernel_name})')
    contour = plt.contourf(D_grid, Z_grid, force_map, np.linspace(-12, 0, 25), cmap='viridis_r')
    plt.scatter(dD_train, zD_train, c=force_train,
                s=250, vmin=-12, vmax=0, cmap='viridis_r',
                edgecolors='red', linewidth=2)
    plt.xlabel('Horizontal Distance $d/D$')
    plt.ylabel('Vertical Distance $z/D$')
    plt.xlim(0.20, 0.55)
    plt.ylim(0.45, 1.05)
    plt.xticks([0.25, 0.375, 0.50])
    plt.yticks([0.50, 0.75, 1.00])
    cbar = plt.colorbar(contour)
    cbar.set_label('Wing Download Force [N]')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

#%% 5. MAIN WORKFLOW

def main():
    """
    Main workflow: load data, scale, hyperparameter search, predict, plot.
    """
    # LOAD CFD DATA
    dD, zD, force = load_cfd_data()
    X_raw = np.column_stack((dD, zD))                     # shape (n_samples,2)

    # SCALING
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)                # shape (n_samples,2)

    # DEFINE RBF CENTERS
    centers_raw = np.array([
        [0.25, 0.50],   # bottom-left
        [0.25, 1.00],   # top-left
        [0.375, 0.75],  # center of the grid
        [0.50, 0.50],   # bottom-right
        [0.50, 1.00]    # top-right
    ])
    
    # TRANSFORM CENTERS
    centers_scaled = scaler.transform(centers_raw)        # shape (n_centers,2)

    # DEFINE HYPERPARAMETER RANGES
    kernels = ['Gaussian', 'Matérn', 'Compact']
    sigmas = np.linspace(0.1, 10, 50)                    # 50 values
    alphas = np.logspace(-6, -1, 50)                     # 50 log-spaced values

    # GRID SEARCH WITH 3‑FOLD CV
    (best_rmse, best_sigma, best_alpha,
     best_kernel, best_model, best_cond) = grid_search_rbf(
         X_scaled, force, centers_scaled, kernels, sigmas, alphas, n_folds=3, random_seed=1)

    # REPORT RESULTS
    print('=' * 50)
    print('BEST MODEL')
    print('=' * 50)
    print(f'Kernel            : {best_kernel}')
    print(f'Sigma             : {best_sigma:.4f}')
    print(f'Alpha             : {best_alpha:.4e}')
    print(f'Average RMSE      : {best_rmse:.4f}')
    print(f'Condition Number  : {best_cond:.4e}')
    print('=' * 50)

    # CREATE FINE PREDICTION GRID
    D_grid, Z_grid, X_plot_raw = create_prediction_grid(
        d_range=(0.25, 0.50), z_range=(0.50, 1.00), n_points=100)
    
    # SCALE THE PREDICTION GRID
    X_plot_scaled = scaler.transform(X_plot_raw)          # shape (n_points^2,2)

    # BUILD DESIGN MATRIX FOR PREDICTION
    Phi_plot = build_design_matrix(X_plot_scaled, centers_scaled,
                                   best_sigma, best_kernel) # shape (n_points^2,n_centers)

    # PREDICT FORCE ON THE GRID
    force_map = best_model.predict(Phi_plot)              # shape (n_points^2,)
    force_map = force_map.reshape(D_grid.shape)           # shape (n_points,n_points)

    # PLOT AND SAVE
    plot_force_map(D_grid, Z_grid, force_map, dD, zD, force, best_kernel,
                   save_path='RBF_Regression_V2.png')
    
if __name__ == "__main__":
    main()