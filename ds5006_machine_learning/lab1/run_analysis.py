"""Run all experiments for DS5006 Lab 1 report."""
import numpy as np
import matplotlib.pyplot as plt
import ds5006_lab1_data as data
import ds5006_lab1 as lab

ROLL = "142602026"  # My Roll Number

def run_sq1_experiments():
    """Run SQ1 experiments."""
    print("=" * 50)
    print("SQ1 - Gradient Descent vs Normal Equations")
    print("=" * 50)
    
    X, y = data.dataset_scaling(ROLL)
    X_std, mean, scale = lab.standardize(X)
    
    # Fit OLS
    w_ols, b_ols = lab.fit_ols_normal(X_std, y)
    print(f"OLS weights: {w_ols[:3]}...")
    print(f"OLS intercept: {b_ols:.6f}")
    
    # Max stable learning rate
    eta_max = lab.max_stable_lr(X_std)
    print(f"\neta_max = {eta_max:.6f}")
    
    # Run GD at different rates
    eta_values = [0.1, 0.5, 0.9, 0.99, 1.01, 1.5]
    histories = {}
    
    for frac in eta_values:
        eta = frac * eta_max
        print(f"Running GD at eta = {frac:.2f} * eta_max = {eta:.6f}")
        w, b, history = lab.fit_gd(X_std, y, eta, 500)
        histories[frac] = history
        print(f"  Final MSE: {history[-1]:.6f}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    for frac, hist in histories.items():
        label = f"{frac:.2f}η_max"
        if np.isfinite(hist[-1]) and hist[-1] < 1e10:
            plt.plot(hist, label=label)
        else:
            plt.plot(hist[:50], label=f"{label} (diverging)")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.yscale('log')
    plt.legend()
    plt.title("SQ1: Gradient Descent Convergence")
    plt.grid(True, alpha=0.3)
    plt.savefig("sq1_convergence.png", dpi=150)
    plt.close()
    print("Saved sq1_convergence.png")
    
    return eta_max, w_ols

def run_sq2_experiments():
    """Run SQ2 experiments."""
    print("\n" + "=" * 50)
    print("SQ2 - Feature Scaling")
    print("=" * 50)
    
    X, y = data.dataset_scaling(ROLL)
    
    # Create scaling matrix A with entries spanning orders of magnitude
    np.random.seed(42)  # for reproducibility
    a_values = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    A = np.diag(a_values)
    XA = X @ A
    
    # Fit OLS on both
    w_X, b_X = lab.fit_ols_normal(X, y)
    w_XA, b_XA = lab.fit_ols_normal(XA, y)
    
    # Compare predictions
    pred_X = lab.predict(X, w_X, b_X)
    pred_XA = lab.predict(XA, w_XA, b_XA)
    pred_diff = np.max(np.abs(pred_X - pred_XA))
    
    print(f"OLS predictions relative diff: {pred_diff/np.std(pred_X):.2e}")
    print(f"OLS coefficients relative diff: {np.linalg.norm(w_X - w_XA)/np.linalg.norm(w_X):.2e}")
    
    # Ridge comparison
    lam = 1.0
    w_ridge_X, b_ridge_X = lab.fit_ridge(X, y, lam)
    w_ridge_XA, b_ridge_XA = lab.fit_ridge(XA, y, lam)
    
    pred_ridge_X = lab.predict(X, w_ridge_X, b_ridge_X)
    pred_ridge_XA = lab.predict(XA, w_ridge_XA, b_ridge_XA)
    ridge_pred_diff = np.max(np.abs(pred_ridge_X - pred_ridge_XA))
    
    print(f"Ridge predictions relative diff: {ridge_pred_diff/np.std(pred_ridge_X):.2e}")
    
    # Standardized ridge comparison
    X_std, _, _ = lab.standardize(X)
    XA_std, _, _ = lab.standardize(XA)
    w_ridge_std, b_ridge_std = lab.fit_ridge(X_std, y, lam)
    w_ridge_XA_std, b_ridge_XA_std = lab.fit_ridge(XA_std, y, lam)
    
    pred_ridge_std = lab.predict(X_std, w_ridge_std, b_ridge_std)
    pred_ridge_XA_std = lab.predict(XA_std, w_ridge_XA_std, b_ridge_XA_std)
    ridge_std_diff = np.max(np.abs(pred_ridge_std - pred_ridge_XA_std))
    
    print(f"Standardized ridge predictions relative diff: {ridge_std_diff/np.std(pred_ridge_std):.2e}")
    
    return

def run_sq3_experiments():
    """Run SQ3 experiments."""
    print("\n" + "=" * 50)
    print("SQ3 - Numerical Stability")
    print("=" * 50)
    
    degrees = range(2, 17, 2)
    errors_normal = []
    errors_qr = []
    errors_svd = []
    cond_numbers = []
    
    for degree in degrees:
        X, y, w_true, b_true = data.dataset_conditioning(ROLL, degree)
        
        # Center the design for condition number
        Xc = X - X.mean(axis=0)
        cond = lab.condition_number(Xc)
        cond_numbers.append(cond)
        
        # Fit with each method
        w_n, b_n = lab.fit_ols_normal(X, y)
        w_q, b_q = lab.fit_ols_qr(X, y)
        w_s, b_s = lab.fit_ols_svd(X, y)
        
        # Compute relative errors
        errors_normal.append(np.linalg.norm(w_n - w_true) / np.linalg.norm(w_true))
        errors_qr.append(np.linalg.norm(w_q - w_true) / np.linalg.norm(w_true))
        errors_svd.append(np.linalg.norm(w_s - w_true) / np.linalg.norm(w_true))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.loglog(cond_numbers, errors_normal, 'o-', label='Normal equations', linewidth=2)
    plt.loglog(cond_numbers, errors_qr, 's-', label='QR', linewidth=2)
    plt.loglog(cond_numbers, errors_svd, '^-', label='SVD', linewidth=2)
    plt.xlabel("Condition number κ")
    plt.ylabel("Relative error in coefficients")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("SQ3: Numerical Stability of OLS Solvers")
    plt.savefig("sq3_stability.png", dpi=150)
    plt.close()
    print("Saved sq3_stability.png")
    
    # Print degree where normal equations fail
    for i, err in enumerate(errors_normal):
        if err >= 1.0:
            print(f"\nNormal equations fail at degree {degrees[i]}")
            print(f"  Condition number: {cond_numbers[i]:.2e}")
            print(f"  κ²ε: {cond_numbers[i]**2 * 2.2e-16:.2e}")
            break
    
    return

def run_sq4_experiments():
    """Run SQ4 experiments."""
    print("\n" + "=" * 50)
    print("SQ4 - Bias-Variance Decomposition")
    print("=" * 50)
    
    # Parameters
    sigma = 0.30
    n_train = 40
    M = 500  # Use smaller for faster execution, 500 for final report
    test_points = 101
    
    f = data.truth_function(ROLL)
    x_test = np.linspace(-0.95, 0.95, test_points)
    f_test = f(x_test)
    
    degrees = range(1, 13)
    results = []
    
    for degree in degrees:
        # Collect predictions across replicates
        preds = np.zeros((M, test_points))
        
        for rep in range(M):
            x_train, y_train = data.sample_training_set(ROLL, n_train, sigma, rep)
            X_train = lab.polynomial_design(x_train, degree)
            
            # Fit using SVD for stability
            w, b = lab.fit_ols_svd(X_train, y_train)
            
            # Predict on test points
            X_test = lab.polynomial_design(x_test, degree)
            preds[rep] = lab.predict(X_test, w, b)
        
        # Decompose
        decomposition = lab.bias_variance_decompose(preds, f_test, sigma)
        
        results.append({
            'degree': degree,
            'bias2': decomposition['mean_bias2'],
            'variance': decomposition['mean_variance'],
            'noise': decomposition['noise'],
            'expected_mse': decomposition['expected_mse']
        })
        
        print(f"Degree {degree:2d}: bias²={decomposition['mean_bias2']:.3f}, "
              f"var={decomposition['mean_variance']:.3f}, "
              f"MSE={decomposition['expected_mse']:.3f}")
    
    # Find best degree
    best_idx = np.argmin([r['expected_mse'] for r in results])
    print(f"\nBest degree: {results[best_idx]['degree']}")
    print(f"Minimum expected MSE: {results[best_idx]['expected_mse']:.6f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    degrees_plot = [r['degree'] for r in results]
    plt.plot(degrees_plot, [r['bias2'] for r in results], 'o-', label='Bias²', linewidth=2)
    plt.plot(degrees_plot, [r['variance'] for r in results], 's-', label='Variance', linewidth=2)
    plt.plot(degrees_plot, [r['noise'] for r in results], '--', label='Noise', linewidth=2)
    plt.plot(degrees_plot, [r['expected_mse'] for r in results], '^-', label='Expected MSE', linewidth=2)
    plt.xlabel("Polynomial Degree")
    plt.ylabel("MSE")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("SQ4: Bias-Variance Decomposition")
    plt.savefig("sq4_bias_variance.png", dpi=150)
    plt.close()
    print("Saved sq4_bias_variance.png")
    
    return results

def run_sq5_sq6_experiments():
    """Run SQ5 and SQ6 experiments."""
    print("\n" + "=" * 50)
    print("SQ5 and SQ6 - Ridge Regression and LOOCV")
    print("=" * 50)
    
    X, y = data.dataset_wide(ROLL)
    n, d = X.shape
    
    # Lambda grid
    lams = np.logspace(-4, 5, 100)
    
    # SQ5: Effective degrees of freedom
    dofs = lab.effective_dof(X, lams)
    
    print(f"\nSQ5 Results:")
    print(f"  d = {d} features, n = {n} observations")
    print(f"  df(λ→0) = {dofs[0]:.3f} (should be ~{d+1})")
    print(f"  df(λ→∞) = {dofs[-1]:.3f} (should be ~1)")
    
    # Find λ where df is half of max
    half_df = (d + 1) / 2
    idx_half = np.argmin(np.abs(dofs - half_df))
    lam_half = lams[idx_half]
    print(f"  λ at half df: {lam_half:.4f}")
    
    # SQ6: LOOCV
    print("\nSQ6 Results:")
    print("Computing LOOCV RMSE (naive and fast)...")
    
    # Test a few lambda values for agreement
    test_lams = np.array([1e-3, 1e-1, 1.0, 10.0, 1e3])
    for lam in test_lams:
        fast = lab.loocv_rmse_fast(X, y, lam)
        naive = lab.loocv_rmse_naive(X, y, lam)
        rel_diff = abs(fast - naive) / abs(naive)
        print(f"  λ={lam:6.1f}: fast={fast:.6f}, naive={naive:.6f}, rel_diff={rel_diff:.2e}")
    
    # Compute LOOCV across lambda grid
    loocv_errors = np.zeros(len(lams))
    for i, lam in enumerate(lams):
        loocv_errors[i] = lab.loocv_rmse_fast(X, y, lam)
    
    # Find best lambda
    best_idx = np.argmin(loocv_errors)
    lam_star = lams[best_idx]
    dof_star = dofs[best_idx]
    
    print(f"\nBest λ*: {lam_star:.6f}")
    print(f"  df(λ*): {dof_star:.3f}")
    print(f"  LOOCV RMSE: {loocv_errors[best_idx]:.6f}")
    
    # OLS LOOCV (λ→0)
    ols_rmse = lab.loocv_rmse_fast(X, y, 1e-10)
    print(f"  OLS LOOCV RMSE: {ols_rmse:.6f}")
    print(f"  Improvement: {((ols_rmse - loocv_errors[best_idx]) / ols_rmse * 100):.1f}%")
    
    # Plot LOOCV
    plt.figure(figsize=(10, 6))
    plt.semilogx(lams, loocv_errors, linewidth=2)
    plt.axvline(lam_star, color='red', linestyle='--', label=f'λ*={lam_star:.3f}')
    plt.xlabel("λ (log scale)")
    plt.ylabel("LOOCV RMSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("SQ6: LOOCV Error vs Ridge Penalty")
    plt.savefig("sq6_loocv.png", dpi=150)
    plt.close()
    print("Saved sq6_loocv.png")
    
    return lams, dofs, loocv_errors

def run_all():
    """Run all experiments."""
    print(f"Running experiments for roll number: {ROLL}")
    print("\nNote: Using M=500 for final report.")
    
    eta_max, w_ols = run_sq1_experiments()
    run_sq2_experiments()
    run_sq3_experiments()
    run_sq4_experiments()
    run_sq5_sq6_experiments()
    
    print("\n" + "=" * 50)
    print("All experiments complete!")
    print("Check the generated PNG files for plots.")
    print("=" * 50)

if __name__ == "__main__":
    run_all()
