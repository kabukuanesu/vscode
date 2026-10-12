import numpy as np
import ds5006_lab2 as lab
import ds5006_lab2_data as data

ROLL = 142602026
X_raw, y = data.dataset_scaling(ROLL)
X_std = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)

X_raw_i = lab.add_intercept(X_raw)
X_std_i = lab.add_intercept(X_std)

# --- Newton, lam = 0 ---
n_raw = lab.fit_logistic_newton(X_raw_i, y, lam=0.0, tol=1e-10, max_iter=200)
n_std = lab.fit_logistic_newton(X_std_i, y, lam=0.0, tol=1e-10, max_iter=200)
p_raw_n = lab.sigmoid(X_raw_i @ n_raw["theta"])
p_std_n = lab.sigmoid(X_std_i @ n_std["theta"])
print("Newton | raw iter =", n_raw["n_iter"], "std iter =", n_std["n_iter"])
print("Newton | max |p_raw - p_std| =", np.max(np.abs(p_raw_n - p_std_n)))

# --- Gradient descent, lam = 1e-3, step = 0.9 * bound ---
lam_gd = 1e-3
step_raw = 0.9 * lab.gd_step_bound(X_raw_i, lam_gd)
step_std = 0.9 * lab.gd_step_bound(X_std_i, lam_gd)
print("step_raw =", step_raw, "step_std =", step_std)

g_raw = lab.fit_logistic_gd(X_raw_i, y, lam_gd, step_raw, tol=1e-10, max_iter=200000)
g_std = lab.fit_logistic_gd(X_std_i, y, lam_gd, step_std, tol=1e-10, max_iter=200000)
p_raw_g = lab.sigmoid(X_raw_i @ g_raw["theta"])
p_std_g = lab.sigmoid(X_std_i @ g_std["theta"])
print("GD     | raw iter =", g_raw["n_iter"], "std iter =", g_std["n_iter"])
print("GD     | max |p_raw - p_std| =", np.max(np.abs(p_raw_g - p_std_g)))

# --- Newton with lam = 1e-3, raw vs standardised ---
lam_n = 1e-3
n_raw_lam = lab.fit_logistic_newton(X_raw_i, y, lam_n, tol=1e-10, max_iter=200)
n_std_lam = lab.fit_logistic_newton(X_std_i, y, lam_n, tol=1e-10, max_iter=200)
p_raw_n_lam = lab.sigmoid(X_raw_i @ n_raw_lam["theta"])
p_std_n_lam = lab.sigmoid(X_std_i @ n_std_lam["theta"])
print("Newton lam=1e-3 | raw iter =", n_raw_lam["n_iter"], "std iter =", n_std_lam["n_iter"])
print("Newton lam=1e-3 | max |p_raw - p_std| =",
      np.max(np.abs(p_raw_n_lam - p_std_n_lam)))
