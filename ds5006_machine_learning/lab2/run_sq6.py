import numpy as np
import ds5006_lab2 as lab
import ds5006_lab2_data as data

ROLL = "142602026"
X, y = data.dataset_multiclass(ROLL)
X_i = lab.add_intercept(X)
Y = lab.one_hot(y, 4)
p = X_i.shape[1]; K = 4

# --- Sanity check on the SQ1 dataset ---
Xm, ym = data.dataset_masking(ROLL)
Xm_i = lab.add_intercept(Xm)
Ym = lab.one_hot(ym, 3)
out_m = lab.fit_multinomial_gd(Xm_i, Ym, lam=1e-3, step=0.5,
                               max_iter=200000)
pred_m = np.argmax(Xm_i @ out_m["theta"], axis=1)
print("SQ1 dataset: multinomial accuracy =",
      lab.accuracy(ym, pred_m))

# --- Unpenalised, zero start ---
init_zero = np.zeros((p, K))
init_const = np.full((p, K), 0.4)
r0 = lab.fit_multinomial_gd(X_i, Y, lam=0.0, step=0.05,
                            theta_init=init_zero, max_iter=200000)
r1 = lab.fit_multinomial_gd(X_i, Y, lam=0.0, step=0.05,
                            theta_init=init_const, max_iter=200000)

# --- Penalised, both starts ---
r0l = lab.fit_multinomial_gd(X_i, Y, lam=0.1, step=0.05,
                             theta_init=init_zero, max_iter=200000)
r1l = lab.fit_multinomial_gd(X_i, Y, lam=0.1, step=0.05,
                             theta_init=init_const, max_iter=200000)

def compare(name_a, ra, name_b, rb):
    dtheta = ra["theta"] - rb["theta"]
    pa = lab.softmax_rows(X_i @ ra["theta"])
    pb = lab.softmax_rows(X_i @ rb["theta"])
    print(f"\n{name_a} vs {name_b}")
    print(f"  max|dTheta|        = {np.max(np.abs(dtheta)):.6f}")
    print(f"  max|dTheta| per col = "
          f"{np.max(np.abs(dtheta), axis=0)}")
    print(f"  max|P_A - P_B|     = {np.max(np.abs(pa - pb)):.3e}")
    print(f"  rowsum A           = "
          f"{ra['theta'].sum(axis=1)}")
    print(f"  rowsum B           = "
          f"{rb['theta'].sum(axis=1)}")

compare("unpen zero-start", r0, "unpen const-start", r1)
compare("pen  zero-start", r0l, "pen  const-start", r1l)
