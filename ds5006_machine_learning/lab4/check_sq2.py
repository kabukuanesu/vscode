import numpy as np

from svmlab_provided import (
    make_dataset,
    C_DEFAULT, S2_DEFAULT,
    C_GRID, TOL
)

from svm_lab import fit, error_rate


X_train, y_train, X_test, y_test = make_dataset("142602026")

print("\nSQ2: What does C buy?")
print("=" * 90)

header = (
    f"{'C':>10s} {'width':>10s} {'sv_frac':>8s} "
    f"{'n_free':>6s} {'n_bnd':>6s} {'train_err':>10s} {'test_err':>9s}"
)
print(header)
print("-" * 90)

rows = []
for C in C_GRID:
    m = fit(X_train, y_train, C=C, s2=S2_DEFAULT)

    sv_frac = m["n_sv"] / X_train.shape[0]

    tr_err = error_rate(m, X_train, y_train, X_train, y_train,
                        s2=S2_DEFAULT)
    te_err = error_rate(m, X_train, y_train, X_test,  y_test,
                        s2=S2_DEFAULT)

    rows.append({
        "C": float(C),
        "margin_width": m["margin_width"],
        "sv_fraction": float(sv_frac),
        "n_free": m["n_free"],
        "n_bounded": m["n_bounded"],
        "train_error": float(tr_err),
        "test_error":  float(te_err),
    })

    print(f"{C:>10.4g} {m['margin_width']:>10.4f} {sv_frac:>8.4f} "
          f"{m['n_free']:>6d} {m['n_bounded']:>6d} "
          f"{tr_err:>10.4f} {te_err:>9.4f}")

# ---- summary statistics for the report -------------------------------
widths = np.array([r["margin_width"]    for r in rows])
te     = np.array([r["test_error"]      for r in rows])

best_test = te.min()
best_idx  = int(np.argmin(te))

print("\nSUMMARY")
print("-" * 45)
print(f"width ratio  (C_min / C_max):   {widths[0] / widths[-1]:.4f}")
print(f"test-error improvement        :   {te[0] - best_test:.4f}")
print(f"best C on grid                :   {C_GRID[best_idx]:g}")
print(f"best test error               :   {best_test:.4f}")

# test-error flat range: |te - best| <= 0.02
flat = np.abs(te - best_test) <= 0.02
print(f"C values with test error within 0.02 of best: "
      f"{C_GRID[flat].tolist()}")

# where the margin width stops changing by more than 1e-2 relative
rel_change = np.abs(np.diff(widths) / widths[:-1])
print("relative width change at each C (consecutive):",
      np.round(rel_change, 4).tolist())

print("\nFREE SUPPORT VECTORS AT SMALLEST C")
print("-" * 45)
print(f"C = {C_GRID[0]:g} -> n_free = {rows[0]['n_free']}, "
      f"n_bounded = {rows[0]['n_bounded']}")

print("\nAt the smallest C, the box 0 <= alpha_i <= C and the equality "
      "\nsum_i alpha_i y_i = 0 force essentially every multiplier to a bound,")
print("so there are (almost) no free SVs and theta_0 must be taken from the ")
print("KKT-interval midpoint [L, U] rather than a free-SV average.")