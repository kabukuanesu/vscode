import numpy as np

from svmlab_provided import (
    make_dataset,
    C_DEFAULT,
    S2_DEFAULT,
    TOL
)

from svm_lab import (
    fit,
    rbf_kernel,
    decision_values
)


# 1. Generate the dataset
X_train, y_train, X_test, y_test = make_dataset("142602026")


# 2. Fit the full model at the default C and s2
model_full = fit(
    X_train,
    y_train,
    C=C_DEFAULT,
    s2=S2_DEFAULT
)

alpha_full   = model_full["alpha"]
theta0_full  = model_full["theta0"]
regimes_full = model_full["regimes"]


# 3. Identify support vectors (regime 1 or 2) and non-support vectors (regime 0)
sv_mask  = regimes_full > 0
non_sv_mask = regimes_full == 0

n_before = X_train.shape[0]
n_sv     = int(sv_mask.sum())
n_non_sv = int(non_sv_mask.sum())


# 4. Delete every non-support-vector and refit
X_reduced = X_train[sv_mask]
y_reduced = y_train[sv_mask]

model_reduced = fit(
    X_reduced,
    y_reduced,
    C=C_DEFAULT,
    s2=S2_DEFAULT
)

alpha_red  = model_reduced["alpha"]
theta0_red = model_reduced["theta0"]


# 5. Evaluate both discriminants on the TEST set
K_test_full = rbf_kernel(X_test, X_train, S2_DEFAULT)
K_test_red  = rbf_kernel(X_test, X_reduced, S2_DEFAULT)

g_full = decision_values(
    alpha_full,
    theta0_full,
    y_train,
    K_test_full
)

g_red = decision_values(
    alpha_red,
    theta0_red,
    y_reduced,
    K_test_red
)


# 6. Compare
max_g_diff = float(np.max(np.abs(g_full - g_red)))

label_full = np.sign(g_full)
label_red  = np.sign(g_red)

label_agreement = float(np.mean(label_full == label_red))

test_err_full = float(np.mean(label_full != y_test))
test_err_red  = float(np.mean(label_red  != y_test))


# 7. Display results
print("\nSQ1(d): Refit After Deleting Non-Support-Vectors")
print("=" * 60)

print("\nTRAINING SET SIZES")
print("-" * 45)
print("n_before (full training set):", n_before)
print("n_after  (support vectors only):", n_sv)
print("non-support vectors removed:", n_non_sv)
print("fraction retained:",
      f"{n_sv / n_before:.4f}")

print("\nDISCRIMINANT COMPARISON ON THE TEST SET")
print("-" * 45)
print("max |g_full - g_reduced|:", max_g_diff)
print("label agreement fraction:",
      f"{label_agreement:.6f}")

print("\nTEST ERROR")
print("-" * 45)
print("full model:   ", test_err_full)
print("reduced model:", test_err_red)
print("difference:   ", test_err_red - test_err_full)


print("\nVERIFICATION")
print("-" * 45)
print("discriminants agree (atol=1e-6):",
      np.allclose(g_full, g_red, atol=1e-6))
print("predicted labels agree exactly:",
      np.array_equal(label_full, label_red))