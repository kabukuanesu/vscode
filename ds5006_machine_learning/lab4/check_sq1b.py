
import numpy as np

from svmlab_provided import (
    make_dataset,
    C_DEFAULT,
    TOL
)

from svm_lab import (
    fit,
    linear_kernel,
    decision_values,
    margin_width
)


# 1. Generate the dataset
X_train, y_train, X_test, y_test = make_dataset("142602026")


# 2. Fit the SVM using the linear kernel
model = fit(
    X_train,
    y_train,
    C=C_DEFAULT,
    s2=None
)

alpha = model["alpha"]
theta0 = model["theta0"]


# 3. Recover the explicit weight vector
theta = X_train.T @ (alpha * y_train)

print("\nSQ1(b): Explicit Weight Vector")
print("=" * 55)

print("Weight vector theta:")
print(theta)

print("\nIntercept theta0:")
print(theta0)


# 4. Calculate the explicit margin width
theta_norm = np.linalg.norm(theta)

if theta_norm == 0:
    explicit_width = float("inf")
else:
    explicit_width = 2.0 / theta_norm


# 5. Calculate the kernel-based margin width
K_train = linear_kernel(X_train, X_train)

kernel_width = margin_width(
    alpha,
    y_train,
    K_train
)


# 6. Compare the margin widths
print("\nMARGIN WIDTH COMPARISON")
print("=" * 55)

print("Explicit margin width:", explicit_width)
print("Kernel margin width:  ", kernel_width)

print(
    "Absolute difference:",
    abs(explicit_width - kernel_width)
)


# 7. Calculate explicit decision values
g_explicit = X_train @ theta + theta0


# 8. Calculate kernel decision values
g_kernel = decision_values(
    alpha,
    theta0,
    y_train,
    K_train
)


# 9. Compare decision values
decision_difference = np.max(
    np.abs(g_explicit - g_kernel)
)

print("\nDECISION VALUE COMPARISON")
print("=" * 55)

print("Maximum absolute difference:")
print(decision_difference)


# 10. Check whether both formulations agree
print("\nVERIFICATION")
print("=" * 55)

print(
    "Margin widths agree:",
    np.isclose(explicit_width, kernel_width)
)

print(
    "Decision values agree:",
    np.allclose(g_explicit, g_kernel, atol=1e-10)
)
