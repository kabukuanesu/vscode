
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
    recover_intercept,
    kkt_residual,
    classify_regimes
)


# 1. Generate the dataset
X_train, y_train, X_test, y_test = make_dataset("142602026")


# 2. Fit the correct RBF SVM
model = fit(
    X_train,
    y_train,
    C=C_DEFAULT,
    s2=S2_DEFAULT
)

alpha = model["alpha"]
correct_theta0 = model["theta0"]


# 3. Calculate the training Gram matrix
K_train = rbf_kernel(
    X_train,
    X_train,
    S2_DEFAULT
)


# 4. Calculate the kernel contribution for each observation
f = K_train @ (alpha * y_train)


# 5. Identify all support vectors
support_vectors = alpha > TOL


# 6. Calculate the incorrect intercept using ALL support vectors
wrong_theta0 = np.mean(
    y_train[support_vectors] - f[support_vectors]
)


# 7. Calculate the intercept using FREE support vectors
# This independently checks the intercept returned by fit().
free_vectors = (
    (alpha > TOL) &
    (alpha < C_DEFAULT - TOL)
)

verified_theta0 = recover_intercept(
    alpha,
    y_train,
    K_train,
    C_DEFAULT,
    TOL
)


# 8. Calculate KKT residuals
correct_residual = kkt_residual(
    alpha,
    correct_theta0,
    y_train,
    K_train,
    C_DEFAULT,
    TOL
)

wrong_residual = kkt_residual(
    alpha,
    wrong_theta0,
    y_train,
    K_train,
    C_DEFAULT,
    TOL
)


# 9. Display results
print("\nSQ1(c): Effect of Using the Wrong Intercept")
print("=" * 60)

print("\nSUPPORT VECTOR COUNTS")
print("-" * 45)

print("All support vectors:", np.sum(support_vectors))
print("Free support vectors:", np.sum(free_vectors))
print("Bounded support vectors:",
      np.sum(support_vectors & ~free_vectors))


print("\nINTERCEPT COMPARISON")
print("-" * 45)

print("Correct intercept:", correct_theta0)
print("Verified intercept:", verified_theta0)
print("Wrong intercept:  ", wrong_theta0)

print("\nDifference between intercepts:")
print(abs(correct_theta0 - wrong_theta0))


print("\nKKT RESIDUAL COMPARISON")
print("-" * 45)

print("Correct KKT residual:", correct_residual)
print("Wrong KKT residual:  ", wrong_residual)


print("\nVERIFICATION")
print("-" * 45)

print(
    "Correct intercepts agree:",
    np.isclose(correct_theta0, verified_theta0)
)

print(
    "Wrong intercept has larger KKT residual:",
    wrong_residual > correct_residual
)
