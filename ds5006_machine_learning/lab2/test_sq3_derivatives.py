import numpy as np
import ds5006_lab2 as lab
import ds5006_lab2_data as data


ROLL = 142602026

X, y = data.dataset_margin(
    ROLL,
    "overlapping"
)

X = lab.add_intercept(X)

theta = np.array([
    0.3,
    -0.7,
    0.5
])

lam = 0.3
h = 1e-6


def numerical_gradient(f, theta, h=1e-6):
    grad = np.zeros_like(theta, dtype=float)

    for j in range(theta.size):
        step = np.zeros_like(theta, dtype=float)
        step[j] = h

        grad[j] = (
            f(theta + step)
            - f(theta - step)
        ) / (2.0 * h)

    return grad


def numerical_hessian(gradient_function, theta, h=1e-6):
    p = theta.size
    hessian = np.zeros((p, p))

    for j in range(p):
        step = np.zeros(p)
        step[j] = h

        hessian[:, j] = (
            gradient_function(theta + step)
            - gradient_function(theta - step)
        ) / (2.0 * h)

    return hessian


def objective(t):
    return lab.logistic_objective(
        X, y, t, lam
    )


def gradient_function(t):
    return lab.logistic_gradient(
        X, y, t, lam
    )


analytic_gradient = lab.logistic_gradient(
    X, y, theta, lam
)

numeric_gradient = numerical_gradient(
    objective,
    theta,
    h
)

analytic_hessian = lab.logistic_hessian(
    X, theta, lam
)

numeric_hessian = numerical_hessian(
    gradient_function,
    theta,
    h
)

print("Gradient")
print("--------")
print("Analytic:", analytic_gradient)
print("Numeric :", numeric_gradient)

print(
    "Max gradient error:",
    np.max(
        np.abs(
            analytic_gradient
            - numeric_gradient
        )
    )
)

print("\nHessian")
print("-------")
print("Analytic:")
print(analytic_hessian)

print("\nNumeric:")
print(numeric_hessian)

print(
    "\nMax Hessian error:",
    np.max(
        np.abs(
            analytic_hessian
            - numeric_hessian
        )
    )
)

print(
    "Hessian symmetry error:",
    np.max(
        np.abs(
            analytic_hessian
            - analytic_hessian.T
        )
    )
)