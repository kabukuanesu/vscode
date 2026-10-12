import numpy as np
import ds5006_lab2_data as data
import ds5006_lab2 as lab


ROLL = 142602026


# ---------------------------------------------------------
# Original features
# ---------------------------------------------------------
X, y = data.dataset_masking(ROLL)

X_intercept = lab.add_intercept(X)
Y = lab.one_hot(y, 3)

coefficients = lab.fit_indicator_regression(X_intercept, Y)
predicted = lab.indicator_predict(X_intercept, coefficients)

print("SQ1 - LINEAR INDICATOR REGRESSION")
print("----------------------------------")

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Coefficient shape:", coefficients.shape)

print("\nOverall accuracy:", lab.accuracy(y, predicted))

print("\nPer-class accuracy:")

for k in range(3):
    mask = y == k
    class_accuracy = lab.accuracy(y[mask], predicted[mask])
    print(f"Class {k}: {class_accuracy:.6f}")

print("\nPredicted class counts:")
for k in range(3):
    print(f"Class {k}: {(predicted == k).sum()}")

print("\nActual class counts:")
for k in range(3):
    print(f"Class {k}: {(y == k).sum()}")


# ---------------------------------------------------------
# Quadratic expansion
# ---------------------------------------------------------
X_quad = lab.quadratic_expansion(X)
X_quad_intercept = lab.add_intercept(X_quad)

quad_coefficients = lab.fit_indicator_regression(
    X_quad_intercept,
    Y
)

quad_predicted = lab.indicator_predict(
    X_quad_intercept,
    quad_coefficients
)

print("\n\nSQ1 - QUADRATIC EXPANSION")
print("-------------------------")

print("Quadratic X shape:", X_quad.shape)
print("Coefficient shape:", quad_coefficients.shape)

print("\nOverall accuracy:",
      lab.accuracy(y, quad_predicted))

print("\nPer-class accuracy:")

for k in range(3):
    mask = y == k
    class_accuracy = lab.accuracy(
        y[mask],
        quad_predicted[mask]
    )
    print(f"Class {k}: {class_accuracy:.6f}")

print("\nPredicted class counts:")
for k in range(3):
    print(f"Class {k}: {(quad_predicted == k).sum()}")
