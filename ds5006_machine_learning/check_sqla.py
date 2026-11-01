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
    decision_values,
    classify_regimes
)


# 1. Generate the dataset
X_train, y_train, X_test, y_test = make_dataset("142602026")


# 2. Fit the RBF SVM using default parameters
model = fit(
    X_train,
    y_train,
    C=C_DEFAULT,
    s2=S2_DEFAULT
)

alpha = model["alpha"]
theta0 = model["theta0"]

# 3. Calculate the training Gram matrix
K_train = rbf_kernel(
    X_train,
    X_train,
    S2_DEFAULT
)

# 4. Calculate decision values and signed margins
g = decision_values(
    alpha,
    theta0,
    y_train,
    K_train
)

margins = y_train * g

# 5. Classify observations into the three regimes
regimes = classify_regimes(
    alpha,
    C_DEFAULT,
    TOL
)

regime_names = {
    0: "Non-support vectors",
    1: "Free support vectors",
    2: "Bounded support vectors"
}

print("\nSQ1(a): SVM Margin Analysis")
print("=" * 55)

# 6. Display the margin ranges and counts
for code, name in regime_names.items():

    selected = margins[regimes == code]

    print(f"\n{name}")
    print("-" * 40)

    print("Count:", len(selected))

    if len(selected) > 0:
        print("Minimum margin:", np.min(selected))
        print("Maximum margin:", np.max(selected))

# 7. Check KKT margin conditions with slack 1e-6
slack = 1e-6

non_sv = margins[regimes == 0]
free_sv = margins[regimes == 1]
bounded_sv = margins[regimes == 2]

violations = {
    "Non-SV (margin >= 1)": np.sum(non_sv < 1.0 - slack),
    "Free SV (margin = 1)": np.sum(
        np.abs(free_sv - 1.0) > slack
    ),
    "Bounded SV (margin <= 1)": np.sum(
        bounded_sv > 1.0 + slack
    )
}

print("\nKKT CONDITION VIOLATIONS")
print("=" * 55)

for condition, count in violations.items():
    print(f"{condition}: {count}")

print("\nTotal violations:", sum(violations.values()))
