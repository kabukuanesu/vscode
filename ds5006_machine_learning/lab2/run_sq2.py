import numpy as np
import ds5006_lab2_data as data
import ds5006_lab2 as lab


ROLL = 142602026


# ============================================================
# Load overlapping two-class dataset
# ============================================================

X, y = data.dataset_margin(ROLL, "overlapping")

print("SQ2 - DATASET")
print("-------------")
print("X shape:", X.shape)
print("Class 0:", np.sum(y == 0))
print("Class 1:", np.sum(y == 1))


# ============================================================
# LDA
# ============================================================

lda_model = lab.lda_fit(X, y, 2)
lda_pred = lab.lda_predict(X, lda_model)

print("\nLDA")
print("---")
print("Means:")
print(lda_model["means"])

print("\nCovariance:")
print(lda_model["cov"])

print("\nPriors:")
print(lda_model["priors"])


# ============================================================
# Compare directions for different codings
# ============================================================

codings = [
    (0, 1),
    (-1, 1),
    (7, -3)
]

print("\nDIRECTION COMPARISON")
print("--------------------")

lda_direction = np.linalg.solve(
    lda_model["cov"],
    lda_model["means"][1] - lda_model["means"][0]
)

for code_zero, code_one in codings:

    ls_direction = lab.ls_direction_two_class(
        X,
        y,
        code_zero,
        code_one
    )

    cosine = (
        np.dot(lda_direction, ls_direction)
        / (
            np.linalg.norm(lda_direction)
            * np.linalg.norm(ls_direction)
        )
    )

    print(
        f"Coding ({code_zero}, {code_one}): "
        f"cosine = {cosine:.12f}"
    )

    print("  LDA direction:", lda_direction)
    print("  LS direction :", ls_direction)


# ============================================================
# Balanced decision-rule comparison
# ============================================================

print("\nBALANCED DECISION RULES")
print("-----------------------")

code_zero, code_one = 0, 1

ls_direction = lab.ls_direction_two_class(
    X,
    y,
    code_zero,
    code_one
)

# Refit OLS including intercept so we can actually classify.
targets = np.where(y == 0, code_zero, code_one)

X_int = lab.add_intercept(X)

Q, R = np.linalg.qr(X_int, mode="reduced")
beta = np.linalg.solve(R, Q.T @ targets)

ls_scores = X_int @ beta

# Threshold halfway between the two coded values.
threshold = (code_zero + code_one) / 2

ls_pred = (ls_scores > threshold).astype(int)

agreement_balanced = np.mean(
    lda_pred == ls_pred
)

print("LDA vs LS agreement:",
      agreement_balanced)

print("Number agreeing:",
      np.sum(lda_pred == ls_pred),
      "out of",
      len(y))


# ============================================================
# Unbalanced subset
# ============================================================

class_zero_indices = np.where(y == 0)[0]
class_one_indices = np.where(y == 1)[0]

# Keep all class 0 observations but only 20 class 1 observations.
unbalanced_indices = np.concatenate([
    class_zero_indices,
    class_one_indices[:20]
])

X_unbalanced = X[unbalanced_indices]
y_unbalanced = y[unbalanced_indices]

print("\nUNBALANCED SAMPLE")
print("-----------------")
print("Class 0:",
      np.sum(y_unbalanced == 0))

print("Class 1:",
      np.sum(y_unbalanced == 1))


# LDA on unbalanced data
lda_unbalanced = lab.lda_fit(
    X_unbalanced,
    y_unbalanced,
    2
)

lda_unbalanced_pred = lab.lda_predict(
    X_unbalanced,
    lda_unbalanced
)


# LS on unbalanced data
targets_unbalanced = np.where(
    y_unbalanced == 0,
    code_zero,
    code_one
)

X_unbalanced_int = lab.add_intercept(
    X_unbalanced
)

Q, R = np.linalg.qr(
    X_unbalanced_int,
    mode="reduced"
)

beta_unbalanced = np.linalg.solve(
    R,
    Q.T @ targets_unbalanced
)

ls_scores_unbalanced = (
    X_unbalanced_int @ beta_unbalanced
)

ls_pred_unbalanced = (
    ls_scores_unbalanced > threshold
).astype(int)


agreement_unbalanced = np.mean(
    lda_unbalanced_pred == ls_pred_unbalanced
)

print("LDA priors:",
      lda_unbalanced["priors"])

print("LDA vs LS agreement:",
      agreement_unbalanced)

print("Number agreeing:",
      np.sum(
          lda_unbalanced_pred ==
          ls_pred_unbalanced
      ),
      "out of",
      len(y_unbalanced))
