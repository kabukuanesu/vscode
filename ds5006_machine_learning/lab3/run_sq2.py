import numpy as np

import ds5006_lab3 as L
import ds5006_lab3_data as D


ROLL = 142602026


# ============================================================
# Load data
# ============================================================

X_train, y_train, X_test, y_test = D.load_split(ROLL)


# ============================================================
# Find the exact best Gini depth
# ============================================================

best_depth = None
best_error = float("inf")

for depth in D.DEPTH_GRID:
    tree = L.grow_tree(
        X_train,
        y_train,
        None,
        L.gini,
        max_depth=depth,
        min_samples_split=2
    )

    predictions = L.tree_predict(tree, X_test)
    error = np.mean(predictions != y_test)

    if error < best_error:
        best_error = error
        best_depth = depth


print("\n==============================")
print("SQ2 - BEST GINI DEPTH")
print("==============================")

print(f"Best depth : {best_depth}")
print(f"Best error : {best_error:.6f}")

tree20 = L.grow_tree(
    X_train,
    y_train,
    None,
    L.gini,
    max_depth=20,
    min_samples_split=2
)

pred20 = L.tree_predict(tree20, X_test)
error20 = np.mean(pred20 != y_test)

print(f"Depth 20 error : {error20:.6f}")
print(f"Gap : {error20 - best_error:.6f}")


# ============================================================
# Bootstrap stability experiment
# ============================================================

rng = D.make_rng(ROLL, "stability")

n_boot = D.N_BOOT_STABILITY

fully_grown_trees = []
shallow_trees = []

root_counts = np.zeros(X_train.shape[1], dtype=int)

fully_grown_errors = []


for b in range(n_boot):

    # Draw one bootstrap sample
    indices = L.bootstrap_indices(rng, X_train.shape[0])

    X_boot = X_train[indices]
    y_boot = y_train[indices]

    # --------------------------------------------------------
    # Fully grown tree
    # --------------------------------------------------------

    full_tree = L.grow_tree(
        X_boot,
        y_boot,
        None,
        L.gini,
        max_depth=D.BAG_DEPTH,
        min_samples_split=2
    )

    fully_grown_trees.append(full_tree)

    # Count root feature
    if not full_tree["leaf"]:
        root_feature = full_tree["feature"]
        root_counts[root_feature] += 1

    # Test error of this individual tree
    full_pred = L.tree_predict(full_tree, X_test)
    full_error = np.mean(full_pred != y_test)

    fully_grown_errors.append(full_error)

    # --------------------------------------------------------
    # Shallow tree
    # --------------------------------------------------------

    shallow_tree = L.grow_tree(
        X_boot,
        y_boot,
        None,
        L.gini,
        max_depth=best_depth,
        min_samples_split=2
    )

    shallow_trees.append(shallow_tree)


# ============================================================
# Predictions
# ============================================================

full_predictions = L.forest_predict_matrix(
    fully_grown_trees,
    X_test
)

shallow_predictions = L.forest_predict_matrix(
    shallow_trees,
    X_test
)


# ============================================================
# Mean pairwise disagreement
# ============================================================

def mean_pairwise_disagreement(pred_matrix):
    n_trees = pred_matrix.shape[0]

    disagreements = []

    for a in range(n_trees):
        for b in range(a + 1, n_trees):
            disagreement = np.mean(
                pred_matrix[a] != pred_matrix[b]
            )

            disagreements.append(disagreement)

    return np.mean(disagreements)


full_disagreement = mean_pairwise_disagreement(
    full_predictions
)

shallow_disagreement = mean_pairwise_disagreement(
    shallow_predictions
)


# ============================================================
# Print results
# ============================================================

print("\n==============================")
print("SQ2 - BOOTSTRAP STABILITY")
print("==============================")


print("\nRoot feature counts:")

for feature, count in enumerate(root_counts):
    print(f"Feature {feature}: {count}")


print("\nMean pairwise disagreement:")

print(
    f"Fully grown : "
    f"{full_disagreement:.6f}"
)

print(
    f"Depth {best_depth} : "
    f"{shallow_disagreement:.6f}"
)


print("\nIndividual fully-grown tree test errors:")

print(
    f"Minimum : "
    f"{np.min(fully_grown_errors):.6f}"
)

print(
    f"Maximum : "
    f"{np.max(fully_grown_errors):.6f}"
)

print(
    f"Mean    : "
    f"{np.mean(fully_grown_errors):.6f}"
)
