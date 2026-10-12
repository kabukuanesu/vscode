import numpy as np
import matplotlib.pyplot as plt

import ds5006_lab3 as L
import ds5006_lab3_data as D


ROLL = 142602026

# ------------------------------------------------------------
# Load the data
# ------------------------------------------------------------

X_train, y_train, X_test, y_test = D.load_split(ROLL)

B_GRID = D.B_GRID
N_TREES = D.N_TREES
N_REPEATS = D.N_REPEATS

test_curves = []
oob_curves = []
test_error_B4 = []

individual_tree_errors = []

# ------------------------------------------------------------
# Fit each forest once, then reuse its predictions
# ------------------------------------------------------------

for repeat in range(N_REPEATS):
    rng = D.make_rng(ROLL, f"bag{repeat}")

    trees, oob_mask = L.fit_bagged_forest(
        X_train,
        y_train,
        rng,
        n_trees=N_TREES,
        max_depth=D.BAG_DEPTH,
        min_samples_split=2,
        impurity=L.gini
    )

    # Predictions for all trees, computed once per forest.
    test_matrix = L.forest_predict_matrix(trees, X_test)
    train_matrix = L.forest_predict_matrix(trees, X_train)

    # Individual-tree test errors.
    tree_errors = np.mean(
        test_matrix != y_test[None, :],
        axis=1
    )
    individual_tree_errors.extend(tree_errors.tolist())

    test_errors_this_forest = []
    oob_errors_this_forest = []

    # Evaluate each requested ensemble size by slicing.
    for B in B_GRID:
        test_prediction = L.majority_vote(
            test_matrix[:B],
            n_classes=2
        )

        test_error = np.mean(test_prediction != y_test)

        oob_error = L.oob_error(
            train_matrix[:B],
            oob_mask[:B],
            y_train,
            n_classes=2
        )

        test_errors_this_forest.append(test_error)
        oob_errors_this_forest.append(oob_error)

    test_curves.append(test_errors_this_forest)
    oob_curves.append(oob_errors_this_forest)

    # The instructions also require the test error at B = 4.
    prediction_B4 = L.majority_vote(
        test_matrix[:4],
        n_classes=2
    )
    test_error_B4.append(
        np.mean(prediction_B4 != y_test)
    )


# ------------------------------------------------------------
# Average the curves over the three independent forests
# ------------------------------------------------------------

mean_test = np.mean(test_curves, axis=0)
mean_oob = np.mean(oob_curves, axis=0)

test_at = {
    B: mean_test[list(B_GRID).index(B)]
    for B in B_GRID
}
oob_at = {
    B: mean_oob[list(B_GRID).index(B)]
    for B in B_GRID
}


# ------------------------------------------------------------
# Single-tree comparisons
# ------------------------------------------------------------

single_tree = L.grow_tree(
    X_train,
    y_train,
    None,
    L.gini,
    max_depth=D.BAG_DEPTH,
    min_samples_split=2
)

single_prediction = L.tree_predict(single_tree, X_test)
single_tree_error = np.mean(single_prediction != y_test)

mean_bagged_tree_error = np.mean(individual_tree_errors)


# ------------------------------------------------------------
# Print the main results
# ------------------------------------------------------------

print("\n======================================")
print("SQ3 - BAGGING AND OOB ESTIMATION")
print("======================================")

print("\nError curves averaged over the forests:")
print(
    f"{'B':>5} {'Test error':>14} {'OOB error':>14}"
)
print("-" * 37)

for i, B in enumerate(B_GRID):
    print(
        f"{B:5d} "
        f"{mean_test[i]:14.6f} "
        f"{mean_oob[i]:14.6f}"
    )

print("\nAdditional comparisons:")
print(f"Test error at B = 4: {np.mean(test_error_B4):.6f}")
print(f"One full-data tree:  {single_tree_error:.6f}")
print(f"Mean individual bagged-tree error: {mean_bagged_tree_error:.6f}")


# ------------------------------------------------------------
# Part 2(a): Level difference and estimated sampling noise
# ------------------------------------------------------------

p_oob = oob_at[150]
p_test = test_at[150]

level_difference = p_oob - p_test

se_oob = np.sqrt(p_oob * (1 - p_oob) / len(y_train))
se_test = np.sqrt(p_test * (1 - p_test) / len(y_test))

se_difference = np.sqrt(se_oob**2 + se_test**2)

print("\nPart 2(a) - Level at B = 150:")
print(f"OOB error - test error: {level_difference:.6f}")
print(f"OOB standard error:     {se_oob:.6f}")
print(f"Test standard error:    {se_test:.6f}")
print(f"Difference SE:           {se_difference:.6f}")
print(f"Absolute difference / SE: {abs(level_difference) / se_difference:.3f}")


# ------------------------------------------------------------
# Part 2(b): Excess above each curve's own B = 150 plateau
# ------------------------------------------------------------

print("\nPart 2(b) - Excess error over the B = 150 plateau:")

for B in (10, 20):
    test_excess = test_at[B] - test_at[150]
    oob_excess = oob_at[B] - oob_at[150]

    print(
        f"B = {B:3d}: "
        f"test excess = {test_excess:.6f}, "
        f"OOB excess = {oob_excess:.6f}"
    )

oob_excess_10 = oob_at[10] - oob_at[150]
test_excess_4 = np.mean(test_error_B4) - test_at[150]
test_excess_10 = test_at[10] - test_at[150]

print("\nCompare the B = 10 OOB excess:")
print(f"OOB excess at 10 trees:  {oob_excess_10:.6f}")
print(f"Test excess at 4 trees:  {test_excess_4:.6f}")
print(f"Test excess at 10 trees: {test_excess_10:.6f}")
print(
    "Difference from test excess at 4:  "
    f"{abs(oob_excess_10 - test_excess_4):.6f}"
)
print(
    "Difference from test excess at 10: "
    f"{abs(oob_excess_10 - test_excess_10):.6f}"
)


# ------------------------------------------------------------
# Plot both curves
# ------------------------------------------------------------

plt.figure(figsize=(9, 5.5))

plt.plot(
    B_GRID, mean_test,
    marker="o",
    label="Test error"
)

plt.plot(
    B_GRID, mean_oob,
    marker="s",
    label="Out-of-bag error"
)

plt.xscale("log")
plt.xlabel("Number of trees (B)")
plt.ylabel("Misclassification error")
plt.title("SQ3: Test Error and Out-of-Bag Error vs Number of Trees")
plt.xticks(B_GRID, [str(B) for B in B_GRID])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("sq3_error_curves.png", dpi=200)
plt.show()
