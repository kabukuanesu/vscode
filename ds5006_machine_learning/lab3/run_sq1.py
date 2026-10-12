import numpy as np

import ds5006_lab3 as L
import ds5006_lab3_data as D


ROLL = 142602026


# ============================================================
# Part 1: Tricky node
# ============================================================

X_node, y_node = D.tricky_node(ROLL)

criteria = [
    ("Gini", L.gini),
    ("Entropy", L.entropy),
    ("Misclassification", L.misclassification),
]

print("\n==============================")
print("SQ1 - TRICKY NODE")
print("==============================")

for name, criterion in criteria:
    feature, threshold, gain = L.best_split(
        X_node,
        y_node,
        None,
        criterion
    )

    print(
        f"{name:20s} "
        f"feature={feature}, "
        f"threshold={threshold:.6f}, "
        f"gain={gain:.6f}"
    )


# ============================================================
# Part 2: Full training/test experiment
# ============================================================

X_train, y_train, X_test, y_test = D.load_split(ROLL)

depths = D.DEPTH_GRID

print("\n==============================")
print("SQ1 - FULL TREE EXPERIMENT")
print("==============================")

print(
    f"{'Depth':>5} "
    f"{'Gini Err':>10} {'Gini Leaves':>12} "
    f"{'Entropy Err':>12} {'Entropy Leaves':>15} "
    f"{'Misclass Err':>13} {'Misclass Leaves':>16}"
)

print("-" * 100)

for depth in depths:

    results = []

    for _, criterion in criteria:

        tree = L.grow_tree(
            X_train,
            y_train,
            None,
            criterion,
            max_depth=depth,
            min_samples_split=2
        )

        predictions = L.tree_predict(tree, X_test)

        test_error = np.mean(predictions != y_test)

        n_leaves, actual_depth = L.tree_stats(tree)

        results.append((test_error, n_leaves))

    print(
        f"{depth:5d} "
        f"{results[0][0]:10.4f} {results[0][1]:12d} "
        f"{results[1][0]:12.4f} {results[1][1]:15d} "
        f"{results[2][0]:13.4f} {results[2][1]:16d}"
    )
