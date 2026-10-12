"""DS5006 Lab 3 - decision trees and ensembles.

Fill in every function below.  NumPy only: no scikit-learn, no scipy, no pandas.

The docstrings are contracts, not suggestions.  Where a docstring pins a
convention (a tie-breaking rule, a normalisation, an argument order) the hidden
tests check that exact behaviour, and a different but "equally reasonable"
choice will be marked wrong.  The conventions are listed together in Section 4
of the instructions; read them once before you start.

Name: Anesu Kabuku
Roll number: 142602026
"""

import numpy as np

#: A split is only accepted if its gain is strictly greater than this.
GAIN_TOL = 1e-12

#: Boosting clips the weighted error into [EPS, 1 - EPS] before taking logs.
EPS = 1e-12


# ======================================================================
# Part A - impurity and splitting
# ======================================================================

def class_distribution(y, w=None, n_classes=2):
    """Return the weighted class proportions of a label vector.

    p[k] = (sum of w[i] over i with y[i] == k) / (sum of all w[i]).

    Parameters
    ----------
    y : (n,) integer array with values in {0, ..., n_classes - 1}
    w : (n,) float array of non-negative weights, or None for uniform weights.
        None must behave exactly like np.ones(n), not like np.ones(n) / n.
    n_classes : int

    Returns
    -------
    (n_classes,) float array summing to 1.

    Convention C1: if the total weight is zero (which includes the empty
    label vector) return an all-zero array of length n_classes.  Do not
    return NaN and do not raise.
    """
    y = np.asarray(y, dtype=int)

    if w is None:
        w = np.ones(y.size, dtype=float)
    else:
        w = np.asarray(w, dtype=float)

    total = np.sum(w)

    if total == 0:
        return np.zeros(n_classes, dtype=float)

    counts = np.zeros(n_classes, dtype=float)

    for k in range(n_classes):
        counts[k] = np.sum(w[y == k])

    return counts / total


def gini(p):
    """Gini impurity 1 - sum_k p_k^2.

    Convention C2: p may have any shape (..., K); the impurity is computed
    along the LAST axis and the result has shape (...).  A scalar comes back
    only when p is one dimensional.  You will call this on a whole stack of
    candidate distributions at once in best_split, so do not write it as a
    loop over a single distribution.
    """
    p = np.asarray(p, dtype=float)
    return 1.0 - np.sum(p ** 2, axis=-1)


def entropy(p):
    """Shannon entropy -sum_k p_k log(p_k), in NATS.

    Convention C3: natural logarithm, not log2.  Use the convention
    0 * log(0) = 0, and do not emit a NumPy warning while doing so.
    Same shape rule as gini (Convention C2).
    """
    p = np.asarray(p, dtype=float)

    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(p > 0.0, p * np.log(p), 0.0)

    return -np.sum(terms, axis=-1)


def misclassification(p):
    """Misclassification impurity 1 - max_k p_k.

    Same shape rule as gini (Convention C2).
    """
    p = np.asarray(p, dtype=float)
    return 1.0 - np.max(p, axis=-1)


def candidate_thresholds(x):
    """Return the candidate split thresholds for one feature column.

    The candidates are the midpoints between consecutive DISTINCT sorted
    values of x, in increasing order.  A feature that takes fewer than two
    distinct values contributes no candidates.

    Parameters
    ----------
    x : (n,) float array

    Returns
    -------
    (m,) float array, ascending.  m = (number of distinct values) - 1, or 0.

    Example: x = [1.0, 3.0, 1.0, 7.0] gives [2.0, 5.0].
    """
    x = np.asarray(x, dtype=float)

    values = np.unique(x)

    if values.size < 2:
        return np.empty(0, dtype=float)

    return (values[:-1] + values[1:]) / 2.0


def best_split(X, y, w, impurity, n_classes=2):
    """Find the best axis aligned split of a node.

    A split is the pair (j, t) meaning "send row i left if X[i, j] <= t".
    Its gain is

        gain = I(parent) - (W_L / W) I(p_L) - (W_R / W) I(p_R)

    where W_L, W_R are the total weights of the two children, W = W_L + W_R,
    and p_L, p_R are their weighted class distributions.

    Parameters
    ----------
    X : (n, d) float array
    y : (n,) integer array
    w : (n,) float array of weights, or None for uniform weights
    impurity : one of gini, entropy, misclassification (any callable obeying
        Convention C2)
    n_classes : int

    Returns
    -------
    (feature, threshold, gain) : (int, float, float)

    Convention C4: only splits with gain > GAIN_TOL are admissible.  If no
    admissible split exists - no candidate thresholds anywhere, or fewer than
    two rows, or zero total weight - return exactly (-1, float('nan'), 0.0).

    Convention C5 (tie-breaking): among admissible splits of equal gain,
    prefer the SMALLEST feature index; within a feature, prefer the SMALLEST
    threshold.  Ties are common with the misclassification criterion, so this
    rule is tested.

    Performance.  The obvious triple loop over features, thresholds and rows
    is far too slow to survive the rest of the lab: you will fit tens of
    thousands of nodes before you are done.  For each feature, sort the column
    once and take a cumulative sum of the per-class weights; every candidate
    split is then a prefix of that sorted order and all the gains can be
    evaluated in one vectorised expression.  Aim for one pass per feature per
    node, not one pass per threshold.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    n, d = X.shape

    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float)

    total_weight = np.sum(w)

    if n < 2 or total_weight == 0:
        return -1, float('nan'), 0.0

    parent_dist = class_distribution(y, w, n_classes)
    parent_impurity = float(impurity(parent_dist))

    best_feature = -1
    best_threshold = float('nan')
    best_gain = 0.0

    for j in range(d):
        order = np.argsort(X[:, j], kind='mergesort')

        x_sorted = X[order, j]
        y_sorted = y[order]
        w_sorted = w[order]

        valid = x_sorted[:-1] < x_sorted[1:]

        if not np.any(valid):
            continue

        class_weights = np.zeros((n, n_classes), dtype=float)

        for k in range(n_classes):
            class_weights[:, k] = w_sorted * (y_sorted == k)

        cumulative = np.cumsum(class_weights, axis=0)

        boundary_indices = np.flatnonzero(valid)

        left_weights = cumulative[boundary_indices]
        left_total = np.sum(left_weights, axis=1)

        total_class_weights = np.sum(class_weights, axis=0)

        right_weights = total_class_weights - left_weights
        right_total = total_weight - left_total

        left_dist = left_weights / left_total[:, None]
        right_dist = right_weights / right_total[:, None]

        left_impurity = impurity(left_dist)
        right_impurity = impurity(right_dist)

        gains = (
            parent_impurity
            - (left_total / total_weight) * left_impurity
            - (right_total / total_weight) * right_impurity
        )

        thresholds = (
            x_sorted[boundary_indices]
            + x_sorted[boundary_indices + 1]
        ) / 2.0

        for threshold, gain in zip(thresholds, gains):
            if gain > GAIN_TOL and gain > best_gain:
                best_feature = j
                best_threshold = float(threshold)
                best_gain = float(gain)

    if best_feature == -1:
        return -1, float('nan'), 0.0

    return best_feature, best_threshold, best_gain


def grow_tree(X, y, w, impurity, max_depth, min_samples_split, n_classes=2):
    """Grow a binary decision tree greedily and return its root node.

    A node is a plain dict.  Convention C6 pins the keys exactly.

    Every node has:
        'leaf'      bool
        'pred'      int, the predicted class
        'n'         int, the number of rows that reached this node
        'dist'      (n_classes,) array, the weighted class distribution

    An internal node additionally has:
        'feature'   int
        'threshold' float
        'gain'      float
        'left'      dict, the subtree for rows with X[i, feature] <= threshold
        'right'     dict, the subtree for the rest

    Convention C7: 'pred' is the class with the largest weighted proportion,
    with ties broken towards the SMALLER class label.

    Convention C8 (stopping): the node becomes a leaf if any of the following
    holds, checked in this order and BEFORE any split is attempted -
        max_depth <= 0,
        y.size < min_samples_split,
        all labels in the node are equal.
    Otherwise call best_split; if it reports no admissible split the node
    becomes a leaf as well.

    Convention C9: max_depth counts remaining depth.  The recursive calls
    receive max_depth - 1, so grow_tree(..., max_depth=0, ...) always returns
    a single leaf and grow_tree(..., max_depth=1, ...) returns a stump.

    Rows with X[i, feature] exactly equal to the threshold go LEFT.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    n = y.size

    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float)

    dist = class_distribution(y, w, n_classes)

    pred = int(np.argmax(dist))

    if (
        max_depth <= 0
        or n < min_samples_split
        or np.all(y == y[0])
    ):
        return {
            'leaf': True,
            'pred': pred,
            'n': n,
            'dist': dist,
        }

    feature, threshold, gain = best_split(X, y, w, impurity, n_classes)

    if feature == -1:
        return {
            'leaf': True,
            'pred': pred,
            'n': n,
            'dist': dist,
        }

    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask

    left = grow_tree(
        X[left_mask], y[left_mask], w[left_mask],
        impurity, max_depth - 1, min_samples_split, n_classes
    )

    right = grow_tree(
        X[right_mask], y[right_mask], w[right_mask],
        impurity, max_depth - 1, min_samples_split, n_classes
    )

    return {
        'leaf': False,
        'pred': pred,
        'n': n,
        'dist': dist,
        'feature': int(feature),
        'threshold': float(threshold),
        'gain': float(gain),
        'left': left,
        'right': right,
    }


def tree_predict(node, X):
    """Return the predicted label of every row of X, as an (n,) integer array.

    Performance.  Do not walk the tree once per row in Python.  Recurse on
    index arrays instead: at an internal node, split the incoming index array
    with a boolean mask and pass the two halves down.  The work is then a
    handful of NumPy operations per node rather than one per row.
    """
    X = np.asarray(X, dtype=float)
    predictions = np.empty(X.shape[0], dtype=int)

    def recurse(current, indices):
        if indices.size == 0:
            return

        if current['leaf']:
            predictions[indices] = current['pred']
            return

        feature = current['feature']
        threshold = current['threshold']

        left_mask = X[indices, feature] <= threshold

        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        recurse(current['left'], left_indices)
        recurse(current['right'], right_indices)

    recurse(node, np.arange(X.shape[0]))

    return predictions


def tree_stats(node):
    """Return (n_leaves, depth) for the tree rooted at node.

    Convention C10: a tree consisting of a single leaf has n_leaves = 1 and
    depth = 0.  Depth is the number of splits on the longest root-to-leaf path.
    """
    if node['leaf']:
        return 1, 0

    left_leaves, left_depth = tree_stats(node['left'])
    right_leaves, right_depth = tree_stats(node['right'])

    leaves = left_leaves + right_leaves
    depth = 1 + max(left_depth, right_depth)

    return leaves, depth


# ======================================================================
# Part B - bagging and out of bag estimation
# ======================================================================

def bootstrap_indices(rng, n):
    """Return one bootstrap sample of row indices, as an (n,) integer array.

    Convention C11: exactly one call to rng.integers(0, n, size=n), and
    nothing else drawn from rng.  Everyone's out of bag sets must be
    reproducible from the roll number, so the number and order of draws
    matters.  Do not use rng.choice.
    """
    return rng.integers(0, n, size=n)


def fit_bagged_forest(X, y, rng, n_trees, max_depth, min_samples_split,
                      impurity=gini):
    """Fit a bagged ensemble of trees.

    For b = 0, 1, ..., n_trees - 1: draw a bootstrap sample of the rows with
    bootstrap_indices, and grow an unweighted tree on those rows.

    Returns
    -------
    trees : list of n_trees node dicts, in the order they were fitted
    oob_mask : (n_trees, n) boolean array, True where row i did NOT appear in
        the bootstrap sample used for tree b

    Convention C12: the trees are fitted with w = None (uniform weights).
    Bagging resamples rows; it does not reweight them.
    """
    n = X.shape[0]

    trees = []
    oob_mask = np.empty((n_trees, n), dtype=bool)

    for b in range(n_trees):
        indices = bootstrap_indices(rng, n)

        mask = np.ones(n, dtype=bool)
        mask[indices] = False
        oob_mask[b] = mask

        tree = grow_tree(
            X[indices], y[indices], None,
            impurity, max_depth, min_samples_split
        )

        trees.append(tree)

    return trees, oob_mask


def forest_predict_matrix(trees, X):
    """Stack per-tree predictions into a (B, n) integer array.

    Row b holds the predictions of trees[b] on all of X.  Building this matrix
    once and slicing it is how you produce a whole error-versus-B curve
    without refitting anything.
    """
    if len(trees) == 0:
        return np.empty((0, X.shape[0]), dtype=int)

    predictions = [
        tree_predict(tree, X) for tree in trees
    ]

    return np.asarray(predictions, dtype=int)


def majority_vote(pred_matrix, n_classes=2):
    """Column-wise majority vote over a (B, n) prediction matrix.

    Returns an (n,) integer array.

    Convention C13: ties are broken towards the SMALLER class label, so with
    B even and a 50-50 split the answer is class 0.  If B == 0, return an
    all-zero array of length n.
    """
    pred_matrix = np.asarray(pred_matrix, dtype=int)

    if pred_matrix.shape[0] == 0:
        return np.zeros(pred_matrix.shape[1], dtype=int)

    counts = np.zeros(
        (n_classes, pred_matrix.shape[1]), dtype=int
    )

    for k in range(n_classes):
        counts[k] = np.sum(pred_matrix == k, axis=0)

    return np.argmax(counts, axis=0)


def oob_error(pred_matrix, oob_mask, y, n_classes=2):
    """Out of bag misclassification rate of a bagged ensemble.

    For each training row i, take the majority vote over only those trees b
    with oob_mask[b, i] True, then compare with y[i].

    Convention C14: rows for which no tree is out of bag contribute nothing -
    they are dropped from both the numerator and the denominator.  If every
    row is dropped, or there are no trees at all, return float('nan').
    Ties within a row are broken as in Convention C13.

    Parameters
    ----------
    pred_matrix : (B, n) predictions of the B trees on the TRAINING rows
    oob_mask : (B, n) boolean array as returned by fit_bagged_forest
    y : (n,) training labels
    """
    pred_matrix = np.asarray(pred_matrix, dtype=int)
    oob_mask = np.asarray(oob_mask, dtype=bool)
    y = np.asarray(y, dtype=int)

    if pred_matrix.shape[0] == 0:
        return float('nan')

    n = y.size

    valid = np.any(oob_mask, axis=0)

    if not np.any(valid):
        return float('nan')

    counts = np.zeros((n_classes, n), dtype=int)

    for k in range(n_classes):
        counts[k] = np.sum(
            (pred_matrix == k) & oob_mask, axis=0
        )

    oob_predictions = np.argmax(counts, axis=0)

    return np.mean(oob_predictions[valid] != y[valid])


# ======================================================================
# Part C - AdaBoost
# ======================================================================

def adaboost_fit(X, y, n_rounds, max_depth, min_samples_split):
    """Run discrete AdaBoost with trees as the base learner.

    Labels arrive as {0, 1}.  Internally the algorithm works with the signed
    encoding s_i = 2 y_i - 1 in {-1, +1}; the base learners are still fitted
    on the {0, 1} labels.

    Round m = 1, ..., n_rounds:
        1. grow a tree on (X, y) with weights w, impurity = gini, the given
           max_depth and min_samples_split
        2. eps_m = sum_i w_i [h_m(x_i) != y_i]
        3. clip eps_m into [EPS, 1 - EPS]
        4. beta_m = 0.5 * log((1 - eps_m) / eps_m)
        5. w_i <- w_i * exp(-beta_m * s_i * (2 h_m(x_i) - 1))
        6. renormalise w so that it sums to 1

    Convention C15: the initial weights are 1/n each, and step 6 renormalises
    after EVERY round, so the weights always sum to 1 and eps_m is directly
    comparable across rounds.

    Convention C16: there is no early stopping.  All n_rounds rounds are run
    even if eps_m creeps up towards 0.5.  The factor 0.5 in beta_m is part of
    the contract; some textbooks fold it elsewhere.

    Returns
    -------
    trees : list of n_rounds node dicts
    betas : (n_rounds,) float array
    epsilons : (n_rounds,) float array, the CLIPPED weighted errors
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    n = y.size

    s = 2 * y - 1  # signed labels in {-1, +1}

    w = np.ones(n, dtype=float) / n

    trees = []
    betas = np.empty(n_rounds, dtype=float)
    epsilons = np.empty(n_rounds, dtype=float)

    for m in range(n_rounds):
        tree = grow_tree(
            X, y, w, gini,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )

        h = tree_predict(tree, X)
        signed_h = 2 * h - 1
        wrong = (h != y).astype(float)
        eps = float(np.sum(w * wrong))

        eps_clipped = min(max(eps, EPS), 1.0 - EPS)
        beta = 0.5 * np.log((1.0 - eps_clipped) / eps_clipped)
        w = w * np.exp(-beta * s * signed_h)
        w /= np.sum(w)

        trees.append(tree)
        betas[m] = beta
        epsilons[m] = eps_clipped

    return trees, betas, epsilons


def adaboost_stage_scores(trees, betas, X):
    """Cumulative boosting scores for every round.

    Returns an (M, n) float array whose row m - 1 is

        F_m(x) = sum_{k <= m} beta_k * (2 h_k(x) - 1)

    evaluated at every row of X.  One array gives you the whole
    error-versus-rounds curve, so build it once per configuration.

    If trees is empty, return an array of shape (0, n).
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    M = len(trees)

    if M == 0:
        return np.empty((0, n), dtype=float)

    signed = np.empty((M, n), dtype=float)
    for m, tree in enumerate(trees):
        h = tree_predict(tree, X)
        signed[m] = 2 * h - 1

    weighted = np.asanyarray(betas, dtype=float)[:, None] * signed
    return np.cumsum(weighted, axis=0)


def adaboost_predict(scores):
    """Convert scores to {0, 1} labels.

    Convention C17: class 1 where the score is STRICTLY positive, class 0
    otherwise.  A score of exactly 0.0 is class 0.  The input may have any
    shape; the output has the same shape.
    """
    scores = np.asarray(scores, dtype=float)
    return (scores > 0).astype(int)


def exp_loss_bound(epsilons):
    """Cumulative training error bound of AdaBoost.

    Returns an (M,) array whose entry m - 1 is

        prod_{k <= m} 2 sqrt(eps_k (1 - eps_k)).
    """
    eps = np.asarray(epsilons, dtype=float)
    factors = 2.0 * np.sqrt(eps * (1.0 - eps))
    return np.cumprod(factors)


def normalised_margins(scores, y, betas):
    """Normalised margins of the round-m ensemble, for every round.

    Returns an (M, n) array whose entry (m - 1, i) is

        s_i * F_m(x_i) / sum_{k <= m} |beta_k|,     s_i = 2 y_i - 1.

    The normalisation makes margins comparable across rounds: every entry
    lies in [-1, 1] and the sign says whether the point is classified
    correctly.

    Convention C18: if the cumulative sum of |beta| is zero for some round,
    use a denominator of 1.0 for that round rather than dividing by zero.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    betas = np.asarray(betas, dtype=float)

    s = 2 * y - 1  # signed labels in {-1, +1}

    cum_abs_beta = np.cumsum(np.abs(betas))
    denom = np.where(cum_abs_beta > 0, cum_abs_beta, 1.0)
    return scores * s[None, :] / denom[:, None]
