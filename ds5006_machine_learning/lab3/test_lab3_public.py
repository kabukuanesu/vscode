"""DS5006 Lab 3 - public self-checks.

Run this from the folder containing your ds5006_lab3.py:

    python3 test_lab3_public.py

These checks test the CONTRACTS, not the science.  Passing all of them does
not mean your answers are right; failing any of them means the hidden tests
will fail too.  No pytest needed.
"""

import sys
import time
import warnings

import numpy as np

import ds5006_lab3 as L

RESULTS = []


def check(name):
    """Decorator registering a single named check."""
    def wrap(fn):
        RESULTS.append((name, fn))
        return fn
    return wrap


def close(a, b, tol=1e-9):
    """Assert two arrays or scalars agree to tolerance."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape}"
    assert np.allclose(a, b, atol=tol, rtol=0.0), f"{a} != {b}"


# ---------------------------------------------------------------- Part A

@check("class_distribution: uniform weights")
def _():
    close(L.class_distribution(np.array([0, 0, 1, 1, 1])), [0.4, 0.6])


@check("class_distribution: w=None equals ones, not ones/n")
def _():
    y = np.array([0, 1, 1, 1])
    close(L.class_distribution(y), L.class_distribution(y, np.ones(4)))


@check("class_distribution: weighted")
def _():
    y = np.array([0, 1])
    close(L.class_distribution(y, np.array([3.0, 1.0])), [0.75, 0.25])


@check("class_distribution: zero total weight returns zeros (C1)")
def _():
    close(L.class_distribution(np.array([0, 1]), np.zeros(2)), [0.0, 0.0])
    close(L.class_distribution(np.array([], dtype=int)), [0.0, 0.0])


@check("impurities: values on a known distribution")
def _():
    p = np.array([0.25, 0.75])
    close(L.gini(p), 0.375)
    close(L.entropy(p), -(0.25 * np.log(0.25) + 0.75 * np.log(0.75)))
    close(L.misclassification(p), 0.25)


@check("entropy: nats not bits (C3)")
def _():
    close(L.entropy(np.array([0.5, 0.5])), np.log(2.0))


@check("impurities: zeros give 0 without warnings (C3)")
def _():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            close(L.entropy(np.array([0.0, 1.0])), 0.0)
            close(L.entropy(np.array([0.0, 0.0])), 0.0)


@check("impurities: broadcast over a stack (C2)")
def _():
    P = np.array([[0.5, 0.5], [1.0, 0.0], [0.25, 0.75]])
    for fn, want in ((L.gini, [0.5, 0.0, 0.375]),
                     (L.misclassification, [0.5, 0.0, 0.25])):
        out = np.asarray(fn(P))
        assert out.shape == (3,), f"{fn.__name__} returned shape {out.shape}"
        close(out, want)


@check("candidate_thresholds: midpoints of distinct values")
def _():
    close(L.candidate_thresholds(np.array([1.0, 3.0, 1.0, 7.0])), [2.0, 5.0])


@check("candidate_thresholds: constant column gives none")
def _():
    out = np.asarray(L.candidate_thresholds(np.full(5, 2.0)))
    assert out.shape == (0,), f"expected shape (0,), got {out.shape}"


@check("best_split: finds the obvious split")
def _():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    j, t, g = L.best_split(X, y, None, L.gini)
    assert j == 0, f"feature {j}"
    close(t, 1.5)
    close(g, 0.5)


@check("best_split: no admissible split returns (-1, nan, 0.0) (C4)")
def _():
    X = np.array([[1.0], [1.0], [1.0]])
    y = np.array([0, 1, 0])
    j, t, g = L.best_split(X, y, None, L.gini)
    assert j == -1 and np.isnan(t) and g == 0.0, f"got {(j, t, g)}"


@check("best_split: pure node has no admissible split (C4)")
def _():
    X = np.arange(6.0).reshape(6, 1)
    j, _, g = L.best_split(X, np.zeros(6, dtype=int), None, L.gini)
    assert j == -1 and g == 0.0, f"got {(j, g)}"


@check("best_split: ties go to the smaller feature index (C5)")
def _():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 0, 1])           # column 1 alone decides the label
    X = X[:, [1, 1]]                     # two identical perfect columns
    j, _, _ = L.best_split(X, y, None, L.gini)
    assert j == 0, f"expected feature 0 on a tie, got {j}"


@check("best_split: respects sample weights")
def _():
    # Weighted, the node is balanced and the split at 0.5 is perfect: gain 1/2.
    # Unweighted the same split has gain 4/9, so this distinguishes the two.
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 1])
    j, t, g = L.best_split(X, y, np.array([2.0, 1.0, 1.0]), L.gini)
    assert j == 0, f"feature {j}"
    close(t, 0.5)
    close(g, 0.5)


@check("grow_tree: max_depth=0 gives a single leaf (C9)")
def _():
    X = np.arange(8.0).reshape(8, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    node = L.grow_tree(X, y, None, L.gini, 0, 2)
    assert node["leaf"] is True, "expected a leaf"
    assert node["n"] == 8 and node["pred"] in (0, 1)
    close(node["dist"], [0.5, 0.5])


@check("grow_tree: node keys are exactly as specified (C6)")
def _():
    X = np.arange(8.0).reshape(8, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    node = L.grow_tree(X, y, None, L.gini, 3, 2)
    assert node["leaf"] is False
    for key in ("feature", "threshold", "gain", "left", "right",
                "pred", "n", "dist"):
        assert key in node, f"missing key {key!r}"
    assert node["left"]["leaf"] and node["right"]["leaf"], "expected a stump"


@check("grow_tree: leaf prediction breaks ties to the smaller label (C7)")
def _():
    X = np.zeros((4, 1))
    y = np.array([0, 0, 1, 1])
    assert L.grow_tree(X, y, None, L.gini, 3, 2)["pred"] == 0


@check("grow_tree: min_samples_split is checked before splitting (C8)")
def _():
    X = np.arange(4.0).reshape(4, 1)
    y = np.array([0, 0, 1, 1])
    assert L.grow_tree(X, y, None, L.gini, 5, 5)["leaf"] is True


@check("tree_predict: exact on a separable problem")
def _():
    X = np.arange(8.0).reshape(8, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    node = L.grow_tree(X, y, None, L.gini, 4, 2)
    out = np.asarray(L.tree_predict(node, X))
    assert out.shape == (8,), f"shape {out.shape}"
    assert np.array_equal(out, y), f"{out} != {y}"


@check("tree_predict: values equal to the threshold go left")
def _():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    node = L.grow_tree(X, y, None, L.gini, 1, 2)
    t = node["threshold"]
    probe = np.array([[t]])
    assert L.tree_predict(node, probe)[0] == node["left"]["pred"]


@check("tree_stats: a single leaf is (1, 0) (C10)")
def _():
    node = L.grow_tree(np.zeros((4, 1)), np.zeros(4, dtype=int), None,
                       L.gini, 0, 2)
    assert tuple(L.tree_stats(node)) == (1, 0), L.tree_stats(node)


@check("tree_stats: a stump is (2, 1)")
def _():
    X = np.arange(8.0).reshape(8, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    node = L.grow_tree(X, y, None, L.gini, 1, 2)
    assert tuple(L.tree_stats(node)) == (2, 1), L.tree_stats(node)


# ---------------------------------------------------------------- Part B

@check("bootstrap_indices: exactly one rng.integers call (C11)")
def _():
    a = L.bootstrap_indices(np.random.default_rng(0), 7)
    b = np.random.default_rng(0).integers(0, 7, size=7)
    a = np.asarray(a)
    assert a.shape == (7,), f"shape {a.shape}"
    assert np.array_equal(a, b), f"{a} != {b}"


@check("bootstrap_indices: the stream advances by one draw")
def _():
    rng = np.random.default_rng(5)
    L.bootstrap_indices(rng, 11)
    ref = np.random.default_rng(5)
    ref.integers(0, 11, size=11)
    close(rng.random(), ref.random())


@check("fit_bagged_forest: shapes and oob consistency (C12)")
def _():
    X = np.arange(40.0).reshape(20, 2)
    y = (np.arange(20) % 2).astype(int)
    trees, oob = L.fit_bagged_forest(X, y, np.random.default_rng(1), 6, 3, 2)
    oob = np.asarray(oob)
    assert len(trees) == 6, len(trees)
    assert oob.shape == (6, 20) and oob.dtype == bool, (oob.shape, oob.dtype)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, 20, size=20)
    want = np.ones(20, dtype=bool)
    want[idx] = False
    assert np.array_equal(oob[0], want), "oob mask of tree 0 is wrong"


@check("forest_predict_matrix: shape (B, n)")
def _():
    X = np.arange(40.0).reshape(20, 2)
    y = (np.arange(20) % 2).astype(int)
    trees, _ = L.fit_bagged_forest(X, y, np.random.default_rng(2), 4, 3, 2)
    P = np.asarray(L.forest_predict_matrix(trees, X))
    assert P.shape == (4, 20), P.shape


@check("majority_vote: ties go to the smaller label (C13)")
def _():
    P = np.array([[0, 1, 1], [1, 1, 0]])
    close(np.asarray(L.majority_vote(P)), [0, 1, 0])


@check("oob_error: rows with no out of bag tree are dropped (C14)")
def _():
    P = np.array([[1, 1, 1], [1, 1, 1]])
    mask = np.array([[True, False, False], [False, False, False]])
    y = np.array([0, 0, 1])
    close(L.oob_error(P, mask, y), 1.0)      # only row 0 counts, and it is wrong
    assert np.isnan(L.oob_error(P, np.zeros_like(mask), y))


# ---------------------------------------------------------------- Part C

@check("adaboost_fit: shapes, normalised weights, clipped errors")
def _():
    X = np.arange(60.0).reshape(30, 2)
    y = (np.arange(30) % 2).astype(int)
    trees, betas, eps = L.adaboost_fit(X, y, 5, 1, 2)
    betas, eps = np.asarray(betas), np.asarray(eps)
    assert len(trees) == 5 and betas.shape == (5,) and eps.shape == (5,)
    assert np.all(eps >= 0.0) and np.all(eps <= 1.0), eps
    close(betas, 0.5 * np.log((1.0 - eps) / eps), tol=1e-8)


@check("adaboost_fit: round 1 uses uniform weights (C15)")
def _():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] + 0.3 * rng.normal(size=40) > 0).astype(int)
    _, _, eps = L.adaboost_fit(X, y, 1, 1, 2)
    stump = L.grow_tree(X, y, None, L.gini, 1, 2)
    want = float(np.mean(L.tree_predict(stump, X) != y))
    close(np.asarray(eps)[0], want, tol=1e-8)


@check("adaboost_stage_scores: cumulative in the rounds")
def _():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    trees, betas, _ = L.adaboost_fit(X, y, 4, 1, 2)
    F = np.asarray(L.adaboost_stage_scores(trees, betas, X))
    assert F.shape == (4, 40), F.shape
    one = np.asarray(L.adaboost_stage_scores(trees[:1], np.asarray(betas)[:1], X))
    close(F[0], one[0])


@check("adaboost_predict: a score of exactly zero is class 0 (C17)")
def _():
    close(np.asarray(L.adaboost_predict(np.array([-1.0, 0.0, 1e-12]))),
          [0.0, 0.0, 1.0])


@check("exp_loss_bound: cumulative product")
def _():
    eps = np.array([0.25, 0.4])
    want = np.cumprod(2.0 * np.sqrt(eps * (1.0 - eps)))
    close(np.asarray(L.exp_loss_bound(eps)), want)


@check("normalised_margins: sign and range (C18)")
def _():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    trees, betas, _ = L.adaboost_fit(X, y, 6, 1, 2)
    F = L.adaboost_stage_scores(trees, betas, X)
    M = np.asarray(L.normalised_margins(F, y, betas))
    assert M.shape == (6, 40), M.shape
    assert np.all(np.abs(M) <= 1.0 + 1e-9), "margins must lie in [-1, 1]"
    pred = np.asarray(L.adaboost_predict(np.asarray(F)))
    assert np.array_equal(M[-1] > 0, pred[-1] == y), "sign disagrees with prediction"


@check("PERFORMANCE: 20 trees on the real training set")
def _():
    try:
        import ds5006_lab3_data as D
    except ImportError:
        print("     (data module not found, skipping)")
        return
    X, y, _, _ = D.load_split("112201001")
    t0 = time.time()
    L.fit_bagged_forest(X, y, np.random.default_rng(0), 20, D.BAG_DEPTH, 2)
    dt = time.time() - t0
    print(f"     20 trees took {dt:.1f}s -> about {dt * 45 / 60:.1f} min "
          f"for the whole lab")
    assert dt < 45.0, (
        f"{dt:.1f}s for 20 trees is far too slow.  The lab fits roughly 900 "
        "trees; re-read the performance notes in best_split and tree_predict.")


def main():
    """Run every check and print a summary."""
    passed, failed = 0, []
    for name, fn in RESULTS:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:                    # pylint: disable=broad-except
            print(f"  FAIL  {name}\n          {type(exc).__name__}: {exc}")
            failed.append(name)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
