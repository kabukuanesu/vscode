"""
DS5006 Fall 2026 -- Lab 5 public self-check.

    python test_lab5_public.py

Put this file next to ds5006_lab5.py and clusterlab_provided.py and run
it.  No pytest needed.

These checks test the CONTRACT (shapes, types, the conventions on tiny
hand-checkable inputs, and the worked examples from the lecture notes).
Passing all of them does NOT guarantee full marks: the grader runs
different, larger hidden tests.  Failing any of them almost certainly
costs marks.
"""

import sys
import time
import traceback

import numpy as np

try:
    import ds5006_lab5 as S
except Exception:                                    # noqa: BLE001
    print("Could not import ds5006_lab5.py:")
    traceback.print_exc()
    sys.exit(1)

from clusterlab_provided import euclidean_distance_matrix

RESULTS = []


def check(name):
    def deco(fn):
        def run():
            t = time.time()
            try:
                fn()
                RESULTS.append((name, True, "", time.time() - t))
            except NotImplementedError:
                RESULTS.append((name, False, "not implemented", 0.0))
            except AssertionError as e:
                RESULTS.append((name, False, str(e) or "assertion failed",
                                time.time() - t))
            except Exception as e:                   # noqa: BLE001
                RESULTS.append((name, False, f"{type(e).__name__}: {e}",
                                time.time() - t))
        run.__name__ = fn.__name__
        CHECKS.append(run)
        return run
    return deco


CHECKS = []
X1 = np.array([[1.0], [2.0], [6.0], [7.0], [9.0]])   # lecture-notes example
D1 = np.abs(X1 - X1.T)


def close(a, b, tol=1e-8):
    return np.allclose(np.asarray(a, float), np.asarray(b, float),
                       rtol=tol, atol=tol)


# ---------------------------------------------------------------- A1
@check("A1 pairwise_sq_dists: shape (n_A, n_B) and values")
def _():
    A = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 4.0]])
    B = np.array([[0.0, 0.0], [3.0, 0.0]])
    D = S.pairwise_sq_dists(A, B)
    assert np.shape(D) == (3, 2), f"shape {np.shape(D)}, expected (3, 2)"
    assert close(D, [[0, 9], [2, 5], [25, 16]]), f"got\n{D}"


@check("A1 pairwise_sq_dists: never negative")
def _():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, 3)) * 1e4
    D = S.pairwise_sq_dists(A, A)
    assert np.min(D) >= 0.0, "negative squared distance (clip at 0)"


@check("A1 within_cluster_scatter: lecture example")
def _():
    W = S.within_cluster_scatter(D1, np.array([0, 0, 1, 1, 1]))
    assert isinstance(W, float), f"return a Python float, got {type(W)}"
    assert close(W, 7.0), f"got {W}, expected 7.0"


# ---------------------------------------------------------------- A2
@check("A2 kmeans_e_step: nearest centre, ties go to smaller index")
def _():
    X = np.array([[0.0], [1.5], [3.0]])
    C = np.array([[1.0], [2.0]])
    lab = S.kmeans_e_step(X, C)
    assert np.issubdtype(np.asarray(lab).dtype, np.integer), "labels must be int"
    assert list(lab) == [0, 0, 1], f"got {list(lab)}, expected [0, 0, 1]"


@check("A2 kmeans_m_step: empty cluster keeps its centre, no in-place edit")
def _():
    X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0]])
    C = np.array([[1.0, 0.0], [5.0, 5.0], [-7.0, 3.0]])
    C0 = C.copy()
    new = S.kmeans_m_step(X, np.array([0, 0, 1]), C)
    assert close(new, [[1, 0], [10, 10], [-7, 3]]), f"got\n{new}"
    assert close(C, C0), "kmeans_m_step modified its `centres` argument"


@check("A2 kmeans_loss: sum, not mean")
def _():
    J = S.kmeans_loss(X1, np.array([0, 1, 1, 1, 1]), np.array([[1.0], [2.0]]))
    assert close(J, 90.0), f"got {J}, expected 90.0"


@check("A2 kmeans: lecture example from m = (1, 2)")
def _():
    lab, C, hist, n_iter = S.kmeans(X1, np.array([[1.0], [2.0]]))
    assert list(lab) == [0, 0, 1, 1, 1], f"labels {list(lab)}"
    assert close(np.ravel(C), [1.5, 22 / 3]), f"centres {np.ravel(C)}"
    assert n_iter == 2, f"n_iter {n_iter}, expected 2"
    assert len(hist) == 1 + 2 * n_iter, "history length must be 1 + 2*n_iter"
    assert close(hist, [90, 26, 11, 31 / 6, 31 / 6]), f"history {hist}"


@check("A2 kmeans: loss never increases")
def _():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    _, _, hist, _ = S.kmeans(X, X[:4])
    assert np.all(np.diff(hist) <= 1e-9), "history went up somewhere"


@check("A2 kmeans: stops at max_iter")
def _():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 2))
    _, _, hist, n_iter = S.kmeans(X, X[:6], max_iter=1)
    assert n_iter == 1 and len(hist) == 3, f"n_iter {n_iter}, len {len(hist)}"


# ---------------------------------------------------------------- A3
@check("A3 kmedoids: lecture data from medoids (0, 1)")
def _():
    lab, med, hist, n_iter = S.kmedoids(D1, np.array([0, 1]))
    assert list(lab) == [0, 0, 1, 1, 1], f"labels {list(lab)}"
    assert list(med) == [0, 3], f"medoids {list(med)}, expected [0, 3]"
    assert close(hist, [16, 8, 5, 4, 4]), f"history {hist}"
    assert n_iter == 2


@check("A3 kmedoids: medoids are row indices of data points")
def _():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 2))
    D = euclidean_distance_matrix(X)
    _, med, _, _ = S.kmedoids(D, np.array([0, 1, 2]))
    med = np.asarray(med)
    assert np.issubdtype(med.dtype, np.integer), "medoids must be int indices"
    assert med.shape == (3,) and med.min() >= 0 and med.max() < 60


# ---------------------------------------------------------------- A4
@check("A4 gaussian_logpdf: standard normal in 2-D")
def _():
    X = np.array([[0.0, 0.0], [1.0, 2.0]])
    lp = S.gaussian_logpdf(X, np.zeros(2), np.eye(2))
    assert np.shape(lp) == (2,), f"shape {np.shape(lp)}"
    want = -np.log(2 * np.pi) - 0.5 * np.array([0.0, 5.0])
    assert close(lp, want), f"got {lp}, expected {want}"


@check("A4 gmm_e_step: rows sum to 1, total log-likelihood")
def _():
    X = np.array([[0.0], [1.0], [4.0]])
    pis = np.array([0.5, 0.5])
    mus = np.array([[0.0], [4.0]])
    Sig = np.array([[[1.0]], [[1.0]]])
    R, ll = S.gmm_e_step(X, pis, mus, Sig)
    assert np.shape(R) == (3, 2)
    assert close(R.sum(axis=1), 1.0), "rows of R must sum to 1"
    lp0 = -0.5 * np.log(2 * np.pi) - 0.5 * (X[:, 0] - 0) ** 2
    lp1 = -0.5 * np.log(2 * np.pi) - 0.5 * (X[:, 0] - 4) ** 2
    want = np.sum(np.log(0.5 * np.exp(lp0) + 0.5 * np.exp(lp1)))
    assert isinstance(ll, float), "loglik must be a Python float"
    assert close(ll, want), f"loglik {ll}, expected {want} (a SUM over points)"


@check("A4 gmm_e_step: a very distant point does not give NaN")
def _():
    X = np.array([[0.0, 0.0], [1e3, 1e3]])
    pis = np.array([0.5, 0.5])
    mus = np.array([[0.0, 0.0], [1.0, 1.0]])
    Sig = np.array([np.eye(2) * 0.01, np.eye(2) * 0.01])
    R, ll = S.gmm_e_step(X, pis, mus, Sig)
    assert np.all(np.isfinite(R)) and np.isfinite(ll), "NaN/inf -- see convention 8"
    assert close(R.sum(axis=1), 1.0)


@check("A4 gmm_m_step: shapes, weights sum to 1, reg on the diagonal")
def _():
    X = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    R = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    pis, mus, Sig = S.gmm_m_step(X, R, reg=0.5)
    assert np.shape(pis) == (2,) and np.shape(mus) == (2, 2)
    assert np.shape(Sig) == (2, 2, 2)
    assert close(pis, [0.5, 0.5]) and close(mus, [[1, 0], [1, 2]])
    assert close(Sig[0], [[1.5, 0], [0, 0.5]]), f"Sigma_1 =\n{Sig[0]}"


@check("A4 gmm_init_from_kmeans: covariance around the GIVEN centre")
def _():
    X = np.array([[0.0], [2.0], [10.0], [12.0]])
    lab = np.array([0, 0, 1, 1])
    C = np.array([[0.0], [11.0]])            # first centre is not the mean
    pis, mus, Sig = S.gmm_init_from_kmeans(X, lab, C, reg=0.0)
    assert close(pis, [0.5, 0.5]) and close(np.ravel(mus), [0, 11])
    assert close(np.ravel(Sig), [2.0, 1.0]), f"got {np.ravel(Sig)}"


@check("A4 gmm_em: history starts at the initial log-likelihood, never falls")
def _():
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(0, 1, (80, 2)), rng.normal(4, 1, (40, 2))])
    pis = np.array([0.5, 0.5])
    mus = X[[0, 100]].copy()
    Sig = np.array([np.eye(2), np.eye(2)])
    _, ll0 = S.gmm_e_step(X, pis, mus, Sig)
    out = S.gmm_em(X, pis, mus, Sig)
    assert len(out) == 6, "gmm_em returns 6 things"
    p, m, s, R, hist, n_iter = out
    assert len(hist) == 1 + n_iter, "history length must be 1 + n_iter"
    assert close(hist[0], ll0), "history[0] must be the INITIAL log-likelihood"
    assert np.all(np.diff(hist) >= -1e-6), "log-likelihood decreased"
    _, llf = S.gmm_e_step(X, p, m, s)
    assert close(hist[-1], llf), "last history entry must match final params"


# ---------------------------------------------------------------- A5
@check("A5 agglomerative: lecture data, all linkages, tie rule")
def _():
    want = {
        "single":   [[0, 1, 1, 2], [2, 3, 1, 2], [4, 6, 2, 3], [5, 7, 4, 5]],
        "complete": [[0, 1, 1, 2], [2, 3, 1, 2], [4, 6, 3, 3], [5, 7, 8, 5]],
        "average":  [[0, 1, 1, 2], [2, 3, 1, 2], [4, 6, 2.5, 3],
                     [5, 7, 35 / 6, 5]],
    }
    for link, w in want.items():
        mg = S.agglomerative(D1, link)
        assert np.shape(mg) == (4, 4), f"{link}: shape {np.shape(mg)}"
        assert close(mg, w), f"{link}: got\n{np.asarray(mg)}\nexpected\n{np.array(w)}"


@check("A5 cut_dendrogram: canonical labels")
def _():
    mg = np.array([[0, 1, 1, 2], [2, 3, 1, 2], [4, 6, 2, 3], [5, 7, 4, 5]], float)
    assert list(S.cut_dendrogram(mg, 5, 2)) == [0, 0, 1, 1, 1]
    assert list(S.cut_dendrogram(mg, 5, 3)) == [0, 0, 1, 1, 2]
    assert list(S.cut_dendrogram(mg, 5, 5)) == [0, 1, 2, 3, 4]
    assert list(S.cut_dendrogram(mg, 5, 1)) == [0, 0, 0, 0, 0]


# ---------------------------------------------------------------- B
@check("B  experiments return the right keys (needs ROLL_NUMBER)")
def _():
    roll = S.ROLL_NUMBER
    assert roll, "set ROLL_NUMBER at the top of ds5006_lab5.py"
    keys = {
        S.experiment_1: {"best_J", "worst_J"},
        S.experiment_2: {"kmeans_centres", "kmedoids_medoids",
                         "kmeans_shift", "kmedoids_shift"},
        S.experiment_3: {"kmeans_acc", "gmm_acc", "gmm_pis_sorted",
                         "gmm_initial_ll", "gmm_final_ll", "gmm_n_iter"},
        S.experiment_4: {"acc", "top_height"},
    }
    import json
    for fn, k in keys.items():
        out = fn(roll)
        assert set(out) == k, f"{fn.__name__}: keys {sorted(out)}, expected {sorted(k)}"
        json.dumps(out)        # must be JSON-serialisable


if __name__ == "__main__":
    for c in CHECKS:
        c()
    w = max(len(n) for n, *_ in RESULTS)
    npass = 0
    for name, ok, msg, dt in RESULTS:
        npass += ok
        tag = "PASS" if ok else "FAIL"
        line = f"[{tag}] {name:<{w}}  {dt:5.2f}s"
        if not ok:
            line += "\n        " + msg.replace("\n", "\n        ")
        print(line)
    print(f"\n{npass}/{len(RESULTS)} checks passed")
    sys.exit(0 if npass == len(RESULTS) else 1)
