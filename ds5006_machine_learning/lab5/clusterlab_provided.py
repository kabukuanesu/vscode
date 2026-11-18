"""
DS5006 Fall 2026 -- Lab 5: Clustering.  PROVIDED MODULE -- DO NOT EDIT.

This file is replaced with a clean copy when your work is graded, so any
change you make here is lost and your code must not depend on it.

What is here
------------
* make_dataset(roll, name)   -> the dataset you use for each experiment.
                                 Every roll number gets a different one.
* initial_centre_indices(roll, name, K, r)
                              -> row indices of X to use as the r-th set of
                                 K initial centres / medoids.  Use exactly
                                 these; do not invent your own starts.
* euclidean_distance_matrix(X)
                              -> (N, N) matrix of (unsquared) Euclidean
                                 distances.  Use it for K-medoids and for
                                 hierarchical clustering in Part B.
* matching_accuracy(labels, y_true)
                              -> fraction of points correctly grouped, under
                                 the best one-to-one matching of cluster
                                 labels to true labels.
* Experiment constants (K_RANGE, N_RESTARTS, GMM_REG, GMM_TOL, ...).
"""

import hashlib
import itertools

import numpy as np

# ----------------------------------------------------------------------
# Experiment constants.  Use these names; do not hard-code the values.
# ----------------------------------------------------------------------
K_RANGE = list(range(1, 9))   # Experiment 1: K = 1, ..., 8
N_RESTARTS = 10               # Experiment 1: initialisations per K
KMEANS_MAX_ITER = 100
KMEDOIDS_MAX_ITER = 100
GMM_REG = 1e-6                # added to every covariance diagonal
GMM_TOL = 1e-6                # EM stops when the log-likelihood gain < tol
GMM_MAX_ITER = 500
LINKAGES = ["single", "complete", "average"]

DATASETS = ["blobs", "outliers", "aniso", "moons"]


# ----------------------------------------------------------------------
# Seeding
# ----------------------------------------------------------------------
def _seed(*parts):
    """Deterministic 64-bit seed from strings; identical on every machine."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16)


def _norm_roll(roll):
    roll = str(roll).strip().upper()
    if not roll:
        raise ValueError("roll number is empty -- set ROLL_NUMBER")
    return roll


# ----------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------
def _blobs(rng):
    k_true = int(rng.integers(3, 6))                 # 3, 4 or 5 groups
    # centres on a jittered circle so that groups are well separated
    phase = rng.uniform(0, 2 * np.pi)
    ang = phase + 2 * np.pi * np.arange(k_true) / k_true
    ang = ang + rng.uniform(-0.15, 0.15, k_true)
    radius = rng.uniform(5.0, 6.0)
    centres = np.c_[radius * np.cos(ang), radius * np.sin(ang)]
    sizes = np.full(k_true, 300 // k_true)
    sizes[: 300 - sizes.sum()] += 1
    X = np.vstack([c + rng.normal(0, 1.0, (s, 2)) for c, s in zip(centres, sizes)])
    y = np.repeat(np.arange(k_true), sizes)
    return X, y


def _outliers(rng):
    # two compact groups and a tight knot of 6 far-away outliers
    ang = rng.uniform(0, 2 * np.pi)
    u = np.array([np.cos(ang), np.sin(ang)])
    v = np.array([-u[1], u[0]])
    c1, c2 = -3.0 * u, 3.0 * u
    n = 120
    X1 = c1 + rng.normal(0, 1.0, (n, 2))
    X2 = c2 + rng.normal(0, 1.0, (n, 2))
    far = 6.0 * u + 28.0 * v + rng.normal(0, 0.5, (6, 2))
    X = np.vstack([X1, X2, far])
    y = np.r_[np.zeros(n, int), np.ones(n, int), np.full(6, 2)]
    return X, y


def _aniso(rng):
    # two long, parallel, closely spaced groups and one small compact group
    sizes = np.array([240, 120, 40])
    gap = rng.uniform(2.3, 2.7)
    tilt = rng.uniform(-0.12, 0.12, 2)          # small, independent tilts
    centres = np.array([[0.0, 0.0], [0.0, gap], [rng.uniform(7.5, 8.5), gap / 2]])
    parts = []
    for k, (c, s) in enumerate(zip(centres, sizes)):
        if k < 2:
            th = tilt[k]
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            A = R @ np.diag([3.0, 0.35])
        else:
            A = np.diag([0.5, 0.5])
        parts.append(c + rng.normal(0, 1.0, (s, 2)) @ A.T)
    rot = rng.uniform(0, 2 * np.pi)
    R0 = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    X = np.vstack(parts) @ R0.T + rng.uniform(-3, 3, 2)
    y = np.repeat(np.arange(3), sizes)
    return X, y


def _moons(rng):
    n = 100
    t1 = np.sort(rng.uniform(0, np.pi, n))
    t2 = np.sort(rng.uniform(0, np.pi, n))
    A = np.c_[np.cos(t1), np.sin(t1)]
    B = np.c_[1.0 - np.cos(t2), 0.5 - np.sin(t2)]
    X = np.vstack([A, B]) * 3.0 + rng.normal(0, 0.10, (2 * n, 2))
    rot = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    X = X @ R.T
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    return X, y


_MAKERS = {"blobs": _blobs, "outliers": _outliers, "aniso": _aniso,
           "moons": _moons}


def make_dataset(roll, name):
    """Return (X, y_true) for dataset `name` and this roll number.

    X has shape (N, 2).  y_true holds the generating group of every point.
    y_true is for EVALUATION ONLY (matching_accuracy, plots); none of the
    clustering functions may use it.

    The rows are shuffled, so the groups are not stored in blocks.
    """
    if name not in _MAKERS:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    rng = np.random.default_rng(_seed("ds5006-lab5", _norm_roll(roll), name))
    X, y = _MAKERS[name](rng)
    perm = rng.permutation(len(X))
    return X[perm].astype(float), y[perm].astype(int)


def initial_centre_indices(roll, name, K, r=0):
    """Row indices of K distinct points of make_dataset(roll, name)[0].

    Use X[idx] as initial centres (K-means, and hence GMM) or idx itself as
    initial medoids (K-medoids).  r = 0, 1, ... selects the restart.

    For the "outliers" dataset with K = 2 the two starting points are
    always drawn one from each of the two real groups.
    """
    X, y = make_dataset(roll, name)
    rng = np.random.default_rng(_seed("ds5006-lab5-init", _norm_roll(roll),
                                      name, int(K), int(r)))
    if name == "outliers" and int(K) == 2:
        # one start inside each of the two real groups, never an outlier
        a = rng.choice(np.flatnonzero(y == 0))
        b = rng.choice(np.flatnonzero(y == 1))
        return np.sort(np.array([a, b]))
    return np.sort(rng.choice(len(X), size=int(K), replace=False))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def euclidean_distance_matrix(X):
    """(N, N) unsquared Euclidean distances, exactly symmetric, zero diagonal."""
    X = np.asarray(X, float)
    sq = np.sum(X * X, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def matching_accuracy(labels, y_true):
    """Best-permutation agreement between cluster labels and true labels.

    Works when the numbers of clusters differ: unmatched clusters count as
    wrong.  Brute force over permutations, fine for up to ~7 labels.
    """
    labels = np.asarray(labels).ravel()
    y_true = np.asarray(y_true).ravel()
    a = np.unique(labels)
    b = np.unique(y_true)
    C = np.zeros((len(a), len(b)))
    for i, la in enumerate(a):
        for j, lb in enumerate(b):
            C[i, j] = np.sum((labels == la) & (y_true == lb))
    best = 0.0
    if len(a) <= len(b):
        for perm in itertools.permutations(range(len(b)), len(a)):
            best = max(best, sum(C[i, perm[i]] for i in range(len(a))))
    else:
        for perm in itertools.permutations(range(len(a)), len(b)):
            best = max(best, sum(C[perm[j], j] for j in range(len(b))))
    return float(best / len(labels))
