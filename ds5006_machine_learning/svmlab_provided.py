"""
DS5006 Fall 2026 -- Lab 4 (Support Vector Machines)
PROVIDED MODULE.  Do not modify this file.  It is replaced by a clean copy
when your submission is graded, so any change you make here is discarded.

It gives you two things:

    make_dataset(roll_number)   your personal train / test split
    solve_dual(K, y, C)         the solution of the SVM dual quadratic
                                programme, as a vector of multipliers alpha

Everything else in the lab you write yourself, in svm_lab.py.

-------------------------------------------------------------------------
On solve_dual
-------------------------------------------------------------------------
The function wraps `cvxopt.solvers.qp`, which minimises

        (1/2) x^T P x + q^T x     subject to    G x <= h,   A x = b.

Section 6.5 of the lecture notes states the dual as a MAXIMISATION

        max_a  sum_i a_i - (1/2) sum_i sum_j a_i a_j y_i y_j K(x_i, x_j)
        subject to   sum_i a_i y_i = 0,   0 <= a_i <= C.

Mapping one onto the other is the part worth reading the source for: the
sign flip that turns the maximisation into a minimisation, the two stacked
blocks of G that encode a two-sided bound as one-sided inequalities, and
the single row of A that carries sum_i a_i y_i = 0.

The solver tolerances below are deliberately tighter than the cvxopt
defaults.  At the default settings a multiplier that should sit exactly at
the upper bound C can land about 1e-6 away from it, which is enough to be
misread as a free support vector and to corrupt the intercept.  Tightened,
those multipliers land within about 2e-15 of C.  You do not need to change
these; you do need to know that "alpha_i equals C" is a statement about
floating point, not about arithmetic.
"""

import numpy as np

try:
    from cvxopt import matrix, solvers
except ImportError as exc:                                   # pragma: no cover
    raise ImportError(
        "This lab needs cvxopt.  Install it with\n"
        "    pip install cvxopt\n"
        "and run `python selftest_lab4.py` to check your setup."
    ) from exc


# ----------------------------------------------------------------------
# The quadratic programme
# ----------------------------------------------------------------------
_QP_OPTIONS = {
    "show_progress": False,
    "abstol":   1e-12,
    "reltol":   1e-12,
    "feastol":  1e-12,
    "maxiters":  200,
}


def solve_dual(K, y, C):
    """Solve the SVM dual for the multipliers alpha.

    Parameters
    ----------
    K : (n, n) ndarray
        Kernel matrix of the training inputs with themselves.  Must be
        symmetric; it is symmetrised defensively before use.
    y : (n,) ndarray
        Labels, each either -1.0 or +1.0.
    C : float
        Upper bound on each multiplier (the soft-margin penalty).

    Returns
    -------
    alpha : (n,) ndarray
        The maximiser of the dual objective, satisfying
        sum_i alpha_i y_i = 0 and 0 <= alpha_i <= C.

    Raises
    ------
    ValueError
        If the solver cannot factor the problem.  This happens when the
        kernel matrix is far from positive semidefinite.
    """
    K = np.asarray(K, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = y.shape[0]

    if K.shape != (n, n):
        raise ValueError(f"K has shape {K.shape}, expected ({n}, {n})")
    if not np.all(np.isin(y, (-1.0, 1.0))):
        raise ValueError("y must contain only -1.0 and +1.0")
    if C <= 0:
        raise ValueError(f"C must be positive, got {C}")

    K = 0.5 * (K + K.T)                      # symmetrise
    P = np.outer(y, y) * K                   # Q_ij = y_i y_j K_ij
    q = -np.ones(n)                          # the -sum_i a_i term

    # 0 <= a_i <= C written as two stacked one-sided blocks
    G = np.vstack([-np.eye(n), np.eye(n)])
    h = np.concatenate([np.zeros(n), np.full(n, float(C))])

    A = y.reshape(1, n)                      # sum_i a_i y_i = 0
    b = np.zeros(1)

    saved = {k: solvers.options.get(k) for k in _QP_OPTIONS}
    solvers.options.update(_QP_OPTIONS)
    try:
        sol = solvers.qp(matrix(P), matrix(q),
                         matrix(G), matrix(h),
                         matrix(A), matrix(b))
    finally:
        for k, v in saved.items():
            if v is None:
                solvers.options.pop(k, None)
            else:
                solvers.options[k] = v

    alpha = np.array(sol["x"]).ravel()
    return np.clip(alpha, 0.0, float(C))     # kill interior-point overshoot


# ----------------------------------------------------------------------
# The data
# ----------------------------------------------------------------------
def _two_moons(n, noise, rng):
    """Two interleaving half-circles with Gaussian jitter."""
    m = n // 2
    t1 = np.pi * rng.random(m)
    lower = np.column_stack([np.cos(t1), np.sin(t1)])
    t2 = np.pi * rng.random(n - m)
    upper = np.column_stack([1.0 - np.cos(t2), 0.5 - np.sin(t2)])
    X = np.vstack([lower, upper]) + noise * rng.standard_normal((n, 2))
    y = np.concatenate([-np.ones(m), np.ones(n - m)])
    return X, y


def make_dataset(roll_number):
    """Return (X_train, y_train, X_test, y_test) for one student.

    The sample is drawn from two interleaved noisy half-moons in the plane.
    The classes overlap, so no hyperplane -- and no kernel -- separates the
    training set perfectly; this is deliberate.

        X_train : (240, 2)      y_train : (240,)   labels in {-1, +1}
        X_test  : (2000, 2)     y_test  : (2000,)

    The seed is derived from your roll number, so your data differ from
    everyone else's while remaining identical on every machine you run on.
    Pass your roll number exactly as it appears on Moodle, as a string.
    """
    import hashlib

    key = str(roll_number).strip().upper().encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")

    rng_tr = np.random.default_rng(seed)
    rng_te = np.random.default_rng(seed ^ 0x5DEECE66D)

    X_tr, y_tr = _two_moons(240, 0.40, rng_tr)
    X_te, y_te = _two_moons(2000, 0.40, rng_te)
    return X_tr, y_tr, X_te, y_te


# ----------------------------------------------------------------------
# The experiment grids.  Use these exactly; do not invent your own.
# ----------------------------------------------------------------------
C_GRID = np.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1000.0])

S2_GRID = np.array([1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1e4])

TOL = 1e-6          # the threshold separating the three multiplier regimes
S2_DEFAULT = 1.0    # kernel width to hold fixed when sweeping C
C_DEFAULT = 1.0     # penalty to hold fixed when sweeping s^2
