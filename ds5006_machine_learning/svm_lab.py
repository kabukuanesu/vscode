"""
DS5006 Fall 2026 -- Lab 4: Support Vector Machines
================================================================

YOUR NAME : Anesu Kabuku
YOUR ROLL : 142602026

Fill in the seven functions in Part A, then the three experiments in
Part C, then run this file to write results.json.

Rules
-----
* NumPy only, plus the provided module.  Do not import scikit-learn,
  scipy, cvxopt (directly), or any other learning or optimisation
  library.  You may import matplotlib in your plotting code, but not
  inside the seven functions.
* Do not modify svmlab_provided.py.  It is replaced with a clean copy
  when your work is graded.
* Vectorise.  rbf_kernel is called with a 2000 x 240 argument; a Python
  loop there will turn a one-minute lab into a twenty-minute one.

Conventions -- these are graded exactly as written
--------------------------------------------------
 1. The Gaussian kernel is parameterised by s^2, NOT by gamma:
        K(x, x') = exp( -||x - x'||^2 / (2 * s2) )
 2. Both kernels take (A, B) and return shape (n_A, n_B), in that
    order.  K_test is therefore (n_test, n_train).
 3. Regime codes are the integers
        0 = not a support vector
        1 = free support vector
        2 = bounded support vector
 4. The regime tests are strict below and inclusive above:
        non-SV   : alpha <= tol
        free     : alpha  > tol   AND   alpha < C - tol
        bounded  : alpha >= C - tol
 5. margin_width returns 2 / ||theta||, not 1 / ||theta||.
 6. recover_intercept averages over FREE support vectors only.  When
    there are none it returns the midpoint of the KKT interval --
    Section 3 of the lab document derives it.
 7. kkt_residual returns 0.0 when there are no free support vectors.

A note on what you are and are not implementing
-----------------------------------------------
solve_dual is given to you.  The quadratic programme is a solved problem
and typing an optimiser is not what this lab is testing.  What you
implement is everything that turns its output into a classifier -- and in
particular the intercept, which the dual discards.
"""

import json
import numpy as np

from svmlab_provided import (
    make_dataset, solve_dual,
    C_GRID, S2_GRID, TOL, S2_DEFAULT, C_DEFAULT,
)

ROLL_NUMBER = 142602026


# ======================================================================
# Part A -- the seven functions                                    (40)
# ======================================================================

def linear_kernel(A, B):
    """K(a, b) = a^T b.

    A is (n_A, d), B is (n_B, d); return shape (n_A, n_B).
    """
    return np.asarray(A) @ np.asarray(B).T


def rbf_kernel(A, B, s2):
    """K(a, b) = exp( -||a - b||^2 / (2 * s2) ).

    Vectorise.  Expanding ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a^T b gives
    the whole matrix in three lines and no Python loop.  Clip the squared
    distances at zero before the exponential -- rounding can make them
    slightly negative.

    Return shape (n_A, n_B).
    """
    A = np.asarray(A, dtype= float)
    B = np.asarray(B, dtype= float)

    if s2 <= 0:
        raise ValueError("s2 must be positive")

    A2 = np.sum(A * A, axis=1)[:, None]
    B2 = np.sum(B * B, axis=1)[None, :]

    D2 = A2 + B2 - 2.0 * (A @ B.T)
    D2 = np.maximum(D2, 0.0)

    return np.exp(-D2 / (2 * s2))


def classify_regimes(alpha, C, tol):
    """Label each training point 0, 1 or 2 by its multiplier.

    See conventions 3 and 4 above, and Proposition 6.1 of the notes.
    Return a (n,) array of int.
    """
    alpha = np.asarray(alpha, dtype=float)

    code = np.zeros(alpha.shape, dtype=int)
    code[(alpha > tol) & (alpha < C - tol)] = 1
    code[alpha >= C - tol] = 2

    return code


def recover_intercept(alpha, y, K, C, tol):
    """theta_0, averaged over the FREE support vectors.

    Each free support vector satisfies y_i g(x_i) = 1 exactly, which
    rearranges to theta_0 = y_i - f_i, where

        f_i = sum_j alpha_j y_j K_ij

    is the discriminant WITHOUT the intercept.  In exact arithmetic every
    free support vector gives the same answer; in floating point they do
    not, so average them.

    Bounded support vectors satisfy y_i g(x_i) <= 1, not = 1, so they
    must be excluded.  Work out why before you write the line.

    If there are no free support vectors the KKT conditions constrain
    theta_0 only to an interval [L, U]; return its midpoint.  Section 3
    of the lab document gives L and U.  This is not a hypothetical case:
    it occurs at the small-C end of the sweep in Experiment 2.

    Return a float.
    """
    alpha = np.asarray(alpha, dtype=float)
    y = np.asarray(y, dtype=float)
    K = np.asarray(K, dtype=float)

    f = K @ (alpha * y)

    free = (alpha > tol) & (alpha < C - tol)

    if np.any(free):
        return float(np.mean(y[free] - f[free]))

    below_C = alpha < C - tol
    above_zero = alpha > tol

    lower_mask = ((y > 0) & below_C) | ((y < 0) & above_zero)
    upper_mask = ((y > 0) & above_zero) | ((y < 0) & below_C)

    lower_values = y[lower_mask] - f[lower_mask]
    upper_values = y[upper_mask] - f[upper_mask]

    L = float(np.max(lower_values)) if lower_values.size else -np.inf
    U = float(np.min(upper_values)) if upper_values.size else np.inf

    if np.isfinite(L) and np.isfinite(U):
        return float((L + U) / 2.0)
    if np.isfinite(L):
        return L
    if np.isfinite(U):
        return U

    raise ValueError("Cannot determine an intercept interval")


def decision_values(alpha, theta0, y, K_test):
    """g(x) for every row of K_test.

    K_test is (n_test, n_train); return shape (n_test,).
    """
    alpha = np.asarray(alpha, dtype=float)
    y = np.asarray(y, dtype=float)
    K_test = np.asarray(K_test, dtype=float)

    return K_test @ (alpha * y) + theta0


def margin_width(alpha, y, K):
    """2 / ||theta||, computed from the Gram matrix alone.

    Under a kernel you cannot form theta -- it lives in the feature
    space, which for the Gaussian kernel is infinite-dimensional.  But
    ||theta||^2 is still available from the multipliers and K alone.
    Derive the expression from theta = sum_i alpha_i y_i phi(x_i).

    Return float('inf') if ||theta||^2 is non-positive.
    """
    alpha = np.asarray(alpha, dtype=float)
    y = np.asarray(y, dtype=float)
    K = np.asarray(K, dtype=float)

    ay = alpha * y
    norm_squared = float(ay @ K @ ay)

    if norm_squared <= 0.0:
        return float('inf')

    return float(2.0 / np.sqrt(norm_squared))


def kkt_residual(alpha, theta0, y, K, C, tol):
    """max over FREE support vectors of | y_i g(x_i) - 1 |.

    A measure of how well the fitted model satisfies the condition that
    defines a free support vector.  With the correct intercept it should
    sit at solver precision (around 1e-10 or smaller).  Experiment 1 uses
    it to tell a correct intercept from a plausible-looking wrong one.

    Return 0.0 if there are no free support vectors.
    """
    alpha = np.asarray(alpha, dtype=float)
    y = np.asarray(y, dtype=float)
    K = np.asarray(K, dtype=float)

    free = (alpha > tol) & (alpha < C - tol)

    if not np.any(free):
        return 0.0

    f = K @ (alpha * y)
    margins =  y[free] * (f[free] + theta0)

    return float(np.max(np.abs(margins - 1.0)))


# ======================================================================
# Part B -- convenience wrappers (already written; read them)
# ======================================================================

def fit(X_train, y_train, C, s2=None, tol=TOL):
    """Fit one SVM.  s2=None selects the linear kernel."""
    K = linear_kernel(X_train, X_train) if s2 is None \
        else rbf_kernel(X_train, X_train, s2)
    alpha = solve_dual(K, y_train, C)
    theta0 = recover_intercept(alpha, y_train, K, C, tol)
    code = classify_regimes(alpha, C, tol)
    return {
        "alpha": alpha, "theta0": theta0, "regimes": code, "K": K,
        "n_non_sv": int((code == 0).sum()),
        "n_free": int((code == 1).sum()),
        "n_bounded": int((code == 2).sum()),
        "n_sv": int((code > 0).sum()),
        "margin_width": margin_width(alpha, y_train, K),
        "kkt_residual": kkt_residual(alpha, theta0, y_train, K, C, tol),
    }


def error_rate(model, X_train, y_train, X_eval, y_eval, s2=None):
    Kt = linear_kernel(X_eval, X_train) if s2 is None \
        else rbf_kernel(X_eval, X_train, s2)
    g = decision_values(model["alpha"], model["theta0"], y_train, Kt)
    return float((np.sign(g) != y_eval).mean())


# ======================================================================
# Part C -- the three experiments
#
# Each must return a dict.  The required keys are listed in Table 2 of
# the lab document; the grader reads them by name, so spell them exactly.
# ======================================================================

def experiment_1(X_tr, y_tr, X_te, y_te):
    """SQ1: which points determine the hyperplane, and is theta_0 right?

    Fit once at C = C_DEFAULT, s2 = S2_DEFAULT, then produce:

    (a) The three regimes.  For each, the range of y_i g(x_i) over its
        members, and the number of members violating the condition
        Proposition 6.1 predicts for it (slack 1e-6).

    (b) With the LINEAR kernel at the same C, form theta explicitly as
        sum_i alpha_i y_i x_i and check it against your kernel-only
        code: that 2/||theta|| matches margin_width, and that
        X theta + theta_0 matches decision_values.

    (c) Recompute the intercept the WRONG way -- averaging y_i - f_i
        over ALL support vectors -- and report kkt_residual under each.

    (d) Delete every non-support-vector, refit at the same C and s2, and
        compare the two discriminants on the TEST set: the largest
        absolute difference, and the fraction of test points on which
        the predicted labels agree.

    Required keys: regime_counts, free_margin_min, free_margin_max,
    bounded_margin_max, nonsv_margin_min, violations, theta_width_diff,
    theta_g_diff, theta0_free, theta0_all_sv, theta0_abs_diff,
    kkt_residual_free, kkt_residual_all_sv, refit_n_before,
    refit_n_after, refit_max_g_diff, refit_label_agreement.
    """
    raise NotImplementedError


def experiment_2(X_tr, y_tr, X_te, y_te):
    """SQ2: what does C buy, and when does it stop buying anything?

    Sweep C over C_GRID at s2 = S2_DEFAULT.  At each C record the margin
    width, the support-vector fraction, the counts of free and bounded
    support vectors, and the training and test error.

    Required keys: C_grid, margin_width, sv_fraction, n_free, n_bounded,
    train_error, test_error, width_ratio, test_error_improvement.

    width_ratio is the first margin width divided by the last.
    test_error_improvement is the first test error minus the smallest.
    """
    raise NotImplementedError


def experiment_3(X_tr, y_tr, X_te, y_te):
    """SQ3: the two degenerate limits of the Gaussian kernel.

    Sweep s2 over S2_GRID at C = C_DEFAULT.  At each s2 record the
    support-vector count, the free-support-vector count, the training and
    test error, and two properties of the training Gram matrix: its
    condition number, and the fraction of the total eigenvalue mass
    carried by the single largest eigenvalue.

    Clip tiny negative eigenvalues to zero first, and use
    numpy.linalg.eigvalsh rather than eig.

    Required keys: s2_grid, n_sv, n_free, train_error, test_error,
    cond_K, top_eig_fraction, best_s2, test_error_left_rise,
    test_error_right_rise.

    best_s2 is the s2 with the smallest test error; the two rises are
    the first and last test errors minus that smallest one.
    """
    raise NotImplementedError


# ======================================================================
# Part D -- driver (already written)
# ======================================================================

def run_experiments(roll_number=None):
    roll = roll_number or ROLL_NUMBER
    if roll.startswith("REPLACE"):
        raise SystemExit("Set ROLL_NUMBER at the top of this file first.")
    X_tr, y_tr, X_te, y_te = make_dataset(roll)
    out = {
        "roll_number": str(roll),
        "experiment_1": experiment_1(X_tr, y_tr, X_te, y_te),
        "experiment_2": experiment_2(X_tr, y_tr, X_te, y_te),
        "experiment_3": experiment_3(X_tr, y_tr, X_te, y_te),
    }
    with open("results.json", "w") as fh:
        json.dump(out, fh, indent=2, default=_jsonify)
    print("wrote results.json")
    return out


def _jsonify(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


if __name__ == "__main__":
    run_experiments()
