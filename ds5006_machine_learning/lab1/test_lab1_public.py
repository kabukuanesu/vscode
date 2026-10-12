"""DS5006 Lab 1 -- public self-checks.

Run with::

    python -m pytest test_lab1_public.py -v

These tests are a SUBSET of what the grader runs. Passing all of them does not
guarantee full marks; the grader adds hidden numerical cases, efficiency
counters and static checks. Failing any of them, however, guarantees you will
lose marks, so fix them first.

Set your roll number in the environment before running::

    export DS5006_ROLL=112201042        # Linux / macOS
    set DS5006_ROLL=112201042           # Windows cmd
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import ds5006_lab1 as lab
import ds5006_lab1_data as data

ROLL = os.environ.get("DS5006_ROLL", "000000000")
ATOL = 1e-8


@pytest.fixture(scope="module")
def scaling_data():
    """Badly scaled dataset used by the Tier A checks."""
    return data.dataset_scaling(ROLL)


@pytest.fixture(scope="module")
def wide_data():
    """Short wide dataset used by the Tier C checks."""
    return data.dataset_wide(ROLL)


# --------------------------------------------------------------------------- #
# Tier A
# --------------------------------------------------------------------------- #


def test_standardize_moments(scaling_data):
    """Standardised columns have zero mean and unit population variance."""
    x_mat, _ = scaling_data
    x_std, mean, scale = lab.standardize(x_mat)
    assert x_std.shape == x_mat.shape
    assert mean.shape == (x_mat.shape[1],) and scale.shape == (x_mat.shape[1],)
    assert np.abs(x_std.mean(axis=0)).max() < 1e-10
    assert np.abs(x_std.std(axis=0, ddof=0) - 1.0).max() < 1e-10


def test_standardize_reuses_supplied_stats():
    """Supplied mean and scale are used verbatim, not recomputed."""
    x_mat = np.arange(12.0).reshape(6, 2)
    mean, scale = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    x_std, out_m, out_s = lab.standardize(x_mat, mean=mean, scale=scale)
    assert np.allclose(x_std, (x_mat - mean) / scale)
    assert np.allclose(out_m, mean) and np.allclose(out_s, scale)


def test_standardize_constant_column():
    """A constant column is centred but not divided by ~zero."""
    x_mat = np.column_stack([np.ones(10), np.arange(10.0)])
    x_std, _, scale = lab.standardize(x_mat)
    assert scale[0] == 1.0
    assert np.all(np.isfinite(x_std))
    assert np.allclose(x_std[:, 0], 0.0)


def test_mse_matches_definition():
    """MSE divides by n."""
    y_true, y_pred = np.array([1.0, 2.0, 3.0]), np.array([1.0, 4.0, 3.0])
    assert np.isclose(lab.mse(y_true, y_pred), 4.0 / 3.0)


def test_ols_normal_residual_orthogonality(scaling_data):
    """OLS residuals are orthogonal to every column and to the constant."""
    x_mat, y_vec = scaling_data
    x_std, _, _ = lab.standardize(x_mat)
    weights, intercept = lab.fit_ols_normal(x_std, y_vec)
    resid = y_vec - lab.predict(x_std, weights, intercept)
    assert np.abs(x_std.T @ resid).max() < 1e-6
    assert abs(resid.sum()) < 1e-6


def test_ols_recovers_exact_fit():
    """With noise-free data the fit is exact."""
    rng = np.random.default_rng(0)
    x_mat = rng.normal(size=(50, 3))
    w_true, b_true = np.array([1.5, -2.0, 0.25]), 4.0
    weights, intercept = lab.fit_ols_normal(x_mat, x_mat @ w_true + b_true)
    assert np.allclose(weights, w_true, atol=ATOL)
    assert np.isclose(intercept, b_true, atol=ATOL)


def test_gd_history_shape_and_first_entry(scaling_data):
    """History has n_iters + 1 entries, and entry 0 is J at w = 0, b = 0."""
    x_mat, y_vec = scaling_data
    x_std, _, _ = lab.standardize(x_mat)
    _, _, history = lab.fit_gd(x_std, y_vec, 1e-3, 25)
    assert history.shape == (26,)
    assert np.isclose(history[0], np.mean(y_vec**2))


def test_gd_converges_to_closed_form(scaling_data):
    """On standardised data GD reaches the normal-equation solution."""
    x_mat, y_vec = scaling_data
    x_std, _, _ = lab.standardize(x_mat)
    w_ref, b_ref = lab.fit_ols_normal(x_std, y_vec)
    w_gd, b_gd, history = lab.fit_gd(x_std, y_vec, 0.9 * lab.max_stable_lr(x_std), 20000)
    assert np.allclose(w_gd, w_ref, atol=1e-5)
    assert np.isclose(b_gd, b_ref, atol=1e-5)
    assert np.all(np.diff(history) <= 1e-12), "MSE must be non-increasing below the threshold"


def test_max_stable_lr_brackets_divergence(scaling_data):
    """GD converges just below the threshold and blows up just above it."""
    x_mat, y_vec = scaling_data
    x_std, _, _ = lab.standardize(x_mat)
    thresh = lab.max_stable_lr(x_std)
    _, _, lo = lab.fit_gd(x_std, y_vec, 0.95 * thresh, 300)
    _, _, hi = lab.fit_gd(x_std, y_vec, 1.05 * thresh, 300)
    assert lo[-1] < lo[0]
    assert (not np.isfinite(hi[-1])) or hi[-1] > hi[0]


# --------------------------------------------------------------------------- #
# Tier B
# --------------------------------------------------------------------------- #


def test_three_solvers_agree_when_well_conditioned(scaling_data):
    """On a benign design all three OLS routes give the same answer."""
    x_mat, y_vec = scaling_data
    x_std, _, _ = lab.standardize(x_mat)
    w_n, b_n = lab.fit_ols_normal(x_std, y_vec)
    w_q, b_q = lab.fit_ols_qr(x_std, y_vec)
    w_s, b_s = lab.fit_ols_svd(x_std, y_vec)
    assert np.allclose(w_n, w_q, atol=1e-7) and np.allclose(w_n, w_s, atol=1e-7)
    assert np.isclose(b_n, b_q, atol=1e-7) and np.isclose(b_n, b_s, atol=1e-7)


def test_condition_number_known_value():
    """Condition number of a diagonal matrix is the ratio of its entries."""
    assert np.isclose(lab.condition_number(np.diag([4.0, 2.0, 0.5])), 8.0)


def test_polynomial_design_shape_and_columns():
    """Design has `degree` columns starting at x**1."""
    x_vals = np.array([2.0, 3.0])
    des = lab.polynomial_design(x_vals, 3)
    assert des.shape == (2, 3)
    assert np.allclose(des, np.array([[2, 4, 8], [3, 9, 27]], dtype=float))


def test_bias_variance_identity_on_synthetic_input():
    """Decomposition reproduces hand-computable values."""
    preds = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = lab.bias_variance_decompose(preds, np.array([1.0, 1.0]), 0.5)
    assert np.allclose(out["bias2"], [1.0, 4.0])
    assert np.allclose(out["variance"], [1.0, 1.0])
    assert np.isclose(out["noise"], 0.25)
    assert np.isclose(out["expected_mse"], 2.5 + 1.0 + 0.25)


# --------------------------------------------------------------------------- #
# Tier C
# --------------------------------------------------------------------------- #


def test_ridge_reduces_to_ols_at_zero(wide_data):
    """lam = 0 must give back the OLS fit."""
    x_mat, y_vec = wide_data
    w_r, b_r = lab.fit_ridge(x_mat, y_vec, 0.0)
    w_o, b_o = lab.fit_ols_normal(x_mat, y_vec)
    assert np.allclose(w_r, w_o, atol=1e-6) and np.isclose(b_r, b_o, atol=1e-6)


def test_ridge_shrinks_monotonically(wide_data):
    """Larger penalties give smaller slope norms."""
    x_mat, y_vec = wide_data
    norms = [np.linalg.norm(lab.fit_ridge(x_mat, y_vec, lam)[0]) for lam in [1e-2, 1.0, 1e2, 1e4]]
    assert all(a > b for a, b in zip(norms, norms[1:]))


def test_ridge_intercept_is_not_shrunk(wide_data):
    """As lam -> infinity the intercept tends to mean(y), not to zero."""
    x_mat, y_vec = wide_data
    _, intercept = lab.fit_ridge(x_mat, y_vec, 1e12)
    assert np.isclose(intercept, y_vec.mean(), atol=1e-4)


def test_leverages_sum_to_effective_dof(wide_data):
    """sum_i h_ii equals trace(H) equals effective_dof."""
    x_mat, _ = wide_data
    for lam in (1e-2, 1.0, 50.0):
        assert np.isclose(lab.hat_diagonal(x_mat, lam).sum(), lab.effective_dof(x_mat, [lam])[0])


def test_effective_dof_limits(wide_data):
    """dof -> d + 1 as lam -> 0 and dof -> 1 as lam -> infinity."""
    x_mat, _ = wide_data
    dofs = lab.effective_dof(x_mat, np.array([1e-10, 1e14]))
    assert np.isclose(dofs[0], x_mat.shape[1] + 1, atol=1e-3)
    assert np.isclose(dofs[1], 1.0, atol=1e-3)


def test_loocv_fast_matches_naive(wide_data):
    """The shortcut is exact, not approximate."""
    x_mat, y_vec = wide_data
    for lam in (1e-2, 1.0, 25.0, 1e3):
        fast, naive = lab.loocv_rmse_fast(x_mat, y_vec, lam), lab.loocv_rmse_naive(x_mat, y_vec, lam)
        assert abs(fast - naive) <= 1e-9 * abs(naive), f"mismatch at lam={lam}"


def test_ridge_path_matches_individual_fits(wide_data):
    """Every row of the path equals the corresponding single fit."""
    x_mat, y_vec = wide_data
    lams = np.array([1e-3, 0.1, 5.0, 200.0])
    weights, intercepts = lab.ridge_path_svd(x_mat, y_vec, lams)
    assert weights.shape == (4, x_mat.shape[1]) and intercepts.shape == (4,)
    for k, lam in enumerate(lams):
        w_ref, b_ref = lab.fit_ridge(x_mat, y_vec, lam)
        assert np.allclose(weights[k], w_ref, atol=1e-7)
        assert np.isclose(intercepts[k], b_ref, atol=1e-7)
