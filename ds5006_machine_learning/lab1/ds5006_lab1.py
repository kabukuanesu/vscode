"""DS5006 Lab 1 -- Linear regression from scratch.

Replace every ``raise NotImplementedError`` with your own implementation.
Do not rename functions, do not change signatures, do not add or remove
parameters. The grader imports this module by name and calls these functions
directly; a renamed function scores zero even if it is perfectly correct.

Permitted imports: ``numpy`` only (plus the standard library, and
``ds5006_lab1_data`` if you need it). ``scipy``, ``sklearn``, ``pandas``,
``statsmodels``, ``torch``, ``jax`` and friends are forbidden anywhere in this
file. ``numpy.linalg.lstsq`` and ``numpy.linalg.pinv`` are also forbidden --
the point of the lab is to build those yourself.

Fill in the header below before you submit.

    Name: Anesu Kabuku
    Roll number: 142602026
    Date: 12-8-2026

Declaration of assistance (required -- see Section 9 of the instructions).
List every person, tool, website and AI system you used, and say what you used
each one for. An honest, specific entry here costs you nothing. An empty or
false entry found to be false is an academic integrity violation.

    Assistance used:
    •	Used NumPy documentation for linear algebra functions (np.linalg.solve, np.linalg.svd, np.linalg.qr)
    •	Consulted lecture notes on bias-variance decomposition and ridge regression
    •	Used the lab instructions and public tests for validation
    •	Geeksforgeeks.org
    •	Tpointtech.com
    •	Mathematics for Machine Learning (Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong)
    •	Used Claude for SQ6 question interpretation and code generation.

"""

from __future__ import annotations

import numpy as np
# import ds5006_lab1_data as data

# --------------------------------------------------------------------------- #
# Tier A -- everybody should finish this section
# --------------------------------------------------------------------------- #


def standardize(x_mat, mean=None, scale=None):
    """Centre and scale the columns of a design matrix.

    When ``mean`` or ``scale`` is None, compute it from ``x_mat``; otherwise
    use the value supplied. This is what lets you apply *training* statistics
    to *test* data, which you will need in SQ4.

    Use the POPULATION standard deviation (``ddof=0``). Any column whose
    standard deviation is below ``1e-12`` gets a scale of exactly ``1.0``, so
    that constant columns are centred but not divided by (almost) zero.

    Args:
        x_mat: ``(n, d)`` array.
        mean: ``(d,)`` array of column means, or None to compute from ``x_mat``.
        scale: ``(d,)`` array of column scales, or None to compute.

    Returns:
        Tuple ``(x_std, mean, scale)`` where ``x_std`` has shape ``(n, d)`` and
        ``mean``, ``scale`` have shape ``(d,)``.
        """

    if mean is None:
        mean = x_mat.mean(axis=0)
    if scale is None:
        scale = x_mat.std(axis=0, ddof=0)
        # Handle constant columns
        scale[scale < 1e-12] = 1.0

    x_std = (x_mat - mean) / scale
    return x_std, mean, scale


def mse(y_true, y_pred):
    """Mean squared error.

    Args:
        y_true: ``(n,)`` observed values.
        y_pred: ``(n,)`` predicted values.

    Returns:
        Python ``float``, the mean over the ``n`` squared errors.
    """
    return np.mean((y_true - y_pred) ** 2)


def predict(x_mat, weights, intercept):
    """Predict with a fitted linear model.

    Args:
        x_mat: ``(n, d)`` design matrix, no column of ones.
        weights: ``(d,)`` slope coefficients.
        intercept: scalar intercept.

    Returns:
        ``(n,)`` array of predictions.
    """
    return x_mat @ weights + intercept


def fit_ols_normal(x_mat, y_vec):
    """Fit ordinary least squares by solving the normal equations.

    Minimise ``||y - Xw - b1||^2`` over ``(w, b)``. Handle the intercept by
    centring: solve for ``w`` on the centred data, then recover
    ``b = mean(y) - mean(X) @ w``. Do NOT append a column of ones.

    Solve the linear system with ``np.linalg.solve``. Do not form an explicit
    matrix inverse -- SQ3 is about exactly why that matters.

    Args:
        x_mat: ``(n, d)`` design matrix.
        y_vec: ``(n,)`` responses.

    Returns:
        Tuple ``(weights, intercept)`` of shapes ``(d,)`` and scalar float.
    """
    # Center both X and y
    x_mean = x_mat.mean(axis=0)
    y_mean = y_vec.mean()

    Xc = x_mat - x_mean
    yc = y_vec - y_mean

    # Solve (Xc^T Xc) w = Xc^T yc
    weights = np.linalg.solve(Xc.T @ Xc, Xc.T @ yc)

    # Recover intercept
    intercept = y_mean - x_mean @ weights

    return weights, intercept


def fit_gd(x_mat, y_vec, learning_rate, n_iters):
    """Fit a linear model by full-batch gradient descent.

    Objective, fixed for the whole lab::

        J(w, b) = (1/n) * sum_i (y_i - x_i @ w - b)^2

    Note the ``1/n`` and the absence of any ``1/2``. Get this wrong and your
    gradients are off by a constant factor, which changes the stable learning
    rate and every number you report.

    Start from ``w = 0`` and ``b = 0``. Update ``w`` and ``b`` simultaneously
    from the same residual vector -- do not update ``w`` and then recompute the
    residual before updating ``b``.

    Args:
        x_mat: ``(n, d)`` design matrix.
        y_vec: ``(n,)`` responses.
        learning_rate: positive step size.
        n_iters: number of gradient steps to take.

    Returns:
        Tuple ``(weights, intercept, history)``. ``history`` has length
        ``n_iters + 1``: ``history[0]`` is ``J`` at the initial parameters and
        ``history[t]`` is ``J`` after ``t`` updates.
    """
    n, d = x_mat.shape
    weights = np.zeros(d)
    intercept = 0.0
    history = np.zeros(n_iters + 1)

    # Initial loss
    history[0] = np.mean(y_vec ** 2) # w=0, b=0 => y_pred = 0

    for t in range(1, n_iters + 1):
        # Compute predictions
        y_pred = x_mat @ weights + intercept
        residuals = y_pred - y_vec # Negative gradient direction

        # Compute gradients (derivatives of J)
        # dJ/dw = (2/n) * X^T (y_pred - y)
        # dJ/db = (2/n) * sum(y_pred - y)
        grad_w = (2 / n) * x_mat.T @ residuals
        grad_b = (2 / n) * residuals.sum()

        # Update simultaneously
        weights -= learning_rate * grad_w
        intercept -= learning_rate * grad_b

        # Record loss (in-sample MSE)
        y_pred = x_mat @ weights + intercept
        history[t] = np.mean((y_vec - y_pred) ** 2)

    return weights, intercept, history


def max_stable_lr(x_mat):
    """Largest learning rate for which ``fit_gd`` does not diverge.

    Derive this yourself before you code it. Write the objective ``J(w, b)``
    from ``fit_gd`` in terms of the augmented design ``Xa = [1, X]`` and the
    augmented parameter ``beta = (b, w)``, take the Hessian, and use the
    standard result that gradient descent on a quadratic with Hessian ``H``
    converges if and only if ``0 < lr < 2 / lambda_max(H)``.

    Your derivation goes in the report. The function returns the threshold.

    Args:
        x_mat: ``(n, d)`` design matrix.

    Returns:
        Python ``float``, the supremum of stable learning rates.
    """
    n = x_mat.shape[0]
    Xa = np.column_stack([np.ones(n), x_mat])
    H = (2 / n) * Xa.T @ Xa
    eigvals = np.linalg.eigvalsh(H)
    lambda_max = eigvals.max()
    return 2.0 / lambda_max  # Maximum stable learning rate


# --------------------------------------------------------------------------- #
# Tier B
# --------------------------------------------------------------------------- #


def fit_ols_qr(x_mat, y_vec):
    """Fit OLS through a QR factorisation of the centred design.

    Factor the centred design as ``Xc = QR`` with ``np.linalg.qr`` (reduced
    mode, the default), then solve the triangular system ``R w = Q^T yc``.
    Never form ``Xc.T @ Xc``. Handle the intercept exactly as in
    ``fit_ols_normal``.

    Args:
        x_mat: ``(n, d)`` design matrix.
        y_vec: ``(n,)`` responses.

    Returns:
        Tuple ``(weights, intercept)``.
    """
    x_mean = x_mat.mean(axis=0)
    y_mean = y_vec.mean()

    Xc = x_mat - x_mean
    yc = y_vec - y_mean

    # QR: Xc = Q R (reduced mode by default)
    Q, R = np.linalg.qr(Xc)

    # Solve R w = Q^T yc
    weights = np.linalg.solve(R, Q.T @ yc)
    intercept = y_mean - x_mean @ weights

    return weights, intercept


def fit_ols_svd(x_mat, y_vec, rcond=1e-12):
    """Fit minimum-norm OLS through a truncated SVD of the centred design.

    Take the thin SVD ``Xc = U S V^T`` (``full_matrices=False``). Discard every
    singular value satisfying ``s <= rcond * max(s)``, invert the rest, and
    form ``w = V @ diag(1/s_kept) @ U^T @ yc``. Discarded directions contribute
    zero, which is what makes the solution minimum-norm when ``Xc`` is rank
    deficient.

    You are building the pseudo-inverse. ``np.linalg.pinv`` and
    ``np.linalg.lstsq`` are forbidden.

    Args:
        x_mat: ``(n, d)`` design matrix.
        y_vec: ``(n,)`` responses.
        rcond: relative singular-value cutoff.

    Returns:
        Tuple ``(weights, intercept)``.
    """
    x_mean = x_mat.mean(axis=0)
    y_mean = y_vec.mean()

    Xc = x_mat - x_mean
    yc = y_vec - y_mean

    # SVD: Xc = U S V^T
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    # Determine which singular values to keep
    s_max = S.max()
    keep = S > rcond * s_max
    S_inv = np.zeros_like(S)
    S_inv[keep] = 1.0 / S[keep]

    # w = V @ diag(1/s_kept) @ U^T @ yc
    weights = Vt.T @ np.diag(S_inv) @ U.T @ yc
    intercept = y_mean - x_mean @ weights

    return weights, intercept


def condition_number(x_mat):
    """Two-norm condition number of a matrix, from its singular values.

    Return ``max(s) / min(s)``, or ``float("inf")`` if the smallest singular
    value is exactly zero. Compute the singular values with
    ``np.linalg.svd(..., compute_uv=False)``; do not call ``np.linalg.cond``.

    Args:
        x_mat: ``(n, d)`` array.

    Returns:
        Python ``float``.
    """
    s = np.linalg.svd(x_mat, compute_uv=False)
    s_min = s.min()
    if s_min < 1e-15: # Numerical zero threshold
        return float("inf")
    return float(s.max() / s_min)


def polynomial_design(x_vals, degree):
    """Build a polynomial design matrix WITHOUT a constant column.

    Column ``j`` (zero-indexed) is ``x ** (j + 1)``, so the returned matrix has
    ``degree`` columns. The constant term is the model's intercept and is
    handled by the fitting routines, not by this function.

    Args:
        x_vals: ``(n,)`` array of scalar inputs.
        degree: positive integer.

    Returns:
        ``(n, degree)`` array.
    """
    n = len(x_vals)
    X = np.zeros((n, degree))
    for j in range(degree):
        X[:, j] = x_vals ** (j + 1)
    return X


def bias_variance_decompose(preds, f_vals, noise_sd):
    """Decompose expected test error into squared bias, variance and noise.

    Args:
        preds: ``(M, T)`` array. Row ``m`` holds the predictions of the model
            fitted on replicate training set ``m``, evaluated at the ``T``
            fixed test inputs.
        f_vals: ``(T,)`` values of the TRUE regression function at those same
            test inputs.
        noise_sd: standard deviation of the observation noise.

    Returns:
        Dict with keys:
          ``"bias2"``        -- ``(T,)`` pointwise squared bias.
          ``"variance"``     -- ``(T,)`` pointwise variance, POPULATION
                                convention (``ddof=0``).
          ``"mean_bias2"``   -- float, mean of ``bias2`` over test points.
          ``"mean_variance"``-- float, mean of ``variance`` over test points.
          ``"noise"``        -- float, ``noise_sd ** 2``.
          ``"expected_mse"`` -- float, the sum of the previous three scalars.
    """
    # M, T = preds.shape

    # mean prediction across replicates
    pred_mean = preds.mean(axis=0)

    # bias^2 = (E[pred] - f)^2
    bias2 = (pred_mean - f_vals) ** 2

    # variance: Var(pred) using ddof=0 (population variance)
    variance = preds.var(axis=0, ddof=0)

    noise = noise_sd ** 2

    expected_mse = bias2.mean() + variance.mean() + noise

    return {
        "bias2": bias2,
        "variance": variance,
        "mean_bias2": bias2.mean(),
        "mean_variance": variance.mean(),
        "noise": noise,
        "expected_mse": expected_mse
    }


# --------------------------------------------------------------------------- #
# Tier C -- the hard section
# --------------------------------------------------------------------------- #


def fit_ridge(x_mat, y_vec, lam):
    """Fit ridge regression with an UNPENALISED intercept.

    Minimise ``||y - Xw - b1||^2 + lam * ||w||^2``. Three things to get right:

      1. The penalty is on ``w`` only. The intercept ``b`` is never shrunk.
      2. The data-fit term is a SUM of squared errors, not a mean. There is no
         ``1/n`` anywhere, so ``lam`` is on the same scale as ``X^T X``.
      3. There is no standardisation inside this function. If you want
         standardised features, standardise before you call it.

    Centre, solve ``(Xc^T Xc + lam I) w = Xc^T yc``, then recover ``b``.

    Args:
        x_mat: ``(n, d)`` design matrix.
        y_vec: ``(n,)`` responses.
        lam: non-negative penalty. ``lam = 0`` must reproduce OLS.

    Returns:
        Tuple ``(weights, intercept)``.
    """
    # Centre the data
    x_mean = x_mat.mean(axis=0)
    y_mean = y_vec.mean()

    Xc = x_mat - x_mean
    yc = y_vec - y_mean

    # Solve (Xc^T Xc + lam I) w = Xc^T yc
    d = x_mat.shape[1]
    gram = Xc.T @ Xc + lam * np.eye(d)
    weights = np.linalg.solve(gram, Xc.T @ yc)

    # Recover intercept
    intercept = y_mean - x_mean @ weights

    return weights, intercept


def hat_diagonal(x_mat, lam):
    """Diagonal of the ridge hat matrix, for the unpenalised-intercept fit.

    The fitted values of ``fit_ridge`` are a linear function of ``y``, so
    ``yhat = H y`` for some ``n x n`` matrix ``H`` that depends only on ``X``
    and ``lam``. Work out what ``H`` is for the model in ``fit_ridge`` -- the
    intercept contributes a term, and getting that term wrong is the single
    most common way to fail SQ6.

    Return only the diagonal. Do NOT build the full ``n x n`` matrix and then
    call ``np.diag`` on it: for a large ``n`` that is wasteful, and the grader
    checks your peak memory behaviour indirectly through the efficiency test in
    SQ6.

    Args:
        x_mat: ``(n, d)`` design matrix.
        lam: non-negative ridge penalty.

    Returns:
        ``(n,)`` array of leverages ``h_ii``.
    """
    n, d = x_mat.shape

    # Centre the design
    X_mean = x_mat.mean(axis=0)
    Xc = x_mat - X_mean

    # Compute gram matrix
    gram = Xc.T @ Xc + lam * np.eye(d)

    # Solve gram * B = Xc.T to get B = gram^{-1} @ Xc.T
    # B has shape (d, n)
    B = np.linalg.solve(gram, Xc.T) # shape: (d, n)

    # Xc @ B gives (n, n) but we only need diagonal
    # diag(Xc @ B) = sum_k Xc[i,k] * B[k,i]
    h_ii_x = np.sum(Xc * B.T, axis=1)  # shape: (n,)

    # Intercept contribution
    h_ii = (1.0 / n) + h_ii_x

    return h_ii


def loocv_rmse_naive(x_mat, y_vec, lam):
    """Leave-one-out CV RMSE, computed the obvious expensive way.

    Loop over ``i``, refit ``fit_ridge`` on the ``n - 1`` remaining rows,
    predict the held-out point, and collect the ``n`` held-out residuals.
    Return the root mean of their squares.

    This is your ground truth for SQ6. Write it first, and write it simply.

    Args:
        x_mat: ``(n, d)`` design matrix.
        y_vec: ``(n,)`` responses.
        lam: non-negative ridge penalty.

    Returns:
        Python ``float``.
    """
    n = len(y_vec)
    residuals = np.zeros(n)

    for i in range(n):
        # Create training set without point i
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        X_train = x_mat[mask]
        y_train = y_vec[mask]

        # Fit ridge on training set
        w, b = fit_ridge(X_train, y_train, lam)

        # Predict held-out point
        y_pred = predict(x_mat[i:i+1], w, b)[0]

        # Store residual
        residuals[i] = y_vec[i] - y_pred

    return np.sqrt(np.mean(residuals ** 2))


def loocv_rmse_fast(x_mat, y_vec, lam):
    """Leave-one-out CV RMSE from a SINGLE fit, via the leverage identity.

    There is an exact shortcut. It is not an approximation, and it is not a
    large-``n`` limit: for penalised least squares with the row removed from
    the design as well as from the response, the held-out residual satisfies a
    closed-form relation involving ``h_ii``. Derive it -- the derivation is
    worth marks in the report -- and implement it here.

    Requirements:
      * Must agree with ``loocv_rmse_naive`` to a relative tolerance of
        ``1e-9`` on every dataset the grader tries.
      * Must make at most FOUR calls in total to the numpy linear-algebra
        routines (``solve``, ``inv``, ``svd``, ``qr``, ``eigh``, ``eig``,
        ``cholesky``), no matter how large ``n`` is. The grader counts them.
        Calling ``loocv_rmse_naive`` from here, or looping over rows, will
        blow that budget and score zero for this function.

    Args:
        x_mat: ``(n, d)`` design matrix.
        y_vec: ``(n,)`` responses.
        lam: non-negative ridge penalty.

    Returns:
        Python ``float``.
    """
    # Fit on full data
    w, b = fit_ridge(x_mat, y_vec, lam)
    y_pred = predict(x_mat, w, b)
    residuals = y_vec - y_pred

    # Get hat diagonal
    h_diag = hat_diagonal(x_mat, lam)

    # LOOCV residuals: e_i / (1 - h_ii)
    loo_residuals = residuals / (1.0 - h_diag)

    return np.sqrt(np.mean(loo_residuals ** 2))


def ridge_path_svd(x_mat, y_vec, lams):
    """Compute a whole ridge path from ONE singular value decomposition.

    Naively you would solve a fresh ``d x d`` system for each penalty value.
    Instead, decompose the centred design once and reuse it: in the SVD basis
    the ridge solution is a per-direction shrinkage of the OLS solution, and
    changing ``lam`` only changes the shrinkage factors.

    Requirement: exactly ONE call to ``np.linalg.svd`` and NO calls to
    ``np.linalg.solve``, ``inv``, ``qr``, ``eigh`` or ``cholesky``, regardless
    of how many penalties are requested. The grader counts them.

    Args:
        x_mat: ``(n, d)`` design matrix.
        y_vec: ``(n,)`` responses.
        lams: ``(L,)`` array of non-negative penalties.

    Returns:
        Tuple ``(weights, intercepts)`` of shapes ``(L, d)`` and ``(L,)``.
        Row ``k`` must equal what ``fit_ridge(x_mat, y_vec, lams[k])`` returns.
    """
    x_mean = x_mat.mean(axis=0)
    y_mean = y_vec.mean()

    Xc = x_mat - x_mean
    yc = y_vec - y_mean

    # One SVD decomposition
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    # Pre-compute U^T @ yc
    Uty = U.T @ yc

    # Pre-compute factor that's independent of lam
    factor = Vt.T @ np.diag(S) # V^T * diag(S)

    L = len(lams)
    d = x_mat.shape[1]

    weights = np.zeros((L, d))
    intercepts = np.zeros(L)

    for k, lam in enumerate(lams):
        # Shrinkage factors: s_i / (s_i^2 + lam)
        # But we need s_i / (s_i^2 + lam) for the solution
        # w = sum_i (s_i / (s_i^2 + lam)) * v_i * (u_i^T yc)
        shrink = S / (S**2 + lam)
        weights[k] = factor @ np.diag(shrink) @ Uty
        intercepts[k] = y_mean - x_mean @ weights[k]

    return weights, intercepts


def effective_dof(x_mat, lams):
    """Effective degrees of freedom of the ridge fit, as a function of ``lam``.

    Defined as ``trace(H)`` for the same ``H`` as in ``hat_diagonal``. Express
    it through the singular values of the centred design so that you can
    evaluate the whole vector of penalties from one decomposition, and so that
    the two limits ``lam -> 0`` and ``lam -> inf`` are obvious by inspection.

    Args:
        x_mat: ``(n, d)`` design matrix.
        lams: ``(L,)`` array of non-negative penalties.

    Returns:
        ``(L,)`` array.
    """
    x_mean = x_mat.mean(axis=0)
    Xc = x_mat - x_mean

    # Get singular values of centred design
    s = np.linalg.svd(Xc, compute_uv=False)

    L = len(lams)
    dofs = np.zeros(L)

    for k, lam in enumerate(lams):
        # Intercept contributes 1 degree of freedom
        if lam == 0:
            dofs[k] = 1.0 + len(s) # All directions + intercept
        else:
            dofs[k] = 1.0 + np.sum(s**2 / (s**2 + lam))

    return dofs
