"""DS5006 Lab 2 -- Linear classification and logistic regression.

Replace every ``raise NotImplementedError`` with your own implementation.
Do not rename functions, do not change signatures, do not add or remove
parameters. The grader imports this module by name and calls these functions
directly; a renamed function scores zero even if it is perfectly correct.

Permitted imports: ``numpy``, the Python standard library, and
``ds5006_lab2_data``. Forbidden anywhere in this file: ``scipy``,
``sklearn``, ``pandas``, ``statsmodels``, ``torch``, ``jax``, ``cvxpy``.
Also forbidden: ``numpy.linalg.lstsq``, ``numpy.linalg.pinv`` and
``numpy.linalg.inv`` -- building those yourself is part of the point.
``numpy.linalg.qr``, ``numpy.linalg.solve``, ``numpy.linalg.svd``,
``numpy.linalg.eigh``, ``numpy.linalg.eigvalsh`` and ``numpy.linalg.cholesky``
are all allowed.

Fill in the header below before you submit.

    Name: Anesu Kabuku
    Roll number: 142602026
    Date: 02/09/2026

Declaration of assistance (required -- see Section 9 of the instructions).
List every person, tool, website and AI system you used, and say what you used
each one for. An honest, specific entry here costs you nothing.

    Assistance used:
        •	Used NumPy documentation for linear algebra functions
        •	Consulted lecture notes on logistic regression
        •	Used the lab instructions and public tests for validation
        •	Geeksforgeeks.org
        •	Tpointtech.com
        •	Mathematics for Machine Learning (Marc P. Deisenroth, A.A Faisal, and Cheng S. Ong)(2020)
"""

# The disable below silences a warning about numpy being unused while the
# file is still all stubs. You may delete the comment once you start using it.
import numpy as np
import ds5006_lab2_data as data

# ---------------------------------------------------------------------------
# Conventions that the whole file obeys. Read these once, carefully.
#
#   * A design matrix ``X`` has shape (n, p) and NEVER carries an implicit
#     column of ones. If a model needs an intercept you add it yourself with
#     ``add_intercept``, which puts the ones column LAST.
#   * Class labels ``y`` are integers 0, 1, ..., K-1 in a 1-D array.
#   * Every average loss in this file is a MEAN over the n observations,
#     not a sum.
#   * Every penalty in this file is ``0.5 * lam * <squared norm of the whole
#     parameter>``. The intercept IS penalised. (Common libraries do not
#     penalise it. We do, and the numerical checks are built around that.)
#   * ``lam`` is always the coefficient in that ``0.5 * lam * ...`` term.
# ---------------------------------------------------------------------------


# ===========================================================================
# S0 -- Building blocks (6 marks)
# ===========================================================================


def add_intercept(design):
    """Append a column of ones to ``design`` as the LAST column.

    Parameters
    ----------
    design : ndarray, shape (n, p)

    Returns
    -------
    ndarray, shape (n, p + 1)
        Columns 0..p-1 are ``design`` unchanged; column p is all ones.

    Notes
    -----
    Last, not first. Every hidden test indexes the intercept as ``theta[-1]``.
    """
    ones = np.ones((design.shape[0], 1))

    return np.hstack((design, ones))


def one_hot(labels, n_classes):
    """Return the indicator (one-hot) response matrix for ``labels``.

    Parameters
    ----------
    labels : ndarray of int, shape (n,), values in ``range(n_classes)``
    n_classes : int

    Returns
    -------
    ndarray of float, shape (n, n_classes)
        Row ``i`` is all zeros except a single 1.0 in column ``labels[i]``.
    """
    result = np.zeros((labels.size, n_classes), dtype=float)
    result[np.arange(labels.size), labels] = 1.0

    return result


def sigmoid(values):
    """Elementwise logistic function, computed without overflow.

    Parameters
    ----------
    values : ndarray

    Returns
    -------
    ndarray of the same shape, with entries in (0, 1).

    Notes
    -----
    ``1 / (1 + np.exp(-values))`` overflows for very negative input and will
    emit a RuntimeWarning. The grader calls this with entries as extreme as
    -1500 and +1500 and requires finite, correct output with no warnings.
    Handle the two signs separately.
    """
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)

    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))

    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)

    return result


def softmax_rows(scores):
    """Row-wise softmax, computed without overflow.

    Parameters
    ----------
    scores : ndarray, shape (n, K)

    Returns
    -------
    ndarray, shape (n, K)
        Every row is non-negative and sums to 1.

    Notes
    -----
    Subtract each row's maximum before exponentiating. The grader calls this
    with entries as large as 1e4.
    """
    scores = np.asarray(scores, dtype=float)

    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)

    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def quadratic_expansion(design):
    """Expand a two-column design with its quadratic terms, in a fixed order.

    Parameters
    ----------
    design : ndarray, shape (n, 2), columns ``a`` and ``b``

    Returns
    -------
    ndarray, shape (n, 5)
        Columns, in exactly this order:
        ``a``, ``b``, ``a**2``, ``a*b``, ``b**2``.

    Notes
    -----
    No intercept column. The order is pinned and is checked directly.
    """
    a = design[:, 0]
    b = design[:, 1]

    return np.column_stack((a, b, a**2, a*b, b**2))

# ===========================================================================
# SQ1 -- Can you classify by running a regression? (8 marks)
# ===========================================================================


def fit_indicator_regression(design, targets):
    """Least-squares fit of an indicator response matrix.

    Solves, separately for each class column, the ordinary least-squares
    problem ``min_theta ||design @ theta - targets[:, k]||^2``.

    Parameters
    ----------
    design : ndarray, shape (n, p)
        Already includes an intercept column if you want one.
    targets : ndarray, shape (n, K)
        Typically the output of ``one_hot``.

    Returns
    -------
    ndarray, shape (p, K)
        Column ``k`` is the coefficient vector for class ``k``.

    Notes
    -----
    ``lstsq`` and ``pinv`` are forbidden. Use a QR or SVD factorisation.
    ``design`` may be rank deficient in principle; the datasets you are given
    are full rank, so a QR-based solve is enough.
    """
    q_matrix, r_matrix = np.linalg.qr(design, mode='reduced')
    return np.linalg.solve(r_matrix, q_matrix.T @ targets)


def indicator_predict(design, coefficients):
    """Predict class labels by taking the largest fitted response.

    Parameters
    ----------
    design : ndarray, shape (n, p)
    coefficients : ndarray, shape (p, K)

    Returns
    -------
    ndarray of int, shape (n,)
        ``argmax`` across the K fitted values, ties broken towards the
        SMALLEST class index.
    """
    return np.argmax(design @ coefficients, axis=1)


def accuracy(true_labels, predicted_labels):
    """Fraction of positions at which the two label arrays agree.

    Returns
    -------
    float in [0, 1]
    """
    return float(np.mean(true_labels == predicted_labels))


# ===========================================================================
# SQ2 -- Does the fitting criterion decide the boundary? (7 marks)
# ===========================================================================


def lda_fit(design, labels, n_classes):
    """Fit linear discriminant analysis by maximum likelihood.

    Parameters
    ----------
    design : ndarray, shape (n, p)
        NO intercept column. LDA supplies its own offset through the
        discriminant function.
    labels : ndarray of int, shape (n,)
    n_classes : int

    Returns
    -------
    dict with exactly these keys
        ``"means"``  : ndarray (K, p), class means, row k for class k.
        ``"cov"``    : ndarray (p, p), the POOLED within-class covariance,
                       ``sum_k sum_{i in k} (x_i - mu_k)(x_i - mu_k)^T``
                       divided by ``(n - n_classes)``.
        ``"priors"`` : ndarray (K,), the class proportions in ``labels``.

    Notes
    -----
    The pooled denominator is ``n - n_classes``, not ``n``. This is the
    unbiased choice and it is what every hidden test compares against.
    """
    n_samples, n_features = design.shape

    means = np.zeros((n_classes, n_features))
    scatter = np.zeros((n_features, n_features))

    for k in range(n_classes):
        class_points = design[labels == k]
        means[k] = np.mean(class_points, axis=0)

        centered = class_points - means[k]
        scatter += centered.T @ centered

    cov = scatter / (n_samples - n_classes)

    priors = np.bincount(labels, minlength=n_classes) / n_samples

    return {"means": means, "cov": cov, "priors": priors}


def lda_scores(design, model):
    """Linear discriminant scores for every point and class.

    Parameters
    ----------
    design : ndarray, shape (n, p)
    model : dict
        As returned by ``lda_fit``.

    Returns
    -------
    ndarray, shape (n, K)
        Entry (i, k) is

            x_i^T S^-1 mu_k  -  0.5 * mu_k^T S^-1 mu_k  +  log(prior_k)

        with ``S`` the pooled covariance.

    Notes
    -----
    Do not call ``np.linalg.inv``. Solve linear systems instead.
    """
    means = model["means"]
    cov = model["cov"]
    priors = model["priors"]

    solved_means = np.linalg.solve(cov, means.T).T

    linear = design @ solved_means.T
    quadratic = 0.5 * np.sum(means * solved_means, axis=1)

    return linear - quadratic + np.log(priors)


def lda_predict(design, model):
    """Predict labels as the argmax of ``lda_scores``, ties to the smallest index.

    Returns
    -------
    ndarray of int, shape (n,)
    """
    return np.argmax(lda_scores(design, model), axis=1)


def ls_direction_two_class(design, labels, code_zero, code_one):
    """Least-squares coefficient vector for a two-class problem, intercept dropped.

    Regress the coded target (``code_zero`` where ``labels == 0``,
    ``code_one`` where ``labels == 1``) on ``add_intercept(design)`` by
    ordinary least squares, then return only the p slope coefficients.

    Parameters
    ----------
    design : ndarray, shape (n, p) -- NO intercept column
    labels : ndarray of int, shape (n,), values in {0, 1}
    code_zero, code_one : float
        The numerical targets assigned to the two classes.

    Returns
    -------
    ndarray, shape (p,)
        The slope block only. The fitted intercept is discarded.
    """
    targets = np.where(labels == 0, code_zero, code_one)

    design_intercept = add_intercept(design)

    q_matrix, r_matrix = np.linalg.qr(design_intercept, mode='reduced')

    coefficients = np.linalg.solve(r_matrix, q_matrix.T @ targets)

    return coefficients[:-1]


# ===========================================================================
# SQ3 -- When does the logistic fit fail to exist? (12 marks)
# ===========================================================================
#
# For every function in this section:
#   ``design`` has shape (n, p) and carries no implicit intercept;
#   ``labels`` are 0/1;
#   ``theta`` has shape (p,);
# and the objective being minimised is
#
#     J(theta) = (1/n) * sum_i [ log(1 + exp(m_i)) - y_i * m_i ]
#                + 0.5 * lam * ||theta||^2 ,     m_i = x_i^T theta
#
# which is the MEAN negative log-likelihood plus a penalty on the WHOLE
# parameter vector, intercept included.
# ---------------------------------------------------------------------------


def logistic_objective(design, labels, theta, lam):
    """Value of the penalised mean negative log-likelihood defined above.

    Returns
    -------
    float

    Notes
    -----
    The grader evaluates this at ``theta`` large enough that ``exp(m)``
    overflows. ``np.logaddexp`` is the clean way to stay finite.
    """
    margins = design @ theta

    loss = np.mean(np.logaddexp(0.0, margins) - labels * margins)

    penalty = 0.5 * lam * np.dot(theta, theta)

    return loss + penalty


def logistic_gradient(design, labels, theta, lam):
    """Gradient of ``logistic_objective`` with respect to ``theta``.

    Returns
    -------
    ndarray, shape (p,)
        ``design.T @ (sigmoid(design @ theta) - labels) / n + lam * theta``.
    """
    margins = design @ theta
    probabilities = sigmoid(margins)

    return design.T @ (probabilities - labels) / design.shape[0] + lam * theta


def logistic_hessian(design, theta, lam):
    """Hessian of ``logistic_objective`` with respect to ``theta``.

    Parameters
    ----------
    design : ndarray, shape (n, p)
    theta : ndarray, shape (p,)
    lam : float

    Returns
    -------
    ndarray, shape (p, p), symmetric
        ``design.T @ diag(w) @ design / n + lam * I`` with
        ``w_i = s_i (1 - s_i)`` and ``s = sigmoid(design @ theta)``.

    Notes
    -----
    This function deliberately takes no ``labels`` argument. If you find
    yourself wanting one, re-derive the second derivative: the labels enter
    the gradient linearly and therefore drop out. Do not form ``diag(w)`` as
    an (n, n) matrix -- scale the rows instead.
    """
    margins = design @ theta
    probabilities = sigmoid(margins)

    weights = probabilities * (1.0 - probabilities)

    weighted_design = design * weights[:, None]

    hessian = design.T @ weighted_design / design.shape[0] + lam * np.eye(design.shape[1])

    return hessian


def fit_logistic_newton(design, labels, lam, tol=1e-10, max_iter=100):
    """Minimise ``logistic_objective`` by Newton's method from ``theta = 0``.

    At each iteration: check the stopping rule FIRST, then take the step
    ``theta <- theta - H^-1 g``. Use ``np.linalg.solve``; do not invert.

    Stopping rule
    -------------
    Stop when ``max(abs(gradient)) < tol``.

    Parameters
    ----------
    design : ndarray, shape (n, p)
    labels : ndarray of int, shape (n,)
    lam : float
    tol : float
    max_iter : int

    Returns
    -------
    dict with exactly these keys
        ``"theta"``     : ndarray (p,), the final iterate.
        ``"n_iter"``    : int, the number of Newton STEPS actually taken.
        ``"converged"`` : bool, True if the stopping rule fired within
                          ``max_iter`` steps.
        ``"objective"`` : float, ``logistic_objective`` at the final iterate.

    Notes
    -----
    Because the check comes first, a problem already solved at ``theta = 0``
    reports ``n_iter == 0``. The grader checks this exact off-by-one.
    """
    theta = np.zeros(design.shape[1], dtype=float)

    for iteration in range(max_iter):
        gradient = logistic_gradient(design, labels, theta, lam)

        if np.max(np.abs(gradient)) < tol:
            return {
                "theta": theta,
                "n_iter": iteration,
                "converged": True,
                "objective": logistic_objective(design, labels, theta, lam)
            }

        hessian = logistic_hessian(design, theta, lam)
        step = np.linalg.solve(hessian, gradient)
        theta -= step

    return {
        "theta": theta,
        "n_iter": max_iter,
        "converged": False,
        "objective": logistic_objective(design, labels, theta, lam)
    }


def min_signed_margin(design, labels, theta):
    """Smallest signed margin over the sample.

    Parameters
    ----------
    design, labels, theta : as above

    Returns
    -------
    float
        ``min_i s_i * (x_i^T theta)`` where ``s_i = 2 * labels[i] - 1``.

    Notes
    -----
    Strictly positive means ``theta`` separates the two classes perfectly.
    """
    signs = 2.0 * labels - 1.0
    margins = signs * (design @ theta)

    return np.min(margins)


def is_separable(design, labels):
    """Decide whether the two classes can be separated by an affine hyperplane.

    Parameters
    ----------
    design : ndarray, shape (n, p) -- NO intercept column
    labels : ndarray of int, shape (n,), values in {0, 1}

    Returns
    -------
    bool

    Method (follow it exactly -- the grader checks agreement with an
    independent linear-programming oracle on cases where the true margin is
    as small as 1e-5)
    -------------------------------------------------------------------
    1. Build the signed design ``Z`` whose row ``i`` is
       ``s_i * add_intercept(design)[i]`` with ``s_i = 2 * labels[i] - 1``.
    2. Scale each row of ``Z`` to unit Euclidean norm.
    3. Fit ``theta`` by minimising the objective of this section on ``Z``
       against an all-ones label vector -- equivalently, run your Newton
       solver with ``lam = 1e-12`` and ``max_iter = 400`` on the problem
       whose signed design is already ``Z``.
    4. Return ``True`` if and only if ``min(Z @ theta) > 0``.

    Why this works is the substance of SQ3, and you are asked to explain it
    in the report. A very lightly penalised logistic fit will drive every
    margin positive precisely when it is possible to do so.
    """
    augmented = add_intercept(design)

    signed = (2 * labels - 1)[:, None] * augmented

    row_norms = np.linalg.norm(signed, axis=1, keepdims=True)
    signed = signed / row_norms

    ones = np.ones(labels.shape, dtype=int)

    result = fit_logistic_newton(
        signed,
        ones,
        lam=1e-12,
        max_iter=400
    )

    theta = result["theta"]

    return np.min(signed @ theta) > 0


def separable_fraction(roll, n_points, n_features, n_trials):
    """Fraction of random labellings that turn out to be linearly separable.

    For each ``trial`` in ``range(n_trials)``, draw one dataset with
    ``ds5006_lab2_data.random_label_trial(roll, n_points, n_features, trial)``
    and test it with ``is_separable``. Return the proportion that were.

    Parameters
    ----------
    roll : str
    n_points, n_features : int
    n_trials : int

    Returns
    -------
    float in [0, 1]

    Notes
    -----
    The labels in those datasets are fair coin flips, completely unrelated to
    the features. So this measures how often a linear boundary can fit pure
    noise -- which is exactly how much a perfect separation is worth as
    evidence about your data.

    This is five lines. The interesting part is the experiment you run with
    it, not the function itself.
    """
    count = 0

    for trial in range(n_trials):
        design, labels = (
            data.random_label_trial(
                roll, n_points, n_features, trial
            )
        )

        if is_separable(design, labels):
            count += 1

    return count / n_trials


# ===========================================================================
# SQ4 -- Is a second-order method worth what it costs? (8 marks)
# ===========================================================================


def gd_step_bound(design, lam):
    """A provably safe constant step size for gradient descent on the SQ3 objective.

    Parameters
    ----------
    design : ndarray, shape (n, p)
    lam : float

    Returns
    -------
    float
        ``2.0 / L`` where ``L`` is the smallest constant you can justify as
        an upper bound on the largest eigenvalue of the Hessian, uniformly
        in ``theta``. Derive ``L`` from the fact that ``s(1 - s) <= 1/4``.

    Notes
    -----
    Use ``np.linalg.eigvalsh``. Do not estimate this by trial and error and
    do not use a power iteration with a loose tolerance; the returned number
    is compared to a reference to within a relative tolerance of 1e-9.
    """
    top_eigenvalue = np.linalg.eigvalsh(design.T @ design)[-1]
    lipschitz = top_eigenvalue / (4.0 * design.shape[0]) + lam
    return 2.0 / lipschitz


def fit_logistic_gd(design, labels, lam, step, tol=1e-10, max_iter=200000):
    """Minimise ``logistic_objective`` by constant-step gradient descent from zero.

    Same stopping rule, same off-by-one convention and the same return
    dictionary as ``fit_logistic_newton``: check ``max(abs(gradient)) < tol``
    BEFORE each step, and report the number of steps actually taken.

    Parameters
    ----------
    design, labels, lam : as above
    step : float
        Constant step size.
    tol : float
    max_iter : int

    Returns
    -------
    dict with keys ``"theta"``, ``"n_iter"``, ``"converged"``, ``"objective"``.

    Notes
    -----
    This must be a vectorised loop over iterations only. A Python loop over
    the n observations inside each iteration will exceed the grader's time
    limit on the SQ4 tests.
    """
    theta = np.zeros(design.shape[1], dtype=float)

    for iteration in range(max_iter):
        gradient = logistic_gradient(design, labels, theta, lam)

        if np.max(np.abs(gradient)) < tol:
            return {
                "theta": theta,
                "n_iter": iteration,
                "converged": True,
                "objective": logistic_objective(design, labels, theta, lam)
            }

        theta -= step * gradient

    return {
        "theta": theta,
        "n_iter": max_iter,
        "converged": False,
        "objective": logistic_objective(design, labels, theta, lam)
    }


# ===========================================================================
# SQ5 -- Which model should you reach for, and does the answer change? (5 marks)
# ===========================================================================


def error_gap(roll, regime, n_train, n_reps):
    """Paired comparison of LDA and logistic regression at one training size.

    For each repetition ``rep`` in ``range(n_reps)``:

      * draw a training set with
        ``ds5006_lab2_data.sample_two_class(roll, regime, n_train, rep)``;
      * fit LDA with ``lda_fit`` / predict with ``lda_predict``;
      * fit logistic regression on ``add_intercept`` of the features with
        ``fit_logistic_newton(..., lam=1e-4)``, predicting class 1 when
        ``x^T theta > 0``;
      * evaluate BOTH on the single fixed test set
        ``ds5006_lab2_data.test_set(roll, regime)``;
      * record ``gap_rep = err_lda - err_logistic``.

    Skip (do not count) any repetition in which the training draw contains
    fewer than two points of either class.

    Parameters
    ----------
    roll : str
    regime : {"gaussian", "contaminated"}
    n_train : int
    n_reps : int

    Returns
    -------
    dict with exactly these keys
        ``"mean_gap"``  : float, the mean of the recorded ``gap_rep``.
        ``"se_gap"``    : float, the standard error of that mean, using the
                          sample standard deviation with ``ddof=1``.
        ``"ci_low"``, ``"ci_high"`` : float, the endpoints of the interval
                          ``mean_gap +/- 1.96 * se_gap``.
        ``"n_used"``    : int, the number of repetitions actually counted.
        ``"err_lda"``, ``"err_logistic"`` : float, the mean test error of
                          each method across the counted repetitions.

    Notes
    -----
    A NEGATIVE ``mean_gap`` means LDA had the lower test error. The pairing
    matters: both methods see the same training draw and the same test set,
    so the repetition-to-repetition noise largely cancels and the standard
    error is far smaller than it would be for two independent experiments.
    """
    X_test, y_test = data.test_set(roll, regime)

    gaps = []
    err_lda = []
    err_log = []

    for rep in range(n_reps):
        X_train, y_train = data.sample_two_class(roll, regime, n_train, rep)

        if np.bincount(y_train, minlength=2).min() < 2:
            continue

        # LDA
        lda_model = lda_fit(X_train, y_train, 2)
        lda_pred = lda_predict(X_test, lda_model)
        e_lda = 1.0 - accuracy(y_test, lda_pred)

        # Logistic regression
        X_train_i = add_intercept(X_train)
        X_test_i = add_intercept(X_test)
        result = fit_logistic_newton(X_train_i, y_train, lam=1e-4)
        theta = result["theta"]
        log_scores = X_test_i @ theta
        log_pred = (log_scores > 0).astype(int)
        e_log = 1.0 - accuracy(y_test, log_pred)

        gaps.append(e_lda - e_log)
        err_lda.append(e_lda)
        err_log.append(e_log)

    gaps = np.array(gaps)
    n_used = len(gaps)
    mean_gap = float(gaps.mean())
    se_gap = float(gaps.std(ddof=1) / np.sqrt(n_used))

    return {
        "mean_gap": mean_gap,
        "se_gap": se_gap,
        "ci_low": mean_gap - 1.96 * se_gap,
        "ci_high": mean_gap + 1.96 * se_gap,
        "n_used": n_used,
        "err_lda": float(np.mean(err_lda)),
        "err_logistic": float(np.mean(err_log))
    }


# ===========================================================================
# SQ6 -- Is the model you fitted the only one that fits? (8 marks)
# ===========================================================================
#
# The multinomial objective, for ``Theta`` of shape (p, K):
#
#     J(Theta) = -(1/n) * sum_i sum_k Y[i,k] * log P[i,k]
#                + 0.5 * lam * ||Theta||_F^2 ,   P = softmax_rows(X @ Theta)
# ---------------------------------------------------------------------------


def multinomial_objective(design, targets, coefficients, lam):
    """Value of the penalised multinomial cross-entropy defined above.

    Parameters
    ----------
    design : ndarray, shape (n, p)
    targets : ndarray, shape (n, K) -- one-hot
    coefficients : ndarray, shape (p, K)
    lam : float

    Returns
    -------
    float

    Notes
    -----
    Must stay finite when a fitted probability underflows to zero. Work with
    log-probabilities computed stably rather than taking ``log`` of a softmax
    output that has already rounded to 0.
    """
    scores = design @ coefficients
    max_score = np.max(scores, axis=1, keepdims=True)
    shifted = scores - max_score
    log_normaliser = max_score[:, 0] + np.log(np.sum(np.exp(shifted), axis=1))
    log_probs = scores - log_normaliser[:, None]
    nll = -np.mean(np.sum(targets * log_probs, axis=1))
    penalty = 0.5 * lam * np.sum(coefficients ** 2)
    return nll + penalty


def multinomial_gradient(design, targets, coefficients, lam):
    """Gradient of ``multinomial_objective`` with respect to ``coefficients``.

    Returns
    -------
    ndarray, shape (p, K)
        ``design.T @ (P - targets) / n + lam * coefficients``.
    """
    scores = design @ coefficients
    probs = softmax_rows(scores)
    return design.T @ (probs - targets) / design.shape[0] + lam * coefficients


def fit_multinomial_gd(design, targets, lam, step, theta_init=None,
                       tol=1e-11, max_iter=300000):
    """Minimise ``multinomial_objective`` by constant-step gradient descent.

    Parameters
    ----------
    design : ndarray, shape (n, p)
    targets : ndarray, shape (n, K)
    lam : float
    step : float
    theta_init : ndarray of shape (p, K), or None
        Starting point. ``None`` means start from the all-zeros matrix.
        This argument exists so that SQ6 can start the SAME problem from
        two different places; do not ignore it.
    tol : float
        Stop when ``max(abs(gradient)) < tol``, checked BEFORE each step.
    max_iter : int

    Returns
    -------
    dict with keys ``"theta"`` (ndarray (p, K)), ``"n_iter"``,
    ``"converged"``, ``"objective"``.
    """
    if theta_init is None:
        theta = np.zeros((design.shape[1], targets.shape[1]), dtype=float)
    else:
        theta = np.array(theta_init, dtype=float, copy=True)

    for iteration in range(max_iter):
        gradient = multinomial_gradient(design, targets, theta, lam)

        if np.max(np.abs(gradient)) < tol:
            return {
                "theta": theta,
                "n_iter": iteration,
                "converged": True,
                "objective": multinomial_objective(design, targets, theta, lam)
            }

        theta -= step * gradient

    return {
        "theta": theta,
        "n_iter": max_iter,
        "converged": False,
        "objective": multinomial_objective(design, targets, theta, lam)
    }
