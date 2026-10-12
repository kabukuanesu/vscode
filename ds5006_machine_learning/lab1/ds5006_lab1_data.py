"""DS5006 Lab 1 -- per-student data generation.

DO NOT MODIFY THIS FILE. The grader replaces it with a pristine copy before
grading. Any edits you make here will be silently discarded, and code that
depends on those edits will fail.

Every dataset in this lab is seeded from your roll number, so your numbers
differ from everybody else's. Report a number that belongs to somebody else's
seed and it will be obvious.

The seed derives from a SHA-256 digest rather than Python's built-in ``hash``,
because ``hash`` is randomised per interpreter session and would give you a
different dataset on every run.
"""

from __future__ import annotations

import hashlib

import numpy as np


def student_seed(roll_number: str) -> int:
    """Map a roll number to a deterministic 32-bit seed.

    Args:
        roll_number: Your institute roll number, e.g. ``"112201015"``.
            Case and surrounding whitespace are ignored.

    Returns:
        An integer in ``[0, 2**32)``, stable across machines and sessions.
    """
    key = str(roll_number).strip().upper().encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def _rng(roll_number: str, stream: int) -> np.random.Generator:
    """Return an independent generator for one dataset stream."""
    return np.random.default_rng(student_seed(roll_number) + 1000 * stream)


def dataset_scaling(roll_number: str) -> tuple[np.ndarray, np.ndarray]:
    """Badly scaled regression data. Used in SQ1 and SQ2.

    Five features whose standard deviations span roughly four orders of
    magnitude, plus a non-zero intercept and moderate Gaussian noise.

    Returns:
        ``(X, y)`` with ``X`` of shape ``(240, 5)`` and ``y`` of shape ``(240,)``.
    """
    rng = _rng(roll_number, 1)
    n, d = 240, 5
    # Correlated latent structure first, so that the design stays non-trivially
    # conditioned even after standardisation, and only then wildly different
    # column scales on top. The two effects are deliberately separable: SQ2 asks
    # you to undo the second and see what survives.
    corr = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            corr[i, j] = corr[j, i] = 0.86 ** (j - i)
    chol = np.linalg.cholesky(corr)
    latent = rng.normal(size=(n, d)) @ chol.T
    exponents = rng.permutation(np.array([-2.0, -0.5, 0.0, 1.0, 2.5]))
    scales = 10.0**exponents
    offsets = rng.uniform(-3.0, 3.0, size=d) * scales
    x_mat = latent * scales + offsets
    w_true = rng.normal(size=d) / scales
    y_vec = x_mat @ w_true + rng.uniform(-5.0, 5.0) + 0.35 * rng.normal(size=n)
    return x_mat, y_vec


def dataset_conditioning(
    roll_number: str, degree: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Noise-free polynomial design of a given degree. Used in SQ3.

    Columns are ``x, x^2, ..., x^degree``. There is deliberately no constant
    column: every fitting routine in this lab handles the intercept separately,
    and a constant column would vanish under centring and make the centred
    design exactly singular.

    The response is generated with *no* noise, so the exact minimiser is known
    to be ``(w_true, b_true)``. Any deviation your solver produces is pure
    numerical error, which is what SQ3 asks you to measure.

    Args:
        roll_number: Your roll number.
        degree: Polynomial degree; the design has ``degree`` columns.

    Returns:
        ``(X, y, w_true, b_true)`` where ``X`` has shape ``(80, degree)`` and
        ``y == X @ w_true + b_true`` up to rounding in the matrix product.
    """
    rng = _rng(roll_number, 2)
    knots = np.sort(rng.uniform(0.0, 1.0, size=80))
    x_mat = np.stack([knots**p for p in range(1, degree + 1)], axis=1)
    w_true = rng.normal(size=degree)
    b_true = float(rng.normal())
    return x_mat, x_mat @ w_true + b_true, w_true, b_true


def truth_function(roll_number: str):
    """Return the true regression function ``f`` used in SQ4.

    A smooth, non-polynomial function. You are told its analytic form so that
    you can compute bias exactly rather than estimating it.

    Returns:
        A vectorised callable mapping ``x`` in ``[-1, 1]`` to ``f(x)``.
    """
    rng = _rng(roll_number, 3)
    amp, freq, phase, tilt = (
        rng.uniform(0.8, 1.6),
        rng.uniform(2.0, 3.5),
        rng.uniform(0.0, 2.0 * np.pi),
        rng.uniform(-1.0, 1.0),
    )

    def f_true(x_vals: np.ndarray) -> np.ndarray:
        x_vals = np.asarray(x_vals, dtype=float)
        return amp * np.sin(freq * x_vals + phase) + tilt * x_vals

    return f_true


def sample_training_set(
    roll_number: str, n_samples: int, noise_sd: float, replicate: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one training set from the SQ4 data-generating process.

    Args:
        roll_number: Your roll number.
        n_samples: Number of points to draw.
        noise_sd: Standard deviation of the additive Gaussian noise.
        replicate: Replicate index. Different values give independent draws;
            the same value always gives the same draw.

    Returns:
        ``(x, y)``, both of shape ``(n_samples,)``, with ``x`` uniform on
        ``[-1, 1]`` and ``y = f(x) + noise``.
    """
    rng = np.random.default_rng(student_seed(roll_number) + 7_000_003 * (replicate + 1))
    x_vals = rng.uniform(-1.0, 1.0, size=n_samples)
    return x_vals, truth_function(roll_number)(x_vals) + noise_sd * rng.normal(size=n_samples)


def dataset_wide(roll_number: str) -> tuple[np.ndarray, np.ndarray]:
    """Short, wide, strongly correlated data. Used in SQ5 and SQ6.

    Fewer observations than would be comfortable for the number of features,
    with the features generated from a small number of latent factors so that
    the design is close to rank-deficient.

    Returns:
        ``(X, y)`` with ``X`` of shape ``(70, 40)`` and ``y`` of shape ``(70,)``.
    """
    rng = _rng(roll_number, 4)
    n, d, k = 70, 40, 6
    latent = rng.normal(size=(n, k))
    loadings = rng.normal(size=(k, d))
    x_mat = latent @ loadings + 0.02 * rng.normal(size=(n, d))
    w_true = np.zeros(d)
    active = rng.choice(d, size=8, replace=False)
    w_true[active] = rng.normal(size=8) * 2.0
    y_vec = x_mat @ w_true + rng.uniform(-2.0, 2.0) + 1.5 * rng.normal(size=n)
    return x_mat, y_vec
