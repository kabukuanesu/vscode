"""
DS5006 Fall 2026 -- Lab 0 warm-up. NOT GRADED, but required.

Fill in every function below. NumPy and the standard library only.
Answer student_info() honestly -- including the low numbers. See
LAB0_INSTRUCTIONS.md for why that matters.
"""

import numpy as np


def student_info():
    """
    Return a dict with exactly these keys. Every field is required.

    Confidence fields are 1-5: 1 = "never done this", 5 = "very comfortable".
    Answer for where you are TODAY, not where you hope to be.
    """
    return {
        "name": "Anesu Kabuku",                          # str
        "roll_number": "142602026",                   # str
        "languages": ["python", "javascript", "C#"],                     # list of str, e.g. ["python", "c"]
        "years_programming": 3,              # number, any language, roughly
        "seen_svd": False,                   # bool: met the SVD in a course?
        "seen_eigendecomposition": False,    # bool
        "seen_pseudoinverse": False,         # bool
        "prior_ml_course": True,            # bool: DS3010 or equivalent
        "confidence_python": 3,              # 1-5
        "confidence_numpy": 2,               # 1-5
        "confidence_linear_algebra": 3,      # 1-5
        "hours_spent": 5.0,                  # fill in at the end
        "help_used": "Online Resources, Youtube Videos",                     # honest and specific; "none" is fine
    }


# ---- Part B: plain Python (do not use NumPy here) ------------------------

def running_max(xs):
    """
    xs: a list of numbers. Return a list of the same length whose i-th entry is
    the maximum of xs[0..i].

    running_max([1, 3, 2, 5, 4]) -> [1, 3, 3, 5, 5]
    running_max([])              -> []
    """
    result = []
    current_max = None

    for x in xs:
        if current_max is None or x > current_max:
            current_max = x
        result.append(current_max)
    return result


def count_words(text):
    """
    Return {word: count} for the words in text. Lower-case everything and strip
    non-alphanumeric characters from the ends of words.

    count_words("The cat, the CAT; a dog.") -> {"the": 2, "cat": 2, "a": 1, "dog": 1}
    """
    counts = {}

    for word in text.split():
        # Convert to lower case and strip non-alphanumeric characters from the ends
        word = word.lower().strip(".,;!?\"'()[]{}")

        if word:  # Only count non-empty words
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1

    return counts


# ---- Part C: NumPy basics ------------------------------------------------

def column_stats(X):
    """
    X: (n, p) array. Return {"mean": ..., "min": ..., "max": ..., "std": ...},
    each a (p,) array computed down the columns.

    Use the POPULATION standard deviation (divide by n, not n-1). NumPy's default
    already does this; the point is to notice that it is a choice.
    """
    return {
        "mean": np.mean(X, axis=0),
        "min": np.min(X, axis=0),
        "max": np.max(X, axis=0),
        "std": np.std(X, axis=0),
    }


def zscore_columns(X):
    """
    X: (n, p) array with no constant columns. Return the (n, p) array whose
    columns have mean 0 and population standard deviation 1.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return (X - mean) / std


def top_k_rows(X, k):
    """
    X: (n, p) array, k: int. Return the indices of the k rows with the largest
    Euclidean norm, ordered from largest to smallest.
    """
    norms = np.linalg.norm(X, axis=1)

    indices = np.argsort(norms)[::-1]

    return indices[:k] 


# ---- Part D: vectorisation -----------------------------------------------

def pairwise_sq_dists(A, B):
    """
    A: (n, d), B: (m, d). Return the (n, m) array D with
    D[i, j] = ||A[i] - B[j]||^2.

    TIMED: n = m = 2000, d = 10, budget 8 seconds. A loop over the n*m pairs
    takes over a minute. Expand the square and see what is left.
    Results must never be negative.
    """
    A_sq = np.sum(A ** 2, axis=1).reshape(-1, 1)  # (n, 1)
    B_sq = np.sum(B ** 2, axis=1).reshape(1, -1)  # (1, m)

    D = A_sq + B_sq - 2 * np.dot(A, B.T)

    return np.maximum(D, 0)


# ---- Part E: linear algebra ----------------------------------------------

def solve_system(A, b):
    """
    A: (n, n), b: (n,). Return x with A @ x = b.

    Checked on a system with condition number about 1e9. Forming A^{-1}
    explicitly and multiplying will not be accurate enough.
    """
    return np.linalg.solve(A, b)


def rank_and_condition(X):
    """
    Return (rank, cond) for the matrix X: its numerical rank, and the ratio of
    its largest to smallest singular value (np.inf if the smallest is zero).

    Use a tolerance of max(X.shape) * eps * largest_singular_value when deciding
    which singular values count as zero.
    """
    s = np.linalg.svd(X, compute_uv=False)

    if len(s) == 0:
        return 0, np.inf

    largest = s[0]

    tol = max(X.shape) * np.finfo(s.dtype).eps * largest

    rank = np.sum(s > tol)

    if s[-1] <= tol:
        cond = np.inf
    else:
        cond = largest / s[-1]

    return rank, cond


def low_rank_approx(X, r):
    """
    Return the best rank-r approximation to X in the Frobenius norm, with the
    same shape as X.

    If you have not met the singular value decomposition, leave this one and say
    so in student_info(). That is useful information, not a failure.
    """
    raise NotImplementedError


# ---- Part F: numerical care ----------------------------------------------

def log_sum_exp(z):
    """
    Return log(sum(exp(z))) as a float.

    Must stay finite and accurate when the entries of z are around +1000 or
    -1000, where the direct translation of the formula overflows or underflows.
    """
    raise NotImplementedError
