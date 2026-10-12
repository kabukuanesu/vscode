"""
DS5006 Lab 0 -- public self-check.

    python test_public.py

Run this on the untouched stub FIRST. It should report eleven unimplemented
functions and no errors. If it does anything else, fix your environment now.
"""

import sys
import traceback

import numpy as np

import lab0 as S

DONE, TODO, BROKEN = [], [], []
REQUIRED_INFO = [
    "name", "roll_number", "languages", "years_programming", "seen_svd",
    "seen_eigendecomposition", "seen_pseudoinverse", "prior_ml_course",
    "confidence_python", "confidence_numpy", "confidence_linear_algebra",
    "hours_spent", "help_used",
]


def check(name):
    def deco(fn):
        try:
            fn()
            DONE.append(name)
            print(f"  done       {name}")
        except NotImplementedError:
            TODO.append(name)
            print(f"  to do      {name}")
        except AssertionError as e:
            BROKEN.append(name)
            print(f"  NOT YET    {name}\n               {e}")
        except Exception as e:
            BROKEN.append(name)
            print(f"  NOT YET    {name}\n               {type(e).__name__}: {e}")
        return fn
    return deco


def eq(a, b, tol=1e-9, what=""):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, f"{what}: shape {a.shape}, expected {b.shape}"
    assert np.all(np.isfinite(a)), f"{what}: contains nan or inf"
    d = float(np.max(np.abs(a - b))) if a.size else 0.0
    assert d <= tol, f"{what}: off by {d:.3e}"


print("\nPart A -- background")


@check("student_info")
def _():
    i = S.student_info()
    assert isinstance(i, dict), "should return a dict"
    missing = [k for k in REQUIRED_INFO if k not in i]
    assert not missing, f"missing keys: {', '.join(missing)}"
    assert str(i["name"]).strip(), "name is blank"
    assert str(i["roll_number"]).strip(), "roll_number is blank"
    assert isinstance(i["languages"], list), "languages should be a list"


print("\nPart B -- plain Python")


@check("running_max")
def _():
    assert S.running_max([1, 3, 2, 5, 4]) == [1, 3, 3, 5, 5]
    assert S.running_max([]) == [], "empty input should give []"
    assert S.running_max([-1, -5, -2]) == [-1, -1, -1], "all-negative input"


@check("count_words")
def _():
    got = S.count_words("The cat, the CAT; a dog.")
    assert got == {"the": 2, "cat": 2, "a": 1, "dog": 1}, f"got {got}"
    assert S.count_words("") == {}, "empty text should give {}"


print("\nPart C -- NumPy basics")


@check("column_stats")
def _():
    X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 60.0]])
    s = S.column_stats(X)
    assert isinstance(s, dict), "should return a dict"
    for k in ("mean", "std", "min", "max"):
        assert k in s, f"missing key '{k}'"
    eq(s["mean"], [3.0, 30.0], what="mean")
    eq(s["std"], np.std(X, axis=0, ddof=0), what="std -- population, ddof=0")
    eq(s["min"], [1.0, 10.0], what="min")
    eq(s["max"], [5.0, 60.0], what="max")


@check("zscore_columns")
def _():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4)) * np.array([1.0, 50.0, 0.1, 3.0]) + 7.0
    Z = np.asarray(S.zscore_columns(X), float)
    eq(Z.mean(0), np.zeros(4), tol=1e-10, what="column means")
    eq(Z.std(0, ddof=0), np.ones(4), tol=1e-10, what="column sds")


@check("top_k_rows")
def _():
    X = np.array([[3.0, 4.0], [0.0, 1.0], [6.0, 8.0], [1.0, 1.0]])
    eq(np.asarray(S.top_k_rows(X, 2)).ravel(), [2, 0],
       what="indices, largest norm first")


print("\nPart D -- vectorisation")


@check("pairwise_sq_dists")
def _():
    rng = np.random.default_rng(1)
    A, B = rng.normal(size=(7, 3)), rng.normal(size=(5, 3))
    ref = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    got = np.asarray(S.pairwise_sq_dists(A, B), float)
    assert got.shape == (7, 5), f"shape {got.shape}, expected (7, 5)"
    eq(got, ref, what="distances")
    d = np.asarray(S.pairwise_sq_dists(A, A), float)
    assert np.all(d >= -1e-12), "negative squared distance -- clip at zero"
    eq(np.diag(d), np.zeros(7), what="diagonal of self-distances")


print("\nPart E -- linear algebra")


@check("solve_system")
def _():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(6, 6))
    x = rng.normal(size=6)
    eq(S.solve_system(A, A @ x), x, tol=1e-8, what="well-conditioned solve")
    U, _, Vt = np.linalg.svd(rng.normal(size=(30, 30)))
    Aill = (U * np.logspace(0, -9, 30)) @ Vt
    xi = rng.normal(size=30)
    got = np.asarray(S.solve_system(Aill, Aill @ xi), float)
    err = float(np.max(np.abs(got - xi)))
    assert err < 1e-4, (f"off by {err:.2e} on a cond~1e9 system -- an explicit "
                        "matrix inverse loses too much accuracy")


@check("rank_and_condition")
def _():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(20, 4))
    r, c = S.rank_and_condition(np.hstack([A, A[:, :2]]))
    assert int(r) == 4, f"rank {r}, expected 4"
    assert not np.isfinite(c) or c > 1e10, f"condition number {c:.2e} for a singular matrix"


@check("low_rank_approx")
def _():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(15, 8))
    s = np.linalg.svd(X, compute_uv=False)
    for r in (1, 3):
        Xr = np.asarray(S.low_rank_approx(X, r), float)
        assert Xr.shape == X.shape, f"shape {Xr.shape}"
        best = float(np.sqrt(np.sum(s[r:] ** 2)))
        got = float(np.linalg.norm(X - Xr))
        assert got <= best * (1 + 1e-6) + 1e-9, (
            f"r={r}: error {got:.6f}, but {best:.6f} is achievable")


print("\nPart F -- numerical care")


@check("log_sum_exp")
def _():
    z = np.array([1.0, 2.0, 3.0])
    eq(S.log_sum_exp(z), np.log(np.sum(np.exp(z))), tol=1e-10, what="small values")
    got = float(S.log_sum_exp(np.array([1000.0, 1001.0, 999.0])))
    assert np.isfinite(got), "overflowed to inf on inputs around 1000"
    eq(got, 1000.0 + np.log(np.e ** 0 + np.e ** 1 + np.e ** -1),
       tol=1e-9, what="large values")


print("\n" + "-" * 62)
print(f"  done: {len(DONE)}    still to write: {len(TODO)}    "
      f"attempted but not passing: {len(BROKEN)}")
if BROKEN:
    print("  look at: " + ", ".join(BROKEN))
if TODO:
    print("  not started: " + ", ".join(TODO))
if not TODO and not BROKEN:
    print("  everything passes. Fill in hours_spent, write reflection.md, submit.")
print("-" * 62)
print("\nThis lab is not graded. If you could not finish something, submit it")
print("unfinished and say so in reflection.md -- that is the useful outcome.\n")
sys.exit(0)
