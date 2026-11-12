"""
DS5006 Lab 4 -- public self-test.

    python selftest_lab4.py

These are the checks you can run yourself.  They test the conventions,
not the science: passing every one of them means your functions have the
right shapes, signs and edge-case behaviour, which is necessary but not
sufficient for a correct lab.  The grader runs these plus a set of hidden
tests on data you have not seen.

Nothing here needs your roll number.
"""

import sys
import traceback
import numpy as np

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  [ ok ] {name}")
    except AssertionError as e:
        FAIL.append((name, str(e)))
        print(f"  [FAIL] {name}\n         {e}")
    except Exception as e:
        FAIL.append((name, f"{type(e).__name__}: {e}"))
        print(f"  [FAIL] {name}\n         {type(e).__name__}: {e}")


# ----------------------------------------------------------------------
print("=" * 66)
print("0. Environment")
print("=" * 66)

try:
    import cvxopt                                        # noqa: F401
    print("  [ ok ] cvxopt is installed")
except ImportError:
    print("  [FAIL] cvxopt is NOT installed.  Run:  pip install cvxopt")
    sys.exit(1)

try:
    import svm_lab as S
    from svmlab_provided import solve_dual, make_dataset, TOL
except Exception:
    print("  [FAIL] could not import your svm_lab.py")
    traceback.print_exc()
    sys.exit(1)
print("  [ ok ] svm_lab.py imports")


# ----------------------------------------------------------------------
print("\n" + "=" * 66)
print("1. Kernels")
print("=" * 66)

A = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
B = np.array([[1.0, 1.0], [-1.0, 0.5]])


def t_lin_shape():
    out = S.linear_kernel(A, B)
    assert out.shape == (3, 2), f"expected shape (3, 2), got {out.shape}"


def t_lin_value():
    out = S.linear_kernel(A, B)
    want = np.array([[0.0, 0.0], [1.0, -1.0], [2.0, 1.0]])
    assert np.allclose(out, want), f"expected\n{want}\ngot\n{out}"


def t_rbf_shape():
    out = S.rbf_kernel(A, B, 1.0)
    assert out.shape == (3, 2), f"expected shape (3, 2), got {out.shape}"


def t_rbf_diag():
    out = S.rbf_kernel(A, A, 0.7)
    assert np.allclose(np.diag(out), 1.0), \
        "K(x, x) must be exactly 1 for the Gaussian kernel"


def t_rbf_value():
    # ||(0,0)-(1,1)||^2 = 2 ; s2 = 2  ->  exp(-2/4) = exp(-0.5)
    out = S.rbf_kernel(np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]), 2.0)
    assert np.allclose(out, np.exp(-0.5)), (
        f"expected exp(-0.5) = {np.exp(-0.5):.6f}, got {out.ravel()[0]:.6f}.  "
        "Check you are dividing by 2*s2, not by s2 or by 2*s2**2."
    )


def t_rbf_s2_direction():
    small = S.rbf_kernel(A, B, 0.05)
    large = S.rbf_kernel(A, B, 50.0)
    assert large.mean() > small.mean(), (
        "larger s2 must make distant points MORE similar; "
        "you may have s2 and 1/s2 the wrong way round"
    )


def t_rbf_symmetric():
    out = S.rbf_kernel(A, A, 1.3)
    assert np.allclose(out, out.T), "K(X, X) must be symmetric"


for n, f in [("linear_kernel shape (n_A, n_B)", t_lin_shape),
             ("linear_kernel values", t_lin_value),
             ("rbf_kernel shape (n_A, n_B)", t_rbf_shape),
             ("rbf_kernel diagonal is 1", t_rbf_diag),
             ("rbf_kernel uses 2*s2 in the denominator", t_rbf_value),
             ("rbf_kernel: larger s2 = smoother", t_rbf_s2_direction),
             ("rbf_kernel symmetry", t_rbf_symmetric)]:
    check(n, f)


# ----------------------------------------------------------------------
print("\n" + "=" * 66)
print("2. Regime classification")
print("=" * 66)

C_T, TOL_T = 2.0, 1e-6


def t_reg_codes():
    a = np.array([0.0, 1e-9, 1.0, 2.0 - 1e-9, 2.0])
    out = S.classify_regimes(a, C_T, TOL_T)
    want = np.array([0, 0, 1, 2, 2])
    assert np.array_equal(out, want), (
        f"expected {want} for alpha = {a} with C = 2, tol = 1e-6; got {out}.\n"
        "         Recall: non-SV is alpha <= tol, bounded is alpha >= C - tol."
    )


def t_reg_dtype():
    out = S.classify_regimes(np.array([0.0, 1.0, 2.0]), C_T, TOL_T)
    assert np.issubdtype(np.asarray(out).dtype, np.integer), \
        f"regime codes must be integers, got dtype {np.asarray(out).dtype}"


def t_reg_boundary():
    a = np.array([TOL_T, C_T - TOL_T])
    out = S.classify_regimes(a, C_T, TOL_T)
    assert np.array_equal(out, np.array([0, 2])), (
        f"alpha exactly at tol and exactly at C-tol should be 0 and 2, got {out}"
    )


for n, f in [("codes 0 / 1 / 2 assigned correctly", t_reg_codes),
             ("codes are an integer array", t_reg_dtype),
             ("boundary cases are inclusive on the high side", t_reg_boundary)]:
    check(n, f)


# ----------------------------------------------------------------------
print("\n" + "=" * 66)
print("3. Margin width, decision values, intercept")
print("=" * 66)


def t_margin_two_over():
    # theta = sum a_i y_i x_i with a linear kernel; pick numbers by hand.
    X = np.array([[1.0, 0.0], [-1.0, 0.0]])
    y = np.array([1.0, -1.0])
    a = np.array([0.5, 0.5])
    K = S.linear_kernel(X, X)
    th = (a * y) @ X                       # = [1, 0]
    want = 2.0 / np.linalg.norm(th)        # = 2.0
    got = S.margin_width(a, y, K)
    assert np.isclose(got, want), (
        f"expected 2/||theta|| = {want}, got {got}.  "
        f"(1/||theta|| would be {want/2}.)"
    )


def t_margin_zero_theta():
    X = np.array([[1.0, 0.0], [1.0, 0.0]])
    y = np.array([1.0, -1.0])
    a = np.array([0.5, 0.5])
    got = S.margin_width(a, y, S.linear_kernel(X, X))
    assert np.isinf(got), f"theta = 0 must give inf, got {got}"


def t_decision_shape():
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    y = np.array([1.0, -1.0, 1.0])
    a = np.array([0.5, 0.5, 0.0])
    Kt = S.linear_kernel(np.zeros((4, 2)), X)
    g = S.decision_values(a, 0.25, y, Kt)
    assert np.asarray(g).shape == (4,), \
        f"expected shape (4,), got {np.asarray(g).shape}"
    assert np.allclose(g, 0.25), \
        f"at the origin g should equal theta_0 = 0.25, got {g}"


def t_intercept_free():
    X = np.array([[1.0, 0.0], [-1.0, 0.0]])
    y = np.array([1.0, -1.0])
    a = np.array([0.5, 0.5])               # both free when C = 2
    K = S.linear_kernel(X, X)
    b = S.recover_intercept(a, y, K, 2.0, TOL_T)
    assert np.isclose(b, 0.0, atol=1e-12), \
        f"by symmetry theta_0 should be 0, got {b}"


def t_intercept_excludes_bounded():
    # Three points.  Two free, one bounded and badly placed.  If the
    # bounded point is included the answer moves; it must not.
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [-6.0, 0.0]])
    y = np.array([1.0, -1.0, -1.0])
    C = 1.0
    a = np.array([0.5, 0.5, 1.0])          # third is bounded (alpha = C)
    K = S.linear_kernel(X, X)
    b = S.recover_intercept(a, y, K, C, TOL_T)
    f = K @ (a * y)
    b_free = np.mean(y[:2] - f[:2])
    b_all = np.mean(y - f)
    assert np.isclose(b, b_free, atol=1e-10), (
        f"expected the free-only average {b_free:.6f}, got {b:.6f}.\n"
        f"         (averaging over all three support vectors gives "
        f"{b_all:.6f} -- that is the error this test looks for.)"
    )


def t_intercept_no_free():
    # Every multiplier at the bound: fall back to the KKT interval midpoint.
    X = np.array([[1.0, 0.0], [-1.0, 0.0]])
    y = np.array([1.0, -1.0])
    C = 0.5
    a = np.array([0.5, 0.5])               # both bounded
    K = S.linear_kernel(X, X)
    b = S.recover_intercept(a, y, K, C, TOL_T)
    assert np.isfinite(b), f"must return a finite number, got {b}"
    assert np.isclose(b, 0.0, atol=1e-9), \
        f"by symmetry the midpoint is 0, got {b}"


def t_kkt_zero_when_no_free():
    X = np.array([[1.0, 0.0], [-1.0, 0.0]])
    y = np.array([1.0, -1.0])
    a = np.array([0.5, 0.5])
    K = S.linear_kernel(X, X)
    got = S.kkt_residual(a, 0.0, y, K, 0.5, TOL_T)
    assert got == 0.0, f"no free support vectors must give 0.0, got {got}"


for n, f in [("margin_width returns 2/||theta||", t_margin_two_over),
             ("margin_width returns inf when theta = 0", t_margin_zero_theta),
             ("decision_values shape and intercept", t_decision_shape),
             ("recover_intercept on free support vectors", t_intercept_free),
             ("recover_intercept EXCLUDES bounded SVs", t_intercept_excludes_bounded),
             ("recover_intercept falls back when no free SVs", t_intercept_no_free),
             ("kkt_residual is 0.0 with no free SVs", t_kkt_zero_when_no_free)]:
    check(n, f)


# ----------------------------------------------------------------------
print("\n" + "=" * 66)
print("4. End to end on a small real fit")
print("=" * 66)


def _small():
    # 40 points from each class.  Slicing X[:80] would take one class only
    # and give a degenerate problem in which every multiplier is zero.
    X, y, _, _ = make_dataset("SELFTEST")
    idx = np.r_[0:40, 120:160]
    return X[idx], y[idx]


def t_end_to_end():
    X, y = _small()
    K = S.rbf_kernel(X, X, 1.0)
    a = solve_dual(K, y, 1.0)
    b = S.recover_intercept(a, y, K, 1.0, TOL)
    code = S.classify_regimes(a, 1.0, TOL)
    assert (code == 1).any(), (
        "this fit should produce free support vectors; if it does not, your "
        "classify_regimes or the solver output is wrong"
    )
    assert (code == 2).any(), "this fit should produce bounded support vectors"
    r = S.kkt_residual(a, b, y, K, 1.0, TOL)
    assert r < 1e-6, (
        f"KKT residual is {r:.3e}, which is far too large.  With the correct "
        "intercept it should be near solver precision.  The usual cause is "
        "including bounded support vectors in the intercept average."
    )


def t_regimes_consistent():
    X, y = _small()
    K = S.rbf_kernel(X, X, 1.0)
    a = solve_dual(K, y, 1.0)
    b = S.recover_intercept(a, y, K, 1.0, TOL)
    code = S.classify_regimes(a, 1.0, TOL)
    g = S.decision_values(a, b, y, K)
    m = y * g
    if (code == 0).any():
        assert m[code == 0].min() > 1.0 - 1e-4, \
            "non-support-vectors must satisfy y*g >= 1"
    if (code == 2).any():
        assert m[code == 2].max() < 1.0 + 1e-4, \
            "bounded support vectors must satisfy y*g <= 1"


for n, f in [("fit gives a tiny KKT residual", t_end_to_end),
             ("the three regimes satisfy Proposition 6.1", t_regimes_consistent)]:
    check(n, f)


# ----------------------------------------------------------------------
print("\n" + "=" * 66)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
print("=" * 66)
if FAIL:
    print("\nStill to fix:")
    for n, e in FAIL:
        print(f"  - {n}")
    sys.exit(1)
print("\nConventions look right.  That is not the same as the science being "
      "right:\nthe hidden tests also check your experiment outputs.")
