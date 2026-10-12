"""DS5006 Lab 2 -- public self-check tests.

Run this from the folder containing your ``ds5006_lab2.py``::

    python test_lab2_public.py

You will see one line per check:

    [ ok ]  the check passed
    [    ]  not implemented yet (the function still raises NotImplementedError)
    [FAIL]  implemented, but wrong -- the failing line is printed

These checks test CONTRACTS: shapes, key names, conventions, edge cases.
They do NOT test whether your numbers are right on the real datasets.
Passing everything here is necessary, not sufficient. The grader adds
hidden numerical tests, timing limits and static analysis.

You may read this file. You may not modify it -- the grader uses its own copy.
"""

import sys
import traceback

import numpy as np

try:
    import ds5006_lab2 as lab
except Exception as exc:                                # pylint: disable=broad-except
    print(f"Could not import ds5006_lab2.py: {exc!r}")
    sys.exit(1)


CHECKS = []


def check(name):
    """Register a check function under a display name."""
    def wrap(func):
        CHECKS.append((name, func))
        return func
    return wrap


# --------------------------------------------------------------- S0


@check("add_intercept: shape and ones column LAST")
def _c1():
    out = lab.add_intercept(np.arange(6.0).reshape(3, 2))
    assert out.shape == (3, 3), out.shape
    assert np.allclose(out[:, -1], 1.0), "ones column must be last"
    assert np.allclose(out[:, :2], np.arange(6.0).reshape(3, 2))


@check("one_hot: shape, dtype and row sums")
def _c2():
    out = lab.one_hot(np.array([0, 2, 1, 2]), 3)
    assert out.shape == (4, 3), out.shape
    assert np.allclose(out.sum(axis=1), 1.0)
    assert out[1, 2] == 1.0 and out[1, 0] == 0.0


@check("one_hot: handles a class with no members")
def _c3():
    out = lab.one_hot(np.array([0, 0, 2]), 4)
    assert out.shape == (3, 4), out.shape
    assert out[:, 1].sum() == 0.0 and out[:, 3].sum() == 0.0


@check("sigmoid: midpoint, monotone, and range")
def _c4():
    assert abs(float(lab.sigmoid(np.array([0.0]))[0]) - 0.5) < 1e-14
    vals = lab.sigmoid(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    assert np.all(np.diff(vals) > 0), "must be increasing"
    assert np.all((vals > 0) & (vals < 1))


@check("sigmoid: no overflow at +/-1500")
def _c5():
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        out = lab.sigmoid(np.array([-1500.0, -50.0, 50.0, 1500.0]))
    assert np.all(np.isfinite(out)), out
    assert out[0] >= 0.0 and out[-1] <= 1.0
    assert out[0] < 1e-300 or out[0] == 0.0


@check("softmax_rows: rows sum to one at extreme scale")
def _c6():
    out = lab.softmax_rows(np.array([[1e4, 0.0, -1e4], [1.0, 1.0, 1.0]]))
    assert out.shape == (2, 3), out.shape
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.allclose(out[1], 1.0 / 3.0)


@check("quadratic_expansion: pinned column order a, b, a^2, ab, b^2")
def _c7():
    out = lab.quadratic_expansion(np.array([[2.0, 3.0], [-1.0, 4.0]]))
    assert out.shape == (2, 5), out.shape
    assert np.allclose(out[0], [2.0, 3.0, 4.0, 6.0, 9.0]), out[0]
    assert np.allclose(out[1], [-1.0, 4.0, 1.0, -4.0, 16.0]), out[1]


# --------------------------------------------------------------- SQ1


@check("fit_indicator_regression: shape (p, K)")
def _c8():
    rng = np.random.default_rng(0)
    design = lab.add_intercept(rng.normal(size=(30, 2)))
    targets = lab.one_hot(rng.integers(0, 3, 30), 3)
    assert lab.fit_indicator_regression(design, targets).shape == (3, 3)


@check("fit_indicator_regression: reproduces an exact linear target")
def _c9():
    rng = np.random.default_rng(1)
    design = lab.add_intercept(rng.normal(size=(40, 3)))
    truth = rng.normal(size=(4, 2))
    got = lab.fit_indicator_regression(design, design @ truth)
    assert np.allclose(got, truth, atol=1e-8), np.abs(got - truth).max()


@check("indicator_predict: argmax with ties to the smallest index")
def _c10():
    design = np.eye(2)
    coef = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    assert np.array_equal(lab.indicator_predict(design, coef), [0, 0])


@check("accuracy: returns a float in [0, 1]")
def _c11():
    got = lab.accuracy(np.array([0, 1, 2, 1]), np.array([0, 1, 1, 1]))
    assert abs(float(got) - 0.75) < 1e-12, got


# --------------------------------------------------------------- SQ2


@check("lda_fit: exact key names and shapes")
def _c12():
    rng = np.random.default_rng(2)
    model = lab.lda_fit(rng.normal(size=(50, 3)), rng.integers(0, 2, 50), 2)
    assert set(model) == {"means", "cov", "priors"}, sorted(model)
    assert model["means"].shape == (2, 3)
    assert model["cov"].shape == (3, 3)
    assert model["priors"].shape == (2,)
    assert abs(model["priors"].sum() - 1.0) < 1e-12


@check("lda_fit: pooled covariance uses denominator n - K")
def _c13():
    design = np.array([[0.0], [2.0], [10.0], [14.0]])
    labels = np.array([0, 0, 1, 1])
    model = lab.lda_fit(design, labels, 2)
    # scatter = 1+1+4+4 = 10 ; n-K = 2 ; so cov = 5.0 (n would give 2.5)
    assert abs(float(model["cov"][0, 0]) - 5.0) < 1e-12, model["cov"][0, 0]


@check("lda_fit: means and priors on an unbalanced sample")
def _c14():
    design = np.array([[1.0], [3.0], [8.0]])
    model = lab.lda_fit(design, np.array([0, 0, 1]), 2)
    assert np.allclose(model["means"].ravel(), [2.0, 8.0])
    assert np.allclose(model["priors"], [2 / 3, 1 / 3])


@check("lda_scores / lda_predict: shape and separation of far classes")
def _c15():
    design = np.array([[-5.0], [-4.0], [4.0], [5.0]])
    labels = np.array([0, 0, 1, 1])
    model = lab.lda_fit(design, labels, 2)
    assert lab.lda_scores(design, model).shape == (4, 2)
    assert np.array_equal(lab.lda_predict(design, model), labels)


@check("ls_direction_two_class: shape is p, intercept dropped")
def _c16():
    rng = np.random.default_rng(3)
    out = lab.ls_direction_two_class(rng.normal(size=(40, 3)),
                                     rng.integers(0, 2, 40), 0.0, 1.0)
    assert out.shape == (3,), out.shape


@check("ls_direction_two_class: rescaling the codes only rescales the vector")
def _c17():
    rng = np.random.default_rng(4)
    design, labels = rng.normal(size=(60, 3)), rng.integers(0, 2, 60)
    one = lab.ls_direction_two_class(design, labels, 0.0, 1.0)
    two = lab.ls_direction_two_class(design, labels, 0.0, 5.0)
    assert np.allclose(two, 5.0 * one, atol=1e-9), np.abs(two - 5 * one).max()


# --------------------------------------------------------------- SQ3


@check("logistic_objective: value at theta = 0 is log 2")
def _c18():
    rng = np.random.default_rng(5)
    design = rng.normal(size=(20, 3))
    val = lab.logistic_objective(design, rng.integers(0, 2, 20),
                                 np.zeros(3), 0.0)
    assert abs(float(val) - np.log(2.0)) < 1e-12, val


@check("logistic_objective: penalty term is 0.5 * lam * ||theta||^2, intercept included")
def _c19():
    design = np.zeros((5, 2))
    theta = np.array([3.0, 4.0])
    val = lab.logistic_objective(design, np.zeros(5, dtype=int), theta, 2.0)
    # margins are 0 -> mean nll = log 2 ; penalty = 0.5*2*25 = 25
    assert abs(float(val) - (np.log(2.0) + 25.0)) < 1e-12, val


@check("logistic_gradient: matches a finite difference")
def _c20():
    rng = np.random.default_rng(6)
    design, labels = rng.normal(size=(25, 3)), rng.integers(0, 2, 25)
    theta, lam, eps = rng.normal(size=3), 0.3, 1e-6
    ana = lab.logistic_gradient(design, labels, theta, lam)
    num = np.empty(3)
    for j in range(3):
        step = np.zeros(3)
        step[j] = eps
        num[j] = (lab.logistic_objective(design, labels, theta + step, lam)
                  - lab.logistic_objective(design, labels, theta - step, lam)) / (2 * eps)
    assert np.allclose(ana, num, atol=1e-6), np.abs(ana - num).max()


@check("logistic_hessian: symmetric, shape (p, p), matches a finite difference")
def _c21():
    rng = np.random.default_rng(7)
    design, labels = rng.normal(size=(25, 3)), rng.integers(0, 2, 25)
    theta, lam, eps = rng.normal(size=3), 0.3, 1e-6
    hess = lab.logistic_hessian(design, theta, lam)
    assert hess.shape == (3, 3), hess.shape
    assert np.allclose(hess, hess.T, atol=1e-12), "must be symmetric"
    num = np.empty((3, 3))
    for j in range(3):
        step = np.zeros(3)
        step[j] = eps
        num[:, j] = (lab.logistic_gradient(design, labels, theta + step, lam)
                     - lab.logistic_gradient(design, labels, theta - step, lam)) / (2 * eps)
    assert np.allclose(hess, num, atol=1e-6), np.abs(hess - num).max()


@check("fit_logistic_newton: return keys, and n_iter == 0 when already converged")
def _c22():
    design = np.zeros((6, 2))
    out = lab.fit_logistic_newton(design, np.ones(6, dtype=int) * 0, 1.0)
    assert set(out) == {"theta", "n_iter", "converged", "objective"}, sorted(out)
    # gradient at theta=0 here is design.T @ (0.5 - 0)/n + 0 = 0 exactly
    assert out["n_iter"] == 0, out["n_iter"]
    assert out["converged"] is True or out["converged"] == np.True_


@check("fit_logistic_newton: solves a well-posed problem to a tiny gradient")
def _c23():
    rng = np.random.default_rng(8)
    design = lab.add_intercept(rng.normal(size=(80, 3)))
    labels = rng.integers(0, 2, 80)
    out = lab.fit_logistic_newton(design, labels, 1e-2)
    grad = lab.logistic_gradient(design, labels, out["theta"], 1e-2)
    assert np.max(np.abs(grad)) < 1e-9, np.max(np.abs(grad))


@check("min_signed_margin: sign convention")
def _c24():
    design = np.array([[1.0], [-1.0]])
    assert abs(float(lab.min_signed_margin(design, np.array([1, 0]),
                                           np.array([2.0]))) - 2.0) < 1e-12
    assert float(lab.min_signed_margin(design, np.array([0, 1]),
                                       np.array([2.0]))) < 0


@check("is_separable: an obvious yes and an obvious no")
def _c25():
    yes_x = np.array([[-3.0], [-2.0], [2.0], [3.0]])
    assert lab.is_separable(yes_x, np.array([0, 0, 1, 1])) is True
    no_x = np.array([[-3.0], [-2.0], [2.0], [3.0]])
    assert lab.is_separable(no_x, np.array([0, 1, 0, 1])) is False


@check("separable_fraction: in range, and monotone in the sample size")
def _c26():
    few = lab.separable_fraction("112201042", 4, 3, 40)
    many = lab.separable_fraction("112201042", 20, 3, 40)
    assert 0.0 <= few <= 1.0 and 0.0 <= many <= 1.0, (few, many)
    assert few > many, \
        "more points with random labels should be HARDER to separate"
    assert abs(lab.separable_fraction("112201042", 3, 5, 20) - 1.0) < 1e-12, \
        "with more parameters than points, every labelling is separable"


# --------------------------------------------------------------- SQ4


@check("gd_step_bound: exact value on a hand-checkable design")
def _c27():
    # X = 2 * I_2, n = 2, lam = 0  ->  X^T X = 4 I, top eig 4,
    # L = 4/(4*2) = 0.5, bound = 4.0
    assert abs(float(lab.gd_step_bound(2.0 * np.eye(2), 0.0)) - 4.0) < 1e-9


@check("fit_logistic_gd: reaches the same answer as Newton")
def _c28():
    rng = np.random.default_rng(9)
    design = lab.add_intercept(rng.normal(size=(60, 2)))
    labels = rng.integers(0, 2, 60)
    lam = 1e-2
    newton = lab.fit_logistic_newton(design, labels, lam)
    step = 0.9 * lab.gd_step_bound(design, lam)
    grad_d = lab.fit_logistic_gd(design, labels, lam, step, max_iter=200000)
    assert set(grad_d) == {"theta", "n_iter", "converged", "objective"}
    assert np.allclose(grad_d["theta"], newton["theta"], atol=1e-5), \
        np.abs(grad_d["theta"] - newton["theta"]).max()


# --------------------------------------------------------------- SQ5


@check("error_gap: exact key names and a consistent interval")
def _c29():
    out = lab.error_gap("112201042", "gaussian", 40, 6)
    need = {"mean_gap", "se_gap", "ci_low", "ci_high", "n_used",
            "err_lda", "err_logistic"}
    assert set(out) == need, sorted(out)
    assert out["ci_low"] <= out["mean_gap"] <= out["ci_high"]
    assert abs((out["err_lda"] - out["err_logistic"]) - out["mean_gap"]) < 1e-12
    assert 0 < out["n_used"] <= 6


# --------------------------------------------------------------- SQ6


@check("multinomial_objective: value at Theta = 0 is log K")
def _c30():
    rng = np.random.default_rng(10)
    design = rng.normal(size=(20, 3))
    targets = lab.one_hot(rng.integers(0, 4, 20), 4)
    val = lab.multinomial_objective(design, targets, np.zeros((3, 4)), 0.0)
    assert abs(float(val) - np.log(4.0)) < 1e-12, val


@check("multinomial_gradient: matches a finite difference")
def _c31():
    rng = np.random.default_rng(11)
    design = rng.normal(size=(25, 3))
    targets = lab.one_hot(rng.integers(0, 3, 25), 3)
    coef, lam, eps = rng.normal(size=(3, 3)), 0.2, 1e-6
    ana = lab.multinomial_gradient(design, targets, coef, lam)
    assert ana.shape == (3, 3), ana.shape
    num = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            step = np.zeros((3, 3))
            step[i, j] = eps
            num[i, j] = (lab.multinomial_objective(design, targets, coef + step, lam)
                         - lab.multinomial_objective(design, targets, coef - step, lam)) / (2 * eps)
    assert np.allclose(ana, num, atol=1e-6), np.abs(ana - num).max()


@check("fit_multinomial_gd: honours theta_init and returns the right shape")
def _c32():
    rng = np.random.default_rng(12)
    design = lab.add_intercept(rng.normal(size=(40, 2)))
    targets = lab.one_hot(rng.integers(0, 3, 40), 3)
    init = np.full((3, 3), 0.4)
    out = lab.fit_multinomial_gd(design, targets, 0.5, 0.5,
                                 theta_init=init, max_iter=0)
    assert out["theta"].shape == (3, 3), out["theta"].shape
    assert np.allclose(out["theta"], init), \
        "with max_iter=0 the result must be theta_init untouched"


def main():
    """Run every registered check and print a summary."""
    n_ok = n_todo = n_bad = 0
    for name, func in CHECKS:
        try:
            func()
        except NotImplementedError:
            print(f"[    ]  {name}")
            n_todo += 1
        except AssertionError as exc:
            frame = traceback.extract_tb(sys.exc_info()[2])[-1]
            print(f"[FAIL]  {name}")
            print(f"          line {frame.lineno}: {frame.line}")
            if str(exc):
                print(f"          got: {exc}")
            n_bad += 1
        except Exception as exc:                        # pylint: disable=broad-except
            frame = traceback.extract_tb(sys.exc_info()[2])[-1]
            print(f"[FAIL]  {name}")
            print(f"          line {frame.lineno}: {frame.line}")
            print(f"          {type(exc).__name__}: {exc}")
            n_bad += 1
        else:
            print(f"[ ok ]  {name}")
            n_ok += 1
    total = len(CHECKS)
    print(f"\n  {n_ok}/{total} passed | {n_todo} not implemented | {n_bad} failing")
    if n_bad:
        print("  Fix the [FAIL] lines: those are implemented but wrong.")
    elif n_todo:
        print("  Keep going: the [    ] lines are still stubs.")
    else:
        print("  All contract checks pass. Now check your NUMBERS on the real data.")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
