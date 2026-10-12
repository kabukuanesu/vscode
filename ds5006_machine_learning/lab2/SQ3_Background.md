# DS5006 Lab 2 — Background notes for SQ3

SQ3 needs two things lectures have not covered.

**Part A — checking derivatives numerically.** A practical skill, and the single most useful
debugging tool in this lab. You will use it in SQ3, SQ4 and SQ6. **Read this before you start
SQ3.**

**Part B — why separation breaks maximum likelihood.** Three pages of theory that make SQ3 one
coherent question instead of three disconnected exercises. Read it when you reach the penalty
sweep.

**Part C — the theorem behind the separability sweep.** Half a page, **entirely optional**, for
anyone curious about where the crossing point in their plot comes from. Nothing in the lab
requires it and nothing in the grader tests it.

---

# Part A — Checking derivatives numerically

## A.1 The problem this solves

You are about to write `logistic_gradient` and `logistic_hessian` by hand. Hand-derived
derivatives are wrong surprisingly often — a missing factor of `n`, a sign error, a transpose in
the wrong place — and a wrong gradient does not announce itself. Your optimiser will still run.
It will still produce numbers. They will just be the wrong numbers, and you will lose a day
hunting for the bug somewhere else entirely.

The fix takes ten lines and five minutes: compare your analytic derivative against a numerical
approximation built only from the *objective function*. If they agree, both are almost certainly
right, because it is very unlikely that two independent mistakes agree to eight decimal places.

This is called **gradient checking**, and it is standard practice in every serious numerical
codebase.

## A.2 The idea

The derivative is defined as a limit:

```
f'(x) = lim   [ f(x + h) - f(x) ] / h
        h->0
```

So the obvious approximation is to pick a small `h` and not take the limit:

```
forward difference:   f'(x)  ~  [ f(x + h) - f(x) ] / h
```

How good is it? Taylor expand:

```
f(x + h) = f(x) + h f'(x) + (h^2/2) f''(x) + ...
```

Rearranged:

```
[ f(x + h) - f(x) ] / h  =  f'(x) + (h/2) f''(x) + ...
```

The error is proportional to `h`. Halve `h`, halve the error. That is **first-order accurate**.

Now do the same with a step in each direction and subtract:

```
f(x + h) = f(x) + h f'(x) + (h^2/2) f''(x) + (h^3/6) f'''(x) + ...
f(x - h) = f(x) - h f'(x) + (h^2/2) f''(x) - (h^3/6) f'''(x) + ...
```

Subtracting kills `f(x)` *and* the `f''` term:

```
central difference:   f'(x)  ~  [ f(x + h) - f(x - h) ] / (2h)
```

with error proportional to `h^2`. Halve `h` and the error drops by a factor of four. That is
**second-order accurate**, it costs one extra function evaluation, and you should essentially
always use it.

## A.3 Why you cannot just make h tiny

Mathematically, smaller `h` is always better. Numerically, it is not, and this is the part that
catches people.

Doubles carry about 16 significant digits. When you compute `f(x + h)` and `f(x - h)` for tiny
`h`, the two values are nearly identical, so their difference cancels almost all the significant
digits — and then you divide that mangled difference by a tiny number, amplifying whatever
noise survived. This is **catastrophic cancellation**.

So there are two competing error sources:

| Source | Behaves like | Wants |
|---|---|---|
| Truncation (the Taylor remainder) | `h^2` for central differences | `h` small |
| Round-off (cancellation) | `eps / h`, with `eps ~ 1e-16` | `h` large |

Their sum is a U-shaped curve. Minimising `h^2 + eps/h` gives an optimum around
`h ~ eps^(1/3) ~ 6e-6` for central differences, and `h ~ sqrt(eps) ~ 1e-8` for forward
differences.

Here is that U-curve measured on the actual Lab 2 logistic objective, reporting the largest
relative error across the three components of the gradient:

| `h` | forward difference | central difference |
|---|---|---|
| `1e-1` | 1.80e-02 | 7.36e-05 |
| `1e-2` | 1.80e-03 | 7.37e-07 |
| `1e-3` | 1.80e-04 | 7.37e-09 |
| `1e-4` | 1.80e-05 | 7.32e-11 |
| **`1e-5`** | 1.80e-06 | **6.98e-12** ← best |
| `1e-6` | 1.80e-07 | 2.71e-10 |
| **`1e-7`** | **1.65e-08** ← best | 1.62e-09 |
| `1e-8` | 3.95e-08 | 7.15e-09 |
| `1e-10` | 4.23e-06 | 4.47e-07 |
| `1e-12` | 3.30e-04 | 1.17e-04 |

Read the columns downward and you can see both effects. The forward column falls by exactly a
factor of 10 per row (error `∝ h`); the central column falls by a factor of 100 per row
(error `∝ h^2`) — until round-off takes over and both start climbing again.

**Practical rule: use central differences with `h = 1e-6`.** It is within a factor of a few of
optimal and it is what the public tests use. If you see agreement to about `1e-6` or better,
your analytic derivative is right.

## A.4 Doing it for a vector argument

Your `theta` is a vector, so the gradient has one component per coordinate. Perturb one
coordinate at a time:

```python
def numerical_gradient(f, theta, h=1e-6):
    """Central-difference approximation to the gradient of f at theta."""
    grad = np.zeros_like(theta, dtype=float)
    for j in range(theta.size):
        step = np.zeros_like(theta, dtype=float)
        step[j] = h
        grad[j] = (f(theta + step) - f(theta - step)) / (2 * h)
    return grad
```

Then, for your objective at some fixed data and `lam`:

```python
def f(t):
    return logistic_objective(X, y, t, lam)

analytic = logistic_gradient(X, y, theta, lam)
numeric  = numerical_gradient(f, theta)
print(np.max(np.abs(analytic - numeric)))      # want ~1e-8 or smaller
```

This costs `2p` objective evaluations, which is far too slow to use inside an optimiser — that
is exactly why you derive gradients analytically. It is used **once**, as a test.

## A.5 Doing it for the Hessian

The Hessian is the derivative of the gradient, so difference the *gradient*, not the objective.
Differencing the objective twice would square the round-off problem and give you about four
correct digits on a good day.

Column `j` of the Hessian is the derivative of the whole gradient vector with respect to
`theta[j]`:

```python
H_numeric = np.zeros((p, p))
for j in range(p):
    step = np.zeros(p)
    step[j] = h
    H_numeric[:, j] = (logistic_gradient(X, y, theta + step, lam)
                       - logistic_gradient(X, y, theta - step, lam)) / (2 * h)
```

On the Lab 2 objective at `h = 1e-6` this agrees with the analytic Hessian to **7.7e-11**, and
the numerical Hessian is symmetric to **6.9e-11** even though nothing forced it to be. That
symmetry is a free extra check: if your numerical Hessian is visibly asymmetric, your
*gradient* is wrong.

## A.6 How to judge the result

Compare **relative**, not absolute, error when the gradient is large:

```python
denom = max(1.0, np.max(np.abs(analytic)))
rel = np.max(np.abs(analytic - numeric)) / denom
```

| Relative error | Verdict |
|---|---|
| `< 1e-7` | Correct. |
| `1e-7` to `1e-4` | Suspicious. Usually a real bug; occasionally just a badly scaled problem. Retry at `h = 1e-5`. |
| `> 1e-4` | Wrong. Find it. |
| Exactly 0 in some components | Suspicious — check you are not perturbing a coordinate the objective ignores. |

## A.7 Five ways this goes wrong

1. **Testing at `theta = 0`.** Many bugs vanish there because terms multiply out to zero. Always
   test at a random `theta`, and test at more than one.
2. **Testing with `lam = 0` only.** That skips the entire penalty term. Test with `lam = 0`
   *and* `lam = 0.3`. A missing factor of `0.5` in the penalty shows up nowhere else.
3. **Integer arrays.** If `theta` is an integer array, `theta + step` silently truncates. Force
   `dtype=float`.
4. **Reusing a mutated array.** If your objective modifies its input in place, the second
   evaluation sees different data. Do not modify arguments.
5. **Forgetting this is a test, not a method.** Never leave a numerical gradient inside
   `fit_logistic_newton`. It will be roughly `2p` times slower and the grader's timing budget
   will notice.

Checks 20, 21 and 31 in `test_lab2_public.py` do exactly this for you. Get them green before
you write a single line of `fit_logistic_newton` — debugging Newton's method with a wrong
gradient underneath it is genuinely miserable.

---

# Part B — Why separation breaks maximum likelihood

This is the spine of SQ3. It is short.

The logistic negative log-likelihood, in terms of the signed margins
`m_i = s_i * (x_i^T theta)` with `s_i = 2 y_i - 1`, is

```
J(theta)  =  (1/n) * sum_i  log(1 + exp(-m_i))
```

Every term is positive, decreasing in `m_i`, and tends to 0 as `m_i -> +inf`.

Now suppose the data is separable: there exists some `theta*` with **every** `m_i > 0`. Consider
scaling it up, `theta = c * theta*` for `c > 0`. Every margin scales to `c * m_i`, so every term
`log(1 + exp(-c m_i))` decreases towards 0 as `c` grows.

So `J(c theta*) -> 0` as `c -> infinity`, and `0` is the infimum. **But it is never attained at
any finite `theta`.** The likelihood has no maximum. The optimiser does not fail loudly; it just
keeps walking, with `||theta||` growing without bound and the objective creeping towards zero.

Add the penalty and the picture changes completely:

```
J_lam(theta)  =  (1/n) * sum_i  log(1 + exp(-m_i))  +  (lam/2) * ||theta||^2
```

Now growing `||theta||` costs something quadratic while gaining something that decays
exponentially. The quadratic wins eventually, so a finite minimiser exists for any `lam > 0`.

But how large is it? Roughly: the fit keeps scaling up until the marginal gain from the
likelihood term balances `lam ||theta||`. The likelihood gain decays like `exp(-c*gamma)` where
`gamma` is the margin, so the balance point sits at `||theta|| ~ log(1/lam) / gamma`.

**That is the prediction SQ3 asks you to test: under separation, `||theta||` should grow
*linearly* in `log(1/lam)`, with a slope set by the margin. Under overlap, no such pressure
exists and `||theta||` should converge to a finite limit as `lam -> 0`.**

Plot both on the same axes and the two behaviours are unmistakable. That plot is the core
evidence for your SQ3 report.

It also explains `is_separable`: set `lam` extremely small (`1e-12`) and fit. If the data is
separable, the fit will have pushed every margin positive, because that is the direction of
descent. If it is not separable, no `theta` makes every margin positive, so the minimum signed
margin stays negative no matter how small the penalty. Checking the sign of the smallest margin
is therefore a separability test — and it is the *same* phenomenon you just plotted, used as a
tool.

---

# Part C — Where the crossing point comes from (optional)

**You do not need this section.** SQ3 asks you to *measure* the fraction of random labellings
that are separable and find the crossing empirically. This is just an answer to "is there a
formula?", for the curious.

There is, and it is exact. Thomas Cover proved it in 1965.

Take `n` points in `R^p` in **general position** — meaning no `p` of them lie in a common
hyperplane through the origin, which happens automatically when you draw them from any
continuous distribution. Label them `+1`/`-1` in all `2^n` possible ways. The number of those
labellings a hyperplane through the origin can reproduce is

```
                p-1
C(n, p)  =  2 * sum  binom(n - 1, k)          for n > p,     and  2^n  for n <= p
                k=0
```

so a random labelling is separable with probability `C(n, p) / 2^n`.

Two things about it are worth knowing even if you never use the formula.

**First, `p` counts parameters, not features.** Fitting an intercept means appending a 1 to
every point, so `d` features with an intercept is a homogeneous problem in `p = d + 1`
dimensions. That is why the crossing in your plot sits where it does.

**Second, the crossing is exactly one half at `n = 2p`.** Setting `n = 2p` makes the sum run
over the first half of a symmetric row of binomial coefficients, which is exactly half their
total — so the probability is `1/2` precisely, for every `p`, with no approximation:

| `p` | `n = 2p` | probability |
|---|---|---|
| 2 | 4 | 8/16 = 0.5 |
| 3 | 6 | 32/64 = 0.5 |
| 5 | 10 | 512/1024 = 0.5 |
| 10 | 20 | 524288/1048576 = 0.5 |

`2p` is the classical **capacity** of a linear classifier. Below it, separation is likely and
tells you nothing about your data; above it, separation becomes rapidly unlikely and starts to
be real evidence.

For the `p = 3` case your sweep measures (`d = 2` plus an intercept), the exact values are
`1.0, 1.0, 1.0, 0.875, 0.6875, 0.5, 0.3438, 0.2266` for `n = 1 ... 8`. If you plot those on top
of your empirical curve, they should agree within your error bars. That is a nice optional
figure for your report, and it is the only use this section has.

The proof is a one-line recursion, `C(n+1, p) = C(n, p) + C(n, p-1)`, and it is on the second
page of Cover's paper, which is unusually readable.

## Further reading

- Cover (1965), "Geometrical and statistical properties of systems of linear inequalities with
  applications in pattern recognition", *IEEE Trans. Electronic Computers* EC-14(3), 326–334.
  The source of Part C. Optional, and only two pages.
- Albert & Anderson (1984), "On the existence of maximum likelihood estimates in logistic
  regression models", *Biometrika* 71(1), 1–10. The complete classification of when the MLE
  exists: separation, quasi-separation, and overlap.
- Nocedal & Wright, *Numerical Optimization*, §8.1 — finite-difference derivative approximation
  and the step-size trade-off, done properly.
- Boyd & Vandenberghe, *Convex Optimization*, §9.5 — Newton's method, for SQ4.
