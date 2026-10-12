# DS5006 Machine Learning — Lab 1

## Linear Regression from First Principles

**Released:** Monday 10 August 2026 &nbsp;&nbsp;|&nbsp;&nbsp; **Due:** Friday 21 August 2026, 23:55 IST
**Weight:** 100 points &nbsp;&nbsp;|&nbsp;&nbsp; **Take-home, individual** &nbsp;&nbsp;|&nbsp;&nbsp; **Language:** Python 3.10+, **NumPy only**

---

## 1. What this lab is actually about

You already know that linear regression minimises squared error and that the answer is
$\hat{w} = (X^\top X)^{-1} X^\top y$. That formula is four lines of code and it is not what
this lab is testing.

This lab is about the gap between a formula on a slide and a method that works. Every question
below exists because there is something that *goes wrong* when you take the formula literally,
and the interesting part is understanding **why** it goes wrong, **when** it goes wrong, and
**how much** it matters.

So the deliverable is not "code that runs." It is **evidence for an answer to a question**.

### The habit we want you to build

Throughout this lab you will see the same six-part structure. Learn it; you will use it for the
rest of the course and for your project.

| Step | What it means | Common failure |
|---|---|---|
| **Question** | A question with a *possible answer you would find surprising*. | "Implement ridge regression" — not a question. |
| **Prediction** | What you expect, **written down before you run anything**, with a reason. | Writing the prediction after seeing the plot. |
| **Experiment** | The specific thing you will measure, and the control that isolates it. | Changing three things at once. |
| **Evidence** | The numbers. Yours, from your seed. | A plot with no axis labels and no numbers in the text. |
| **Answer** | A direct answer to the question you asked. | Restating the plot in words. |
| **Limitation** | Where your evidence stops being valid. | "Further research is needed." |

> **The prediction step is graded, and it is graded whether or not you predicted correctly.**
> A confidently-argued wrong prediction followed by an honest "I was wrong, and here is why"
> earns full marks. A prediction that mysteriously always matches the result earns suspicion.

---

## 2. Your data is yours alone

Every dataset is generated from your roll number. Your condition numbers, your optimal penalty,
your bias-variance crossover point — all different from everyone else's.

```python
import ds5006_lab1_data as data

ROLL = "112201042"          # <-- YOUR roll number, as a string
X, y = data.dataset_scaling(ROLL)
```

Two consequences:

1. **Every number in your report must come from your own seed.** If your report contains a number
   that belongs to another student's seed, that is not a coincidence and it will be treated as
   an integrity matter.
2. **Do not modify `ds5006_lab1_data.py`.** The grader overwrites it with a clean copy. Anything
   you change there is discarded, and code relying on your change will fail.

---

## 3. Files

### Given to you

| File | What it is |
|---|---|
| `ds5006_lab1.py` | **Your working file.** Stubs with full specifications in the docstrings. |
| `ds5006_lab1_data.py` | Data generation. Read it, do not edit it. |
| `test_lab1_public.py` | Self-checks. A subset of what the grader runs. |
| `pylintrc` | The lint configuration the grader uses. Keep it beside your code. |
| `report_template.md` | Skeleton for your report. |

### What you upload

A single ZIP named with your roll number, e.g. `112201042.zip`, containing exactly:

```
112201042.zip
├── ds5006_lab1.py          # your implementations. Name it EXACTLY this.
├── lab1_report.pdf         # max 8 pages, PDF only
└── figures/                # optional; source images if your PDF references them
```

**Do not** include `ds5006_lab1_data.py`, virtual environments, `__pycache__`, datasets, or
`.ipynb` files. Upload to Moodle. Late policy is on the course page.

> Renaming `ds5006_lab1.py`, submitting a notebook instead of a `.py`, or nesting the file inside
> extra folders all still get graded — the grader digs for it — but each costs marks under
> Part 0. Just follow the layout.

---

## 4. Rules

**Permitted:** `numpy`, the Python standard library, and `matplotlib` **for figures in your
report only** (not inside `ds5006_lab1.py`).

**Forbidden anywhere in `ds5006_lab1.py`:** `scipy`, `scikit-learn`, `pandas`, `statsmodels`,
`torch`, `jax`, `tensorflow`, `cvxpy` — and also `np.linalg.lstsq`, `np.linalg.pinv`,
`np.linalg.cond`, `np.polyfit`. Those four are exactly the things you are being asked to build.

The grader parses your file. **A forbidden import or call voids all 54 code marks**, not just
the marks for that function.

**Permitted and encouraged:** `np.linalg.solve`, `qr`, `svd`, `eigh`, `norm`, `einsum`, and all
of NumPy's array machinery. This lab is not about avoiding libraries; it is about knowing which
routine to reach for and why.

---

## 5. The six scientific questions

Points: **54 code + 36 report + 10 code quality = 100.**

Tiers are guidance, not a schedule. SQ1–SQ2 are approachable with a solid first course in Python.
SQ3–SQ5 need you to have understood the lectures. SQ6 is genuinely hard.

---

### SQ1 — Is the optimisation algorithm irrelevant to the answer? *(15 pts: 9 code + 6 report)*

**Functions:** `mse`, `predict`, `fit_ols_normal`, `fit_gd`, `max_stable_lr`
**Data:** `data.dataset_scaling(ROLL)`

#### The question

Least squares has a closed-form solution. It also has an iterative one. Textbooks present these
as two routes to the same place. That claim contains two separate assertions that are worth
separating:

> **(a)** Does gradient descent converge to the same parameters the normal equations give?
> **(b)** Is there any choice of learning rate for which it does not converge at all — and can
> you predict, from $X$ alone and without running anything, exactly where that boundary is?

Part (b) is the real question. A boundary you can *predict* is a much stronger claim than a
boundary you *discover by trial and error*, and the difference between those two ways of
choosing a learning rate is the difference between engineering and guessing.

#### What to do

1. Standardise the design (`standardize`), because the raw data in this dataset is not usable
   for gradient descent — you will find out why in SQ2.
2. Fit with `fit_ols_normal`.
3. **Derive** the stability threshold before writing `max_stable_lr`. Write $J(w,b)$ from the
   `fit_gd` docstring in terms of the augmented design $X_a=[\mathbf{1}, X]$ and augmented
   parameter $\beta=(b,w)$. Compute $\nabla^2 J$. Apply the standard result that gradient descent
   on a convex quadratic converges iff $0 < \eta < 2/\lambda_{\max}(\nabla^2 J)$. Your derivation
   goes in the report.
4. Run `fit_gd` at $\eta \in \{0.1, 0.5, 0.9, 0.99, 1.01, 1.5\}\times\eta_{\max}$ and record the
   loss trajectory.
5. Plot $\log_{10}\lVert w_t - w^\star\rVert$ against $t$ for the stable rates.

#### What to report

- Your derivation of $\eta_{\max}$, and its numerical value **for your data**.
- A single figure showing convergence and divergence either side of the threshold.
- The number of iterations to reach $\lVert w_t-w^\star\rVert < 10^{-6}$ at $\eta = 0.9\eta_{\max}$.
- **The question to actually answer:** the error decays geometrically,
  $\lVert \beta_t - \beta^\star\rVert \approx C\rho^t$. Theory says the rate for step size $\eta$
  on a quadratic with Hessian eigenvalues in $[\lambda_{\min}, \lambda_{\max}]$ is
  $\rho(\eta) = \max\big(|1-\eta\lambda_{\min}|,\ |1-\eta\lambda_{\max}|\big)$, minimised at
  $\eta^{\rm opt} = 2/(\lambda_{\min}+\lambda_{\max})$, where it becomes exactly
  $(\kappa-1)/(\kappa+1)$ with $\kappa = \lambda_{\max}/\lambda_{\min}$.

  Estimate $\rho$ empirically at **two** step sizes — $\eta = 0.9\,\eta_{\max}$ and
  $\eta = \eta^{\rm opt}$ — and compare each to its predicted value. Report both to four decimals.

  *Practical warning:* fit $\rho$ over a window where the error is still well above machine
  precision (say between $10^{-1}$ and $10^{-10}$). Once the iterate has converged you are
  measuring the ratio of two rounding errors, and you will get $\rho \approx 1$ and conclude
  something false. Say in your report which window you used.

  Then state what the relation $\rho^{\rm opt} = (\kappa-1)/(\kappa+1)$ implies for a badly
  conditioned problem — how does the iteration count needed for a fixed accuracy scale with
  $\kappa$? You should be able to answer that *before* running SQ2, and SQ2 will then test your
  answer.

---

### SQ2 — Does rescaling a feature change the model, or only the arithmetic? *(15 pts: 9 code + 6 report)*

**Functions:** `standardize`, plus everything from SQ1 and `fit_ridge`
**Data:** `data.dataset_scaling(ROLL)`

#### The question

"Always standardise your features" is advice you have heard. It is repeated so often that it is
rarely examined. Examine it.

Replace $X$ by $XA$ for an invertible diagonal $A$ — measuring a length in millimetres instead of
metres, say. This changes nothing about the world. So:

> Which of the following are unchanged by $X \mapsto XA$: the fitted coefficients, the
> predictions, the training MSE, the number of gradient descent iterations needed, and the ridge
> solution? For each one that *is* changed, is that a defect in the method or a legitimate
> statement that the method depends on units?

#### What to do

1. Build $A = \mathrm{diag}(a_1,\dots,a_5)$ with entries spanning several orders of magnitude.
2. Fit OLS on $X$ and on $XA$. Compare predictions to machine precision, and compare the
   coefficient vectors. Work out the exact algebraic relationship between $\hat w_{XA}$ and
   $\hat w_X$, and confirm it numerically.
3. Now the same for ridge at some fixed $\lambda > 0$. Then repeat with both designs standardised
   *first*, and compare again.
4. Run gradient descent on **raw** `dataset_scaling` data at $0.9\eta_{\max}$ for 20,000
   iterations. Compare $\lVert w_t - w^\star\rVert/\lVert w^\star\rVert$ to the same run on
   standardised data.

#### What to report

- A small table: for each of the five quantities above, **invariant / not invariant**, with the
  numerical evidence (a relative difference, not a yes/no).
- The algebraic relation between $\hat w_{XA}$ and $\hat w_X$ for OLS, derived not guessed.
- The gradient descent comparison. The raw-data number will look broken. **Explain the mechanism**
  — connect it to the $\rho \approx (\kappa-1)/(\kappa+1)$ result you established in SQ1. This is
  the payoff of SQ1 and you should be able to state it in two sentences.
- **The question to actually answer:** ridge is *not* invariant to rescaling. Standardising first
  makes the whole pipeline invariant again. Is standardisation therefore a fix for a flaw in
  ridge, or is it a modelling assumption you are quietly adopting? Argue one side. Give a concrete
  example of a dataset where you would *not* want to standardise before ridge, and say why.

---

### SQ3 — Why is the formula discussed in the class is not the formula in the library? *(15 pts: 9 code + 6 report)*

**Functions:** `condition_number`, `fit_ols_qr`, `fit_ols_svd`
**Data:** `data.dataset_conditioning(ROLL, degree)` for `degree = 2, 4, 6, ..., 16`

#### The question

The normal equations say to form $X^\top X$ and solve. No serious numerical library does this.
That is a strong claim about a formula every textbook prints, so it deserves evidence.

> Solving via $X^\top X$, via QR, and via SVD are algebraically identical. In exact arithmetic
> they give the same vector. In floating point they do not. **How large is the discrepancy, what
> does it scale with, and is the scaling law different for the three methods?**

The datasets here are **noise-free**: $y = Xw^\star + b^\star$ exactly. So the exact minimiser is
known, and every departure from $w^\star$ your solver produces is pure numerical error. There is
no statistics in this question at all — only arithmetic.

#### What to do

1. For each degree, compute $\kappa$ of the centred design with your `condition_number`.
2. Solve with all three routes and record the relative error
   $\lVert \hat w - w^\star\rVert / \lVert w^\star\rVert$.
3. Plot $\log_{10}(\text{relative error})$ against $\log_{10}\kappa$ for all three, on one axis.
4. Fit a straight line to each. **Report the slopes.**

#### What to report

- The plot, with the three fitted slopes stated numerically.
- **The question to actually answer:** the classical results are that the forward error for a
  backward-stable solver is $O(\kappa \varepsilon)$, while forming $X^\top X$ squares the
  condition number and gives $O(\kappa^2 \varepsilon)$, with $\varepsilon \approx 2.2\times
  10^{-16}$. On a log-log plot those are lines of slope 1 and slope 2. **Do your measured slopes
  match?** Report them to two decimals and say honestly whether they do.
- Identify the degree at which the normal-equation solution has *no correct significant digits*
  (relative error $\ge 1$). Give the condition number there. Show that $\kappa^2\varepsilon
  \approx 1$ at that point, and explain why that is the expected place for it to happen.
- **The judgement call:** given that QR costs roughly twice the flops of the normal equations,
  when — if ever — is forming $X^\top X$ the right engineering decision? Answer in terms of a
  condition number threshold and the precision you actually need, not in terms of "always use QR."

---

### SQ4 — Where does test error come from? *(15 pts: 9 code + 6 report)*

**Functions:** `polynomial_design`, `bias_variance_decompose`
**Data:** `data.truth_function(ROLL)`, `data.sample_training_set(ROLL, n, sigma, replicate)`

#### The question

Expected test error decomposes into squared bias, variance, and irreducible noise. This is stated
as an identity. Identities are checkable.

> Take a model family indexed by complexity. **Measure** all three terms by Monte Carlo, and check
> whether they add up to the test error you actually observe. Then: which term drives the increase
> in test error at high complexity, and is the classical picture — bias falling, variance rising,
> a U-shaped total — what your data actually shows?

You know the true $f$ analytically here, which is the whole point: **you can compute bias exactly
rather than estimating it.** This is a privilege you will never have on real data, and it is why
this experiment is worth doing once.

#### What to do

1. Fix $\sigma = 0.30$, $n = 40$, test grid $x_0 = $ 101 points on $[-0.95, 0.95]$.
2. For each degree $p = 1,\dots,12$: draw $M = 500$ independent training sets
   (`replicate = 0..499`), fit with `fit_ols_svd` on `polynomial_design(x, p)`, predict on the
   test grid, and stack the predictions into an $(M, 101)$ array.
3. Call `bias_variance_decompose`.
4. Separately, estimate the test MSE **empirically**: for each replicate, draw fresh noisy test
   responses at the grid points and average the squared error.
5. Plot bias², variance, $\sigma^2$, their sum, and the empirical MSE against $p$.

#### What to report

- The plot, with a log scale on the vertical axis (you will see why at $p \ge 9$).
- A table at three degrees — underfit, best, overfit — with all five quantities.
- **The identity check:** the sum and the empirical MSE will not agree exactly, because $M$ is
  finite. Report the gap, and report the Monte Carlo standard error of your variance estimate.
  **Is the gap within Monte Carlo error?** If you cannot answer that, you have not computed an
  uncertainty, and an estimate without an uncertainty is not a measurement.
- Your degree with minimum expected MSE, and the corresponding value.
- **The question to actually answer:** at large $p$ your variance will grow by several orders of
  magnitude. Two mechanisms could explain it: genuine statistical variance (the estimator really
  is unstable across training sets), or numerical error (the Vandermonde design is desperately
  ill-conditioned, as you established in SQ3). **Design a check that distinguishes them** and run
  it. Say which mechanism dominates for your data. This is the part most people skip, and it is
  the part that separates a measurement from a plot.

---

### SQ5 — What does a regularisation penalty actually cost you? *(14 pts: 8 code + 6 report)*

**Functions:** `fit_ridge`, `effective_dof`
**Data:** `data.dataset_wide(ROLL)` — 70 observations, 40 correlated features

#### The question

Ridge trades bias for variance. "Complexity" is usually measured by counting parameters, but ridge
never sets a parameter to zero: at every $\lambda > 0$ you still have all 40 coefficients. So
parameter counting cannot be the right notion of complexity here.

> $\mathrm{df}(\lambda) = \mathrm{tr}(H)$ is offered as the replacement. **Is it a sensible one?**
> Specifically: does it interpolate correctly between the two limits, is it a *smooth* dial rather
> than a discrete count, and does a model with $\mathrm{df}(\lambda) = 7.3$ behave like a model
> with 7 free parameters?

#### What to do

1. Take $\lambda$ on a log grid, $10^{-4}$ to $10^{5}$, 100 points.
2. Compute $\mathrm{df}(\lambda)$ with `effective_dof`. Verify the two limits.
3. Plot the coefficient paths: all 40 coefficients against $\log_{10}\lambda$.
4. Plot the same paths against $\mathrm{df}(\lambda)$ instead. Compare.
5. Report $\lVert\hat w(\lambda)\rVert_2$ against $\mathrm{df}(\lambda)$.

#### What to report

- Both path plots. State what changes when you re-parametrise the horizontal axis by df, and why
  that makes df the more useful axis for a reader.
- The exact limits of $\mathrm{df}$ as $\lambda\to 0$ and $\lambda\to\infty$, **with the reason
  the second limit is 1 and not 0.** (If your answer is 0, re-read the `fit_ridge` docstring.)
- $\lambda$ at which $\mathrm{df}(\lambda)$ equals half its maximum, for your data.
- **The question to actually answer:** you have 70 observations and 40 features. Fit OLS on this
  data and compare its training MSE to ridge at your half-df penalty. Training error will be lower
  for OLS — necessarily so, since OLS minimises it. **So what evidence could possibly justify
  preferring the ridge fit?** Name the specific quantity you would need to measure. You are not
  asked to measure it here. You are asked to notice that you cannot answer the question with the
  tools of SQ5, and to say precisely what is missing. SQ6 supplies it.

---

### SQ6 — Can you get *n* fits for the price of one? *(16 pts: 10 code + 6 report)* — **ADVANCED**

**Functions:** `hat_diagonal`, `loocv_rmse_naive`, `loocv_rmse_fast`, `ridge_path_svd`
**Data:** `data.dataset_wide(ROLL)`

> This question is deliberately hard. It is possible to score 84/100 without it. Attempt it only
> once SQ1–SQ5 are finished and correct.

#### The question

SQ5 ended by needing an honest estimate of out-of-sample error. Leave-one-out cross-validation is
the obvious candidate and it is almost never used, because for a grid of 100 penalties on $n=70$
points it means 7,000 model fits. So it gets replaced by 5-fold CV, which is cheaper and noisier.

But this trade is not actually forced:

> For ridge regression there is an **exact, closed-form** identity giving all $n$ leave-one-out
> residuals from a **single** fit. Not an approximation, not a large-$n$ limit — exact to machine
> precision. Find it, prove it, implement it, and verify it.

Two things make this hard, and both are places where a plausible-looking answer is wrong:

**(i) The intercept.** The identity involves the hat matrix $H$ with $\hat y = Hy$. The commonly
quoted form is $H = X(X^\top X + \lambda I)^{-1}X^\top$. That is **not** the hat matrix for the
model in `fit_ridge`, because your intercept is unpenalised. Derive the correct $H$. Equivalently:
write the fit as a penalised least squares problem on the augmented design $\tilde X = [\mathbf 1,
X]$ with penalty matrix $\lambda D$, $D = \mathrm{diag}(0,1,\dots,1)$, and read off $H$ from there.
An implementation that omits the intercept contribution will pass a smoke test and fail every
numerical comparison, which is exactly why the naive version is required.

**(ii) Removing a row changes the design, not just the response.** LOOCV drops $(x_i, y_i)$
together, so the Gram matrix changes too, and the mean $\bar x$ used for centring changes with it.
It is not obvious that a shortcut derived from a *fixed* $H$ survives that. Sherman–Morrison on
$\tilde X_{-i}^\top \tilde X_{-i} = \tilde X^\top \tilde X - \tilde x_i \tilde x_i^\top$ is the
route. **Whether the identity remains exact under refitting is precisely what your numerical check
must establish** — do not assert it, demonstrate it.

#### What to do

1. Implement `hat_diagonal` — the diagonal only, never the full $n\times n$ matrix.
2. Implement `loocv_rmse_naive` by brute force. This is your ground truth. Write it first.
3. Derive and implement `loocv_rmse_fast`. It must use **at most 4 calls** to NumPy's
   factorisation routines regardless of $n$. The grader counts them by instrumenting
   `numpy.linalg`, so a hidden loop will be detected.
4. Verify: for $\lambda \in \{10^{-3}, 10^{-1}, 1, 10, 10^3\}$ report
   $|{\rm fast} - {\rm naive}|/{\rm naive}$. It should be $\sim 10^{-15}$, not $\sim 10^{-3}$.
5. Time both across $n \in \{50, 100, 200, 400, 800\}$ (use your own synthetic designs for the
   larger sizes). Plot both timings on log-log axes and report the fitted slopes.
6. Implement `ridge_path_svd`: the entire penalty path from **one** SVD, no other factorisations.
7. Use `loocv_rmse_fast` over your 100-point $\lambda$ grid to select $\lambda^\star$. **Answer the
   question SQ5 left open.**

#### What to report

- **The derivation.** Both the correct $H$ *and* the leave-one-out identity. This is the largest
  single block of report marks in the lab. Show the Sherman–Morrison step. State clearly where you
  use that the row is removed from $\tilde X$ as well as from $y$.
- Your agreement table. If the relative difference is $10^{-6}$ rather than $10^{-15}$, you have a
  real bug, not a rounding issue — say so rather than calling it "close enough."
- The timing plot with fitted slopes, and the empirical speed-up at $n = 800$.
- Why `ridge_path_svd` needs only one decomposition: state what is $\lambda$-independent in the
  SVD form and what is not.
- **The closing answer:** your $\lambda^\star$, its $\mathrm{df}(\lambda^\star)$, its LOOCV RMSE,
  and the LOOCV RMSE of OLS ($\lambda \to 0$) on the same data. Now return to SQ5's unanswered
  question and answer it with evidence. State the limitation: LOOCV is nearly unbiased but has
  high variance as an estimate of test error, so quantify how much of your OLS-versus-ridge gap
  you would actually trust.

---

## 6. Code quality — Part 0 *(10 pts)*

Graded automatically. Free marks if you spend twenty minutes on them.

| Component | Points | How it is judged |
|---|---|---|
| Pylint score | 4 | $\ge 9.0 \to 4$; $\ge 8.0 \to 3$; $\ge 7.0 \to 2$; $\ge 5.0 \to 1$ |
| Docstrings | 3 | Proportion of implemented functions with a real docstring |
| Header + declaration | 3 | Name, roll number, and assistance declaration filled in |

### Installing pylint

You need it — these are 4 marks that cost you one command.

```bash
pip install pylint
```

If that fails because your Python is system-managed (common on Linux and on newer macOS):

```bash
pip install --user pylint                       # usual fix
python -m pip install pylint                    # if `pip` is not on your PATH
pip install --break-system-packages pylint      # last resort, Debian/Ubuntu
```

Check it worked, then run it:

```bash
pylint --version
pylint ds5006_lab1.py
```

Keep `pylintrc` in the same folder as `ds5006_lab1.py` and pylint picks it up automatically — no
flag needed. (If you prefer to be explicit: `pylint ds5006_lab1.py --rcfile=pylintrc`.) The file
has no leading dot on purpose; dot-files are hidden by default on Windows and awkward to save.

Use the supplied config, because the grader does. Without it the same file scores several points
lower against pylint's stock settings, which are stricter about naming than is reasonable for
linear algebra.

VS Code users: install the Microsoft Python extension, then `Ctrl+Shift+P` →
*Python: Select Linter* → Pylint. Warnings then appear inline as you type, which is far more
useful than running the command at the end.

### If you have never used a linter

A linter reads your code **without running it** and reports things that are legal Python but
probably wrong: a variable you assigned and never used, an import you no longer need, a name that
shadows a builtin, a function that returns a value on one path and nothing on another. It is not a
style police officer — roughly half of what it catches are latent bugs.

Worth twenty minutes:

- **[Python Code Quality: Best Practices and Tools](https://realpython.com/python-code-quality/)**
  (Real Python) — start here. Explains what linters do and why, with runnable examples. Read the
  sections on linters and on PEP 8; skip type checkers and security scanners for now.
- **[Pylint messages overview](https://pylint.readthedocs.io/en/stable/user_guide/messages/messages_overview.html)**
  — the reference. When you get `W0612` and don't know what it means, look it up here.
- **[PEP 8](https://peps.python.org/pep-0008/)** — the style guide pylint is largely enforcing.
  Skim it once; you will absorb most of it by fixing warnings.

A worked example of the difference. This is legal Python and it runs:

```python
def fit(X, y, lam):
    n, d = X.shape
    sum = 0.0
    Xc = X - X.mean(axis=0)
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ (y - y.mean()))
    return w
```

pylint reports three things: `sum` shadows a builtin, `sum` and `n` are assigned but never used.
The unused `n` is harmless. The unused `sum` means you started writing an accumulation and
abandoned it — which is exactly the kind of half-finished edit that later turns into a bug. That
is the value: the linter noticed something about your intent that the interpreter never could.

Fix warnings, do not silence them. A file peppered with `# pylint: disable=` scores 10/10 and will
be marked down by hand.

### Comments and docstrings

Every function you implement keeps a docstring. Google style, as in the stubs:

```python
def fit_ridge(x_mat, y_vec, lam):
    """Fit ridge regression with an unpenalised intercept.

    Args:
        x_mat: (n, d) design matrix.
        y_vec: (n,) responses.
        lam: non-negative penalty.

    Returns:
        Tuple (weights, intercept).
    """
```

For inline comments the rule is: **comment the why, never the what.**

```python
# BAD  -- restates the code
w = np.linalg.solve(gram, xc.T @ yc)   # solve the system

# GOOD -- explains a decision the reader cannot infer
# solve() rather than inv() @ b: forming the inverse explicitly costs an extra
# factor of kappa in accuracy, which SQ3 measures directly.
w = np.linalg.solve(gram, xc.T @ yc)
```

The bar: **a classmate who has not read the lab should understand your non-obvious choices.**
Silent magic constants, unexplained `1e-12` thresholds, and a bare `+ 1` in `effective_dof` are
exactly the places a comment earns its keep.

---

## 7. Running the self-checks

First tell the checks who you are, then run them:

```bash
export DS5006_ROLL=112201042        # Linux / macOS
set DS5006_ROLL=112201042           # Windows cmd
$env:DS5006_ROLL="112201042"        # Windows PowerShell

python test_lab1_public.py
```

**No installation is needed.** If you get `ModuleNotFoundError: No module named 'numpy'`, install
NumPy — that one you do need. Nothing else.

If you already use `pytest` and prefer its output, the same file works unchanged:

```bash
python -m pytest test_lab1_public.py -v
```

If `pytest` is not installed, do not chase it; just use the plain command above. (`pip install
pytest` works if you want it, `pip install --user pytest` if your Python is system-managed.)

Output looks like this:

```
DS5006 Lab 1 self-checks -- roll 112201042
--------------------------------------------------------------------
  [ ok ] test_mse_matches_definition
  [    ] test_ridge_shrinks_monotonically          <- not implemented yet
  [FAIL] test_effective_dof_limits                 <- implemented, but wrong
--------------------------------------------------------------------
```

A blank `[    ]` means you have not written that function yet. `[FAIL]` means you have, and it
does not do what the specification says — those are the ones to look at, and the runner prints the
assertion that failed.

### If the command itself errors

| Message | Cause | Fix |
|---|---|---|
| `__path__ attribute not found` / `Error while finding module specification` | You wrote `python -m test_lab1_public.py`. The `-m` flag takes a **module name**, never a filename. | Drop the `-m`: `python test_lab1_public.py` |
| `No module named 'pytest'` | pytest is not installed. You do not need it. | Use `python test_lab1_public.py` |
| `No module named 'ds5006_lab1'` | `ds5006_lab1.py` is not sitting in the same folder as `test_lab1_public.py`. | Keep all the lab files together in one folder |
| `No module named 'numpy'` | NumPy is genuinely missing. | `pip install numpy` |

The rule for `-m`: it is followed by a module name with no `.py`. In `python -m pytest
test_lab1_public.py` the module being run is **pytest**, and the filename is an argument handed to
it. In `python test_lab1_public.py` there is no `-m` at all.

These 20 checks are a **subset** of the grader. Passing all of them does not guarantee full marks:
the grader adds hidden numerical cases, the efficiency counters, and static analysis. Failing any
of them does guarantee lost marks.

> If `DS5006_ROLL` is unset you will be testing against a default seed, and the runner warns you.
> Your marks come from *your* roll number, so set it.

---

## 8. Report

**PDF, maximum 8 pages including figures. One section per SQ, in order.** Use
`report_template.md` as a skeleton.

What earns marks (36 total, 6 per question):

| | |
|---|---|
| **2 pts** | Prediction stated before the result, with a reason. Being wrong costs nothing. |
| **2 pts** | Evidence: your numbers, from your seed, with units and uncertainties where applicable. |
| **2 pts** | The answer to the question asked, plus a stated limitation. |

What loses marks: figures without axis labels; "as we can see from the plot"; numbers with no
uncertainty in a Monte Carlo experiment; restating the question as the answer; and any
paragraph that would read identically if the data had come out the other way.

**Be concise.** Six well-chosen sentences and one honest number beat two pages of narration.

---

## 9. Academic integrity and the assistance declaration

The course policy is on the course page and applies in full. This lab adds one requirement.

**You must fill in the assistance declaration in the module docstring of `ds5006_lab1.py`.** List
every person, website, textbook, and AI tool you used, and what you used each one for. For
example:

```
Assistance used: discussed the Sherman-Morrison step with <name>; used an LLM to
explain why np.linalg.qr returns the reduced factorisation by default; consulted
Hastie et al. section 3.4.1 for the effective-degrees-of-freedom definition.
```

This is not a trap. Specific, honest declarations cost you nothing — not a single mark. Using
tools to *understand* something is how professionals work, and the declaration is a normal part
of scientific practice, not a confession.

What the declaration does not cover: submitting work you cannot explain. Both lab exams are
proctored and closed-book, and Lab Exam 1 covers this material directly. If you cannot re-derive
the SQ6 identity on paper in week 8, this lab has not done its job. Use that as your standard
for what "I understand this" means.

---

## Appendix A — Conventions, all in one place

Get these wrong and correct mathematics produces wrong numbers. They are stated in the stub
docstrings too; this is a summary you can check against.

| | Convention |
|---|---|
| Design matrix | $(n,d)$, **no column of ones**. Intercept is always separate. |
| Standardisation | Population SD (`ddof=0`). SD $< 10^{-12} \Rightarrow$ scale exactly $1.0$. |
| OLS / ridge intercept | Solve on centred data; $b = \bar y - \bar x^\top w$. |
| Ridge objective | $\lVert y - Xw - b\mathbf 1\rVert^2 + \lambda\lVert w\rVert^2$. **Sum, not mean. Intercept never penalised.** |
| GD objective | $J = \frac1n\lVert y - Xw - b\mathbf 1\rVert^2$. **No factor of $\frac12$.** |
| GD initialisation | $w = 0$, $b = 0$; simultaneous update from one residual vector. |
| GD history | Length `n_iters + 1`; `history[0]` is $J$ at initialisation. |
| `mse` | Divides by $n$. |
| Variance in SQ4 | Population convention, `ddof=0`. |
| Hat matrix | For the **unpenalised-intercept** ridge fit. It has an intercept term. |
| $\mathrm{df}(\lambda)$ | $\mathrm{tr}(H)$. Ranges from $d+1$ down to $1$, not $d$ down to $0$. |
| `polynomial_design` | Columns $x^1,\dots,x^{\text{degree}}$. No constant column. |

## Appendix B — SVD, the minimum you need

SQ3, SQ4 and SQ6 use the SVD. If it has not appeared in lectures yet, this is enough to proceed;
a fuller treatment is in the bridge note on Moodle.

Any real $X$ of shape $(n,d)$ with $n \ge d$ factors as $X = U S V^\top$ where $U$ is $(n,d)$ with
orthonormal columns, $S = \mathrm{diag}(s_1 \ge \dots \ge s_d \ge 0)$, and $V$ is $(d,d)$
orthogonal. In NumPy:

```python
u, s, vt = np.linalg.svd(X, full_matrices=False)   # note: vt is V TRANSPOSED
```

Three facts do all the work in this lab:

1. **Condition number.** $\kappa(X) = s_1/s_d$. It measures how much a relative perturbation in
   the data can be amplified in the solution.
2. **OLS.** With $X$ full rank, $\hat w = V S^{-1} U^\top y$. When some $s_i$ are zero or
   negligible, dropping those directions gives the *minimum-norm* solution — that is all
   `fit_ols_svd` does.
3. **Ridge.** $\hat w(\lambda) = V\,\mathrm{diag}\!\left(\frac{s_i}{s_i^2+\lambda}\right)U^\top y$.
   Compare to OLS's $1/s_i$: ridge replaces it by $s_i/(s_i^2+\lambda)$, which is close to $1/s_i$
   when $s_i \gg \sqrt\lambda$ and close to $0$ when $s_i \ll \sqrt\lambda$. **Ridge is a
   direction-by-direction shrinkage, and it shrinks the poorly-determined directions hardest.**

Fact 3 is why `ridge_path_svd` needs only one decomposition: $U$, $S$, $V$ do not depend on
$\lambda$. Changing $\lambda$ changes only the diagonal shrinkage factors, which is $O(d)$ work.
It is also the cleanest way to see $\mathrm{df}(\lambda) = 1 + \sum_i s_i^2/(s_i^2+\lambda)$ —
each direction contributes between $0$ and $1$, and the $1$ is the unpenalised intercept.
