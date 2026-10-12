# DS5006 Lab 1 — Report

**Name:** &nbsp;&nbsp;&nbsp;&nbsp; **Roll number:** &nbsp;&nbsp;&nbsp;&nbsp; **Date:**

> Maximum 8 pages including figures. Export to PDF. Delete these instruction blocks before
> submitting.
>
> Each section is worth 6 marks: **2** for a prediction made before the result, **2** for
> evidence (your numbers, your seed, uncertainties where applicable), **2** for a direct answer
> plus a stated limitation.
>
> Keep it short. Six good sentences and one honest number beat two pages of narration.

---

## SQ1 — Is the optimisation algorithm irrelevant to the answer?

**Prediction.** *(Before running anything: does GD reach the closed-form solution? Where do you
expect the stability boundary to be, and why?)*

**Derivation of $\eta_{\max}$.**

**Evidence.**

| Quantity | Value (my seed) |
|---|---|
| $\eta_{\max}$ | |
| Iterations to $\lVert w_t-w^\star\rVert<10^{-6}$ at $0.9\eta_{\max}$ | |
| $\kappa$ of the standardised augmented Hessian | |
| $\rho$ measured at $0.9\eta_{\max}$ / predicted | / |
| $\rho$ measured at $\eta^{\rm opt}$ / $(\kappa-1)/(\kappa+1)$ | / |
| Window used for the $\rho$ fit | |

*[Figure 1: convergence and divergence either side of the threshold.]*

**Answer.**

**Limitation.**

---

## SQ2 — Does rescaling a feature change the model, or only the arithmetic?

**Prediction.**

**Evidence.**

| Quantity under $X \mapsto XA$ | Invariant? | Relative difference |
|---|---|---|
| OLS coefficients | | |
| OLS predictions | | |
| OLS training MSE | | |
| GD iterations to fixed accuracy | | |
| Ridge predictions ($\lambda>0$, unstandardised) | | |
| Ridge predictions (standardised first) | | |

Algebraic relation between $\hat w_{XA}$ and $\hat w_X$:

GD on raw vs standardised data, 20,000 iterations, relative error: &nbsp; raw = &nbsp;&nbsp;
standardised =

**Answer.** *(Including: is standardisation a fix for a flaw in ridge, or a modelling assumption?
Give a dataset where you would not standardise.)*

**Limitation.**

---

## SQ3 — Why is the textbook formula not the library's formula?

**Prediction.** *(What slopes do you expect on the log-log plot, for which methods, and why?)*

**Evidence.**

| Method | Fitted slope | Expected slope |
|---|---|---|
| Normal equations | | |
| QR | | |
| SVD | | |

Degree at which the normal-equation solution has no correct significant digits:
&nbsp;&nbsp; $\kappa$ there: &nbsp;&nbsp; $\kappa^2\varepsilon$ there:

*[Figure 2: $\log_{10}$ relative error vs $\log_{10}\kappa$, three methods.]*

**Answer.** *(Including the engineering judgement: when is forming $X^\top X$ acceptable?)*

**Limitation.**

---

## SQ4 — Where does test error come from?

**Prediction.** *(Which degree do you expect to minimise test error? Which term dominates the rise
afterwards?)*

**Evidence.**

| Degree | bias² | variance | $\sigma^2$ | sum | empirical MSE | gap | MC s.e. |
|---|---|---|---|---|---|---|---|
| *(underfit)* | | | | | | | |
| *(best)* | | | | | | | |
| *(overfit)* | | | | | | | |

Degree minimising expected MSE: &nbsp;&nbsp; Minimum value:

Is the gap within Monte Carlo error?

*[Figure 3: bias², variance, noise, sum and empirical MSE vs degree, log vertical axis.]*

**The distinguishing check.** *(How did you separate genuine statistical variance from numerical
error at high degree? What did you find?)*

**Answer.**

**Limitation.**

---

## SQ5 — What does a regularisation penalty cost you?

**Prediction.**

**Evidence.**

$\mathrm{df}(\lambda\to 0) =$ &nbsp;&nbsp; $\mathrm{df}(\lambda\to\infty) =$ &nbsp;&nbsp;
$\lambda$ at half-maximum df $=$

*[Figure 4: coefficient paths vs $\log_{10}\lambda$ and vs $\mathrm{df}(\lambda)$.]*

Training MSE: OLS $=$ &nbsp;&nbsp; ridge at half-df $\lambda$ $=$

**Answer.** *(Why is $\mathrm{df}\to 1$ and not $0$? What quantity would you need to justify
preferring ridge, and why can you not compute it with SQ5's tools alone?)*

**Limitation.**

---

## SQ6 — Can you get *n* fits for the price of one? *(advanced)*

**Prediction.**

**Derivation.** *(The hat matrix for the unpenalised-intercept ridge fit, then the leave-one-out
identity. Show the Sherman–Morrison step. State where you use that the row leaves $\tilde X$ as
well as $y$.)*

**Evidence.**

| $\lambda$ | fast | naive | relative difference |
|---|---|---|---|
| $10^{-3}$ | | | |
| $10^{-1}$ | | | |
| $1$ | | | |
| $10$ | | | |
| $10^3$ | | | |

| $n$ | naive (s) | fast (s) |
|---|---|---|
| 50 | | |
| 100 | | |
| 200 | | |
| 400 | | |
| 800 | | |

Fitted timing slopes: naive $=$ &nbsp;&nbsp; fast $=$ &nbsp;&nbsp; speed-up at $n=800$ $=$

Why `ridge_path_svd` needs only one decomposition:

*[Figure 5: LOOCV RMSE vs $\mathrm{df}(\lambda)$, with $\lambda^\star$ marked.]*

**Closing answer.** $\lambda^\star =$ &nbsp;&nbsp; $\mathrm{df}(\lambda^\star) =$ &nbsp;&nbsp;
LOOCV RMSE at $\lambda^\star$ $=$ &nbsp;&nbsp; LOOCV RMSE for OLS $=$

*(Now answer the question SQ5 left open, with evidence.)*

**Limitation.** *(How much of the OLS-vs-ridge gap do you actually trust, and why?)*

---

## Declaration of assistance

*(Must match the declaration in `ds5006_lab1.py`.)*
