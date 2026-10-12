# DS5006 Lab 2 — Linear classification and logistic regression

**Released:** Tuesday 25 August 2026 **Due:** Tuesday 8 September 2026, 23:59 IST
**Weight:** 100 marks (4% of the course) **Work:** individual

---

## 1. Read this first, then relax

This lab looks long. It is not as long as it looks.

You are being asked six questions. Each one has a short answer that you could write on an
index card. The work is in *earning* the right to write that answer: implementing enough
machinery to run an honest experiment, then reporting what actually happened.

Three things that should lower your blood pressure:

- **Nothing here requires you to invent an algorithm.** Every function you write is described
  in its docstring, with shapes, conventions and edge cases spelled out. If a docstring is
  ambiguous, that is a bug — post on Moodle and I will fix it for everyone.
- **The questions have surprising answers.** That is deliberate. You are not expected to guess
  correctly. Guessing wrong and then working out *why* is worth full marks.
- **You have already written most of the hard numerical code.** Lab 1 gave you QR, SVD and
  gradient descent. This lab reuses all three. The genuinely new machinery is Newton's method,
  which is about fifteen lines.

---

## 2. What this lab is about

The last few lectures covered three ways to draw a linear decision boundary: regression on indicator
responses, linear discriminant analysis, and logistic regression. The lectures presented them
as three methods. This lab asks whether that is the right way to think about them.

The six questions build on each other:

| | Question | Marks | Difficulty |
|---|---|---|---|
| **S0** | (building blocks — no question, just tools) | 6 | — |
| **SQ1** | Can you classify by running a regression? | 8 | gentle |
| **SQ2** | Does the fitting criterion decide the boundary? | 8 | gentle |
| **SQ3** | When does the logistic fit fail to exist? | 11 | moderate |
| **SQ4** | Is a second-order method worth what it costs? | 8 | moderate |
| **SQ5** | Which model should you reach for — and does the answer change? | 5 | moderate |
| **SQ6** | Is the model you fitted the only one that fits? | 8 | hard |

That is 54 marks of code. Add **10 marks** for code quality and **36 marks** for the report
(6 per question) and you have 100.

**You can score 40 without touching SQ6.** You can score 84 by doing everything except SQ6.
SQ6 is the one that will still be interesting to you in five years, so do attempt it — but do
it *last*, and do not let it eat the report.

---

## 3. Setup

Create a folder for the lab and put these files in it:

```
ds5006_lab2.py          <- you edit this one. Only this one.
ds5006_lab2_data.py     <- do not edit; the grader uses its own copy
test_lab2_public.py     <- do not edit; run it constantly
pylintrc                <- do not edit; the code-quality config
SQ3_Background.md       <- background reading, see below
report_template.md      <- skeleton for your report
```

**`SQ3_Background.md` covers what SQ3 needs that lectures have not covered.** **Part A** is how
to check a hand-derived derivative numerically — read this **before you start SQ3**; it will
save you hours here and again in SQ4 and SQ6. **Part B** explains why separation breaks maximum
likelihood, which is the spine of the whole question. **Part C** is half a page of optional
background on the theorem behind the SQ3 sweep; you are not required to read or use it.

Use the same virtual environment as Lab 1. If you need to rebuild it, the Lab 0 environment
guide on Moodle still applies. You need `numpy` and `pylint`, nothing else.

```
python -m pip install numpy pylint
```

Check that everything is wired up:

```
python test_lab2_public.py
```

You should see 32 lines of `[    ]`, meaning "not implemented yet". That is the correct
starting state.

**Every function takes your roll number as a string.** Your data is yours. Two students who
report identical numbers did not generate them independently, and that is straightforward for
me to detect.

```python
import ds5006_lab2_data as data
X, y = data.dataset_masking("112201042")     # your roll number here
```

---

## 4. Conventions — read these once, carefully

Correct mathematics with the wrong convention produces wrong numbers, and the grader compares
numbers. There are only six conventions and they are stated in every relevant docstring; this
is your one-page summary.

| | Convention |
|---|---|
| **Design matrix** | Shape `(n, p)`. **Never** an implicit column of ones — you add it with `add_intercept`, which puts the ones column **last**. |
| **Losses** | Always a **mean** over the `n` observations. Never a sum. |
| **Penalty** | Always `0.5 * lam * ||theta||^2`. Note the one-half, and note `lam` is the coefficient *inside* that expression. |
| **The intercept is penalised** | The penalty applies to the whole parameter vector, intercept included. Most libraries do not do this. We do. |
| **Pooled covariance** | Denominator `n - K`, where `K` is the number of classes. Not `n`. |
| **Ties** | Every `argmax` breaks ties towards the **smallest** class index. |

Two more that catch people:

- `quadratic_expansion` returns columns in the pinned order `a, b, a², ab, b²` and adds **no**
  intercept.
- `logistic_hessian` takes **no labels argument**. If you want one, differentiate the gradient
  again and watch the labels drop out.

---

## 5. The six questions

For each question: **write down your prediction before you run anything.** Then run it. Then
explain the gap. The report marks are structured around exactly that sequence.

### S0 — Building blocks (6 marks)

Five small functions: `add_intercept`, `one_hot`, `sigmoid`, `softmax_rows`,
`quadratic_expansion`.

The only subtle ones are `sigmoid` and `softmax_rows`, which must not overflow. The grader
evaluates `sigmoid` at ±1500 and `softmax_rows` at ±10⁴. Naive formulas raise warnings and
return `nan`. For the sigmoid, handle positive and negative arguments with different (but
algebraically identical) expressions. For the softmax, subtract each row's maximum first.

No report section for S0.

---

### SQ1 — Can you classify by running a regression? (8 marks)

You have three classes. You have a working least-squares solver from Lab 1. So: one-hot encode
the labels into a matrix `Y` with three columns, regress each column on the design, and predict
whichever column has the largest fitted value. This is a completely reasonable thing to try, and
it was standard practice for years.

**Do it on `dataset_masking(roll)` and report the per-class accuracy.**

*Predict first:* three well-separated classes, a linear method, an easy problem. What accuracy
do you expect? How many of the 300 points do you expect to be assigned to class 1?

Then look at what you actually get, and specifically at **how many points are assigned to
class 1 at all**. Something has gone wrong, and it has gone wrong in a very structured way.

Now fit the same method to `quadratic_expansion` of the same features and see whether the
problem survives.

**The question to answer in your report:** what exactly failed, why does the geometry of the
three class means cause it, and why does the quadratic expansion help? Be precise about
"why" — "the model is not flexible enough" is not an answer, because the *boundary* the method
needs was available to it all along. That last clause is a hint, and it is the bridge to SQ2.

---

### SQ2 — Does the fitting criterion decide the boundary? (8 marks)

SQ1 makes it tempting to conclude that least squares points in the wrong direction. SQ2 tests
that conclusion directly, in the two-class case where it can be checked exactly.

Fit LDA to `dataset_margin(roll, "overlapping")`. Separately, code the two classes as numbers
and run an ordinary least-squares regression, keeping only the slope coefficients
(`ls_direction_two_class`). Compare the two coefficient vectors as **directions** — compute the
cosine of the angle between them.

*Predict first:* how close do you expect these to be? Do you expect the answer to depend on the
two numbers you chose to code the classes with?

Try at least three codings, including an asymmetric one like `(7, -3)`.

Then compare the *decision rules* the two methods induce — not the directions, the actual
predicted labels. Do this on a balanced sample and on a deliberately unbalanced one (take a
subset of your data).

**The question to answer:** if the directions agree exactly, what is different about the two
methods, and what does that tell you about the SQ1 failure? State clearly what the least-squares
fit gets right and what it gets wrong.

---

### SQ3 — When does the logistic fit fail to exist? (11 marks)

This is the longest question. Budget three or four days and do not start it on day nine.

Logistic regression has no closed form, so you fit it by Newton's method. Implement
`logistic_objective`, `logistic_gradient`, `logistic_hessian` and `fit_logistic_newton`, and
check them against the finite differences in the public tests before going further.

> **New material.** "Finite differences" means approximating a derivative from values of the
> function alone, and using it to verify a derivative you worked out by hand. If that is not
> already a familiar technique, read **Part A of `SQ3_Background.md` now** — it is four pages,
> it is the single most useful debugging skill in this lab, and checks 20, 21 and 31 in the
> public tests are doing exactly this for you.

Now take `dataset_margin(roll, "separable")` and fit it with a penalty `lam` that you shrink
towards zero: 10⁻¹, 10⁻², … , 10⁻⁹. Record `||theta||` at each step. Do the same for
`dataset_margin(roll, "overlapping")`.

*Predict first:* what happens to the fitted coefficients as the penalty vanishes? Do you expect
the two datasets to behave the same way?

Plot `||theta||` against `log(1/lam)` for both. One of these curves does something the other
does not, and the difference is not subtle.

Then implement `min_signed_margin` and `is_separable`. The docstring of `is_separable` gives you
the method; your job in the report is to explain **why that method works** — why does a barely
penalised logistic fit tell you whether a separating hyperplane exists?

Finally, an experiment. You now have a working detector for "is this dataset separable?", so
you can ask how often separation happens *by accident*.

`random_label_trial(roll, n, d, trial)` gives you `n` points in `d` dimensions with labels
assigned by **fair coin flip** — the labels carry no information about the features at all.
Implement `separable_fraction`, which just runs `is_separable` over many such trials and returns
the proportion that came out separable. It is five lines.

Now sweep: for `d = 2`, `d = 3` and `d = 5`, vary `n` from very small to a few times `d`, using
at least 200 trials per point, and plot the fraction against `n`.

*Predict first:* at what sample size does separating random labels become a coin flip — a
fraction of one half? Write down a guess as a formula in `d` before you plot anything.

The three curves have the same shape, shifted. Find the `n` at which each crosses one half and
look for the pattern. It is exact and it is simpler than you expect. Remember that fitting an
intercept costs you a parameter.

> This crossing point is a known result — Cover's counting theorem (1965) gives the exact
> probability in closed form, and your empirical curve is estimating it. You are not asked to
> derive or implement that formula. If you would like to see it, the reference is in Appendix B
> and Part B of `SQ3_Background.md` states the result in half a page.

Put error bars on your points. With 200 trials the standard error near one half is about 3.5
percentage points, which is large enough to see.

**The questions to answer:** what does separation do to the maximum likelihood estimate, and
why? What does the ridge penalty do about it? And, from your sweep: if you fit a logistic model
and find it separates your training data perfectly, when should you be pleased and when should
you be suspicious? Answer that last one in terms of `n` and `d`.

---

### SQ4 — Is a second-order method worth what it costs? (8 marks)

Newton's method costs `O(p³)` per iteration to solve the linear system; gradient descent costs
`O(np)`. For large `p` that is a serious difference, and the usual advice is to use gradient
descent at scale.

Use `dataset_scaling(roll)`, whose three features live on wildly different scales.

First, derive `gd_step_bound`: the largest constant step size you can *prove* is safe, using the
fact that `s(1-s) ≤ 1/4`. Show the derivation in your report; it is three lines.

Then run four fits:

| | raw features | standardised features |
|---|---|---|
| **Newton**, `lam = 0` | iterations? | iterations? |
| **Gradient descent**, `lam = 1e-3`, step `= 0.9 × bound` | iterations? | iterations? |

*Predict first:* standardising features is standard advice. By how much do you expect it to help
each method? Write down two numbers before you run anything.

One row of that table will surprise you. Compare not just the iteration counts but the **fitted
probabilities** on the two designs — how close are they?

Then check whether the effect survives when you turn the penalty back on (`lam = 1e-3`,
Newton, raw versus standardised). It does not. Explain why not.

**The questions to answer:** what property does Newton's method have that gradient descent
lacks, and what exactly does the penalty do to that property? Given your iteration counts, when
would you actually choose gradient descent here?

---

### SQ5 — Which model should you reach for? (5 marks)

LDA assumes Gaussian classes with a common covariance and estimates that model. Logistic
regression assumes only that the log-odds are linear and fits the boundary directly. The
textbook line is that the generative model wins when its assumptions hold and loses when they
do not. This question asks you to measure it.

Implement `error_gap(roll, regime, n_train, n_reps)` exactly as its docstring specifies — a
**paired** comparison in which both methods see the same training draw and the same fixed test
set. The pairing is the whole reason this experiment is feasible at these sample sizes; say
something about that in your report.

Run it in both regimes across a range of `n_train`, say 25 to 800, with at least 100
repetitions each. Plot the gap against `n_train` with error bars.

*Predict first:* in which regime does each method win, and — the part people get wrong — what
happens to the gap as `n_train` grows in each regime?

**The question to answer:** the two regimes produce gaps with different *signs* and, more
importantly, different *behaviour as n grows*. One gap shrinks towards zero; the other does not.
Explain what distinguishes them, and say which of the two is the more serious problem for a
practitioner. Note also that at the smallest sample sizes the two regimes look alike — what does
that tell you about how much you can learn from a single small experiment?

---

### SQ6 — Is the model you fitted the only one that fits? (8 marks)

Multinomial logistic regression handles `K` classes directly, with a parameter matrix `Theta` of
shape `(p, K)`. Implement `multinomial_objective`, `multinomial_gradient` and
`fit_multinomial_gd`.

First, the easy part: run it on `dataset_masking(roll)` and confirm it does not suffer the SQ1
problem.

Now the real question. Fit `dataset_multiclass(roll)` **with no penalty**, twice: once from
`theta_init = None` (all zeros), and once from a starting matrix in which every column is the
same non-zero vector. Compare the two fitted `Theta` matrices, and separately compare the
**predicted probabilities**.

*Predict first:* the objective is convex. Should two runs of a convex minimisation agree?

Look carefully at the *structure* of the difference between the two `Theta` matrices — not just
its size. It has a very specific form.

Then repeat with a penalty (`lam = 0.1`) from the same two starting points, and examine
`Theta.sum(axis=1)` — the sum across classes, for each feature.

**The questions to answer:** in what sense is the unpenalised model not identifiable, and in
what sense does it not matter? What does the ridge penalty do about it, and is the thing it
picks meaningful or arbitrary? Finally: with no penalty and a zero start, gradient descent
happens to land on a particular representative anyway — which one, and is that the objective's
doing or the algorithm's?

---

## 6. Code quality (10 marks)

```
python -m pylint ds5006_lab2.py --rcfile=pylintrc
```

Your mark is the score out of 10, rounded to one decimal. **The untouched stub already scores
10.00**, so you begin at full marks and lose them only by writing code that is hard to read.

The configuration is deliberately lenient about mathematical style: single-letter names like
`X`, `y`, `H`, `n`, `p` are fine, and long functions with many local variables are fine. What it
objects to is genuinely bad practice — unused variables and imports, bare `except:`, missing
docstrings, deeply nested blocks, dead code.

Run it *before* your last day. Fixing eleven warnings at 23:00 on the 3rd is miserable and
entirely avoidable.

In VS Code: `Ctrl+Shift+P` → "Python: Select Linter" → pylint, and set
`"pylint.args": ["--rcfile=pylintrc"]` in your workspace settings so the editor agrees with the
grader.

---

## 7. Testing your work

```
python test_lab2_public.py
```

Not `python -m test_lab2_public.py` — that is a module path, not a filename, and it will fail
with a confusing error.

You will see one line per check:

- `[ ok ]` — passed
- `[    ]` — still a stub
- `[FAIL]` — implemented but wrong; the failing line is printed for you

These 32 checks test **contracts**: shapes, dictionary key names, conventions, edge cases. They
do **not** check whether your numbers are right on the real datasets. Passing all 32 is
necessary, not sufficient. The grader adds 38 hidden numerical tests, timing limits and static
analysis.

Common problems:

| Symptom | Cause |
|---|---|
| `RuntimeWarning: overflow in exp` | `sigmoid` or `softmax_rows` not stabilised |
| `n_iter` off by exactly one | You are stepping before checking the stopping rule; check first |
| LDA numbers slightly wrong | Pooled covariance divided by `n` instead of `n - K` |
| `logistic_objective` off by a constant | Penalty is `0.5 * lam * ...`, and it includes the intercept |
| Grader says "forbidden call" | You used `lstsq`, `pinv` or `inv`. Use `qr`, `solve` or `svd` |
| Everything in SQ4 times out | A Python loop over observations inside the iteration loop |

---

## 8. What to submit

**One file: `ds5006_lab2.py`.** Plus your report as a separate PDF.

Upload to Moodle:

1. **`ds5006_lab2.py`** — your implementation, with the header and assistance declaration
   filled in.
2. **`<rollnumber>_lab2_report.pdf`** — maximum 8 pages including figures.

Do **not** upload `ds5006_lab2_data.py`, `test_lab2_public.py`, `pylintrc`,
`SQ3_Background.md`, `report_template.md`, a virtual environment, a zip of your whole folder,
or a notebook. The grader can cope with all of those, but it will note the deviation and you
are making work for yourself.

If Moodle insists on a single file, a zip containing exactly the two files above is fine.

**Report structure — 36 marks, 6 per question, SQ1 through SQ6.** Use `report_template.md` as
a skeleton. Each section is marked:

| | |
|---|---|
| **2 marks** | Your prediction, stated **before** the result, with a reason. **Being wrong costs nothing.** |
| **2 marks** | Evidence: your numbers, from your seed, with axis labels and uncertainties where the experiment is stochastic. |
| **2 marks** | The answer to the question asked, plus one honest limitation. |

What loses marks: figures without axis labels; "as we can see from the plot"; Monte Carlo
numbers quoted without error bars; restating the question as the answer; and any paragraph that
would read identically if the data had come out the other way.

**Be concise.** Six well-chosen sentences and one honest number beat two pages of narration.
Several of these questions can be answered well in half a page.

---

## 9. Academic integrity and the assistance declaration

The course policy on the course page applies in full. This lab adds one requirement.

**Fill in the assistance declaration in the module docstring of `ds5006_lab2.py`.** List every
person, website, textbook and AI tool you used (along with the link to the conversation), and what you used each one for:

```

```

This is not a trap. Specific, honest declarations cost you nothing — not one mark. Using tools
to *understand* something is how professionals work, and declaring it is normal scientific
practice, not a confession.

What the declaration does not cover: **submitting work you cannot explain.** Lab Exam 1 is comping up soon, it is proctored and closed-book, and it covers this material directly. If you cannot
re-derive the SQ2 equivalence or the SQ4 invariance argument on paper, this lab has
not done its job. Use that as your standard for what "I understand this" means.

---

## Appendix A — Function checklist

Tick these off as you go. 24 functions, but eleven of them are five lines or fewer.

| Function | Section | Rough size |
|---|---|---|
| `add_intercept` | S0 | 2 lines |
| `one_hot` | S0 | 3 lines |
| `sigmoid` | S0 | 6 lines |
| `softmax_rows` | S0 | 4 lines |
| `quadratic_expansion` | S0 | 3 lines |
| `fit_indicator_regression` | SQ1 | 3 lines |
| `indicator_predict` | SQ1 | 1 line |
| `accuracy` | SQ1 | 1 line |
| `lda_fit` | SQ2 | 12 lines |
| `lda_scores` | SQ2 | 5 lines |
| `lda_predict` | SQ2 | 1 line |
| `ls_direction_two_class` | SQ2 | 3 lines |
| `logistic_objective` | SQ3 | 4 lines |
| `logistic_gradient` | SQ3 | 3 lines |
| `logistic_hessian` | SQ3 | 4 lines |
| `fit_logistic_newton` | SQ3 | 15 lines |
| `min_signed_margin` | SQ3 | 2 lines |
| `is_separable` | SQ3 | 7 lines |
| `separable_fraction` | SQ3 | 5 lines |
| `gd_step_bound` | SQ4 | 3 lines |
| `fit_logistic_gd` | SQ4 | 14 lines |
| `error_gap` | SQ5 | 20 lines |
| `multinomial_objective` | SQ6 | 5 lines |
| `multinomial_gradient` | SQ6 | 4 lines |
| `fit_multinomial_gd` | SQ6 | 15 lines |

Roughly 150 lines of actual implementation. It is a smaller lab than it looks.

---