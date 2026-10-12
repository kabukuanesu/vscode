# DS5006 Lab 2 — Report

**Name: ANESU KABUKU**
**Roll number: 142602026**
**Date: 04/09/2026**

> Delete every instruction line (the indented ones) before you submit.
> Maximum 8 pages including figures. One section per question, in order.
> Each section is marked 2 / 2 / 2 on prediction, evidence, and answer-with-limitation.
> Being wrong in the prediction costs nothing, provided you diagnose why.
> Every figure needs axis labels. Every Monte Carlo number needs an uncertainty.

---

## SQ1 — Can you classify by running a regression?

**Prediction.**
> What accuracy did you expect from indicator regression on three well-separated classes,
> and how many of the 300 points did you expect in class 1? One or two sentences, with a reason.
Since the three classes are well separated and the decision boundaries between neighbouring classes are linear, I expect indicator regression to achieve high classification accuracy. I expect approximately one third of the 300 observations, around 100, to be assigned to class 1 because the classes are equally sized.

**Evidence.**
> Per-class accuracy and the class-1 count, from your seed. A figure showing the data with the
> fitted boundaries helps a lot here. Include the quadratic-expansion result.

**Answer and limitation.**
> What failed, and why the geometry of the class means causes it. Then one honest limitation —
> for example, does your explanation predict what would happen with four collinear classes?
I expected the well-separated classes to produce high accuracy and approximately 100 class-1 predictions. The experiment contradicted this expectation: only 2 observations were assigned to class 1. The failure is explained by the collinearity of the class means and the inability of a single linear indicator score for the middle class to peak between the two outer classes.

**Notes.**
The quadratic expansion is not needed because the true class boundaries are curved. It is useful because it gives the separate regression score functions enough structure to represent the middle class correctly.
The failure comes from fitting independent linear score functions to the one-hot responses. The middle class lies between the two outer class means, so its indicator response requires a score that is high in the middle and low on both sides. A single linear score cannot have that shape.
The quadratic expansion does not necessarily mean the true class boundaries are quadratic. Instead, quadratic features give the individual regression score functions enough flexibility to produce the right ranking between the three classes.

---

## SQ2 — Does the fitting criterion decide the boundary?

**Prediction.**
> How close did you expect the least-squares and LDA directions to be? Did you expect the
> target coding to matter?

**Evidence.**
> Cosines for at least three codings, quoted to enough digits to be convincing.
> Rule agreement on a balanced and an unbalanced sample.
I expect the least-squares and LDA coefficient directions to be very close, with cosine values near \(1\) for different class codings. I expect changing the numerical coding to alter the scale or sign of the least-squares coefficients but not their underlying direction. I expect the decision rules to agree closely on the balanced sample, but I expect differences to appear on the deliberately unbalanced sample because LDA incorporates class priors.

**Answer and limitation.**
> If the directions agree exactly, what differs between the methods, and how does that revise
> your SQ1 explanation? State the limitation — does the equivalence extend beyond two classes?
This equivalence was demonstrated for a two-class problem under the common-covariance LDA setup; it should not be assumed to extend directly to multiclass indicator regression, where the separate class scores can interact differently.

**Notes.**
Least squares can get the separating direction right in the two-class common-covariance setting. The SQ1 failure therefore comes from the multiclass arrangement and the way separate one-vs-all indicator scores are fitted, rather than simply from least squares choosing an incorrect direction.
Even though LDA explicitly contains class priors, ordinary least squares with the standard \(0/1\) coding and a \(0.5\) threshold can incorporate the same class-frequency effect through its fitted intercept.

---

## SQ3 — When does the logistic fit fail to exist?

**Prediction.**
> What did you expect to happen to the coefficients as the penalty vanished, on each dataset?
> And: at what sample size did you expect random labels to stop being separable? State your
> guess as a formula in `d`, written before you plotted anything.
Prediction 1: For the separable dataset, I expect \(\|\theta\|\) to increase approximately linearly with \(\log(1/\lambda)\) as the penalty decreases. For the overlapping dataset, I expect \(\|\theta\|\) to approach a finite value rather than diverge. Therefore I do not expect the two datasets to show the same behaviour.
Prediction 2: Because the logistic model includes an intercept, I expect the probability of separating random labels to fall through 0.5 at approximately \(n=2(d+1)\). Thus I expect the crossing near \(n=6\) for \(d=2\), \(n=8\) for \(d=3\), and \(n=12\) for \(d=5\).
Prediction 1 — penalty sweep

I expect:

Separable: \(\|\theta\|\) will increase as \(\lambda\to0\), approximately linearly with \(\log(1/\lambda)\).
Overlapping: \(\|\theta\|\) will approach a finite value.

This follows because separation allows the likelihood to keep improving as the coefficients grow, while the ridge penalty prevents the coefficients from becoming infinite.

**Evidence.**
> The `||theta||` versus `log(1/lam)` curves for both datasets, on one pair of axes.
> The separable-fraction sweep for `d = 2, 3, 5` with error bars, and the crossing point you
> read off each curve.

**Answer and limitation.**
> What separation does to the MLE and why; what the penalty does about it; why a barely
> penalised fit is a valid separability test; and when a perfect separation should please you
> versus worry you, in terms of `n` and `d`. Limitation: where does your separability test
> become unreliable?

---

## SQ4 — Is a second-order method worth what it costs?

**Prediction.**
> Two numbers, written before you ran anything: how much did you expect standardisation to
> help Newton, and how much to help gradient descent?

**Evidence.**
> The 2×2 table of iteration counts. The step bounds. The largest discrepancy between the
> fitted probabilities on raw and standardised designs. The penalised comparison.
> Include your three-line derivation of the step bound.

**Answer and limitation.**
> The property Newton has that gradient descent lacks; what the penalty does to it; and when
> you would still choose gradient descent. Limitation: at what problem size does your answer
> flip, and what would you need to measure to know?

---

## SQ5 — Which model should you reach for?

**Prediction.**
> Which method wins in each regime, and — the part that matters — what happens to the gap as
> the training size grows in each regime?

**Evidence.**
> The gap against `n_train` for both regimes, with error bars, on one plot.
> State your repetition count. Say one sentence about why the comparison is paired.

**Answer and limitation.**
> What distinguishes the two regimes, which is the more serious problem in practice, and what
> the small-sample behaviour tells you about single small experiments. Limitation: name one
> way the contaminated regime is unrealistic.

---

## SQ6 — Is the model you fitted the only one that fits?

**Prediction.**
> The objective is convex. Did you expect two runs from different starting points to agree?

**Evidence.**
> The difference between the two fitted `Theta` matrices — its size *and* its structure.
> The largest difference in predicted probabilities. `Theta.sum(axis=1)` in the penalised case.
> Confirmation that the multinomial model does not show the SQ1 failure.

**Answer and limitation.**
> In what sense the model is not identifiable and in what sense it does not matter; what the
> penalty selects and whether that choice is meaningful; and whether the zero-start
> representative is chosen by the objective or by the algorithm. Limitation: does anything you
> reported change if the classes are unbalanced?

---

## Appendix (optional, does not count towards the page limit)

> Anything you tried that did not work, or a question you would like answered.
> This is read but not marked.
