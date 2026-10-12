# DS5006 Lab 3 — Report

**Name:** Anesu Kabuku  
**Roll number:** 142602026

## SQ1 — Why do we not split on the thing we actually care about? (6)

### Best split at the tricky node

| Criterion | Feature | Threshold | Gain |
|---|---:|---:|---:|
| Gini | 0 | -1.044356 | 0.043393 |
| Entropy | 0 | -1.044356 | 0.055506 |
| Misclassification | -1 (no split) | NaN | 0.000000 |

### Trees grown with each criterion

| Depth | Gini err | Gini leaves | Entropy err | Entropy leaves | Misclass err | Misclass leaves |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3473 | 2 | 0.3473 | 2 | 0.3667 | 2 |
| 2 | 0.1627 | 4 | 0.1627 | 4 | 0.2467 | 4 |
| 3 | 0.1173 | 8 | 0.1083 | 8 | 0.1350 | 8 |
| 4 | 0.1173 | 13 | 0.1103 | 15 | 0.1250 | 13 |
| 5 | 0.1313 | 23 | 0.1243 | 27 | 0.1350 | 16 |
| 6 | 0.1397 | 36 | 0.1323 | 42 | 0.1307 | 18 |
| 8 | 0.1707 | 65 | 0.1663 | 73 | 0.1307 | 18 |
| 12 | 0.2033 | 104 | 0.1987 | 103 | 0.1307 | 18 |
| 20 | 0.2070 | 106 | 0.1987 | 103 | 0.1307 | 18 |

### What I infer

At the tricky node, Gini and entropy select feature 0 at threshold -1.044356 and achieve positive gains, whereas misclassification impurity returns no admissible split. This is not simply a case where misclassification chooses a worse split: its gain is exactly zero for every candidate split at this node under the implementation's gain tolerance. For binary class proportion \(p\), misclassification impurity is \(M(p)=\min(p,1-p)\), a piecewise-linear function with a kink at \(p=0.5\). Splits that change class proportions without changing the majority class can therefore produce no reduction in this impurity. Gini, \(G(p)=2p(1-p)\), and entropy, \(H(p)=-p\log p-(1-p)\log(1-p)\), are curved functions and can reward such changes. The depth sweep also shows that test error does not improve monotonically with tree depth. Entropy gives the lowest error at depth 3 (0.1083), while misclassification stops growing beyond 6 leaves' depth-6 result of 18 leaves and remains at error 0.1307 at depths 6–20.

## SQ2 — How deep, and how much should you trust it? (8)

The best depth by the Gini test-error sweep is **3**, with test error **0.117333**. At depth 20, test error is **0.207000**, a gap of **0.089667**. This is evidence that growing the tree deeper can overfit: a more complex tree can fit training details that do not generalize to the held-out test set.

**Stability over 15 bootstrap resamples.** Root feature counts for fully grown trees were: feature 0 = 0, feature 1 = 0, feature 2 = 15, feature 3 = 0, feature 4 = 0, feature 5 = 0. Thus, all 15 fully grown trees selected feature 2 at the root in these resamples.

| Measure | Fully grown | At best depth (3) |
|---|---:|---:|
| Mean pairwise disagreement | 0.190667 | 0.069784 |

The minimum and maximum individual test errors among fully grown trees were **0.192000 to 0.227667**.

Because the fully grown and depth-3 trees were trained on the same bootstrap resamples, the lower disagreement among depth-3 trees suggests that much of the instability accumulates in the deeper parts of the trees rather than at the root. The root counts reinforce this for this particular run: all 15 fully grown trees chose feature 2, so root-feature selection was stable even though the complete trees disagreed on about 19.1% of test predictions on average. Pruning at depth 3 reduced disagreement to about 7.0%. The root counts do not imply that all subsequent splits were stable; rather, the instability appears mainly below the shared root.

## SQ3 — Bagging, and what "out of bag" actually estimates (8)

**Figure 1: test error and out-of-bag error against \(B\), averaged over three forests.**

![SQ3 test and OOB curves](sq3_error_curves_report.png)

Single tree at `BAG_DEPTH` on the full training set: **0.207000**.  
Mean test error of the individual bagged trees: **0.212981**.  
Test error of the first 4 trees: **0.176444**.

| B | Test error | OOB error |
|---:|---:|---:|
| 1 | 0.212444 | 0.210928 |
| 2 | 0.225444 | 0.220639 |
| 5 | 0.148444 | 0.198973 |
| 10 | 0.140000 | 0.176891 |
| 20 | 0.132000 | 0.146667 |
| 40 | 0.128000 | 0.134583 |
| 70 | 0.128111 | 0.132083 |
| 100 | 0.127778 | 0.128750 |
| 150 | 0.126556 | 0.130417 |

### Saturation

The test-error curve begins to flatten at approximately 40 trees. It falls from 0.212444 at one tree to 0.126556 at 150 trees, an absolute reduction of 0.085888 (about 8.59 percentage points). Between 40 and 150 trees it changes only from 0.128000 to 0.126556. Bagging reduces the effect of variation among individual trees, but it does not eliminate shared errors or guarantee zero error. The ensemble's error of 0.126556 is below both the single full-data tree error (0.207000) and the mean individual bagged-tree error (0.212981). The single full-data tree is the clearest direct comparison for the practical benefit of replacing one tree with an ensemble; the mean individual tree error shows that voting improves on the typical bootstrap-trained tree. The best depth-3 tree from SQ2 has error 0.117333, which is lower than the 150-tree bagged result, so bagging is not the best of every model considered in this run. That comparison should be interpreted with care because depth 3 was selected from the SQ1 depth sweep, whereas the bagged trees use `BAG_DEPTH`.

### (a) Level

At \(B=150\), OOB minus test error is

\[
0.130417-0.126556=0.003861.
\]

Using the requested approximate standard errors for proportions,

\[
SE_{\mathrm{OOB}}=\sqrt{\frac{0.130417(1-0.130417)}{800}}\approx0.011906,
\]

\[
SE_{\mathrm{test}}=\sqrt{\frac{0.126556(1-0.126556)}{3000}}\approx0.006070.
\]

Combining them in quadrature gives

\[
SE_{\mathrm{difference}}\approx\sqrt{0.011906^2+0.006070^2}=0.013364.
\]

The observed absolute difference is \(0.003861/0.013364\approx0.289\) standard errors, small relative to the estimated sampling noise. Therefore, this run does not provide strong evidence that the OOB and test error levels differ. This is an approximate calculation; OOB predictions are dependent because they share trees, so the standard error is only a practical reference.

### (b) Shape

Excess means each curve's error minus its own value at \(B=150\).

| | OOB excess | Test excess at same B | Test excess of 4 trees |
|---|---:|---:|---:|
| B = 10 | 0.046475 | 0.013444 | 0.049889 |
| B = 20 | 0.016250 | 0.005444 | — |

At \(B=10\), the OOB excess (0.046475) differs from the test excess at \(B=4\) (0.049889) by only 0.003414, while it differs from the test excess at \(B=10\) (0.013444) by 0.033030. This supports the explanation that an OOB prediction uses only about \(0.37B\) trees on average: at \(B=10\), a training point receives about 3.7 OOB votes, roughly the four votes used in the test-ensemble comparison. At \(B=20\), the OOB excess remains larger than the test excess, consistent with slower convergence of the OOB curve. The small-\(B\) OOB curve is therefore not directly equivalent to the test curve at the same number of trees.

## SQ4 — How strong does a weak learner have to be? (8)

**Figure 2: training and test error against boosting round \(m\), for base depths 1 and 3.**

![SQ4 AdaBoost error curves](c3111c87-9de0-4260-ba07-fc45479abb3d.png)

| Base depth | \(m_0\) (train error hits 0) | Train error at \(M=300\) | Test error at \(M=300\) |
|---:|---:|---:|---:|
| 1 | Never | 0.315000 | 0.354667 |
| 3 | 148 | 0.000000 | 0.150000 |

### Why the two depths do not converge

A depth-1 tree (stump) makes one threshold split. A sum of stumps can combine effects of individual features, but has a restricted additive form and may not efficiently represent complex feature interactions. A sum of depth-3 trees can represent a richer family of decision functions because each tree can make several sequential splits. The target in this dataset therefore appears to require structure that the depth-1 boosted ensemble does not capture effectively: after 300 rounds its training error remains 0.315000, while depth 3 reaches zero training error at round 148. More rounds do not guarantee that a restricted family of base learners can close this representational gap.

### The bound

At \(m=300\), depth 3 has exponential-loss bound **0.03156**, training error **0.000000**, and test error **0.150000**. The bound was above training error at every round for both depths. The bound controls an upper bound on training classification error under the standard AdaBoost formulation; it does not bound test error or guarantee that test error decreases with every round. The depth-3 result illustrates the distinction: training error is zero, but test error remains 15%.

### Margins, depth 3, training set

| Round | Minimum | 5th percentile | Median |
|---|---:|---:|---:|
| \(m_0=148\) | 0.0027 | 0.0396 | 0.1266 |
| 300 | 0.0327 | 0.0527 | 0.1059 |

### What I infer

After round 148, every training point is correctly classified, but boosting continues to change the margin distribution. By round 300, the minimum margin and 5th percentile have increased, while the median margin has decreased. Thus, the weakest training example and the lower tail gained margin, but the typical normalized margin did not increase. The margins are normalized by the sum of absolute boosting weights, so adding more rounds changes both the cumulative score and its normalizer. Zero training error therefore does not imply that all margins continue to increase or that test error must improve.

## Notes

The experiments were run using the roll-specific data generated for roll number 142602026. The experiment scripts produced the required tables and plots. **Approximate total time spent:** [fill in your estimate before submitting].
