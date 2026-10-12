# DS5006 Lab 3 — Report

**Name: Anesu Kabuku**
**Roll number:142602026**

Keep this to about four pages. Tables and plots are cheap; paragraphs are what
is marked. Every number you quote must come from your own run on your own roll
number.

---

## SQ1 — Why do we not split on the thing we actually care about? (6)

**Best split at the tricky node.**

| Criterion | Feature | Threshold | Gain |
|---|---|---|---|
| Gini | | | |
| Entropy | | | |
| Misclassification | | | |

**Trees grown with each criterion.**

| Depth | Gini err | Gini leaves | Entropy err | Entropy leaves | Misclass err | Misclass leaves |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |
| 8 | | | | | | |
| 12 | | | | | | |
| 20 | | | | | | |

**What I infer.** (What property of the three impurity functions explains the
tricky node, and what exactly goes wrong when misclassification is used as the
splitting criterion? A sketch of the three curves against `p` helps.)

Gini and entropy are more suitable splitting criteria than misclassification impurity because their curved impurity functions are more sensitive to changes in class proportions. In the tricky-node experiment, both Gini and entropy identified a useful split on feature 0, whereas misclassification impurity produced zero gain and therefore rejected every split. In the full-tree experiment, entropy achieved the lowest test error of 0.1083 at depth 3, followed by Gini with 0.1173 at depth 3–4. Misclassification achieved its lowest error of 0.1250 at depth 4 and produced substantially fewer leaves at greater depths. As tree depth increased beyond the useful range, Gini and entropy showed increasing test error while tree size continued to grow, indicating overfitting. Overall, the results demonstrate why misclassification impurity is generally not preferred for growing decision trees, although it remains useful for evaluating the error of a final classifier.

---

## SQ2 — How deep, and how much should you trust it? (8)

Best depth: ___ with test error ___. Test error at depth 20: ___.

**Stability over 15 bootstrap resamples.**

Root feature counts (fully grown trees): ___

| | Fully grown | At best depth |
|---|---|---|
| Mean pairwise disagreement | | |

Individual test error of the fully grown trees, min and max: ___ to ___

**What I infer.** (Both sets of trees saw the same resamples. What does the gap
between their disagreement rates say about where in a tree the instability
lives? What do your root counts add?)

What I infer. The Gini tree achieved its lowest test error at depth 3, with an error of 0.117333, whereas the depth-20 tree had an error of 0.207000. The increase of 0.089667 provides evidence of overfitting as the tree becomes unnecessarily deep. The bootstrap stability experiment shows that all 15 fully grown trees selected feature 2 as their root feature, so the root split is highly stable despite resampling. However, the mean pairwise disagreement was 0.190667 for the fully grown trees compared with only 0.069784 for the depth-3 trees. Thus, the large increase in disagreement is not caused by instability in the root split; rather, it develops mainly in the deeper branches, where small changes in the bootstrap sample can lead to different subsequent splits. The fully grown trees also had individual test errors ranging from 0.192000 to 0.227667, with a mean of 0.211489, reinforcing the evidence that allowing the tree to grow deeply produces a more variable and less generalizable model.

---

## SQ3 — Bagging, and what "out of bag" actually estimates (8)

*Figure 1: test error and out of bag error against B, averaged over 3 forests.*

Single tree at `BAG_DEPTH` on the full training set: ___
Mean test error of the individual bagged trees: ___
Test error of the first 4 trees: ___

| B | Test error | OOB error |
|---|---|---|
| 1 | | |
| 2 | | |
| 5 | | |
| 10 | | |
| 20 | | |
| 40 | | |
| 70 | | |
| 100 | | |
| 150 | | |

**Saturation.** (Where does the curve flatten, and what does that say about what
bagging is doing? Which comparison is the fair one — the single deep tree, the
mean individual tree, or your best tree from SQ2 — and why?)

**(a) Level.** OOB minus test at B = 150: ___
Typical difference from sampling noise alone: ___ (show the calculation)

(Is the level difference evidence of anything?)

**(b) Shape.** Excess over own plateau:

| | OOB excess | Test excess at same B | Test excess of 4 trees |
|---|---|---|---|
| B = 10 | | | |
| B = 20 | | | — |

(Why does the OOB excess decay more slowly, and how well does the 0.37 B check
support your explanation?)

Bagging substantially reduces the test error of unstable deep decision trees, with most of the improvement occurring within the first few dozen trees; at 150 trees the OOB error is very close to the independently measured test error, while its slower convergence at small \(B\) is explained by each training observation receiving only about 37% as many OOB votes as the total number of trees.

---

## SQ4 — How strong does a weak learner have to be? (8)

*Figure 2: training and test error against round m, for base depths 1 and 3.*

| Base depth | m0 (train error hits 0) | Train err at M | Test err at M |
|---|---|---|---|
| 1 | | | |
| 3 | | | |

**Why the two depths do not converge.** (State the general principle about what
a sum of depth-k trees can represent, and connect it to this target.)

**The bound.** At m = 300, depth 3: bound ___, training error ___, test error ___.
Bound above training error at every round: yes / no.

(What does the bound control, and what does it not?)

**Margins, depth 3, training set.**

| Round | Min | 5th percentile | Median |
|---|---|---|---|
| m0 | | | |
| 300 | | | |

**What I infer.** (These three do not move the same way. What is boosting doing
in the rounds after training error is already zero?)

---

## Notes

Anything that went wrong, anything you could not get working, and roughly how
long the lab took you. This section is not marked and is genuinely useful.
