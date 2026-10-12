# DS5006 Lab 3 — Decision Trees and Ensembles

**Due date:** as announced on Moodle.
**Weight:** 100 marks — 60 code, 30 report, 10 code quality.
**Language:** Python with NumPy only. No scikit-learn, no SciPy, no pandas.

---

## 1. What this lab is about

A decision tree is the least stable model you have met so far. Change a handful
of training points and you can get a visibly different tree. That instability is
usually described as a weakness, and it is — but it is also the raw material
that ensembles are built out of. Averaging things that all agree buys you
nothing.

This lab follows that thread from one end to the other. You will build a tree
from scratch, watch it become unstable as it grows, and then spend the second
half of the lab exploiting that instability in the two standard ways: averaging
many trees fitted to resampled data (bagging), and fitting trees in sequence to
reweighted data (boosting). Along the way four questions come up that are worth
answering properly, and those are your four scientific questions.

You are not being asked to design experiments. Every sweep, every grid and every
ensemble size is specified for you in `ds5006_lab3_data.py` as a module-level
constant. What you are being asked to do is implement the machinery correctly,
run the specified experiments, and say what you make of the numbers.

Your dataset is generated from your roll number, so your numbers are yours. Two
students who report identical tables did not both do the lab.

---

## 2. Files

You are given five files. Keep them in one folder.

| File | What it is |
|---|---|
| `ds5006_lab3.py` | The stub. This is the only file you edit and the only code file you submit. |
| `ds5006_lab3_data.py` | Your dataset and the experiment grids. Do not edit it. |
| `test_lab3_public.py` | Self-checks you can run as often as you like. |
| `pylintrc` | The lint configuration the grader uses. Keep it beside your code. |
| `report_template.md` | The skeleton of your report. |

You submit a single zip named with your roll number, for example
`112201001.zip`, containing exactly two files:

| File | Notes |
|---|---|
| `ds5006_lab3.py` | Your completed stub, with your name and roll number filled in at the top. |
| `report.pdf` | Your report. PDF only. |

No notebooks, no data files, no plotting scripts, no folders inside the zip. If
the grader cannot find `ds5006_lab3.py` at a predictable place it will skip your
submission and you will be marked by hand, late.

Your plotting and experiment code does not go in `ds5006_lab3.py`. Write it in a
separate scratch file that you do not submit. The stub must contain the
19 functions and nothing that runs at import time.

---

## 3. The scientific questions

Read all four before you start coding. They build on each other, and knowing
where you are going will change how you write Part A.

### SQ1 — Why do we not split on the thing we actually care about? (6 marks)

We report a classifier by its misclassification rate. But nobody grows trees by
maximising the reduction in misclassification rate; the standard choice is Gini
or entropy. That is a strange state of affairs and it deserves an explanation.

**Run this.**

1. `ds5006_lab3_data.tricky_node(roll)` hands you a single node of 100 points
   taken from part-way down a tree: 60 of class 0 and 40 of class 1, with only
   column 0 varying. Call `best_split` on it three times, once with each of
   `gini`, `entropy` and `misclassification`. Report the chosen feature,
   threshold and gain in each case.
2. Grow a full tree on your training set with each of the three criteria, for
   every depth in `DEPTH_GRID`, with `min_samples_split=2`. Report a table of
   test error and number of leaves, nine rows by three criteria.

**Report.** The three-way comparison at the tricky node, the table, and then:
state what you infer about why misclassification is not used as a splitting
criterion. Your explanation should say something about the *shape* of the three
impurity functions as functions of the class proportion `p` — sketching them on
one axis will make the point in a way words will not. Be precise about what goes
wrong: it is not that misclassification gives a worse split, it is something
sharper than that.

### SQ2 — How deep, and how much should you trust it? (8 marks)

**Run this.**

1. From the Gini column of your SQ1 table, report the depth that minimises test
   error, the error there, and the error at depth 20. Say what the gap is
   evidence of.
2. Create one generator with `rng = make_rng(roll, "stability")` and use it to
   draw `N_BOOT_STABILITY` bootstrap resamples of your training set, one after
   another, with `bootstrap_indices`. On each resample grow **two** trees with
   Gini and `min_samples_split=2`: one with `max_depth=BAG_DEPTH` (fully
   grown), and one with `max_depth` equal to the best depth from part 1.
   Report:
   - how often each feature index is chosen as the *root* split of the fully
     grown trees, as counts;
   - the mean pairwise disagreement on the test set, separately for the 15
     fully grown trees and the 15 shallow ones. Disagreement is the average,
     over all pairs `(a, b)` of distinct trees, of the fraction of test points
     where the two trees predict differently;
   - the smallest and largest individual test error among the 15 fully grown
     trees.

**Report.** The numbers above, and then: state what you infer about *where* in
a tree the instability lives. The shallow and fully grown trees saw exactly the
same resamples, so the difference between their disagreement rates isolates
something specific. Use the root counts as well — whether your roots are all
the same or not, they tell you something, and different students will see
different things there. One paragraph.

### SQ3 — Bagging, and what "out of bag" actually estimates (8 marks)

Bagging comes with a free validation estimate: each tree sees roughly 63% of the
training rows, so each row is out of bag for roughly 37% of the trees and can be
predicted by them without leakage. The question is whether the estimate is free
in the sense of *costless*, or free in the sense of *worth what you paid*.

**Run this.**

Fit `N_REPEATS` independent forests of `N_TREES` trees each
(`max_depth=BAG_DEPTH`, `min_samples_split=2`), using a different named RNG
stream for each repeat — `make_rng(roll, "bag0")`, `"bag1"`, `"bag2"`. Fit each
forest once and slice it; do not refit for each ensemble size.

Averaged over the three forests, report

- for each `B` in `B_GRID`: the test error of the majority vote of the first
  `B` trees, and the out of bag error computed from the first `B` trees;
- the test error of the majority vote of the first **4** trees (you will need
  it in part 2);
- as single numbers: the test error of one tree grown on the whole training
  set at `BAG_DEPTH`, and the mean test error of the individual bagged trees.

**Report.** One plot with both curves against `B` (log scale on `B` is
sensible), the numbers above, and then answer two things.

1. Where does the test error curve flatten out, and what does the flattening
   tell you about what bagging is and is not doing? Compare against the two
   single numbers — and against your SQ2 results — and be careful about which
   comparison is the fair one.

2. The two curves do not sit on top of each other, and they are computed on
   *different rows*: the out of bag error on your 800 training rows, the test
   error on 3000 fresh ones. So they can differ for two unrelated reasons, and
   the first job is to separate them.

   **(a) Level.** Report the out of bag error minus the test error at
   `B = 150`. Both are proportions estimated from a finite sample. For a
   proportion `p` estimated from `n` points the standard error is about
   `sqrt(p (1 - p) / n)`, and your two numbers have different `n`. Work out how
   large a difference sampling noise alone would typically produce, report it,
   and say whether your level difference is evidence of anything. Either
   verdict can earn full marks; an unquantified verdict cannot.

   **(b) Shape.** Now remove the level: subtract from each curve its own value
   at `B = 150`, giving the *excess* of each curve over its plateau. Report
   both excesses at `B = 10` and `B = 20`. The out of bag excess decays more
   slowly than the test excess. Explain why — think about how many trees
   actually vote on a given training row when you compute an out of bag
   prediction from `B` trees. Then check your explanation directly: compare
   the out of bag excess at `B = 10` with the *test* excess of an ensemble of
   about `0.37 x 10`, that is 4, trees — and with the test excess at 10 trees.
   Which does it match, and how closely? (By `B = 20` both excesses are about
   one point, which is close to the resolution of an estimate made from 800
   rows; report them, but do not lean on them.)

   (We skip `B = 5` here on purpose. `0.37 x 5` is two trees, and a two-tree
   majority vote is settled by the tie-breaking rule whenever the two trees
   disagree, which makes it a poor stand-in for "a small ensemble".)

### SQ4 — How strong does a weak learner have to be? (8 marks)

**Run this.**

For each base-learner depth in `BASE_DEPTHS` (that is, 1 and 3), run
`adaboost_fit` for `N_ROUNDS` rounds with `min_samples_split=2`, and record for
every round `m`: training error, test error, and the bound
`exp_loss_bound(epsilons)[m-1]`.

**Report.**

1. One plot of training and test error against `m` for both depths (four
   curves). Report the round `m0` at which the training error first reaches
   zero, for each depth, or state that it never does.
2. The two depths do not end up in the same place, and the gap does not close
   with more rounds. Explain it. The relevant fact is about what a sum of
   depth-`k` trees can and cannot represent as a function of the input features,
   and what the gap therefore reveals about the target in this dataset — which
   you were never told. State the general principle, not just the observation.
3. Report the bound at `m = N_ROUNDS` for depth 3, alongside the training and
   test errors there. Verify that the bound is above the training error at every
   round and say so. Then state plainly what the bound does and does not control,
   given the three numbers you just reported.
4. For depth 3, report the minimum, the 5th percentile and the median of the
   **training** margins at round `m0` and at round `N_ROUNDS`. These three
   numbers do not all move in the same direction. Say what that means about what
   boosting is doing in the rounds after the training error has already hit zero.

---

## 4. Conventions

Eighteen contracts are pinned in the docstrings. They are collected here so you
can check them in one pass. Where a convention differs from what a library would
do, that is deliberate: the hidden tests check these exact behaviours.

| | Convention |
|---|---|
| C1 | `class_distribution` returns all zeros when the total weight is zero. `w=None` means weights of one, not one over n. |
| C2 | Impurities take an array of shape `(..., K)` and reduce along the last axis. |
| C3 | Entropy is in nats (natural log), with `0 log 0 = 0` and no warnings. |
| C4 | A split is admissible only if its gain exceeds `GAIN_TOL`. Otherwise `best_split` returns `(-1, nan, 0.0)`. |
| C5 | Split ties: smallest feature index first, then smallest threshold. |
| C6 | Node dicts use exactly the keys listed in the `grow_tree` docstring. |
| C7 | Leaf prediction is the largest weighted proportion; ties go to the smaller label. |
| C8 | Stopping conditions are checked before any split is attempted, in the order given. |
| C9 | `max_depth` counts remaining depth; `max_depth=0` is a single leaf. |
| C10 | A single leaf has `tree_stats` equal to `(1, 0)`. |
| C11 | `bootstrap_indices` is exactly one `rng.integers(0, n, size=n)` call. |
| C12 | Bagged trees are fitted unweighted. |
| C13 | Majority-vote ties go to the smaller label. |
| C14 | Rows with no out of bag tree are dropped from the out of bag error entirely. |
| C15 | Boosting weights start at `1/n` and are renormalised after every round. |
| C16 | No early stopping in boosting; `beta = 0.5 log((1-eps)/eps)`. |
| C17 | A boosting score of exactly zero predicts class 0. |
| C18 | Zero cumulative `|beta|` uses a denominator of 1.0. |

Rows with a feature value exactly equal to the threshold go **left**, everywhere.

---

## 5. Performance

This lab fits about 1,100 trees end to end, most of them the 450 fully grown
trees in SQ3. That is nothing if your inner loop is vectorised and painful if
it is not — the difference between the
two is about two orders of magnitude, which is the difference between a two
minute run and a four hour one.

Two places matter and both are flagged in the docstrings:

- **`best_split`.** Sort each column once per node and take a cumulative sum of
  the per-class weights. Every candidate threshold is then a prefix of that
  order, and all the gains for one feature come out of a single vectorised
  expression. Do not loop over thresholds recomputing masks.
- **`tree_predict`.** Recurse on index arrays, not on rows. At an internal node,
  split the incoming array of row indices with a boolean mask and pass the two
  halves down.

The last public check times 20 trees on your real training set and prints an
estimate for the whole lab. Run it early. If it says an hour, stop and fix the
vectorisation before you write any experiment code — you will be running these
fits many times.

---

## 6. Running the checks

```bash
python3 test_lab3_public.py
python3 -m pylint ds5006_lab3.py --rcfile=pylintrc
```

The self-checks test contracts, not correctness. Passing all 36 does not mean
your science is right; failing any of them means the hidden tests will fail too.

The lint configuration is deliberately relaxed about short mathematical names
(`X`, `y`, `w`, `p`, `j`) and strict about everything else: unused variables,
dead code, inconsistent returns, over-long lines. Fix the warnings rather than
silencing them. A file full of `# pylint: disable=` scores 10/10 from pylint and
will be marked down by hand.

**Common problems.**

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: ds5006_lab3_data` | You are not running from the folder containing all five files. |
| `python3 -m test_lab3_public.py` fails | Drop the `-m`. It is `python3 test_lab3_public.py`. |
| Recursion error in `grow_tree` | Your stopping conditions are not being reached. Check C8 and C9, and that `max_depth` decreases on the recursive calls. |
| Out of bag error is `nan` at `B = 1` | Your mask is inverted. At `B = 1` about 37% of rows are out of bag and should be scored; the rest are dropped (C14). |
| Boosting weights drift away from summing to one | You are renormalising in the wrong place. C15. |
| Everything is correct but glacial | See Section 5. |

---

## 7. Marks

| Component | Marks |
|---|---|
| Hidden tests on the 19 functions | 60 |
| SQ1 | 6 |
| SQ2 | 8 |
| SQ3 | 8 |
| SQ4 | 8 |
| Code quality (pylint, scaled) | 10 |

The code marks are spread across all three parts, with roughly half of them in
Part A. A correct Part A and an honest attempt at the rest is a comfortable
pass; the marks in SQ3 and SQ4 reward explanation, not more output.

Report marks are for the reasoning. A table with no paragraph under it earns
close to nothing, and a paragraph that describes the table back to me in words
earns the same. What earns marks is a claim about mechanism that the numbers you
reported actually support. Where a question asks you to state what you infer,
a wrong inference argued from your own numbers will score better than a right
one asserted without them.

## 8. Academic integrity

The course policy applies in full. Discuss ideas with anyone; write your own
code. Every submission is graded against a dataset derived from the submitting
roll number, so code that hard-codes numbers from someone else's run fails
silently and completely.
