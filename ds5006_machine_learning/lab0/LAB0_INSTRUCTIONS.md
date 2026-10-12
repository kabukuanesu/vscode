# DS5006 Machine Learning — Fall 2026
## Lab 0: Warm-up and Calibration

| | |
|---|---|
| **Released** | Wednesday 29 July 2026 |
| **Due** | Wednesday 5 August 2026, 23:59 IST |
| **Marks** | **None. This lab is not graded.** |
| **Required** | **Yes.** A submission is required before Lab 1 will be accepted. |
| **Expected effort** | 2–4 hours |

---

### Why this exists, and what it is not

This course is the mathematically heavier counterpart of DS3010, and it assumes
you can turn linear algebra into working NumPy. Some of you have done that for
years. Some of you have done a lot of mathematics and very little Python, or the
reverse. Both are fine starting points, and neither is obvious to me from your
transcript.

So Lab 0 is a calibration exercise. It tells me, concretely, what the class can
already do — so that the lectures, the Wednesday sessions and the four graded labs
are pitched at the people actually in the room. **It carries no marks and cannot
hurt your grade.**

That has a direct consequence for how you should approach it: **an honest weak
submission is far more useful to me than a polished one you did not write.** If you
get help, or look things up, or use an AI tool, say so in the `help_used` field and
in your reflection. Nobody is penalised for this — the declaration is the point. If
the profile I get back says the class is comfortable with the SVD and it isn't, I
will pitch the September labs at a level that hurts you, and I will not find out
until the marks come in. Please just tell me where you are.

You will also get your own profile back, which is the most useful information you
can have a week before Lab 1.

---

### What to do

Fill in the eleven functions in `lab0.py` (ten tasks plus the information block). They are deliberately small — most are
one to five lines once you know the right tool. NumPy and the standard library
only.

**Part A — who you are.** `student_info()` returns a dictionary describing your
background. Every field is required; answer honestly, including the low numbers.

**Part B — plain Python** (no NumPy). `running_max`, `count_words`. Loops, dicts,
strings. Watch the empty-input cases.

**Part C — NumPy basics.** `column_stats`, `zscore_columns`, `top_k_rows`. Reducing
along an axis, sorting by a computed key, and the `ddof` argument, which will matter
again in Lab 1.

**Part D — vectorisation.** `pairwise_sq_dists(A, B)` returns the matrix of squared
Euclidean distances between every row of `A` and every row of `B`. This is checked
for correctness on a small input and separately for speed on a 2000 × 2000 problem
with an 8-second budget. A loop over pairs takes well over a minute. There are at
least two ways to avoid one, and the faster of the two comes from expanding
$\lVert a - b\rVert^2$ and noticing what is left. Watch for tiny negative values in
the result if you go that way.

**Part E — linear algebra.** `solve_system`, `rank_and_condition`, `low_rank_approx`.
One of these is checked on a system with a condition number near $10^9$, where a
solution obtained by explicitly inverting the matrix will not be accurate enough to
pass. `low_rank_approx` must return the *best* rank-$r$ approximation in the
Frobenius norm, which is a one-line consequence of a decomposition you will see in
week 2. If you have not met the SVD, say so in `student_info` and skip this one —
that is a genuinely useful thing for me to know, and it is not a mark against you.

**Part F — numerical care.** `log_sum_exp(z)` computes $\log \sum_i e^{z_i}$ and must
work when the entries of `z` are around $\pm 1000$, where the obvious implementation
returns `inf` or `-inf`.

**Reflection.** A short `reflection.md`, half a page at most, answering: which task
took you longest and why; anything you had to look up; anything you could not do;
roughly how long the whole thing took; and what help you used. Plain honesty is worth
more here than polish.

---

### Self-checking

`test_public.py` covers most of the same ground as my harness. Run it from the folder
containing your `lab0.py`:

```bash
python test_public.py
```

Run it on the untouched stub first, before you write anything. It should report ten
unimplemented functions, flag `student_info` as blank, and produce no errors. If it does something else, your
environment is wrong, and finding that out today rather than on 13 August is most of
the value of this exercise.

### What to submit

A zip named `<ROLLNUMBER>_lab0.zip` on Moodle, unpacking to exactly two files with no
enclosing folder:

```
lab0.py
reflection.md
```

This is the same submission format as Lab 1, deliberately. Getting it right once,
with nothing at stake, is the point.

### One more thing, on 5 August

There is a short **calibration quiz** on Moodle, open all day on Wednesday 5 August.
Twenty minutes, one attempt, no marks. It asks what small pieces of NumPy code print
and a few questions about singular values. Answer it from memory and move on — a wrong
answer given quickly is more useful to me than a right one looked up, and the timer is
short on purpose so that looking things up costs you more than it gains.

### What happens next

I will send each of you your own profile before Lab 1 opens on 6 August, and I will
post a short write-up of the tasks that gave the class the most trouble. If something
in it surprises you, email me — that is exactly the conversation this exercise is for.

Anything you could not finish, send the question.
