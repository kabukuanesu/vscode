DS5006 Fall 2026 -- Lab 4: Support Vector Machines
==================================================

Files in this folder
--------------------
  LAB4_Instructions.pdf   read this first
  svm_lab.py              the file you edit and submit
  svmlab_provided.py      the solver and your data -- DO NOT MODIFY
  selftest_lab4.py        checks your conventions; run it often
  README.txt              this file


Setup (do this on day one, not the night before)
------------------------------------------------
  pip install cvxopt
  python selftest_lab4.py

The self-test should run and report failures for every function you have
not written yet. If instead it stops with

    [FAIL] cvxopt is NOT installed

then the install did not work. Try

    python -m pip install --user cvxopt

and if it still fails, tell me or the TA early -- do not wait.


Working through the lab
-----------------------
1. Read Sections 1 and 2 of the instructions. The conventions in
   Section 2 are graded exactly as written.
2. Read the source of svmlab_provided.py, particularly solve_dual. The
   comments walk through how the dual of the lecture notes maps onto the
   standard form a QP library expects.
3. Write the seven functions in Part A of svm_lab.py. Run the self-test
   after each one.
4. Write the three experiments in Part C.
5. Run  python svm_lab.py  to produce results.json.
6. Write report.pdf (at most four pages).


What to submit
--------------
A single zip named exactly with your roll number, e.g. CS24M001.zip,
containing:

    svm_lab.py
    results.json
    report.pdf

Do NOT include svmlab_provided.py -- it is replaced with a clean copy
when your work is graded. Do not submit a notebook.


Expected runtime
----------------
The three experiments together solve about 80 quadratic programmes at
n = 240 and should finish in well under a minute. If yours takes
materially longer you have a Python loop where a NumPy expression
belongs, almost always inside rbf_kernel, which gets called with a
2000 x 240 argument.


A warning about the self-test
-----------------------------
Passing every check means your conventions are right. It does not mean
your lab is right: the grader also runs hidden tests on data you have not
seen, and re-runs your experiment_* functions rather than reading your
results.json. A hand-edited results file earns nothing.
