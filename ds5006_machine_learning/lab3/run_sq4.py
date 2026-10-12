import numpy as np
import matplotlib.pyplot as plt

import ds5006_lab3 as L
import ds5006_lab3_data as D

ROLL = 142602026

X_train, y_train, X_test, y_test = D.load_split(ROLL)

M = D.N_ROUNDS
DEPTHS = D.BASE_DEPTHS

results = {}

for depth in DEPTHS:
    trees, betas, eps = L.adaboost_fit(
        X_train, y_train,
        n_rounds=M,
        max_depth=depth,
        min_samples_split=2
    )

    F_train = L.adaboost_stage_scores(trees, betas, X_train)
    F_test = L.adaboost_stage_scores(trees, betas, X_test)

    pred_train = L.adaboost_predict(F_train)      # (M, n_train)
    pred_test = L.adaboost_predict(F_test)        # (M, n_test)

    train_err = np.mean(pred_train != y_train[None, :], axis=1)
    test_err = np.mean(pred_test != y_test[None, :], axis=1)
    bound = L.exp_loss_bound(eps)

    margins = L.normalised_margins(F_train, y_train, betas)  # (M, n_train)

    # First round where training error reaches zero
    zero_hits = np.flatnonzero(train_err == 0.0)
    m0 = int(zero_hits[0]) + 1 if zero_hits.size > 0 else None

    results[depth] = dict(
        trees=trees, betas=betas, eps=eps,
        train_err=train_err, test_err=test_err, bound=bound,
        margins=margins, m0=m0,
    )

    print(f"\n=== depth = {depth} ===")
    print(f"m0 (first zero train err): {m0}")
    print(f"train err at m=1:   {train_err[0]:.6f}")
    print(f"train err at m=300: {train_err[-1]:.6f}")
    print(f"test  err at m=1:   {test_err[0]:.6f}")
    print(f"test  err at m=300: {test_err[-1]:.6f}")
    print(f"bound  at m=300:    {bound[-1]:.3e}")
    print(f"bound above train err at every round: "
          f"{bool(np.all(bound >= train_err))}")

    if m0 is not None:
        row0 = margins[m0 - 1]
        rowM = margins[-1]
        print(f"margins at m0={m0}:   min={row0.min():.4f}  "
              f"p05={np.percentile(row0, 5):.4f}  "
              f"median={np.median(row0):.4f}")
        print(f"margins at m=300:     min={rowM.min():.4f}  "
              f"p05={np.percentile(rowM, 5):.4f}  "
              f"median={np.median(rowM):.4f}")

# Plot
plt.figure(figsize=(10, 6))
for depth, style in zip(DEPTHS, ("-", "--")):
    r = results[depth]
    ms = np.arange(1, M + 1)
    plt.plot(ms, r["train_err"], style, color="C0",
             label=f"train, depth {depth}")
    plt.plot(ms, r["test_err"], style, color="C1",
             label=f"test, depth {depth}")
plt.xlabel("round m")
plt.ylabel("misclassification error")
plt.title("SQ4: AdaBoost training and test error")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("sq4_curves.png", dpi=200)
plt.show()
