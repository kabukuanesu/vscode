import numpy as np
import matplotlib.pyplot as plt
import ds5006_lab2 as lab
import ds5006_lab2_data as data

ROLL = "142602026"
N_REPS = 100
n_grid = [25, 50, 100, 200, 400, 800]

fig, ax = plt.subplots(figsize=(8, 5))
for regime, colour in [("gaussian", "C0"), ("contaminated", "C3")]:
    means, ses = [], []
    for n in n_grid:
        out = lab.error_gap(ROLL, regime, n, N_REPS)
        means.append(out["mean_gap"])
        ses.append(out["se_gap"])
        print(f"{regime:12s} n={n:4d} "
              f"mean_gap={out['mean_gap']:+.4f} "
              f"se={out['se_gap']:.4f} "
              f"err_lda={out['err_lda']:.4f} "
              f"err_log={out['err_logistic']:.4f} "
              f"n_used={out['n_used']}")
    ax.errorbar(n_grid, means, yerr=[1.96*s for s in ses],
                marker="o", capsize=4, label=regime, color=colour)

ax.axhline(0, color="black", lw=0.7)
ax.set_xscale("log")
ax.set_xlabel("n_train")
ax.set_ylabel("mean_gap = err_LDA - err_logistic")
ax.set_title("LDA vs logistic regression: paired test error gap")
ax.legend(); ax.grid(True)
plt.tight_layout(); plt.savefig("sq5_error_gap.png", dpi=200)
plt.show()
