import numpy as np
import matplotlib.pyplot as plt
import ds5006_lab2 as lab
import ds5006_lab2_data as data

ROLL = 142602026

lams = 10.0 ** (-np.arange(1, 10))

print("=" * 60)
print("SQ3 - PENALTY SWEEP")
print("=" * 60)

results = {}

for kind in ["separable", "overlapping"]:

    X, y = data.dataset_margin(ROLL, kind)

    norms = []
    iterations = []
    objectives = []

    print(f"\n{kind.upper()}")
    print("-" * 60)

    for lam in lams:

        result = lab.fit_logistic_newton(
            X,
            y,
            lam=lam,
            tol=1e-10,
            max_iter=100
        )

        theta = result["theta"]
        norm = np.linalg.norm(theta)

        norms.append(norm)
        iterations.append(result["n_iter"])
        objectives.append(result["objective"])

        print(
            f"lambda = {lam:.0e} | "
            f"log(1/lambda) = {np.log(1 / lam):.6f} | "
            f"||theta|| = {norm:.10f} | "
            f"iterations = {result['n_iter']} | "
            f"objective = {result['objective']:.10e}"
        )

    results[kind] = {
        "norms": norms,
        "iterations": iterations,
        "objectives": objectives
    }


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------


x = np.log(1 / lams)

plt.figure(figsize=(8, 5))

plt.plot(
    x,
    results["separable"]["norms"],
    marker="o",
    label="Separable"
)

plt.plot(
    x,
    results["overlapping"]["norms"],
    marker="o",
    label="Overlapping"
)

plt.xlabel("log(1 / lambda)")
plt.ylabel("||theta||")
plt.title("Logistic coefficient norm vs penalty")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("sq3_penalty_sweep.png", dpi=300)

plt.show()
