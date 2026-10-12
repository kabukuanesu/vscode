import numpy as np
import matplotlib.pyplot as plt

import ds5006_lab2 as lab

ROLL = 142602026

TRIALS = 200

# Include enough n values to clearly see the transition.
n_values = {
    2: range(1, 11),
    3: range(1, 13),
    5: range(1, 19)
}

results = {}

print("=" * 60)
print("SQ3 - RANDOM LABEL SEPARABILITY")
print("=" * 60)

for d, ns in n_values.items():

    fractions = []
    standard_errors = []

    print(f"\nFEATURE DIMENSION d = {d}")
    print("-" * 60)

    for n in ns:

        fraction = lab.separable_fraction(
            ROLL,
            n,
            d,
            TRIALS
        )

        # Monte Carlo standard error for a proportion
        se = np.sqrt(
            fraction * (1.0 - fraction) / TRIALS
        )

        fractions.append(fraction)
        standard_errors.append(se)

        print(
            f"n = {n:2d} | "
            f"fraction = {fraction:.4f} | "
            f"SE = {se:.4f} | "
            f"95% approx. CI = "
            f"[{fraction - 1.96 * se:.4f}, "
            f"{fraction + 1.96 * se:.4f}]"
        )

    results[d] = {
        "n": np.array(list(ns)),
        "fraction": np.array(fractions),
        "se": np.array(standard_errors)
    }


# ------------------------------------------------------------
# Plot all three curves
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

for d in [2, 3, 5]:

    result = results[d]

    plt.errorbar(
        result["n"],
        result["fraction"],
        yerr=result["se"],
        marker="o",
        capsize=4,
        label=f"d = {d}"
    )

    # Expected theoretical crossing
    expected_crossing = 2 * (d + 1)

    plt.axvline(
        expected_crossing,
        linestyle="--",
        alpha=0.4
    )


plt.axhline(
    0.5,
    linestyle=":",
    alpha=0.7,
    label="0.5"
)

plt.xlabel("Number of points (n)")
plt.ylabel("Fraction of separable datasets")
plt.title("Separability of random labels")
plt.ylim(-0.05, 1.05)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("sq3_random_separability.png", dpi=300)

plt.show()
