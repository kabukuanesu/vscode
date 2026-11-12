# DS5006 Lab 5 — Clustering: Report

**Name:**  
**Roll number:**

*At most 4 pages including figures. Every figure needs axis labels and a
legend or caption. Each answer: 2–5 sentences.*

---

## Q1. Choosing K and restarts (Experiment 1)

*Figure 1: `best_J` and `worst_J` against K on one set of axes.*

**K I would choose, and why:**

**Why "the K with the smallest loss" does not work:**

**Why `best_J` and `worst_J` differ for some K:**

---

## Q2. Outliers (Experiment 2)

*Figure 2: the data, with the two K-means centres and the two medoids
marked.*

`kmeans_shift` = ……  `kmedoids_shift` = ……

**Why the K-means centre moved and the medoid did not:**

---

## Q3. K-means, then EM (Experiment 3)

*Figure 3a: data coloured by K-means labels. Figure 3b: data coloured by
GMM labels. Figure 3c: EM log-likelihood against iteration.*

`kmeans_acc` = ……  `gmm_acc` = ……  `gmm_n_iter` = ……

`gmm_pis_sorted` = ……  True group sizes (from `y`) = ……

**What K-means got wrong, and what in the GMM corrected it:**

**Mixing coefficients versus group sizes:**

---

## Q4. Linkage (Experiment 4)

*Figure 4: three scatter plots coloured by the two-cluster cut — single,
complete, average.*

| Linkage | acc | top_height |
|---|---|---|
| single | | |
| complete | | |
| average | | |

**Why single linkage recovers the moons and complete linkage does not:**

**Why single linkage's top height is so much smaller:**
