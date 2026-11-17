"""
DS5006 Fall 2026 -- Lab 5: Clustering
================================================================

NAME : Anesu Kabuku
ROLL : 142602026

Fill in the fourteen functions in Part A, then the four experiments in
Part B, then run this file.  It writes results.json.

Rules
-----
* NumPy only, plus the provided module.  Do not import scikit-learn,
  scipy, or any other learning library.  matplotlib is allowed for your
  plots, but not inside the Part A functions.
* Do not modify clusterlab_provided.py.  It is replaced with a clean copy
  when your work is graded.
* Every function receives exactly the arguments shown.  Do not add
  arguments, do not rely on global variables, do not print.
* None of the clustering functions may look at the true labels y.

Conventions -- these are graded exactly as written
--------------------------------------------------
Distances and losses
 1. pairwise_sq_dists returns SQUARED Euclidean distances, shape
    (n_A, n_B), never negative.
 2. within_cluster_scatter is W(C) = 1/2 * sum_k sum_{C(i)=k} sum_{C(i')=k} d_ii'
    computed from a given dissimilarity matrix D (whatever D contains).
 3. kmeans_loss is J = sum_i ||x_i - m_{C(i)}||^2: a SUM, not a mean,
    and without the cluster-size factor N_k.

Ties and empty clusters
 4. When a point is equally close to several centres (or medoids), it is
    assigned to the one with the SMALLEST index.
 5. In the K-means M step, a cluster with no points KEEPS its previous
    centre.
 6. In the K-medoids M step, if several members of a cluster have the same
    smallest total dissimilarity, the one with the SMALLEST row index wins.

Iteration bookkeeping (K-means and K-medoids use the same scheme)
 7. Run one E step on the initial centres, and record the loss.  Then
    repeat: M step, record the loss; E step, record the loss; stop if the
    E step did not change any assignment.  Also stop after max_iter
    M steps.  n_iter is the number of M steps performed.
    The history therefore has length 1 + 2 * n_iter.

Gaussian mixtures
 8. gmm_e_step returns the TOTAL log-likelihood sum_i log p(x_i), not the
    mean.  It must give finite responsibilities that sum to 1 even for a
    point so far away that every component density underflows to 0.
 9. Covariances use the denominator N_k (the effective count), the outer
    product (x - mu)(x - mu)^T around the NEW mean, and then reg * I is
    added.  This applies in gmm_m_step and gmm_init_from_kmeans.
10. gmm_init_from_kmeans: pi_k = N_k / N, mu_k = centres[k], and the
    covariance of cluster k is taken around centres[k] (the centre you
    are given, not a recomputed mean).
11. gmm_em: record the log-likelihood of the initial parameters.  Then
    repeat: M step, E step, record the log-likelihood; stop when
    (latest - previous) < tol.  Note: no absolute value.  Also stop after
    max_iter M steps.  Return the responsibilities computed from the FINAL
    parameters.

Hierarchical clustering
12. agglomerative returns an (N-1, 4) float array.  Row t describes merge t:
        [i, j, height, size]
    where i < j are the ids of the two clusters merged, height is the
    linkage distance between them, and size is the number of points in the
    new cluster.  Points have ids 0..N-1; the cluster created by merge t
    gets id N + t.
13. When several pairs of clusters are equally close, merge the pair with
    the smallest i, and among those the smallest j.
14. cut_dendrogram labels clusters 0..K-1 in order of first appearance:
    point 0 is in cluster 0, the first point not in cluster 0 is in
    cluster 1, and so on.
"""

import json

import numpy as np

from clusterlab_provided import (
    make_dataset, initial_centre_indices, euclidean_distance_matrix,
    matching_accuracy, K_RANGE, N_RESTARTS, KMEANS_MAX_ITER,
    KMEDOIDS_MAX_ITER, GMM_REG, GMM_TOL, GMM_MAX_ITER, LINKAGES)

ROLL_NUMBER = "142602026"          # e.g. "CS24M001" -- REQUIRED


# ======================================================================
# Part A1 -- distances and scatter                    (lab doc Sec. 3.1)
# ======================================================================
def pairwise_sq_dists(A, B):
    """Squared Euclidean distances.

    A : (n_A, d), B : (n_B, d)
    returns (n_A, n_B) array, entry [a, b] = ||A[a] - B[b]||^2, >= 0.
    Vectorise: no Python loop over rows.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    A_sq = np.sum(A ** 2, axis=1)
    B_sq = np.sum(B ** 2, axis=1)

    D = A_sq[:, None] + B_sq[None, :] - 2 * A @ B.T

    return np.maximum(D, 0.0)  # Ensure non-negative distances


def within_cluster_scatter(D, labels):
    """W(C) from a dissimilarity matrix.

    D      : (N, N) dissimilarities d_ii'
    labels : (N,) integer cluster index of every point
    returns float
    """
    D = np.array(D, dtype=float)
    labels = np.asarray(labels)

    total = 0.0

    for k in np.unique(labels):
        indices = np.flatnonzero(labels == k)

        cluster_distances = D[np.ix_(indices, indices)]

        total += 0.5 * np.sum(cluster_distances)

    return float(total)


# ======================================================================
# Part A2 -- K-means                                  (lab doc Sec. 3.2)
# ======================================================================
def kmeans_e_step(X, centres):
    """Assign every point to its nearest centre (convention 4).

    X : (N, d), centres : (K, d)
    returns (N,) int array of cluster indices in 0..K-1
    """
    D = pairwise_sq_dists(X, centres)

    labels = np.argmin(D, axis=1)

    return labels.astype(int)


def kmeans_m_step(X, labels, centres):
    """Move each centre to the mean of its points (convention 5).

    X : (N, d), labels : (N,), centres : (K, d) the CURRENT centres
    returns (K, d) new centres.  Must not modify `centres` in place.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    new_centres = np.array(centres, dtype=float, copy=True)

    K = centres.shape[0]

    for k in range(K):
        cluster_points = X[labels == k]

        if len(cluster_points) > 0:
            new_centres[k] = np.mean(cluster_points, axis=0)

    return new_centres


def kmeans_loss(X, labels, centres):
    """J (convention 3).  returns float."""
    
    assigned_centres = centres[labels]

    differences = X - assigned_centres

    loss = np.sum(differences ** 2)

    return float(loss)


def kmeans(X, init_centres, max_iter=KMEANS_MAX_ITER):
    """Full K-means from the given starting centres (convention 7).

    returns (labels, centres, history, n_iter)
        labels  : (N,) int
        centres : (K, d) float
        history : 1-D float array of length 1 + 2 * n_iter
        n_iter  : int, number of M steps performed
    """
    X = np.asarray(X, dtype=float)
    centres = np.asarray(init_centres, dtype=float, copy=True)

    # Initial E step
    labels = kmeans_e_step(X, centres)

    history = [kmeans_loss(X, labels, centres)]

    n_iter = 0

    for _ in range(max_iter):

        # M-step
        centres = kmeans_m_step(X, labels, centres)

        # Record loss after M-step
        history.append(kmeans_loss(X, labels, centres))

        # E-step
        new_labels = kmeans_e_step(X, centres)

        # Record loss after E-step
        history.append(kmeans_loss(X, new_labels, centres))

        n_iter += 1

        # Stop if assignments have not changed
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break

        labels = new_labels

    return labels, centres, np.array(history, dtype=float), int(n_iter)


# ======================================================================
# Part A3 -- K-medoids                                (lab doc Sec. 3.3)
# ======================================================================
def kmedoids(D, init_medoids, max_iter=KMEDOIDS_MAX_ITER):
    """K-medoids on a dissimilarity matrix (conventions 4, 6, 7).

    D            : (N, N) dissimilarity matrix -- the ONLY view of the data
    init_medoids : (K,) int, row indices of the starting medoids
    The loss is  sum_i D[i, medoid of the cluster of i].
    The new medoid of cluster k is the member i of cluster k that minimises
    sum over members i' of cluster k of D[i, i'].

    returns (labels, medoids, history, n_iter)
        labels  : (N,) int, cluster index 0..K-1 (position in `medoids`)
        medoids : (K,) int row indices
        history : 1-D float array of length 1 + 2 * n_iter
        n_iter  : int
    """
    D = np.asarray(D, dtype=float)
    medoids = np.array(init_medoids, dtype=int, copy=True)

    N = D.shape[0]
    K = len(medoids)

    # E-step: assign points to their nearest medoid
    distances = D[:, medoids]
    labels = np.argmin(distances, axis=1)

    # Initial loss
    loss = np.sum(D[np.arange(N), medoids[labels]])
    history = [float(loss)]

    n_iter = 0

    for _ in range(max_iter):

        # M-step: find the best medoid within each cluster
        new_medoids = medoids.copy()

        for k in range(K):
            members = np.flatnonzero(labels == k)

            if len(members) > 0:
                # Distances from every candidate to all cluster members
                cluster_D = D[np.ix_(members, members)]

                # Candidate with smallest total dissimilarity
                costs = np.sum(cluster_D, axis=1)

                # members are in ascending row-index order,
                # so argmin also implements the required tie rule.
                best_position = np.argmin(costs)

                new_medoids[k] = members[best_position]

        medoids = new_medoids

        # Record loss after M-step (using previous assignments)
        loss = np.sum(D[np.arange(N), medoids[labels]])
        history.append(float(loss))

        # E-step with the updated medoids
        new_labels = np.argmin(D[:, medoids], axis=1)

        # Record loss after E-step
        loss = np.sum(D[np.arange(N), medoids[new_labels]])
        history.append(float(loss))

        n_iter += 1

        # Stop if assignments have not changed
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break

        labels = new_labels

    return labels, medoids, np.array(history, dtype=float), int(n_iter)


# ======================================================================
# Part A4 -- Gaussian mixture models                  (lab doc Sec. 3.4)
# ======================================================================
def gaussian_logpdf(X, mu, Sigma):
    """log N(x_i | mu, Sigma) for every row of X.

    X : (N, d), mu : (d,), Sigma : (d, d) symmetric positive definite
    returns (N,) float array.
    Hint (not a rule): a Cholesky factor of Sigma gives both the
    log-determinant and the quadratic form without forming Sigma^{-1}.
    """
    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    N, d = X.shape

    # Cholesky decomposition: Sigma = L @ L.T
    L = np.linalg.cholesky(Sigma)

    # Log determinant of Sigma
    log_det = 2.0 * np.sum(np.log(np.diag(L)))

    # Differences between observations and the mean
    diff = (X - mu).T

    # Solve L @ Z = diff
    Z = np.linalg.solve(L, diff)

    # Squared Mahalanobis distances
    mahalanobis = np.sum(Z ** 2, axis=0)

    # Log Gaussian density
    log_pdf = -0.5 * (
        d * np.log(2.0 * np.pi)
        + log_det
        + mahalanobis
    )

    return log_pdf


def gmm_e_step(X, pis, mus, Sigmas):
    """Responsibilities and total log-likelihood (convention 8).

    pis : (K,), mus : (K, d), Sigmas : (K, d, d)
    returns (R, loglik)
        R      : (N, K), R[i, k] = gamma(z_ik), rows sum to 1
        loglik : float, sum_i log sum_k pi_k N(x_i | mu_k, Sigma_k)
    """
    X = np.asarray(X, dtype=float)
    pis = np.asarray(pis, dtype=float)
    mus = np.asarray(mus, dtype=float)
    Sigmas = np.asarray(Sigmas, dtype=float)

    N = X.shape[0]
    K = len(pis)

    # Store log(pi_k * Gaussian density) for every point and component
    log_joint = np.empty((N, K), dtype=float)

    for k in range(K):
        log_joint[:, k] = (
            np.log(pis[k])
            + gaussian_logpdf(X, mus[k], Sigmas[k])
        )

    # Log-sum-exp for numerical stability
    row_max = np.max(log_joint, axis=1)

    shifted = np.exp(log_joint - row_max[:, None])

    row_sum = np.sum(shifted, axis=1)

    # Responsibilities
    R = shifted / row_sum[:, None]

    # Total log-likelihood
    loglik = np.sum(row_max + np.log(row_sum))

    return R, float(loglik)


def gmm_m_step(X, R, reg=GMM_REG):
    """Re-estimate the parameters from responsibilities (convention 9).

    returns (pis, mus, Sigmas) with shapes (K,), (K, d), (K, d, d)
    """
    X = np.asarray(X, dtype=float)
    R = np.asarray(R, dtype=float)

    N, d = X.shape
    K = R.shape[1]

    # Effective number of points per component
    Nk = np.sum(R, axis=0)

    # Mixing coefficients
    pis = Nk / N

    # Weighted means
    mus = (R.T @ X) / Nk[:, None]

    # Covariance matrices
    Sigmas = np.empty((K, d, d), dtype=float)

    for k in range(K):
        diff = X - mus[k]

        weighted_cov = (diff.T * R[:, k]) @ diff

        Sigmas[k] = weighted_cov / Nk[k] + reg * np.eye(d)

    return pis, mus, Sigmas


def gmm_init_from_kmeans(X, labels, centres, reg=GMM_REG):
    """Starting GMM parameters from a K-means result (conventions 9, 10).

    You may assume every cluster has at least two points.
    returns (pis, mus, Sigmas)
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    centres = np.asarray(centres, dtype=float)

    N, d = X.shape
    K = centres.shape[0]

    pis = np.empty(K, dtype=float)
    mus = centres.copy()
    Sigmas = np.empty((K, d, d), dtype=float)

    for k in range(K):
        members = X[labels == k]
        Nk = len(members)

        # Mixing coefficient
        pis[k] = Nk / N

        # Covariance around the GIVEN K-means centre
        diff = members - centres[k]

        Sigmas[k] = (
            diff.T @ diff / Nk
            + reg * np.eye(d)
        )

    return pis, mus, Sigmas


def gmm_em(X, pis, mus, Sigmas, reg=GMM_REG, tol=GMM_TOL,
           max_iter=GMM_MAX_ITER):
    """EM for a Gaussian mixture from the given start (convention 11).

    returns (pis, mus, Sigmas, R, history, n_iter)
        history : 1-D float array of length 1 + n_iter
    """
    X = np.asarray(X, dtype=float)
    pis = np.array(pis, dtype=float, copy=True)
    mus = np.array(mus, dtype=float, copy=True)
    Sigmas = np.array(Sigmas, dtype=float, copy=True)

    # Initial E-step
    R, loglik = gmm_e_step(X, pis, mus, Sigmas)

    history = [loglik]
    n_iter = 0

    for _ in range(max_iter):

        # M-step: update parameters
        pis, mus, Sigmas = gmm_m_step(X, R, reg)

        # E-step: calculate new responsibilities and likelihood
        R, loglik = gmm_e_step(X, pis, mus, Sigmas)

        history.append(loglik)
        n_iter += 1

        # Stop when the likelihood gain is smaller than tolerance
        if history[-1] - history[-2] < tol:
            break

    return (
        pis,
        mus,
        Sigmas,
        R,
        np.array(history, dtype=float),
        int(n_iter)
    )



# ======================================================================
# Part A5 -- agglomerative hierarchical clustering    (lab doc Sec. 3.5)
# ======================================================================
def agglomerative(D, linkage):
    """Bottom-up hierarchical clustering (conventions 12, 13).

    D       : (N, N) dissimilarity matrix
    linkage : "single", "complete" or "average"
    returns (N-1, 4) float array of merges
    """
    D = np.asarray(D, dtype=float)

    N = D.shape[0]

    if linkage not in ("single", "complete", "average"):
        raise ValueError("Unknown linkage method")

    # Initially, every point is its own cluster
    clusters = {i: [i] for i in range(N)}

    merges = []

    for t in range(N - 1):

        active_ids = sorted(clusters.keys())

        best_pair = None
        best_distance = np.inf

        # Examine every pair of active clusters
        for a in range(len(active_ids)):
            for b in range(a + 1, len(active_ids)):

                i = active_ids[a]
                j = active_ids[b]

                members_i = clusters[i]
                members_j = clusters[j]

                # Distances between the two clusters
                distances = D[np.ix_(members_i, members_j)]

                # Calculate linkage distance
                if linkage == "single":
                    distance = np.min(distances)

                elif linkage == "complete":
                    distance = np.max(distances)

                else:  # average linkage
                    distance = np.mean(distances)

                # Active IDs are sorted, so this also implements
                # the required smallest-i, then smallest-j tie rule.
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)

        i, j = best_pair

        # Create the new cluster with ID N + t
        new_id = N + t

        new_members = clusters[i] + clusters[j]

        # Record the merge
        merges.append([
            i,
            j,
            best_distance,
            len(new_members)
        ])

        # Remove old clusters and insert the new one
        del clusters[i]
        del clusters[j]

        clusters[new_id] = new_members

    return np.asarray(merges, dtype=float).reshape(N - 1, 4)


def cut_dendrogram(merges, N, K):
    """Cluster labels obtained by keeping only the first N-K merges
    (convention 14).

    returns (N,) int array with values 0..K-1
    """
    merges = np.asarray(merges, dtype=float)

    # Every point and every possible merged cluster has an ID
    parent = np.arange(2 * N - 1, dtype=int)

    # Keep the first N-K merges
    n_merges = N - K

    for t in range(n_merges):

        i = int(merges[t, 0])
        j = int(merges[t, 1])

        new_id = N + t

        parent[i] = new_id
        parent[j] = new_id

    # Find the final cluster containing each original point
    final_clusters = []

    for point in range(N):

        node = point

        while parent[node] != node:
            node = parent[node]

        final_clusters.append(node)

    # Assign canonical labels in order of first appearance
    label_map = {}
    labels = np.empty(N, dtype=int)

    next_label = 0

    for point in range(N):

        cluster_id = final_clusters[point]

        if cluster_id not in label_map:
            label_map[cluster_id] = next_label
            next_label += 1

        labels[point] = label_map[cluster_id]

    return labels


# ======================================================================
# Part B -- experiments                               (lab doc Sec. 4)
# Each returns a dict with EXACTLY the keys listed.  Plain Python floats,
# ints and lists only (use float(), int(), .tolist()).
# ======================================================================
def experiment_1(roll):
    """K-means on "blobs": loss against K, over restarts.

    returns {"best_J": [8 floats], "worst_J": [8 floats]}
    """
    X, y = make_dataset(roll, "blobs")

    best_J = []
    worst_J = []

    for K in K_RANGE:
        losses = []

        for r in range(N_RESTARTS):
            idx = initial_centre_indices(roll, "blobs", K, r)

            init_centres = X[idx]

            labels, centres, history, n_iter = kmeans(
                X, init_centres
            )

            losses.append(float(history[-1]))

        best_J.append(float(np.min(losses)))
        worst_J.append(float(np.max(losses)))

    return {
        "best_J": best_J,
        "worst_J": worst_J
    }


def experiment_2(roll):
    """K-means versus K-medoids on "outliers".

    returns {"kmeans_centres": [[x, y], [x, y]],
             "kmedoids_medoids": [int, int],
             "kmeans_shift": float,
             "kmedoids_shift": float}
    """
    X, y = make_dataset(roll, "outliers")

    # Two initial points, shared by both algorithms
    idx = initial_centre_indices(roll, "outliers", 2, 0)

    # K-means
    labels_km, centres, history_km, n_iter_km = kmeans(
        X, X[idx]
    )

    # K-medoids
    D = euclidean_distance_matrix(X)

    labels_md, medoids, history_md, n_iter_md = kmedoids(
        D, idx
    )

    # Means of the two genuine groups (exclude outliers)
    g0 = np.mean(X[y == 0], axis=0)
    g1 = np.mean(X[y == 1], axis=0)

    group_means = np.array([g0, g1])

    # Calculate distance from each representative to the nearest
    # genuine group mean
    def shifts(points):
        distances = np.linalg.norm(
            points[:, None, :] - group_means[None, :, :],
            axis=2
        )

        return np.min(distances, axis=1)

    kmeans_shift = float(np.max(shifts(centres)))

    medoid_points = X[medoids]
    kmedoids_shift = float(np.max(shifts(medoid_points)))

    return {
        "kmeans_centres": centres.tolist(),
        "kmedoids_medoids": medoids.astype(int).tolist(),
        "kmeans_shift": kmeans_shift,
        "kmedoids_shift": kmedoids_shift
    }


def experiment_3(roll):
    """K-means, then EM for a GMM, on "aniso".

    returns {"kmeans_acc": float, "gmm_acc": float,
             "gmm_pis_sorted": [3 floats, largest first],
             "gmm_initial_ll": float, "gmm_final_ll": float,
             "gmm_n_iter": int}
    """
    X, y = make_dataset(roll, "aniso")

    # Initialisation for K = 3
    idx = initial_centre_indices(roll, "aniso", 3, 0)

    # Run K-means
    kmeans_labels, centres, kmeans_history, kmeans_iter = kmeans(
        X, X[idx]
    )

    kmeans_acc = matching_accuracy(kmeans_labels, y)

    # Initialise GMM from K-means
    pis, mus, Sigmas = gmm_init_from_kmeans(
        X, kmeans_labels, centres, GMM_REG
    )

    # Run GMM EM using default arguments
    pis, mus, Sigmas, R, history, n_iter = gmm_em(
        X, pis, mus, Sigmas
    )

    # Convert responsibilities to hard cluster assignments
    gmm_labels = np.argmax(R, axis=1)

    gmm_acc = matching_accuracy(gmm_labels, y)

    # Sort mixing coefficients from largest to smallest
    gmm_pis_sorted = np.sort(pis)[::-1].tolist()

    return {
        "kmeans_acc": float(kmeans_acc),
        "gmm_acc": float(gmm_acc),
        "gmm_pis_sorted": gmm_pis_sorted,
        "gmm_initial_ll": float(history[0]),
        "gmm_final_ll": float(history[-1]),
        "gmm_n_iter": int(n_iter)
    }


def experiment_4(roll):
    """Single, complete and average linkage on "moons".

    returns {"acc": {"single": float, "complete": float, "average": float},
             "top_height": {"single": float, "complete": float,
                            "average": float}}
    """
    X, y = make_dataset(roll, "moons")

    N = X.shape[0]

    # Construct Euclidean distance matrix
    D = euclidean_distance_matrix(X)

    acc = {}
    top_height = {}

    for linkage in LINKAGES:

        # Construct hierarchical clustering
        merges = agglomerative(D, linkage)

        # Cut the dendrogram into two clusters
        labels = cut_dendrogram(merges, N, 2)

        # Evaluate clustering accuracy
        acc[linkage] = float(matching_accuracy(labels, y))

        # Height of the final merge
        top_height[linkage] = float(merges[-1, 2])

    return {
        "acc": acc,
        "top_height": top_height
    }


# ======================================================================
# Do not edit below this line
# ======================================================================
def run_experiments(roll=None, path="results.json"):
    roll = roll or ROLL_NUMBER
    if not roll:
        raise SystemExit("Set ROLL_NUMBER at the top of the file first.")
    res = {"roll_number": roll,
           "experiment_1": experiment_1(roll),
           "experiment_2": experiment_2(roll),
           "experiment_3": experiment_3(roll),
           "experiment_4": experiment_4(roll)}
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {path}")
    return res


if __name__ == "__main__":
    run_experiments()
