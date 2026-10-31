from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.interpolate import splprep, splev
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import pairwise_distances
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA


def compute_spline_pseudotime(points, ref_vector=None, k=10, smooth=0.5, n_eval=2000, saved_tck=None):
    points = np.asarray(points)

    # --- Spline setup ---
    if saved_tck is not None:
        tck = saved_tck
        is_flipped = False
    else:
        N = len(points)

        # Build a connected graph by combining MST edges with local kNN edges.
        dist_matrix = pairwise_distances(points)

        mst = minimum_spanning_tree(dist_matrix)
        G = nx.from_scipy_sparse_array(mst)

        nbrs = NearestNeighbors(n_neighbors=min(k + 1, N)).fit(points)
        dists, idxs = nbrs.kneighbors(points)
        for i in range(N):
            for j, dist in zip(idxs[i, 1:], dists[i, 1:]):
                G.add_edge(i, int(j), weight=float(dist))

        try:
            # Approximate the graph diameter endpoints.
            lengths0 = nx.single_source_dijkstra_path_length(G, 0, weight='weight')
            a = max(lengths0, key=lengths0.get)
            lengthsa = nx.single_source_dijkstra_path_length(G, a, weight='weight')
            b = max(lengthsa, key=lengthsa.get)

            # Use the longest path as the spline backbone.
            path = nx.shortest_path(G, a, b, weight='weight')
            P = points[path]
        except:
            # Fall back to PCA ordering for degenerate graph cases.
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(points)
            idx_sort = np.argsort(pc1[:, 0])
            P = points[idx_sort]

        # Fit the smoothing spline.
        try:
            (tck, u) = splprep(P.T, s=smooth, k=min(3, len(P) - 1))
        except:
            (tck, u) = splprep(P.T, s=smooth, k=1)

    # --- Point projection ---
    us = np.linspace(0, 1, n_eval)
    xs, ys = splev(us, tck)

    S = np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))
    S = np.concatenate([[0.0], S])
    S = (S - S.min()) / (S.max() - S.min() + 1e-12)

    XYs = np.vstack([xs, ys]).T
    tree = KDTree(XYs)
    dist, idx = tree.query(points, k=1)
    pseudotime = S[idx[:, 0]]

    # --- Direction correction ---
    is_flipped = False
    if ref_vector is not None:
        ref_vector = np.asarray(ref_vector)
        valid_idx = ~np.isnan(ref_vector)
        if np.any(valid_idx):
            corr = np.corrcoef(pseudotime[valid_idx], ref_vector[valid_idx])[0, 1]
            if not np.isnan(corr) and corr < 0:
                is_flipped = True
                pseudotime = 1 - pseudotime

    return pseudotime, tck, is_flipped
