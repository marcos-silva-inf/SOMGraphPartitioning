from minisom import MiniSom, fast_norm
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.metrics import davies_bouldin_score
import numpy as np
from numpy.typing import NDArray

def translate_2d_to_1d(x: int, y: int, shape: tuple[int, int]) -> int:
    n, m = shape
    return x * m + y

def translate_1d_to_2d(i: int, shape: tuple[int, int]) -> tuple[int, int]:
    n, m = shape
    return i // m, i % m

def distance_matrix(
    som: MiniSom
) -> NDArray:
    """
    Computes the neurons' distance matrix.
    """
    ws = som.get_weights()
    n, m = ws.shape[:2]
    D_matrix = np.zeros(shape=(n*m, n*m))
    ii = [[-1, 0, 0, 1]] * 2
    jj = [[0, 1, -1, 0]] * 2
    if som.topology == 'hexagonal':
        ii = [[0, 0, 1, -1, -1, 1], [0, 0, 1, -1, -1, 1]]
        jj = [[-1, 1, 0, 0, -1, -1], [-1, 1, 0, 0, 1, 1]]
    for x in range(n):
        for y in range(m):
            w_1 = ws[x, y]
            e = x % 2 != 0
            for (i, j) in zip(ii[e], jj[e]):
                if (x+i >= 0 and x+i < n and
                   y+j >= 0 and y+j < m):
                   w_2 = ws[x+i, y+j]
                   idx_1 = translate_2d_to_1d(x, y, (n, m))
                   idx_2 = translate_2d_to_1d(x+i, y+j, (n, m))
                   D_matrix[idx_1, idx_2] = fast_norm(w_1-w_2)
    return D_matrix

def prune_smallest_activity(
    som: MiniSom,
    data: NDArray,
    distance_matrix: NDArray,
    min_activity: int | float
) -> NDArray:
    """
    Eliminates the neurons with lesser activity.
    """
    activities = som.activation_response(data).flatten()
    to_drop = np.where(activities < min_activity)[0]
    mask = np.zeros(distance_matrix.shape, dtype=bool)
    mask[to_drop, :] = True
    mask[:, to_drop] = True
    distance_matrix[mask] = np.nan
    return distance_matrix

def dbi_weights(
        w1_index: tuple[int, int],
        w2_index: tuple[int, int],
        win_map: dict[tuple[int, int], list]
    ):
    """
    Compute the DBI between two vectors' associated data.
    """
    associated_data = np.vstack([
        np.array(win_map[w1_index]),
        np.array(win_map[w2_index])
    ])
    labels = np.append(
        np.ones(len(win_map[w1_index]), dtype='int'),
        np.zeros(len(win_map[w2_index]), dtype='int')
    )
    return davies_bouldin_score(associated_data, labels)

def prune_smallest_dbi(
    som: MiniSom,
    data: NDArray,
    k: int,
    mst: NDArray
) -> NDArray:
    """
    Prunes the k edges with lesser DBI
    """
    mst = mst.copy()
    non_zero = np.where(mst != 0)
    shape = som.get_weights().shape[:2]
    win_map = som.win_map(data)
    dbi_array = []
    for i, j in zip(*non_zero):
        idx1 = translate_1d_to_2d(i, shape)
        idx2 = translate_1d_to_2d(j, shape)
        # check for DBI invalid inputs
        if len(win_map[idx1]) > 1  and len(win_map[idx2]) > 1:
            x = dbi_weights(idx1, idx2, win_map), (i, j)
            dbi_array.append(x)
    dbi_array = sorted(dbi_array, key=lambda x: x[0])
    n = min(k-1, len(dbi_array))
    for i in range(n):
        _, (x, y) = dbi_array[i]
        mst[x, y] = 0
    return mst

def make_labels(
    som: MiniSom,
    data: NDArray,
    min_activity: int | float,
    mst: NDArray
) -> NDArray:
    """
    Returns the neurons' labels
    """
    response = som.activation_response(data)
    activities = response.flatten()
    to_drop = np.where(activities < min_activity)[0]
    mask = np.ones(mst.shape, dtype=bool)
    mask[to_drop, :] = False
    mask[:, to_drop] = False
    n, m = mst.shape
    amount_dropped = len(to_drop)
    mst = mst[mask].reshape(n - amount_dropped, m - amount_dropped)
    has_label = np.where(response >= min_activity)
    labels = np.zeros(response.shape) - 1
    _, components = connected_components(mst, directed=False, return_labels=True)
    labels[has_label] = components + 1
    return labels


def cluster(
    som: MiniSom,
    data: NDArray,
    k: int,
    min_activity: int | float | None = None,
    prune_activity: bool = True
) -> NDArray:
    """
    Computes the proposed clustering.
    """
    D_matrix = distance_matrix(som)
    if not prune_activity:
        if min_activity is not None:
            raise ValueError("Cannot pass min_activity argument if prune_activity is False")
        else:
            # We can prune all inactive neurons. This doesn't guarantee k clusters.
            min_activity = 0
    elif min_activity is None:
        min_activity = 0.5 * np.mean(som.activation_response(data))
    # 1st pruning: activity level
    D_matrix = prune_smallest_activity(som, data, D_matrix, min_activity)
    # 2nd pruning: Minimum Spanning Tree
    mst = minimum_spanning_tree(D_matrix, overwrite=True).toarray()
    # 3rd pruning: DBI
    mst = prune_smallest_dbi(som, data, k, mst)
    return make_labels(som, data, min_activity, mst)
