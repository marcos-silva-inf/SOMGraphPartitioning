from minisom import MiniSom
from typing import Literal
import numpy as np
from sklearn.metrics import davies_bouldin_score
# clustering
from minisom import MiniSom, fast_norm

def translate_2d_to_1d(x: int, y: int, shape: tuple[int, int]) -> int:
    n, m = shape
    return x * m + y


def translate_1d_to_2d(i: int, shape: tuple[int, int]) -> tuple[int, int]:
    n, m = shape
    return i // m, i % m

def neighbor(topology, y):
    if (topology == 'rectangular'):
       ii = [0, 0, -1, 1]
       jj = [-1, 1, 0, 0] 
    elif ((topology == 'hexagonal') and (y % 2 == 0)):
        ii = [1, -1, 0, 0, -1, 1]
        jj = [0, 0, -1, 1, -1, -1]
    elif ((topology == 'hexagonal') and (y % 2 != 0)):
        ii = [1, -1, 0, 0, 1, 1]
        jj = [0, 0, -1, 1, -1, 1]
    return ii, jj

def dbi_weights(
        w1_index, 
        w2_index,
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



def compute_DBI(
        som: MiniSom,
        n: int,
        m: int,
        data: np.ndarray
):
    """
    Compute the DBI values and heuristic (mean of DBI).
    """
    ws = som.get_weights()
    adjacency_arr = np.zeros(shape=(n*m, n*m))
    win_map = som.win_map(data)
    
    for x in range(ws.shape[0]):
       for y in range(ws.shape[1]):
           idx_1 = translate_2d_to_1d(x, y, (n, m))
           ii, jj = neighbor(som.topology, y)
           for k, (i, j) in enumerate(zip(ii, jj)):
               if ((x+i >= 0 and x+i < ws.shape[0]) and
                   (y+j >= 0 and y+j < ws.shape[1]) and
                   (len(win_map[(x, y)]) > 1) and
                   (len(win_map[(x+i, y+j)]) > 1)):
                   idx_2 = translate_2d_to_1d(x+i, y+j, (n, m))
                   adjacency_arr[idx_1, idx_2] = dbi_weights((x, y), (x+i, y+j), win_map)
    
    
    mean_DBI = np.mean(adjacency_arr[adjacency_arr!=0])
    return adjacency_arr, mean_DBI



def identify_consistent(
        adjacency_arr,
        v
):
    """
    Compute the consistent edges.
    """
    consistent_arr = np.ones(shape=(adjacency_arr.shape[0], adjacency_arr.shape[1]))
    consistent_arr[adjacency_arr < v] = 0
    
    return consistent_arr



def dfs(consistent, x, y, n, m, visited, labels, label, topology) -> None:
    """
    Performs a DFS for labeling each connected component in the graph.
    """
    idx_1 = translate_2d_to_1d(x, y, (n, m))
    visited[x, y] = True
    labels[x, y] = label
    ii, jj = neighbor(topology, y)
    for (i, j) in zip(ii, jj):
        idx_2 = translate_2d_to_1d(x+i, y+j, (n, m))
        if (
            x+i >= 0 and x+i < visited.shape[0] and
            y+j >= 0 and y+j < visited.shape[1] and
            consistent[idx_1, idx_2]==1 and
            not visited[x+i, y+j]
        ):
            dfs(consistent, x+i, y+j, n, m, visited, labels, label, topology)
    return



def label_in_sequence(labels):
    labels_unique = np.unique(labels)
    if(-1 in labels_unique):
        labels_unique = labels_unique[1:]
    seq = [i for i in range(len(labels_unique))]
    for s, l in zip(seq, labels_unique):
        labels[labels==l] = s
    return labels



def make_labels(consistent, topology, n , m):
    labels = np.zeros(shape=(n, m), dtype=np.int32)
    visited = np.zeros(shape=(n, m), dtype=bool)
    label = 0
    for x in range(n):
        for y in range(m):
            if (labels[x, y] == 0):
                label += 1
                dfs(consistent, x, y, n, m, visited, labels,
                               label, topology)
    
    unique_vals, counts = np.unique(labels, return_counts=True)
    equal_1 = unique_vals[counts < 2]
    labels[np.isin(labels, equal_1)] = -1
    return label_in_sequence(labels)


def force_labels(som, n, m, first_labels):
    weights = som.get_weights()

    if(len(np.unique(first_labels))==1 and (-1 in first_labels)):
        print("Could not force, all neurons are undefined, network: "+str(n)+'x'+str(m))
        return first_labels

    while(-1 in np.unique(first_labels)):
        for x in range(weights.shape[0]):
            for y in range(weights.shape[1]):
                if(first_labels[x,y]==-1):
                    min_dist = 99999
                    closest_neighbor = (0,0)
                    w1 = weights[x, y]
                    ii, jj = neighbor(som.topology, y)
                    for (i, j) in zip(ii, jj):
                        if ((x+i >= 0) and (x+i < n) and (y+j >= 0) and (y+j < m) and (first_labels[x+i, y+j]!=-1)):
                            w2 = weights[x+i, y+j]
                            dist = fast_norm(w1-w2)
                            if(dist < min_dist):
                                closest_neighbor = (x+i, y+j)
                    first_labels[x,y] = first_labels[closest_neighbor[0], closest_neighbor[1]]

    return first_labels


def silva_costa(
        som: MiniSom,
        n: int,
        m: int,
        data: np.ndarray,
        adjacency_arr:  np.ndarray,
        v: float,
        forceAllNeurons = False
):
    """
    Compute the Silva-Costa clustering.

    Parameters
    ----------
    som
        A trained MiniSom
    n
        n-dimension of MiniSom
    m
        m-dimension of MiniSom
    data
        The data to be clusterized
    adjacency_arr
        The generated array with DBI connected edges from function compute_DBI()
    v
        Threshold value 
    forceAllNeurons
        If True, force all neurons to receive defined label
    """
    consistent = identify_consistent(adjacency_arr, v)
    neuron_labels = make_labels(consistent, som.topology, n, m)

    if(forceAllNeurons and (-1 in np.unique(neuron_labels))):
        neuron_labels = force_labels(som, n, m, neuron_labels)

    label_data = np.array(
        [neuron_labels[som.winner(x)] for x in data]
    ) 
    
    return label_data, neuron_labels





