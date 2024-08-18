# data-handling
import pandas as pd
import numpy as np

# clustering
from minisom import MiniSom, fast_norm

# aux functions
def centroid(arr):
    return np.mean(arr, axis=0) 


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


def dist_map(weights, topology):
    um = np.nan * np.zeros((weights.shape[0], weights.shape[1], 6))  # 2 spots more for hexagonal topology
    for x in range(weights.shape[0]):
        for y in range(weights.shape[1]):
            w_2 = weights[x, y]
            ii, jj = neighbor(topology, y)
            for k, (i, j) in enumerate(zip(ii, jj)):
                if (x+i >= 0 and x+i < weights.shape[0] and y+j >= 0 and y+j < weights.shape[1]):
                    w_1 = weights[x+i, y+j]
                    um[x, y, k] = fast_norm(w_2-w_1)
    return np.nanmean(um, axis=2)


def dist_adj(w1, w2, w1_adj, w2_adj, dp) -> bool:
    return (fast_norm(w1- w2) <= dp*w1_adj) and (fast_norm(w1 - w2) <= dp*w2_adj)

#dist(c1, c2) <= dc*dist(w1, w2)
def dist_centroid(w1, w2, centroid_w1, centroid_w2, dc) -> bool:
    return (fast_norm(centroid_w1 - centroid_w2) <= dc*fast_norm(w1 - w2)) 



#criação da matriz com arestas consistentes
def consistent_matrix(data: np.ndarray, som: MiniSom, n: int, m: int, dp: float, omega: float, dc: float) -> np.ndarray:
    """
    Generate the consistent matrix of costa-netto algorithm.

    Parameters
    ----------
    data
        The data used to train the MiniSom
    som
        A trained MiniSom
    n
        n-dimension of MiniSom
    m
        m-dimension of MiniSom
    dp
        parameter of costa-netto alg.: distance between weights (2.1) 
    omega
        omega parameter of costa-netto alg.: neuron activity (2.2) 
    dc
        parameter of costa-netto alg.: distance between centroids (2.3)
    """
    map_act = som.activation_response(data)
    map_win = som.win_map(data)
    weights = som.get_weights()
    mean_dist = dist_map(weights, som.topology)
    Hmed = len(data)/(n*m)
    Hmin = Hmed*omega
    H = 0.5*Hmin
    matrix = np.zeros((n*m, n*m))

    for x in range(weights.shape[0]):
        for y in range(weights.shape[1]):
            w1 = weights[x, y]
            idx_1 = translate_2d_to_1d(x, y, (n, m))
            ii, jj = neighbor(som.topology, y)
            for (i, j) in zip(ii, jj):
                if ((x+i >= 0) and (x+i < n) and (y+j >= 0) and (y+j < m)):
                    w2 = weights[x+i, y+j]
                    if ( (dist_adj(w1, w2, mean_dist[x][y], mean_dist[x+i][y+j], dp)) and
                    (map_act[x][y] >= H or map_act[x+i][y+j] >= H ) and
                    (map_act[x][y] !=0 and map_act[x+i][y+j] !=0) and 
                    (dist_centroid(w1, w2, centroid(map_win[(x,y)]), centroid(map_win[(x+i,y+j)]), dc)) ):
                        idx_2 = translate_2d_to_1d(x+i, y+j, (n, m))
                        matrix[idx_1, idx_2] = 1
    
    return matrix



# aux functions to create labels
def dfs_costa_netto(consistent, x, y, n, m, visited, labels, label, topology) -> None:
    """
    Performs a DFS for labeling each connected component in the graph.
    """
    visited[x, y] = True
    labels[x, y] = label
    ii, jj = neighbor(topology, y)
    for (i, j) in zip(ii, jj):
        idx_1 = translate_2d_to_1d(x, y, (n, m))
        idx_2 = translate_2d_to_1d(x+i, y+j, (n, m))
        if (
            x+i >= 0 and x+i < visited.shape[0] and
            y+j >= 0 and y+j < visited.shape[1] and
            consistent[idx_1, idx_2]==1 and
            not visited[x+i, y+j]
        ):
            dfs_costa_netto(consistent, x+i, y+j, n, m, visited, labels, label, topology)
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
            if labels[x, y] == 0:
                label += 1
                dfs_costa_netto(consistent, x, y, n, m, visited, labels, label, topology)
    
    # components with less than 3 neurons are considered noise/outlier
    unique_vals, counts = np.unique(labels, return_counts=True)
    less_than_3 = unique_vals[counts < 3]
    labels[np.isin(labels, less_than_3)] = -1
    # it's possible to classify the original data points only using the win_map and the labels
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


# returns the cluster sequence in the data sequence
def label_data(som: MiniSom, data: np.ndarray, matrix: np.ndarray, n: int, m: int, forceAllNeurons = False) -> [np.ndarray, np.ndarray]: # type: ignore
    """
    Cluster data with the consistent matrix of costa-netto algorithm. Also returns neurons labels.

    Parameters
    ----------
    som
        A trained MiniSom
    data
        The data to be clusterized
    matrix
        The consistent matrix generated with the function consistent_matrix()
    n
        n-dimension of MiniSom
    m
        m-dimension of MiniSom
    """
    neuron_labels =  make_labels(matrix, som.topology, n, m)
    
    if(forceAllNeurons and (-1 in np.unique(neuron_labels))):
        neuron_labels = force_labels(som, n, m, neuron_labels)

    label_data = np.array(
        [neuron_labels[som.winner(x)] for x in data]
    ) 
    return label_data, neuron_labels