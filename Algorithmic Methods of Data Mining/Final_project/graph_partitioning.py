import numpy as np
from sklearn.cluster import KMeans
import time
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix


def form_graph(filename):
    '''
    form a graph from the .txt file
    :param file: data file
    :return: graph, in the shape used latter
            n, k
    '''
    with open('./data/{}'.format(filename), 'r') as f:
        first_line = f.readline()[:-1]  # remove '\n' at the end
        meta = first_line.split(' ')
        n, e, k = int(meta[2]), int(meta[3]), int(meta[-1])

        lines = f.readlines()

    graph = np.ndarray((e, 2), dtype=np.int32)
    for i, edge in enumerate(lines):
        s, t = edge[:-1].split(' ')
        graph[i] = int(s), int(t)

    return graph, n, k


def generate_adj(graph, n):
    '''
    generate the adjacency matrix of a graph
    :param graph: the edges of a graph
    :param n: the number of vertices in this graph
    :return: adjacency matrix
    '''
    adj = np.zeros((n, n), dtype=np.int32)
    for s, t in graph:
        adj[s, t] = adj[t, s] = 1
    return adj


def generate_dia(adj, n):
    '''
    From adjacency matrix build diagonal matrix
    :param adj: adjacency matrix, a ndarray
    :param n: the number of vertices in this graph
    :return: diagonal matrix
    '''
    dia = np.zeros((n, n), dtype=np.int32)
    for i, row in enumerate(adj):
        dia[i][i] = sum(row)
    return dia


def generate_lap(dia, adj):
    '''
    From adjacency matrix and diagonal matrix build Laplacian matrix
    :param dia: diagonal matrix
    :param adj: adjacency matrix
    :return: Laplacian matrix
    '''
    lap = dia - adj
    # normalize lap
    x = np.linalg.norm(lap)
    lap = lap / x
    return lap


def compute_k_eigenvectors(lap, k):
    '''compute the first k eigenvectors of laplacian matrix
    :param lap: laplacian matrix
    :param k: a number
    :return: The normalized (first k) eigenvectors
    '''
    _, vectors = np.linalg.eig(lap)
    vectors = vectors.real

    return vectors[:k]


def get_U(lap, k):
    '''
    Using scipy.sparse.linalg.eigs to calculate matrix U that we need for kmeans algorithm
    :param lap: laplacian matrix
    :param k: a number
    :return: matrix U
    '''
    s = time.time()
    lap = csr_matrix(lap)
    _, first_k = eigs(lap, k, sigma=0)
    U = first_k.real
    # normalize U
    x = np.linalg.norm(U)
    U = U / x
    t = time.time()
    print(t - s)
    return U


def generate_u(vec_k):
    '''
    from first k vectors generate matrix U
    :param vec_k: first k eigenvectors
    :return: matrix U, using rows of vec_k as columns
    '''
    u = vec_k.T
    x = np.linalg.norm(u)
    u = u / x
    return u


def k_means(data, k):
    '''
    Using K-means algorithm to cluster the data
    :param data: n points
    :param k: number of clusters
    :return: clusters
    '''
    s = time.time()
    kmeans = KMeans(n_clusters=k, algorithm='auto')
    kmeans.fit(data)
    t = time.time()
    print(t-s)
    return kmeans.labels_


def get_clusters(labels, k):
    '''
    return the clusters of vertices
    :param labels: labels generated from kmeans method
    :return: clusters
    '''
    clusters = [[] for _ in range(k)]
    for i, l in enumerate(labels):
        clusters[l].append(i)
    return clusters


def partitioning(filename):
    s = time.time()
    graph, n, k = form_graph(filename)
    adj = generate_adj(graph, n)
    dia = generate_dia(adj, n)
    lap = generate_lap(dia, adj)
    vec_k = compute_k_eigenvectors(lap, k)
    data = generate_u(vec_k)
    labels = k_means(data, k)
    clusters = get_clusters(labels, k)
    print(clusters)
    t = time.time()
    print(t - s)
    return clusters


def partitioning_1(filename):
    s = time.time()
    graph, n, k = form_graph(filename)
    adj = generate_adj(graph, n)
    dia = generate_dia(adj, n)
    lap = generate_lap(dia, adj)
    data = get_U(lap, k)
    labels = k_means(data, k)
    clusters = get_clusters(labels, k)
    t = time.time()
    print(t - s)
    return clusters


if __name__ == '__main__':
    label2 = partitioning_1('soc-Epinions1.txt')
    for row in label2:
        print(row)
        print(len(row))
