import numpy as np
from sklearn.cluster import KMeans
import time
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix


class Graph:

    def __init__(self, data_name):
        self.filename = data_name
        self.n = None
        self.k = None
        self.edges = self.form_graph()
        # self.e = None  # number of edges
        self.adj = None  # adjacency list
        self.lap = None
        self.U = None
        self.labels = None

    def form_graph(self):
        '''
        form a graph from the .txt file
        :param file: data file
        :return: graph, in the shape used latter
                n, k
        '''
        with open('./data/{}'.format(self.filename), 'r') as f:
            first_line = f.readline()[:-1]  # remove '\n' at the end
            meta = first_line.split(' ')
            yield int(meta[2]), int(meta[-1])

            for i, edge in enumerate(f.readlines()):
                s, t = edge[:-1].split(' ')
                yield int(s), int(t)

    def generate_adj(self):
        '''
        generate the adjacency matrix of a graph
        :param graph: the edges of a graph
        :param n: the number of vertices in this graph
        :return: adjacency matrix
        '''
        a = time.time()
        self.n, self.k = next(self.edges)
        adj = [set() for _ in range(self.n)]
        for s, t in self.edges:
            adj[s].add(t)
            adj[t].add(s)
        b = time.time()
        print('Generate adjacency matrix cost: {}s'.format(b-a))
        return adj

    def generate_lap(self):
        '''
        From adjacency matrix and diagonal matrix build Laplacian matrix
        :param dia: diagonal matrix
        :param adj: adjacency matrix
        :return: Laplacian matrix
        '''
        a = time.time()
        self.lap = np.ndarray((self.n, self.n))
        for i, row in enumerate(self.adj):
            row_dia = np.zeros(self.n)
            row_dia[i] = len(row)
            row_adj = [1 if j in row else 0 for j in range(self.n)]
            self.lap[i] = row_dia - row_adj
        x = np.linalg.norm(self.lap)
        self.lap = self.lap / x
        b = time.time()
        print('Genearte Laplacian matrix cost: {}s'.format(b-a))

    def get_U(self):
        '''
        Using scipy.sparse.linalg.eigs to calculate matrix U that we need for kmeans algorithm
        :param lap: laplacian matrix
        :param k: a number
        :return: matrix U
        '''
        s = time.time()
        self.lap = csr_matrix(self.lap)
        _, first_k = eigs(self.lap, self.k, sigma=0)
        U = first_k.real
        # normalize U
        x = np.linalg.norm(U)
        U = U / x
        t = time.time()
        print('Generate U cost: {}s'.format(t - s))
        return U

    def k_means(self):
        '''
        Using K-means algorithm to cluster the data
        :param data: n points
        :param k: number of clusters
        :return: clusters
        '''
        s = time.time()
        kmeans = KMeans(n_clusters=self.k, algorithm='auto')
        kmeans.fit(self.U)
        t = time.time()
        print('Run k-means algorithm cost: {}s'.format(t - s))
        return kmeans.labels_

    def write_clusters(self):
        '''
        return the clusters of vertices
        :param labels: labels generated from kmeans method
        :return: clusters
        '''
        with open('./result/{}_res.txt'.format(self.filename[:-4]), 'w') as f:
            for i, l in enumerate(self.labels):
                f.write('{} {}\n'.format(i, l))

    def main(self):
        self.adj = self.generate_adj()
        self.generate_lap()
        self.U = self.get_U()
        self.labels = self.k_means()
        self.write_clusters()


if __name__ == '__main__':
    graph = Graph('soc-Epinions1.txt')
    graph.main()

