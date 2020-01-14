import random
from find_mst import minimum_spanning_tree


def ca_tsp(data):
    # build a graph
    G = form_graph(data)
    # create a minimum spanning tree T of G.
    T = minimum_spanning_tree(G)
    # find odd vertices
    odd_vertices = find_odd_vertices(T)
    # find a minimum-weight perfect matching
    min_weight_m(T, G, odd_vertices)
    # form an Eulerian path
    eul = find_eul_path(T)
    # form path
    path, length = form_path(G, eul)
    print("Result path: ", path)
    print("Total distance of the path: ", length)

    return path, length


def get_distance(x1, y1, x2, y2):
    '''
    calculate the distance from two points by Pythagorean theorem
    :return: distance of two points
    '''
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def form_graph(data):
    '''
    :param data: a set of vertices, each vertex include its x and y position.
    :return: a graph
    '''
    graph = {}
    le = len(data)
    for cur in range(le):
        for nex in range(le):
            if cur != nex:
                if cur not in graph:
                    graph[cur] = {}
                graph[cur][nex] = get_distance(data[cur][0], data[cur][1], data[nex][0], data[nex][1])
    return graph


def find_odd_vertices(T):
    tmp_g = dict()
    vertices = list()
    for edge in T:
        if edge[0] not in tmp_g:
            tmp_g[edge[0]] = 0

        if edge[1] not in tmp_g:
            tmp_g[edge[1]] = 0

        tmp_g[edge[0]] += 1
        tmp_g[edge[1]] += 1

    for vertex in tmp_g:
        if tmp_g[vertex] % 2 == 1:
            vertices.append(vertex)

    return vertices


def min_weight_m(T, G, odd_vert):
    random.shuffle(odd_vert)

    while odd_vert:
        v = odd_vert.pop()
        length = float("inf")
        u = 1
        closest = 0
        for u in odd_vert:
            if v != u and G[v][u] < length:
                length = G[v][u]
                closest = u

        T.append((v, closest, length))
        odd_vert.remove(closest)


def find_eul_path(MT):
    neighbours = {}
    for edge in MT:
        if edge[0] not in neighbours:
            neighbours[edge[0]] = []

        if edge[1] not in neighbours:
            neighbours[edge[1]] = []

        neighbours[edge[0]].append(edge[1])
        neighbours[edge[1]].append(edge[0])

    # find the hamiltonian circuit
    start_point = MT[0][0]
    EP = [neighbours[start_point][0]]

    while len(MT) > 0:
        for i, v in enumerate(EP):
            if len(neighbours[v]) > 0:
                break

        while len(neighbours[v]) > 0:
            w = neighbours[v][0]

            remove_edge(MT, v, w)

            del neighbours[v][(neighbours[v].index(w))]
            del neighbours[w][(neighbours[w].index(v))]

            i += 1
            EP.insert(i, w)

            v = w

    return EP


def remove_edge(MT, v1, v2):
    for i, item in enumerate(MT):
        if (item[0] == v2 and item[1] == v1) or (item[0] == v1 and item[1] == v2):
            MT.pop(i)

    return MT


def form_path(G, eul):
    current = eul[0]
    path = [current]
    visited = [False] * len(eul)
    visited[0] = True

    length = 0

    for v in eul[1:]:
        if not visited[v]:
            path.append(v)
            visited[v] = True

            length += G[current][v]
            current = v

    return path, length


if __name__ == '__main__':
    data = [[0, 1], [3, 10], [6, 8], [1, 4], [2, 3], [18, 9], [23, 2], [31, 61], [18, 7], [10, 9]]
    ca_tsp(data)
