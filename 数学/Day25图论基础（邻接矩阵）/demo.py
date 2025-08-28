import numpy as np
n = 5
inf = np.inf
adj_matrix = np.array([
    [0, 1, 2,inf,inf],
    [1,0,1,3,inf],
    [2,1,0,inf,inf],
    [inf,3,inf,0,1],
    [inf,inf,inf,1,0],
])
print("邻接矩阵")
print(adj_matrix)

def dijkstra(adj_matrix,start):
    n = adj_matrix.shape[0]
    dist = np.full(n,inf)
    visited = np.zeros(n,dtype=bool)
    dist[start] = 0

    for _ in range(n):
        min_dist = np.inf
        u = -1
        for v in range(n):
            if not visited[v] and dist[v] < min_dist:
                min_dist = dist[v]
                u = v
        if  u ==-1:
            break
        visited[u] = True
        for v in range(n):
            if adj_matrix[u][v] != np.inf and not visited[v]:
                new_dist = dist[u] + adj_matrix[u][v]
                if new_dist < dist[v]:
                    dist[v] = new_dist

    return dist

distances = dijkstra(adj_matrix,0)
print("\n从节点0出发的最短距离:")
for i ,d in enumerate(distances):
    print(f"到节点{i}: {d}")


def dijkstra_with_path(adj_matrix,start):
    n = adj_matrix.shape[0]
    dist = np.full(n,inf)
    visited = np.zeros(n,dtype=bool)
    prev = np.full(n,-1,dtype=int)
    dist[start] = 0
    for _ in range(n):
        min_dist = np.inf
        u= -1
        for v in range(n):
            if not visited[v] and dist[v] < min_dist:
                min_dist = dist[v]
                u = v
        if u ==-1:
            break
        visited[u] = True
        for v in range(n):
            if adj_matrix[u][v] !=np.inf and visited[v]:
                new_dist = dist[u] + adj_matrix[u][v]
                if new_dist <dist[v]:
                    dist[v] = new_dist
                    prev[v] = u

    return dist,prev

def get_path(prev,start,end):
    path = []
    at = end
    while at != -1:
        path.append(at)
        at = prev[at]
    path.reverse()
    return path if path[0] == start else []

distances,prev=dijkstra_with_path(adj_matrix,0)
path_to_4 = get_path(prev,0,4)
print(f"\n从节点0到节点4的最短路径:{'-->'.join(map(str,path_to_4))}")
print(f"路径长度:{distances}")

import networkx as nx
import matplotlib.pyplot as plt

G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
# 将 inf 边去掉
G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d['weight'] == np.inf])

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800, font_size=16)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.0f}" for k, v in edge_labels.items()})
plt.title("图结构")
plt.show()