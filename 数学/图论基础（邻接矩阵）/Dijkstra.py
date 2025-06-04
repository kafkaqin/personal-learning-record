import numpy as np

graph = np.array([
    [np.inf, 1,      4,      np.inf],  # 节点 0 出发的边
    [np.inf, np.inf, 2,      6     ],  # 节点 1 出发的边
    [np.inf, 3,      np.inf, 1     ],  # 节点 2 出发的边
    [np.inf, np.inf, np.inf, np.inf],  # 节点 3 出发的边
])

print("图的邻接矩阵：")
print(graph)

def get_path(prev,target):
    path = []
    while target!=-1:
        path.append(target)
        target = prev[target]
    return path[::-1]

def dijkstra(graph, start_node):
    n_nodes = graph.shape[0]
    dist = np.full(n_nodes, np.inf)
    visited = np.zeros(n_nodes,  dtype=bool)
    prev = np.full(n_nodes,-1, dtype=int)
    dist[start_node] = 0

    for _ in range(n_nodes):
        current  = None
        min_dist = np.inf
        for i in range(n_nodes):
            if not visited[i] and dist[i] < min_dist:
                min_dist = dist[i]
                current = i
        if current is None:
            break

        visited[current] = True

        for neighbor in range(n_nodes):
            if (not visited[neighbor] and
                graph[current][neighbor] !=np.inf and
                dist[current]+graph[current][neighbor] < dist[neighbor]):
                dist[neighbor] = dist[current]+graph[current][neighbor]
                prev[neighbor] = current

    return dist, prev

distances,predecessors = dijkstra(graph,start_node=0)
print("\n从节点 0 出发的最短路径长度：")
for i, d in enumerate(distances):
    print(f"到节点 {i} 的最短距离: {d}")