import math
import random
import heapq

class State:

    def calculate_state_heuristic(self, dists):
        heuristic = 0
        labels = self.nodes
        for index in labels:
            if(index != labels[index]):
                heuristic += dists[index][labels[index]]
        return heuristic

    def is_goal(self):
        if self.heuristic == 0:
            return True
        return False

    def get_F(self):
        return self.heuristic + self.depth

    def swap(self, node, labels):
        temp = labels[node]
        labels[node] = 0
        labels[self.zero_node] = temp
        return labels

    def expand(self):
        for node in State.adjacency_list[self.zero_node]:
            temp_labels = (self.nodes).copy()
            temp_labels = self.swap(node, temp_labels)
            next_state = State(temp_labels, node, self.depth + 1, self)
            if (self.parent == None ):
                heapq.heappush(open_heap, (next_state.get_F(), next_state.heuristic, random.random(), next_state))
            elif not((self.parent).nodes == next_state.nodes):
                heapq.heappush(open_heap, (next_state.get_F(), next_state.heuristic, random.random(), next_state))
            if next_state.is_goal():
                return next_state

    
    def __init__(self, nodes, zero_node, depth, parent) -> None:
        self.nodes = nodes
        self.zero_node = zero_node
        self.heuristic = self.calculate_state_heuristic(State.dists)
        self.depth = depth
        self.parent = parent

def set_distances(mat, size):
    State.dists = floyd_warshall(mat, size)

def set_adjacency_matrix(mat):
    State.mat = mat

def set_adjacency_list(adj):
    State.adjacency_list = adj

def get_input():
    n, m = [int(x) for x in input().split(' ')]
    mat = [[math.inf for i in range(n)] for j in range(n)]
    adj = [[] for i in range(n)]
    for i in range(m):
        u, v = [int(x) for x in input().split(' ')]
        mat[u][v] = 1
        mat[v][u] = 1
        adj[u].append(v)
        adj[v].append(u)
    start_state_nodes = [int(x) for x in input().split(' ')]
    return mat, adj, n, m, start_state_nodes

def floyd_warshall(graph, size):
    dist = list(map(lambda i: list(map(lambda j: j, i)), graph))
    for k in range(size):
         for i in range(size):
            for j in range(size):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

def start_search(open_heap, start_state_nodes, zero_node):
    start_node = State(start_state_nodes, zero_node, 0, None)
    heapq.heappush(open_heap, (start_node.get_F(), start_node.heuristic, random.random(), start_node))

def solve_graph(open_heap, closed_list):
    while (len(open_heap) > 0):
        state = heapq.heappop(open_heap)[3]
        closed_list.append(state)
        if state.is_goal():
            return state.depth
        ans = state.expand()
        if ans != None:
            return ans.depth

if __name__ == '__main__':
    mat, adjaceny_list, n, m, start_state_nodes = get_input()
    for index in start_state_nodes:
        if start_state_nodes[index] == 0:
            zero_node = index
    set_adjacency_matrix(mat)
    set_adjacency_list(adjaceny_list)
    set_distances(mat, n)
    open_heap = []
    heapq.heapify(open_heap)
    closed_list = []
    start_search(open_heap, start_state_nodes, zero_node)
    ans = solve_graph(open_heap, closed_list)
    print(ans)