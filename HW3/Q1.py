import copy
class BN:
    def __init__(self, size) -> None:
        self.graph = dict()
        self.parents = []
        self.children = []
        for i in range(size + 1):
            self.graph[i] = list()
            self.parents.append(list())
            self.children.append(list())
        self.size = size
        self.arcs = []
        self.observed = []
        self.observed_anc = set()
        self.visited = [False] * (size+1) 

    def add_arc(self, parent, child):
        self.arcs.append((parent, child))
        self.parents[child].append(parent)
        self.children[parent].append(child)

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def observe_node(self, node):
        self.observed.append(node)

    def create_observed_anc(self):
        visited = self.observed.copy()
        while len(visited) > 0:
            visiting_node = visited.pop()
            for parent in self.parents[visiting_node] :
                self.observed_anc.add(parent)
                if (parent not in visited) :
                    visited.append(parent)

def check_independency(network, start, goal, current_path, visited):
    visited[start] = True
    current_path.append(start)
    if(len(current_path)>=3):
        if((current_path[len(current_path)-2], current_path[len(current_path)-1]) in network.arcs and (current_path[len(current_path)-3], current_path[len(current_path)-2]) in network.arcs and current_path[len(current_path)-2] in network.observed):
            current_path.pop()
            return False
        elif((current_path[len(current_path)-1], current_path[len(current_path)-2]) in network.arcs and (current_path[len(current_path)-2], current_path[len(current_path)-3]) in network.arcs and current_path[len(current_path)-2] in network.observed):
            current_path.pop()
            return False
        elif((current_path[len(current_path)-2], current_path[len(current_path)-3]) in network.arcs and (current_path[len(current_path)-2], current_path[len(current_path)-1]) in network.arcs and current_path[len(current_path)-2] in network.observed):
            current_path.pop()
            return False
        elif((current_path[len(current_path)-3], current_path[len(current_path)-2]) in network.arcs and (current_path[len(current_path)-1], current_path[len(current_path)-2]) in network.arcs and current_path[len(current_path)-2] not in network.observed and current_path[len(current_path)-2] not in network.observed_anc):
            current_path.pop()
            return False
    if start == goal:
        return True
    else:
        for v in network.graph[start]:
            if visited[v]==False:
                if check_independency(network, v, goal, current_path, visited.copy()):
                    return True
    current_path.pop()
    visited[start] = False
    return False
    

def main():
    (size, edge_count, observed_count) = [int(x) for x in input().split(" ")]
    network = BN(size)
    for _ in range(edge_count):
        (u, v) = [int(x) for x in input().split(' ')]
        network.add_edge(u, v)
        network.add_arc(u, v)
    for _ in range(observed_count):
        node = int(input())
        network.observe_node(node)
    network.create_observed_anc()
    (start, goal) = [int(x) for x in input().split(" ")]
    path = []
    visited = [False] * (size+1)
    if(check_independency(network, start, goal, path, visited)):
        for i in range(len(path)):
            path[i] = str(path[i])
        print(", ".join(path))
    else:
        print("independent")
    
main()