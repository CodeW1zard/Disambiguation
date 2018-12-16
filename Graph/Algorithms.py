from collections import defaultdict
from queue import Queue

class Search():
    def __init__(self):
        self.methods = ['dfs', 'bfs']

    def search(self, G, v, method='dfs'):
        assert method in self.methods, 'unknown method'
        assert G.is_in_nodes(v), 'vertex {} is not in graph'.format(G.node_of_index(v))
        v = G.index_of_node(v)
        self.marked = [False] * G.number_of_nodes

        if method == 'dfs':
            self.__dfs(G, v)
        elif method == 'bfs':
            self.__bfs(G, v)
        return [G.node_of_index(i) for i, x in enumerate(self.marked) if x]

    def __dfs(self, G, v):
        self.marked[v] = True
        for node in G.adjacency(v):
            if not self.marked[node]:
                self.__dfs(G, node)

    def __bfs(self, G, v):
        self.marked[v] = True
        to_search = Queue(maxsize=G.number_of_nodes)
        to_search.put(v)
        while not to_search.empty():
            s = to_search.get()
            for node in G.adjacency(s):
                if not self.marked[node]:
                    self.marked[node] = True
                    to_search.put(node)

class Connectivity():
    def __init__(self, G):
        self.G = G
        self.id = [0] * G.number_of_nodes
        self.group = 0

    def connected_components(self):
        self.__compute_components()
        groups = defaultdict(list)
        for node, group in enumerate(self.id):
            groups[group].append(self.G.node_of_index(node))
        return groups

    def __compute_components(self):
        for node in range(self.G.number_of_nodes):
            if not self.id[node]:
                self.group += 1
                self.__dfs(node)

    def __dfs(self, v):
        self.id[v] = self.group
        for node in self.G.adjacency(v):
            if not self.id[node]:
                self.__dfs(node)

class Path():
    def __init__(self, G):
        self.G = G

    def path_between(self, s, v, method='bfs'):
        methods = ['dfs', 'bfs']
        assert method in methods, 'method %s does not exist'%(method)
        assert s in self.G.nodes, '{} is not in graph'.format(s)
        assert v in self.G.nodes, '{} is not in graph'.format(v)

        s = self.G.index_of_node(s)
        v = self.G.index_of_node(v)

        self.marked = [False] * self.G.number_of_nodes
        self.path_to = {}

        if method == 'dfs':
            self.__dfs(s)
        elif method == 'bfs':
            self.__bfs(s)

        assert self.is_connect(v), '{} and {} is not connected'.format(s, v)
        path = []
        while v!=s:
            path.append(v)
            v = self.path_to[v]
        path.append(s)
        path = [self.G.node_of_index(p) for p in path]
        return path[::-1]

    def __dfs(self, s):
        self.marked[s] = True
        for node in self.G.adjacency(s):
            if not self.marked[node]:
                self.path_to[node] = s
                self.__dfs(node)

    def __bfs(self, s):
        to_search = Queue(maxsize=self.G.number_of_nodes)
        to_search.put(s)
        self.marked[s] = True
        while not to_search.empty():
            v = to_search.get()
            for node in self.G.adjacency(v):
                if not self.marked[node]:
                    self.path_to[node] = v
                    self.marked[node] = True
                    to_search.put(node)


    def is_connect(self, v):
        return self.marked[v]

class Bipartition():
    def bipartite(self, G):
        self.colors = [False] * G.number_of_nodes
        self.marked = [False] * G.number_of_nodes
        self.bipartitionable = True

        self.__dfs(G, 0)
        result = []
        result.append([G.node_of_index(index) for index, color in enumerate(self.colors) if color])
        result.append([G.node_of_index(index) for index, color in enumerate(self.colors) if not color])
        return result

    def __dfs(self, G, v):
        self.marked[v] = True
        for node in G.adjacency(v):
            if not self.marked[node]:
                self.colors[node] = not self.colors[v]
                self.__dfs(G, node)
            elif self.colors[node] == self.colors[v]:
                self.bipartitionable = False

class DegreeOfSeparation():
    def degree_of_separation(self, G, s):
        self.distance = [-1] * G.number_of_nodes
        self.to_search = Queue(maxsize=G.number_of_nodes)
        self.parent = Queue(maxsize=G.number_of_nodes)

        self.to_search.put(s)
        self.parent.put(s)
        while not self.to_search.empty():
            v = self.to_search.get()
            p = self.parent.get()
            self.distance[v] = self.distance[p] + 1
            for node in G.adjacency(v):
                if self.distance[node] == -1:
                    self.to_search.put(node)
                    self.parent.put(v)
        self.distance = dict([(G.node_of_index(index), d) for index, d in enumerate(self.distance)])
        return self.distance




