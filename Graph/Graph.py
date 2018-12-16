from queue import Queue

class Graph():
    '''
    Undirected Graph
    '''
    def __init__(self, G=None):
        if G:
            self.__nodes = G.nodes
            self.__adj = G.adj
            self.__keys = G.keys
        else:
            self.__nodes = []
            self.__keys = {}
            self.__adj = {}

    @property
    def nodes(self):
        return self.__nodes

    @property
    def adj(self):
        return self.__adj

    @property
    def keys(self):
        return self.__keys

    @property
    def max_degree(self):
        return max([len(item) for key, item in self.__adj.items()])

    @property
    def average_degree(self):
        V = self.number_of_nodes
        if V:
            return sum([len(item) for key, item in self.__adj.items()]) / V
        else:
            return 0

    @property
    def number_of_nodes(self):
        return len(self.__nodes)

    @property
    def number_of_edges(self):
        return self.average_degree * self.number_of_nodes

    @property
    def number_of_self_loops(self):
        return len([self.index_to_name(node) for node, adjacency in self.__adj.items() if node in adjacency])

    def is_in_nodes(self, node):
        return node in self.__nodes

    def node_of_index(self, index):
        return self.__nodes[index]

    def index_of_node(self, node):
        return self.__keys[node]

    def from_adjacent_dict(self, adj):
        assert isinstance(adj, dict), 'adjacent_dict type mismatch'
        self.__nodes = list(adj.keys())
        self.__keys = dict([(node, i) for i, node in enumerate(adj.keys())])
        self.__adj = dict([(self.index_of_node(node), dict([(self.index_of_node(v), w) for v, w in adjacency.items()])) \
                         for node, adjacency in adj.items()])

    def add_node(self, node):
        if node not in self.__nodes:
            self.__keys[node] = len(self.__nodes)
            self.__nodes.append(node)
            self.__adj[self.index_of_node(node)] = {}

    def add_nodes_from(self, nodes):
        for node in nodes:
            self.add_node(node)

    def adjacency(self, node, symbol=False):
        self.__check_node(node, symbol)
        if symbol:
            node = self.index_of_node(node)
        return self.__adj[node]

    def degree(self, node, symbol=False):
        self.__check_node(node, symbol)
        if symbol:
            node = self.index_of_node(node)
        return len(self.__adj[node])

    def to_string(self):
        s = '%d verticies, %d edges \n '%(self.number_of_nodes, self.number_of_edges)
        for node in self.__nodes:
            s += 'node {}: '.format(node)
            for e, w in self.__adjacency(node, symbol=True).items():
                s += '{}, '.format(self.index_to_name(e))
            s += '\n '
        return s

    def __check_node(self, node, symbol = False):
        if symbol:
            assert node in self.__nodes, "node {} does not exist".format(node)
        else:
            assert node < self.number_of_nodes, 'node does not exist'

    def add_edges_from(self, edges):
        for edge in edges:
            if len(edge) == 2:
                self.add_edge(edge[0], edge[1])
            elif len(edge) == 3:
                self.add_edge(edge[0], edge[1], edge[2])
            else:
                raise ValueError('edges shape mismatch')

    def add_edge(self, node1, node2, weight=1):
        if not node1 in self.__nodes:
            self.add_node(node1)

        if not node2 in self.__nodes:
            self.add_node(node2)

        self.__adj[self.index_of_node(node1)][self.index_of_node(node2)] = weight
        self.__adj[self.index_of_node(node2)][self.index_of_node(node1)] = weight

    def remove_node(self, node, symbol=False):
        self.__check_node(node, symbol)
        if symbol:
            node = self.index_of_node(node)

        for adj in self.__adjacency(node):
            self.__adj[adj].pop(node, None)
        self.__adj.pop(node)


    def remove_edge(self, node1, node2):
        self.__check_node(node1)
        self.__check_node(node2)

        self.__adj[node1].pop(node2, None)
        self.__adj[node2].pop(node1, None)
        
    @property
    def is_acyclic(self):
        unmarked = self.__nodes.copy()
        marked = []
        to_search = Queue(maxsize=self.number_of_nodes)
        parent = Queue(maxsize=self.number_of_nodes)
        
        initial = next(iter(unmarked))
        to_search.put(initial)
        parent.put(initial) # just put it to avoid empty queue get
        marked.append(initial)

        while unmarked:
            s = to_search.get()
            p = parent.get()
            unmarked.remove(s)
            for node in self.__adjacency(s):
                if node not in marked:
                    to_search.put(node)
                    parent.put(s)
                    marked.append(node)
                elif node != p:
                    return False
        return True


class Digraph(Graph):
    '''
    Directed Graph
    '''

    def add_edge(self, node1, node2, weight=1):
        if not node1 in self.__nodes:
            self.add_node(node1)

        if not node2 in self.__nodes:
            self.add_node(node2)
        self.__adj[self.index_of_node(node1)][self.index_of_node(node2)] = weight

    def remove_node(self, node, symbol=False):
        self.__check_node(node, symbol)
        if symbol:
            node = self.index_of_node(node)

        for adj in self.nodes:
            if self.adjacency(adj):
                self.__adj[adj].pop(node, None)
        self.__adj.pop(node)

    def remove_edge(self, node1, node2):
        self.__check_node(node1)
        self.__check_node(node2)

        self.__adj[node1].pop(node2, None)
        self.__adj[node2].pop(node1, None)

    @property
    def is_acyclic(self):
        unmarked = self.__nodes.copy()
        marked = []
        to_search = Queue(maxsize=self.number_of_nodes)
        parent = Queue(maxsize=self.number_of_nodes)

        initial = next(iter(unmarked))
        to_search.put(initial)
        parent.put(initial)  # just put it to avoid empty queue get
        marked.append(initial)

        while unmarked:
            s = to_search.get()
            p = parent.get()
            unmarked.remove(s)
            for node in self.__adjacency(s):
                if node not in marked:
                    to_search.put(node)
                    parent.put(s)
                    marked.append(node)
                elif node != p:
                    return False
        return True
