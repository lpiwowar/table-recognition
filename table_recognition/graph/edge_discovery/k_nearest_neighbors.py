from rtree import index

from table_recognition.graph.edge_discovery.edge import Edge


class KNearestNeighbors(object):
    K_NEIGHBORS = 4

    def __init__(self, graph):
        self.graph = graph

        self.rtree_index = index.Index()
        self.rtree_index_2_node = {}

    def discover_edges(self):
        self.populate_rtree_index()

        # Discover K nearest neighbors for each node
        for node in self.graph.nodes:
            node_neighbors_idx = list(self.rtree_index.nearest(node.bbox["rtree"],
                                                               KNearestNeighbors.K_NEIGHBORS))
            self.graph.edges = self.graph.edges.union({Edge(node, self.rtree_index_2_node[neighbor_idx])
                                                       for neighbor_idx in node_neighbors_idx})
            self.graph.edges = self.graph.edges.union({Edge(self.rtree_index_2_node[neighbor_idx], node)
                                                       for neighbor_idx in node_neighbors_idx})

        # Remove reflexive edges
        self.graph.edges = {edge for edge in self.graph.edges if not edge.is_reflexive()}

    def populate_rtree_index(self):
        for node in self.graph.nodes:
            self.rtree_index_2_node[node.id] = node
            self.rtree_index.insert(node.id, node.bbox["rtree"])
