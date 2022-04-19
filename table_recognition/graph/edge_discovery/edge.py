import uuid


class Edge(object):
    def __init__(self, node1, node2):
        self.uuid = uuid.uuid1()

        self.node1 = node1
        self.node2 = node2
        self.type = None

        self.input_feature_vector = None
        self.output_feature_vector = None

        # Indicates whether the edge influences the finale shape of the graph
        self.gridify_locked = False

    def __eq__(self, other):
        return self.node1.id == other.node1.id and \
               self.node2.id == other.node2.id

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        return f"({self.node1}, {self.node2})"

    def is_reflexive(self):
        return self.node1 == self.node2

    def connects(self, node1_id, node2_id):
        if node1_id == self.node1.id and node2_id == self.node2.id or \
           node1_id == self.node2.id and node2_id == self.node1.id:
            return True
        else:
            return False

    def get_connected_node(self, node_id):
        if self.node1.id == node_id:
            return self.node2
        else:
            return self.node1
