class Edge(object):
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.type = None

        self.input_feature_vector = None
        self.output_feature_vector = None

    def __eq__(self, other):
        return self.node1.id == other.node1.id and \
               self.node2.id == other.node2.id

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        return f"<Edge: node1={self.node1} node2={self.node2}"

    def is_reflexive(self):
        return self.node1 == self.node2
