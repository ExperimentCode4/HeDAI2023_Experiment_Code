import pandas as pd
import networkx as nx


def reset_node_id():
    Node.NODE_ID = 0


class Node:
    NODE_ID = 0

    def __init__(self, code):
        self.code = code
        self.children = []
        self.embedding = []
        self.parent = None
        self.descr = None
        self.depth = None
        self.index = None

    def set_index(self):
        self.index = Node.NODE_ID
        Node.NODE_ID += 1

    def add_child(self, child):
        if child not in self.children:
            self.children.append(child)

    def remove_child(self, child):
        if child in self.children:
            self.children.remove(child)

    def set_parent(self, parent):
        self.parent = parent

    def is_leaf(self):
        if self.children:
            return False
        else:
            return True

    def get_parent(self):
        return self.parent

    def add_depth(self):
        depth = self.parent.depth
        if depth is not None:
            self.depth = depth + 1
            return self.depth
        else:
            self.depth = self.parent.add_depth() + 1
            return self.depth

    @property
    def leaves(self):
        leaves = set()
        if not self.children:
            return [self]
        for child in self.children:
            leaves.update(child.leaves)
        return list(leaves)

    def __str__(self):
        return f'{self.code}'

    def __hash__(self):
        return hash(str(self))


class GeneralHierarchy:
    def __init__(self, codesfile, index_leaf=False):
        # Reset Node ID
        reset_node_id()

        # Create empty Node Dict
        self.node_dict = dict()

        # Construct root node
        name = 'ROOT'
        root_node = Node(name)
        root_node.depth = 0
        root_node.descr = 'Root node'
        self.node_dict[name] = root_node
        self.graph = None

        # Read codes file
        # A valid codes file have the specific format of
        # CHILD, PARENT, optional(descr)
        codes = pd.read_csv(codesfile, delimiter=';')

        # Construct the tree
        for _, row in codes.iterrows():
            child = row['CHILD']
            parent = row['PARENT']
            descr = row['DESCR'] if hasattr(row, 'DESCR') else None
            self.add_pair(child, parent, descr)

        # Add depth to nodes
        self.add_depth()

        # Add index to each node
        self.index_tree(index_leaf)

        self.generate_graph()

        # Construct embeddings
        self.add_emb()

    def add_pair(self, child, parent, descr):

        # Create the parent nodes if it does not exist
        if parent not in self.node_dict:
            parent_node = Node(parent)
            self.node_dict[parent] = parent_node
        else:
            parent_node = self.node_dict[parent]

        # Create the parent nodes if it does not exist
        if child not in self.node_dict:
            child_node = Node(child)
            self.node_dict[child] = child_node
        else:
            child_node = self.node_dict[child]

        # Connect the child to the parent
        child_node.set_parent(parent_node)
        parent_node.add_child(child_node)
        child_node.descr = descr

    def add_depth(self):
        for leaf in self.node_dict['ROOT'].leaves:
            leaf.add_depth()

        # Some nodes may be isolated, we initialize their depth too
        for node in self.node_dict.values():
            if not node.depth:
                node.depth = 0

    def add_emb(self):
        for node in self.node_dict.values():
            path = nx.shortest_path(self.graph, 'ROOT', node.code)
            for step in path:
                step = self.node_dict[step]
                if step.index is not None and step.index not in node.embedding:
                    node.embedding.append(step.index)

    def index_tree(self, index_leaf):
        for node in self.node_dict.values():
            if not index_leaf and node.is_leaf():
                continue
            else:
                node.set_index()

    def get_max_depth(self):
        return max([node.depth for node in self.node_dict.values()])

    def get_nodes_at_depth(self, depth):
        return [node for node in self.node_dict.values() if node.depth == depth]

    def get_nodes_below_depth(self, depth):
        return [node for node in self.node_dict.values() if node.depth <= depth]

    def get_code(self, code):
        if code in self.node_dict:
            return self.node_dict[code]
        else:
            return None

    def get_node_by_index(self, index):
        for code, node in self.node_dict.items():
            if node.index == index:
                return node
        return None

    def generate_graph(self):
        edges = []
        for node in list(self.node_dict.values()):
            for child in node.children:
                edges.append((node.code, child.code))
            if node.parent is not None:
                edges.append((node.parent.code, node.code))

        self.graph = nx.Graph(edges)
