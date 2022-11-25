from abc import ABC
from dgl.data.utils import load_graphs


class BaseDataset(ABC):
    def __init__(self, ):
        super(BaseDataset, self).__init__()
        self.g = None

    def load_graph_from_disk(self, file_path):
        """
        load graph from disk and the file path of graph is generally stored in ``./openhgnn/dataset/``.

        Parameters
        ----------
        file_path: the file path storing the graph.bin

        Returns
        -------
        g: dgl.DGLHetrograph
        """
        g, _ = load_graphs(file_path)
        return g[0]

