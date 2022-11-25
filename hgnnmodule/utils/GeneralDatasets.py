import os.path
import pickle

from dgl.data import DGLDataset
from dgl.data.utils import load_graphs
import shutil


class GeneralDatasets(DGLDataset):

    def __init__(self, name, force_reload=False, verbose=True):
        assert name in ['mimiciv']
        self.hgnn_data_path = './hgnnmodule/data/' + name + '/'
        self.orig_data_path = './datasetmodule/output/' + name + '/'
        self.graph = 'graph.bin'
        self.extras = 'extras.pickle'
        raw_dir = './hgnnmodule/data'
        super(GeneralDatasets, self).__init__(name=name,
                                              raw_dir=raw_dir,
                                              force_reload=force_reload,
                                              verbose=verbose)

    def process(self):
        # Copy latest graph if any exist
        if os.path.exists(self.orig_data_path + self.graph):
            shutil.copyfile(self.orig_data_path + self.graph, self.hgnn_data_path + self.graph)

        # Copy rollup indexes if any exist
        if os.path.exists(self.orig_data_path + self.extras):
            shutil.copyfile(self.orig_data_path + self.extras, self.hgnn_data_path + self.extras)
            self.extras = pickle.load(open(self.hgnn_data_path + self.extras, 'rb'))

        # process raw data to graphs, labels, splitting masks
        g, _ = load_graphs(self.hgnn_data_path + self.graph)
        self._g = g[0]

    def __getitem__(self, idx):
        # get one example by index
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        # number of data examples
        return 1

    def get_extras(self):
        return self.extras

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass
