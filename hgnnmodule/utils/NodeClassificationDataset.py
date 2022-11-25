import torch as th

from hgnnmodule.utils.GeneralDatasets import GeneralDatasets
from hgnnmodule.utils.BaseDataset import BaseDataset


class NodeClassificationDataset(BaseDataset):
    r"""
    Description
    ------------
    Base datasets for standardized and transparent evaluation

    Dataset Name :
    dblp_subset/ ...
    """
    def __init__(self, dataset_name):
        super(NodeClassificationDataset, self).__init__()
        self.has_feature = False
        self.multi_label = False
        self.hml_loss = False
        self.extras = None
        self.g, self.category, self.num_classes = self.load_dataset(dataset_name)

    def load_dataset(self, name_dataset):

        if name_dataset == 'mimiciv':
            dataset = GeneralDatasets(name='mimiciv')
            category = 'S'
            g = dataset[0].long()
            num_classes = len(g.nodes[category].data['labels'][0])
            self.multi_label = True
            self.extras = dataset.get_extras()
        else:
            return NotImplementedError('Unsupported dataset {}'.format(name_dataset))

        return g, category, num_classes

    def get_idx(self, validation=True):
        if 'train_mask' not in self.g.nodes[self.category].data:
            print("The dataset has no train mask. "
                  "So split the category nodes randomly. And the ratio of train/test is 9:1.")
            num_nodes = self.g.number_of_nodes(self.category)
            n_test = int(num_nodes * 0.2)
            n_train = num_nodes - n_test

            train, test = th.utils.data.random_split(range(num_nodes), [n_train, n_test])
            train_idx = th.tensor(train.indices)
            test_idx = th.tensor(test.indices)
            if validation:
                random_int = th.randperm(len(train_idx))
                val_idx = train_idx[random_int[:len(train_idx) // 10]]
                train_idx = train_idx[random_int[len(train_idx) // 10:]]
            else:
                val_idx = train_idx
        else:
            train_mask = self.g.nodes[self.category].data.pop('train_mask').squeeze()
            test_mask = self.g.nodes[self.category].data.pop('test_mask').squeeze()
            train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
            test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
            if validation:
                if 'val_mask' in self.g.nodes[self.category].data:
                    val_mask = self.g.nodes[self.category].data.pop('val_mask').squeeze()
                    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
                elif 'valid_mask' in self.g.nodes[self.category].data:
                    val_mask = self.g.nodes[self.category].data.pop('valid_mask').squeeze()
                    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
                else:
                    random_int = th.randperm(len(train_idx))
                    val_idx = train_idx[random_int[:len(train_idx) // 10]]
                    train_idx = train_idx[random_int[len(train_idx) // 10:]]
            else:
                val_idx = train_idx
        return train_idx, val_idx, test_idx

    def get_labels(self):
        if 'labels' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('labels')
        elif 'label' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('label')
        else:
            raise ValueError('label in not in the hg.nodes[category].data')
        return labels
