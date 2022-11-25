import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from .FeatureConstructor import FeatureConstructor
from ..utils.GeneralHierarchy import GeneralHierarchy
from ..utils.Graph import Graph
from ..utils.atc_utils import get_atc_hrchy
from ..utils.diag_utils import get_diag_hrchy, roll_to_level
from ..utils.loinc_utils import get_loinc_hrchy
from ..utils.proc_utils import get_proc_hrchy
from ..utils.utils import get_rxnorm_atc_map
import pandas as pd
import dgl
import torch as th


class MIMICIVFeatureConstructor(FeatureConstructor):
    def __init__(self, args, conf):
        super(MIMICIVFeatureConstructor, self).__init__(args)
        self.args = args,
        self.conf = conf
        self.hrchys = dict()

    def construct_features(self, graph: Graph):
        # Extract features using feature specific functions
        for feature in self.conf['aux_features']:
            file_name = feature['data_file']
            entity = feature['entity']
            func = getattr(self, feature['function'])
            feats = func(file_name, graph)

            # Save feature in feature_map
            if entity != 'all':
                self.feature_map[entity] = feats

    def get_emb_dim(self, hrchy: GeneralHierarchy, index_leaf: bool):
        if index_leaf:
            return len(hrchy.node_dict)
        else:
            return len(hrchy.node_dict) - len(hrchy.get_code('ROOT').leaves)

    def medication_atc_hierarchy(self, file_name: str, graph: Graph) -> dict:
        index_leaf = False

        # Load external data needed to construct this feature
        atc_hierarchy = get_atc_hrchy(f'{self.path["data_fold"]}hrchy_subset_atc.csv', index_leaf)
        manual_rxnorm_atc = get_rxnorm_atc_map(f'{self.path["data_fold"]}rxnorm_atc_map.json')
        emb_dim = self.get_emb_dim(atc_hierarchy, index_leaf)

        # Load data file
        omop_rxnorm_atc = pd.read_parquet(f'{self.path["output_fold"]}/{file_name}.parquet')

        # Add manual rxnorm_atc map to the data dataframe
        manual_rxnorm_atc = DataFrame(manual_rxnorm_atc, index=None, columns=['medication', 'atc_code'])
        full_rxnorm_atc = pd.concat([omop_rxnorm_atc, manual_rxnorm_atc])

        embeddings = {}
        # For each RxNorm concept, find the corresponding ATC code embedding and convert to tensor
        for index, row in full_rxnorm_atc.iterrows():
            sparse_codes = atc_hierarchy.get_code(row['atc_code']).embedding
            one_hot = th.zeros(emb_dim, dtype=th.float32)
            one_hot[th.tensor(sparse_codes, dtype=th.int64)] = 1
            embeddings[row['medication']] = one_hot

        self.hrchys['medication'] = atc_hierarchy
        return embeddings

    def person_attributes(self, file_name: str, graph: Graph) -> dict:
        # Read parquet file
        data = pd.read_parquet(f'{self.path["output_fold"]}/{file_name}.parquet')

        # Construct person features
        data['gender_cat'] = pd.Categorical(data['gender'])
        data['ethnicity_cat'] = pd.Categorical(data['ethnicity'])
        data['race_cat'] = pd.Categorical(data['race'])
        data['gender'] = data['gender_cat'].cat.codes
        data['age'] = data['age'] / 92
        data['ethnicity'] = data['ethnicity_cat'].cat.codes
        data['race'] = data['race_cat'].cat.codes
        race_one_hot = pd.get_dummies(data.race, prefix='race')
        data = pd.concat([data, race_one_hot], axis=1)
        data = data[['person', 'gender', 'age', 'ethnicity', 'race_0', 'race_1', 'race_2', 'race_3', 'race_4', 'race_5', 'race_6', 'race_7']]

        feature_lists = data.iloc[:, 1:-1].values.tolist()

        embedding = {}
        # For each RxNorm concept, find the corresponding ATC code embedding and convert to tensor
        for person_id, features in zip(data['person'].values, feature_lists):
            embedding[person_id] = th.tensor(features, dtype=th.float32)

        return embedding

    def labtest_loinc_hierarchy(self, file_name: str, graph: Graph):
        index_leaf = False
        loinc_hrchy = get_loinc_hrchy(f'{self.path["data_fold"]}hrchy_subset_loinc.csv', index_leaf)
        emb_dim = self.get_emb_dim(loinc_hrchy, index_leaf)

        embedding = {}
        one_tensor = th.tensor([1], dtype=th.float32)
        zero_tensor = th.tensor([0], dtype=th.float32)
        # For each loinc leaf node in the hierarchy, construct an embedding
        for node in loinc_hrchy.get_code('ROOT').leaves:
            loinc_code = node.code
            sparse_codes = node.embedding
            one_hot = th.zeros(emb_dim, dtype=th.float32)
            one_hot[th.tensor(sparse_codes, dtype=th.int64)] = 1

            # Two versions of each loinc code exist, one for a normal result
            # and one for an abnormal, we create tensors for both cases
            embedding[f'{loinc_code}-N'] = th.cat((one_hot, zero_tensor), 0)
            embedding[f'{loinc_code}-A'] = th.cat((one_hot, one_tensor), 0)

        self.hrchys['labtest'] = loinc_hrchy
        return embedding

    def procedure_icd_hierarchy(self, file_name: str, graph: Graph):
        index_leaf = False
        proc_hrchy = get_proc_hrchy(f'{self.path["data_fold"]}hrchy_subset_proc.csv', index_leaf)
        emb_dim = self.get_emb_dim(proc_hrchy, index_leaf)

        embedding = {}
        # For each procedure leaf node in the hierarchy, construct an embedding
        for node in proc_hrchy.get_code('ROOT').leaves:
            diag_code = node.code
            sparse_codes = node.embedding
            one_hot = th.zeros(emb_dim, dtype=th.float32)
            one_hot[th.tensor(sparse_codes, dtype=th.int64)] = 1

            embedding[diag_code] = one_hot

        self.hrchys['procedure'] = proc_hrchy
        return embedding

    # Graphlets are expensive to compute, see the graphlet file
    # in utils for how to precompute these features for each node
    def graphlets(self, file_name: str, graph: Graph):

        graphlets = pd.read_parquet(f'{self.path["output_fold"]}/{file_name}.parquet')
        homo_graph = dgl.to_homogeneous(graph.graph)

        groups = graphlets.groupby(['src']).sum()
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(groups), columns=groups.columns, index=groups.index)

        # Calculate edge counts
        edge_counts = [len(homo_graph.out_edges(node_id.item(), form='eid')) for node_id in homo_graph.ndata['_ID']]
        df_edge_counts = pd.DataFrame(scaler.fit_transform(np.array([edge_counts]).reshape(-1, 1)))

        types = {i: ntype for i, ntype in enumerate(graph.graph.ntypes)}

        # Find embedding
        feature_dict = {ntype: dict() for ntype in types.values()}
        for node_id, ntype, reindex_id in zip(homo_graph.ndata['_ID'], homo_graph.ndata['_TYPE'], homo_graph.nodes()):
            if reindex_id.item() in df_scaled.index:
                emb = th.tensor(df_scaled.loc[reindex_id.item()].values[1:], dtype=th.float32)
                feature_dict[types[ntype.item()]][node_id.item()] = th.cat([emb, th.tensor(df_edge_counts.loc[reindex_id.item()], dtype=th.float32)], dim=0)

        # Reindex features
        ntype_dict = {'D': 'diagnosis', 'M': 'medication', 'P': 'procedure', 'S': 'person', 'L': 'labtest'}
        for ntype, feature in feature_dict.items():
            reindex_origin_map = graph.get_entity(ntype_dict[ntype]).maps['reindex-origin']
            new_feature = dict()
            for re_index, feat in feature.items():
                new_feature[reindex_origin_map[re_index]] = feat
            feature_dict[ntype] = new_feature

        for ntype, features in feature_dict.items():
            self.feature_map[ntype_dict[ntype]] = features
