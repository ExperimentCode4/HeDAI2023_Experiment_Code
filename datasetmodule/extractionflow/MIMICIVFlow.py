from .Flow import Flow
from ..utils.diag_utils import rollup_code, get_rollup_map, get_diag_hrchy, rollup_tensor
import torch as th

from ..utils.plotting import plot


class MimicIVExtractor(Flow):
    r"""
    Extraction flow for mimic-iv patient graphs
    """

    def __init__(self, args):
        """
        flow for extracting mimic-iv dataset
        """
        super(MimicIVExtractor, self).__init__(args)

        # Construct icd9 hierarchy as this is often used
        self.diag_hrchy = get_diag_hrchy(f'{self.path["data_fold"]}hrchy_subset_diag.csv')
        self.diag_level = self.conf['general']['diag_level']
        self.aux_relations = self.conf['general']['aux_relations']
        self.aux_features = self.conf['general']['aux_features']
        self.feature_scale = self.conf['general']['feature_scale']
        self.plot_aux_feats = self.conf['general']['plot_aux_feats']

    def run_flow(self):
        self.construct_graph()

        # Generate data split
        self.graph.add_data_split('S', 0.8, 0.1)
        self.gen_labels('S', 'S-D')

        # Drop relations from graph
        self.graph.drop_relations([('S', 'S-D', 'D'), ('D', 'D-S', 'S')])

        # Add domain features to the graph
        self.add_aux_features()

        # Add extra information to graph
        self.graph.add_extra(self.diag_level, 'diag_level')

        # Save Graph
        self.graph.save_graph(self.path["output_fold"])

    def construct_graph(self):
        # Extract from data source
        self.extract_entities()
        self.extract_relations()
        self.extract_features()

        # Construct entities and relations
        self.construct_entities()
        self.construct_relations()

        # The original entity ids are substituted by new ids starting
        # from 0. The relation maps are updated and index-maps are
        # kept for backwards compatability
        self.graph.reindex_graph()

        # Roll to a specific code level of ICD-9
        self.aggregate_icd9(self.diag_level, 'S', 'D')

        # Add auxiliary relations to the graph
        self.add_aux_relations()

        # Create dgl graph representation
        self.graph.create_graph()

        return self.graph.graph

    def add_aux_relations(self):
        if self.aux_relations:
            self.relation_constructor.construct_relations()
            self.graph.add_aux_relations(self.relation_constructor.relation_map)

    def add_aux_features(self):
        if self.aux_features:
            self.feature_constructor.construct_features(self.graph)
            features = self.feature_constructor.feature_map
            hrchys = self.feature_constructor.hrchys

            self.graph.add_aux_features(features)
            if self.feature_scale:
                self.graph.scale_features(features, self.feature_scale, hrchys)

            if self.plot_aux_feats:
                plot(features, hrchys)

    def aggregate_icd9(self, level, sub, obj):
        # Extract relation and diagnosis entity
        relation = self.graph.relations[f'{sub}-{obj}']
        diagnosis = relation.obj

        # Create map between reindex and target level icd9 codes
        reindex_origin_map = relation.obj.maps['reindex-origin']
        origin_rollup_map = {reindex_origin_map[re_id]: rollup_code(
            reindex_origin_map[re_id],
            self.diag_hrchy,
            level) for re_id in relation.obj.ids}

        # Create new index maps
        rollup_reindex_map = {rollup_id: reindex for reindex, rollup_id in enumerate(set(origin_rollup_map.values()))}
        reindex_rollup_map = {reindex: rollup_id for rollup_id, reindex in rollup_reindex_map.items()}

        # Update entity ids
        diagnosis.ids = set(rollup_reindex_map.values())

        # Update relation mapping and drop duplicates
        relation.mapping['diagnosis'] = relation.mapping['diagnosis'].map(reindex_origin_map)
        relation.mapping['diagnosis'] = relation.mapping['diagnosis'].map(origin_rollup_map)
        relation.mapping['diagnosis'] = relation.mapping['diagnosis'].map(rollup_reindex_map)
        relation.mapping.drop_duplicates(inplace=True)

        # Update entity maps
        diagnosis.maps['reindex-origin'] = reindex_rollup_map
        diagnosis.maps['origin-reindex'] = rollup_reindex_map

    def gen_labels(self, ntype, etype):
        # Find all disease edges
        diag_edges = self.graph.graph.edges(etype=etype, form='all', order='srcdst')
        root_diseases = th.unique(self.graph.graph.edges(etype=etype, form='all')[1]).numpy()

        # Find sorted list of subjects and how many diseases they have
        persons, counts = th.unique(diag_edges[0], return_counts=True, sorted=True)

        # Find list of all diseases and disease edges sorted by subjects
        cum_indexes = th.split(diag_edges[1], list(counts.numpy()), dim=0)

        # Create initial zero indexes disease_labels
        labels = th.zeros(size=(persons.shape[0], len(root_diseases)), dtype=th.float32)

        # Populate person disease_labels based on cum_indexes
        for person_id, disease_indexes in zip(persons, cum_indexes):
            labels[person_id][disease_indexes] = 1

        # Add labels to graph
        self.graph.graph.nodes[ntype].data['labels'] = labels

        # diagnosis_map maps from the reindexed labels to the original ICD9 codes
        diagnosis_map = self.graph.entities['diagnosis'].maps['reindex-origin']

        # maps from reindex to original ICD9 code
        init_map = {it: self.diag_hrchy.get_code(diagnosis_map[it]).code for it in root_diseases}

        rollup_maps = dict()
        rollup_labels = dict()
        rollup_labels[f'l{self.diag_level}'] = labels
        for out_depth in range(self.diag_level - 1, 0, -1):

            # Find rolled up tensor and the map from indexes to ICD9 codes
            out_map = get_rollup_map(init_map, out_depth, self.diag_hrchy)

            # Add label tensor and map to dicts
            rollup_maps[f'l{out_depth}'] = out_map

            # Rollup labels
            rollup_labels[f'l{out_depth}'] = rollup_tensor(labels, out_map, th.max)

        # Add data to graph class
        self.graph.add_extra(rollup_maps, 'rollup_maps')
        self.graph.add_extra(rollup_labels, 'rollup_labels')
