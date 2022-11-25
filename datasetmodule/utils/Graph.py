import pickle

import numpy as np
import pandas as pd
import torch as th

from datasetmodule.utils.Entity import Entity
from datasetmodule.utils.Relation import Relation
import dgl


class Graph:
    def __init__(self):
        self.entities = dict()
        self.relations = dict()
        self.graph = None
        self.extras = dict()

    def add_extra(self, extra, name):
        self.extras[name] = extra

    def add_relation(self, relation):
        if type(relation) != Relation:
            exit(f'Only of type Relation can be added to the list of entities')
        elif relation.relation_name in self.entities:
            exit(f'Relation with name {relation.relation_name} already exists')
        else:
            self.relations[relation.relation_name] = relation

    def add_entity(self, entity):
        if type(entity) != Entity:
            exit(f'Only of type Entity can be added to the list of entities')
        elif entity.name in self.entities:
            exit(f'Entity with name {entity.name} already exists')
        else:
            self.entities[entity.name] = entity

    def get_entity(self, name):
        if name in self.entities:
            return self.entities[name]
        else:
            exit(f'Entity with name {name} does not exist. '
                 f'Did you add it to the configuration?')

    def reindex_graph(self):
        # For each entity, find the relations using the entity as either sub or obj.
        # Find the set of all entity ids and create mapping for backward compatability.
        for entity_name in self.entities.keys():
            ids = set()
            for relation in self.relations.values():
                # If entity used in relation it should be re-indexed
                if entity_name in [relation.sub.name, relation.obj.name]:
                    ids = ids.union(set(relation.mapping[entity_name].unique()))

            # Reindex the entity
            self.entities[entity_name].reindex(ids)

        # Reindex relation mappings
        for relation in self.relations.values():
            sub_name = relation.sub.name
            obj_name = relation.obj.name
            relation.mapping[sub_name] = relation.mapping[sub_name].map(relation.sub.maps['origin-reindex'])
            relation.mapping[obj_name] = relation.mapping[obj_name].map(relation.obj.maps['origin-reindex'])

    def create_graph(self):
        graph_relations = dict()

        # Construct relations for subsequent graph construction
        print(f'Adding data relations to graph')
        for relation in [rel for rel in self.relations.values() if rel.aux is False]:
            graph_relations.update(relation.construct_graph_relations())
        # Add auxiliary relations to graph
        aux_rels = [rel for rel in self.relations.values() if rel.aux is True]
        if len(aux_rels) > 0:
            print(f'\nAdding aux relations to graph')
            for relation in aux_rels:
                graph_relations.update(relation.construct_graph_relations())

        # Construct relations for subsequent graph construction
        self.graph = dgl.heterograph(graph_relations)
        print(f'\nCreated DGL graph')

    def change_relation_direction(self, relation_name: str, direction: str):
        if relation_name in self.relations:
            self.relations[relation_name].change_direction(direction)
        else:
            print(f'Relation {relation_name} not in dict of relations')

    def drop_relations(self, canonical_etypes: list):
        for canonical_etype in canonical_etypes:
            self.drop_relation(canonical_etype)
            print(f'Dropped relation {canonical_etype} from graph')

        # Newline print for beautiful output
        print()

    def drop_relation(self, canonical_etype: str):
        edge_dict = {c_type: self.graph.edges(etype=c_type, form='eid')
                     for c_type in self.graph.canonical_etypes}

        edge_dict.pop(canonical_etype, None)
        self.graph = dgl.edge_subgraph(self.graph, edges=edge_dict, relabel_nodes=False)

    def add_data_split(self, ntype: str, train_size: float, eval_size: float):

        # Select all ids
        all_ids = self.graph.nodes(ntype).numpy()

        # Shuffle the dataset
        np.random.shuffle(all_ids)

        # Split into Train, Eval and Test sets
        train_len = int(len(all_ids) * train_size)
        eval_len = int(len(all_ids) * (train_size + eval_size))
        train_ids = all_ids[:train_len]
        eval_ids = all_ids[train_len: eval_len]
        test_ids = all_ids[eval_len:]

        print(f'\nTrain nids: {train_ids.shape[0]}')
        print(f'Evaluation nids: {eval_ids.shape[0]}')
        print(f'Test nids: {test_ids.shape[0]}\n')

        # Add boolean masks to graph for indicating data splits
        for mask, ids in zip(['train_mask', 'val_mask', 'test_mask'], [train_ids, eval_ids, test_ids]):
            bool_mask = th.zeros(len(all_ids))
            bool_mask[ids] = 1
            self.graph.nodes[ntype].data[mask] = bool_mask

    def add_aux_relations(self, relations: dict):
        # Add each feature to the graph
        for (sub, obj), relation in relations.items():
            if sub not in self.entities or obj not in self.entities:
                print(f'Entity {sub} or {obj} not in graph, can\'t add relation')
            else:
                self.add_aux_relation(sub, obj, relation[0], relation[1])

    def add_aux_relation(self, sub, obj, pairs, direction):
        sources, targets = [], []

        subject_map = self.entities[sub].maps['origin-reindex']
        object_map = self.entities[obj].maps['origin-reindex']

        for s, o in pairs:
            if s in subject_map and o in object_map:
                sources.append(subject_map[s])
                targets.append(object_map[o])

        df = pd.DataFrame({sub: sources, obj: targets})
        sub_alias = self.entities[sub].alias
        obj_alias = self.entities[obj].alias

        # Construct new Relation
        new_relation = Relation(
            sub=self.entities[sub],
            obj=self.entities[obj],
            relation_name=f'{sub_alias}-{obj_alias}',
            mapping=df,
            aux=True
        )

        new_relation.set_direction(direction)

        self.add_relation(new_relation)

    def scale_features(self, features, feature_scale, hrchys):
        for entity, feature in features.items():
            if self.is_entity_in_graph(entity):
                if feature_scale == 'naive':
                    if entity in hrchys:
                        self.naive_scale(entity, feature, hrchys[entity])
            else:
                print(f'Entity {entity} not in graph, can\'t add features')

    def naive_scale(self, entity, feature, hrchy):
        scale_tensor = th.tensor([], dtype=th.float32)
        max_depth = hrchy.get_max_depth()
        num_nodes = len(hrchy.node_dict)
        num_feats = next(iter(feature.values())).shape[0]
        alias = self.entities[entity].alias

        for i in range(0, num_nodes):
            node = hrchy.get_node_by_index(i)
            scale_tensor = th.cat((scale_tensor, th.tensor([node.depth / max_depth], dtype=th.float32)), 0)

        # Lab tests have one feature more than other embeddings,
        # Since one feature represents the case of a normal/abnormal result
        for _ in range(0, num_feats - num_nodes):
            scale_tensor = th.cat((scale_tensor, th.tensor([1])))

        self.graph.nodes[alias].data['h'] = self.graph.nodes[alias].data['h'] * scale_tensor[None, :]

    def add_aux_features(self, features: dict):
        # Add each feature to the graph
        print("Adding aux features to graph")
        for entity, feature in features.items():
            if self.is_entity_in_graph(entity):
                self.add_aux_feature(entity, feature)
            else:
                print(f'Entity {entity} not in graph, can\'t add features')

    def add_aux_feature(self, entity: str, feature: dict):
        # Get entity name alias for which we want to add features
        alias = self.entities[entity].alias
        entity_ids = self.entities[entity].ids
        zero_string = ""

        # Map features dict to new keys using the entity origin-reindex map
        # This also automatically drops non_mappable rows
        origin_reindex = self.entities[entity].maps['origin-reindex']
        feature = {origin_reindex[key]: val for key, val in feature.items() if key in origin_reindex}

        # Check that we have a feature for each entity sample
        if len(feature) != len(entity_ids):

            # If we are missing features for some entities, we zero initialize them
            if len(feature) < len(entity_ids):
                zero_string = f'with {len(entity_ids) - len(feature)} zero initialized'
                feature = self.zero_initialize_missing_features(feature, entity_ids)

            # If we have too many features for some entities, we can't continue
            if len(feature) > len(entity_ids):
                exit('Num feature vectors and num of entities do not correspond, can\'t continue')

        # Sort dataframe
        feature = {key: feature[key] for key in sorted(feature.keys())}

        # Add features to graph
        self.graph.nodes[alias].data['h'] = th.stack(list(feature.values()))
        print(f'Added {feature[0].shape[0]} dim {entity} feats to graph {zero_string}')

    def zero_initialize_missing_features(self, feature: dict, entity_ids: set) -> dict:
        # Create zero tensor of same length as the other embeddings
        zero_tensor = th.zeros_like(list(feature.items())[0][1], dtype=th.float32)

        # For entities with no embedding, we create a zero initialized one
        for entity_id in entity_ids:
            if entity_id not in feature:
                feature[entity_id] = zero_tensor

        return feature

    def save_graph(self, out_folder: str):
        # Create Heterogeneous graph and save it
        dgl.save_graphs(f'{out_folder}/graph.bin', [self.graph])

        # Pickle extra data related to evaluation
        # and other usages regarding this graph
        if self.extras:
            pickle.dump(self.extras, open(f'{out_folder}/extras.pickle', 'wb'))

        # print saved graph
        print('\n----------------- Graph Statistics -----------------')
        print(f'\n{self.graph}')

    def is_entity_in_graph(self, entity):
        if entity in self.entities:
            return True
        else:
            return False
