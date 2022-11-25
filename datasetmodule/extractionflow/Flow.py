import os

from datasetmodule.extractionflow.MIMICIVDataExtractor import MimicIVDataExtractor
from datasetmodule.extractionflow.MIMICIVFeatureConstructor import MIMICIVFeatureConstructor
from abc import ABC
import pandas as pd
import json

from datasetmodule.extractionflow.MIMICIVRelationConstructor import MIMICIVRelationConstructor
from datasetmodule.utils.Entity import Entity
from datasetmodule.utils.Graph import Graph
from datasetmodule.utils.Relation import Relation


class Flow(ABC):
    def __init__(self, args):
        super(Flow, self).__init__()
        self.args = args
        self.dataset = args.dataset
        self.ent_extract = args.ent_extract
        self.rel_extract = args.rel_extract
        self.feat_extract = args.feat_extract
        self.path = args.path
        self.feature_constructor = None
        self.relation_constructor = None
        self.data_extractor = None
        self.conf = None
        self.graph = None
        self.relations_set = {}
        self.indexes = {}
        self.graph = Graph()

        self._read_ext_conf()
        self._get_data_extractor()
        self._get_feature_constructor()
        self._get_relation_constructor()

    def _get_data_extractor(self):
        if self.dataset == 'mimiciv':
            self.data_extractor = MimicIVDataExtractor(
                self.args
            )

    def _get_feature_constructor(self):
        if self.dataset == 'mimiciv':
            self.feature_constructor = MIMICIVFeatureConstructor(
                self.args,
                self.conf
            )

    def _get_relation_constructor(self):
        if self.dataset == 'mimiciv':
            self.relation_constructor = MIMICIVRelationConstructor(
                self.args,
                self.conf
            )

    def extract_entities(self):
        if self.ent_extract:
            print("Extracting Entities")
            self.data_extractor.extract_entities(self.conf)

    def extract_relations(self):
        if self.rel_extract:
            print("Extracting Relations")
            self.data_extractor.extract_relations(self.conf)

    def extract_features(self):
        if self.feat_extract:
            print("Extracting Features")
            self.data_extractor.extract_features(self.conf)

    def construct_entities(self):
        for ent in self.conf['entities']:
            # Read entity information from conf
            name = ent['name']
            alias = ent['alias']

            # Create new Entity
            new_entity = Entity(name, alias)

            # Read parquet file with entity ids
            file_path = f'{self.path["output_fold"]}{name}.parquet'
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
            else:
                exit(f"Could not find entity file {name}.parquet, did you extract it?")

            # Populate the new entity with ids
            new_entity.populate(df)

            self.graph.add_entity(new_entity)

    def construct_relations(self):
        for rel in self.conf['relations']:
            # Read relation information
            file_name = rel['file_name']
            relation_name = rel['relation_name']
            direction = rel['direction']
            sub = rel['sub']
            obj = rel['obj']
            entity1 = self.graph.get_entity(sub)
            entity2 = self.graph.get_entity(obj)

            # Read parquet file with entity ids
            df = pd.read_parquet(f'{self.path["output_fold"]}{file_name}.parquet')

            # Construct new Relation
            new_relation = Relation(
                sub=entity1,
                obj=entity2,
                relation_name=relation_name,
                mapping=df,
            )

            # Set directionality
            new_relation.set_direction(direction)

            # Add relation to graph
            self.graph.add_relation(new_relation)

    def _read_ext_conf(self):
        with open(f'{self.path["config_fold"]}{self.dataset}.json') as json_file:
            self.conf = json.load(json_file)
