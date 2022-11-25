from abc import abstractmethod
import json


class RelationConstructor:
    def __init__(
            self,
            args
    ):
        self.args = args
        self.path = args.path
        self.dataset = args.dataset
        self._read_ext_conf()
        self.relation_map = dict()

    @abstractmethod
    def construct_relations(self):
        pass

    def _read_ext_conf(self):
        with open(f'{self.path["config_fold"]}{self.dataset}.json') as json_file:
            self.conf = json.load(json_file)
