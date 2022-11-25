from decouple import config as get_env
import configparser
import os


class Config():
    def __init__(self, file_path, dataset, rel_extract=False, feat_extract=False, ent_extract=False, path=os.getcwd()):
        conf = configparser.ConfigParser()

        try:
            conf.read(file_path)
        except:
            print("Could not read dataset module config.ini file")

        # paths
        self.dataset = dataset
        self.rel_extract = rel_extract
        self.feat_extract = feat_extract
        self.ent_extract = ent_extract
        self.path = {'dataset_fold': './output/datasets/',
                     'input_fold': './dataset/' + self.dataset + '/',
                     'config_fold': path + '/datasetmodule/extractionflow/conf/' + self.dataset + '/',
                     'output_fold': path + '/datasetmodule/output/' + self.dataset + '/',
                     'data_fold': path + '/datasetmodule/data/' + self.dataset + '/'}

        # Read Logger Arguments
        self.use_logging = conf.getboolean('neptune', 'use_logging', fallback=None)
        self.neptune_project_id = conf.get('neptune', 'neptune_project_id', fallback=None)
        self.neptune_token_key = conf.get('neptune', 'neptune_token_key', fallback=None)
        self.neptune_api_token = get_env(self.neptune_token_key) if self.neptune_token_key is not None else None

        if dataset == "mimiciv":
            self.conn_type = conf.get('mimiciv', 'conn_type', fallback=None)
            self.project_id = conf.get('mimiciv', 'project_id', fallback=None)

            self.server = conf.get('mimiciv', 'server', fallback=None)
            self.database = conf.get('mimiciv', 'database', fallback=None)
            self.user = conf.get('mimiciv', 'user', fallback=None)
            self.password_key = conf.get('mimiciv', 'password_key', fallback=None)
            self.password = get_env(self.password_key) if self.password_key is not None else None
            self.port = conf.get('mimiciv', 'port', fallback='5432')

    def __repr__(self):
        return f'[Config Info]\tDataset: {self.dataset}'
