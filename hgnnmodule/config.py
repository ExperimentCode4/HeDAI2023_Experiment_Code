import configparser
import torch as th
from decouple import config as get_env


class Config(object):
    def __init__(self, file_path, model, dataset, gpu):
        conf = configparser.ConfigParser()
        if gpu == -1:
            self.device = th.device('cpu')
        elif gpu >= 0 and th.cuda.is_available():
            self.device = th.device('cuda', int(gpu))
        else:
            raise ValueError("cuda is not available, please set 'gpu' -1")

        # Read Configuration File
        conf.read(file_path)

        # training dataset path
        self.seed = 0
        self.patience = 1
        self.max_epoch = 1
        self.model = model
        self.dataset = dataset
        self.path = {'output_fold': 'hgnnmodule/output/' + self.model + '/' + self.dataset + '/',
                     'input_fold': './dataset/'+self.dataset+'/',
                     'temp_fold': './output/temp/'+self.model+'/'}
        self.optimizer = 'Adam'

        # Read Logger Arguments
        self.use_logging = conf.getboolean('neptune', 'use_logging', fallback=None)
        self.neptune_project_id = conf.get('neptune', 'neptune_project_id', fallback=None)
        self.neptune_token_key = conf.get('neptune', 'neptune_token_key', fallback=None)
        self.neptune_api_token = get_env(self.neptune_token_key) if self.neptune_token_key is not None else None

        # Read general arguments
        self.experiment_name = conf.get('general', 'experiment_name', fallback=None)
        self.flow = conf.get('general', 'flow', fallback='feat_init')
        self.loss_func = conf.get("general", "loss_func", fallback='bce')
        self.optimizer = conf.get("general", "optimizer", fallback='Adam')

        if model == 'HSAGE':
            self.lr = conf.getfloat("HSAGE", "learning_rate")
            self.dropout = conf.getfloat("HSAGE", "dropout")
            self.h_dim = conf.getint("HSAGE", "h_dim")
            self.n_layers = conf.getint("HSAGE", "n_layers")

            self.max_epoch = conf.getint("HSAGE", "max_epoch")
            self.weight_decay = conf.getfloat("HSAGE", "weight_decay")
            self.seed = conf.getint("HSAGE", "seed")
            self.fanout = conf.getint("HSAGE", "fanout")
            self.patience = conf.getint("HSAGE", "patience")
            self.batch_size = conf.getint("HSAGE", "batch_size")
            self.validation = conf.getboolean("HSAGE", "validation")
            self.mini_batch_flag = conf.getboolean("HSAGE", "mini_batch_flag")
            self.use_self_loop = conf.getboolean("HSAGE", "use_self_loop")

    def __repr__(self):
        return f'[Config Info]\tModel: {self.model},\tDataset: {self.dataset}'
