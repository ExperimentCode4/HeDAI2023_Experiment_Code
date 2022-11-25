import argparse
from hgnnmodule.config import Config
from hgnnmodule.start import hgnnmodule

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='HSAGE', type=str, help='name of models')
    parser.add_argument('--dataset', '-d', default='mimiciv', type=str, help='name of datasets')
    parser.add_argument('--gpu', '-g', default='0', type=int, help='-1 means cpu')
    parser.add_argument('--use_hpo', action='store_true', help='hyper-parameter optimization')
    args = parser.parse_args()

    config_file = ["./hgnnmodule/config.ini"]
    config = Config(file_path=config_file, model=args.model, dataset=args.dataset, gpu=args.gpu)
    config.use_hpo = args.use_hpo
    hgnnmodule(args=config)
