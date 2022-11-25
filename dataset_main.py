import argparse
from datasetmodule.config import Config
from datasetmodule.start import datasetmodule

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='mimiciv', type=str, help='name of datasets')
    parser.add_argument('--ent_extract', '-e', action='store_true', help='Re-extract entities?')
    parser.add_argument('--rel_extract', '-r', action='store_true', help='Re-extract relations?')
    parser.add_argument('--feat_extract', '-f', action='store_true', help='Re-extract features?')
    args = parser.parse_args()

    config_file = ["./datasetmodule/config.ini"]
    config = Config(
        file_path=config_file,
        dataset=args.dataset,
        ent_extract=args.ent_extract,
        rel_extract=args.rel_extract,
        feat_extract=args.feat_extract,
    )
    datasetmodule(args=config)
