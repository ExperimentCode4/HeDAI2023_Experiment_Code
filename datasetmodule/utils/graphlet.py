from datasetmodule.extractionflow.MIMICIVFlow import MimicIVExtractor
from datasetmodule.utils import set_random_seed
from sklearn.preprocessing import MinMaxScaler
from datasetmodule.config import Config
import pandas as pd
import dgl
import os


def extract_graph_edges():
    # Setup Configuration
    config_file = ["./datasetmodule/config.ini"]
    config = Config(
        file_path=config_file,
        dataset='mimiciv',
        ent_extract=False,
        rel_extract=False,
        feat_extract=False,
        path='/srv/data/graph_based_hml_loss'
    )

    config.seed = 0
    set_random_seed(config.seed)

    flow = MimicIVExtractor(config)

    graph = flow.construct_graph()

    # Convert dgl graph to homogeneous graph
    homo_graph = dgl.to_homogeneous(graph)

    num_edges = homo_graph.num_edges()
    num_nodes = homo_graph.num_nodes()

    # Print sparse graph representation
    file_path = os.path.join(config.path['data_fold'], 'sparse.csv')
    with open(file_path, 'w') as fp:

        # Writing header information
        fp.write(f'{num_nodes} {num_nodes} {num_edges}\n')

        # Writing edge information
        for i, (s, t) in enumerate(
                zip(homo_graph.edges(form='uv')[0].tolist(), homo_graph.edges(form='uv')[1].tolist())):
            fp.write(f'{s} {t}\n')

    print(f'Graph edges extracted to file {file_path}')


def convert_sparse_to_parquet():
    file_path = os.path.join(os.getcwd(), '../data/mimiciv', 'sparse.micro')
    df = pd.read_csv(file_path)

    out_path = os.path.join(os.getcwd(), '../output/mimiciv/', 'graphlet_features.parquet')
    df.to_parquet(out_path)
