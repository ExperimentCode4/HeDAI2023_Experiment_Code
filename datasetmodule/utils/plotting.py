import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot(features, hrchys):
    edges = []
    for node in list(hrchys['labtest'].node_dict.values()):
        for child in node.children:
            edges.append((node.code, child.code))
        if None not in node.parents:
            for parent in node.parents:
                edges.append((parent.code, node.code))

    G = nx.Graph(edges)

    distances = []
    for i, source in enumerate(hrchys['labtest'].node_dict['ROOT'].leaves):
        for target in hrchys['labtest'].node_dict['ROOT'].leaves:
            if source.code != target.code:
                s_emb = features['labtest'][target.code + '-N']
                t_emb = features['labtest'][source.code + '-N']
                dist = np.linalg.norm(s_emb - t_emb)
                distances.append([source.code, target.code, len(nx.shortest_path(G, source.code, target.code)) - 1, dist])

    distances = sorted(distances, key=lambda tup: tup[2])
    x = [dist[2] for dist in distances]
    y = [dist[3] for dist in distances]

    df = pd.DataFrame(data={'Euclidean Distance': y, 'Tree Distance': x})
    df = df[df['Euclidean Distance'] != 0]

    sns.set_style("whitegrid")
    sns.set_style("ticks")
    sns.scatterplot(data=df, x="Tree Distance", y="Euclidean Distance", palette="Set2")

    plt.show()
