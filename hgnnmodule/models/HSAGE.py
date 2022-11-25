from dgl.nn.pytorch import SAGEConv, HeteroGraphConv
from torch import nn
import torch as th


class HSAGE(nn.Module):
    def __init__(
            self,
            graph,
            feats,
            h_dim,
            out_dim,
            n_layers,
            dropout,
            device,
            loss_func=None,
            extras=None):
        super().__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.etypes = graph.etypes
        self.ntypes = graph.ntypes
        self.extras = extras
        self.loss_func = loss_func
        self.device = device
        self.feats = nn.ParameterDict()
        self.input = nn.ModuleDict({})
        self.out_layer = None

        self.preprocess(graph, feats)

        # Convolutions
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(SAGEConvLayer(
                self.h_dim,
                self.h_dim,
                self.etypes,
                self.dropout))

        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # linear output layer for base predictions
        self.set_output_layer(self.out_dim)

        # Here we modify the model if something special is required
        if self.loss_func == 'hmcn_out':
            # Here we create a layer to transform the final output
            # into hierarchical classifications (one for each hierarchy)
            self.flat_hml = nn.ModuleDict({
                name: nn.Linear(self.h_dim, len(dim)) for name, dim in self.extras['rollup_maps'].items()
            })

    def freeze_gcn_layer(self, layer):
        for param in self.layers[layer].parameters():
            param.requires_grad = False

    def set_output_layer(self, out_dim):
        self.out_layer = nn.Linear(self.h_dim, out_dim)

    def add_gcn_layer(self):
        self.layers.append(SAGEConvLayer(
            self.h_dim,
            self.h_dim,
            self.etypes,
            self.dropout))

    def preprocess(self, graph, feats):
        # Keep track of dims for subsequent linear transformations
        linear_dict = {}

        # Assign trainable embeddings
        if feats == {}:
            print("Assigning trainable embeddings to nodes")
            for ntype in graph.ntypes:
                self.feats[ntype] = self.create_type_emb(graph.number_of_nodes(ntype), self.h_dim)
                linear_dict[ntype] = [self.h_dim, self.h_dim]

        # Extract existing embeddings
        else:
            print('Embeddings already preinitialized')
            for ntype in graph.ntypes:
                if feats.get(ntype) is None:
                    self.feats[ntype] = self.create_type_emb(graph.number_of_nodes(ntype), self.h_dim)
                    linear_dict[ntype] = [self.h_dim, self.h_dim]
                else:
                    self.feats[ntype] = nn.Parameter(feats.get(ntype), requires_grad=False)
                    linear_dict[ntype] = [self.feats[ntype].shape[1], self.h_dim]

        # Type-specific linear transformations for semantic integration
        self.add_linear_trans(linear_dict)

    def add_linear_trans(self, linear_dict):
        for ntype, (in_dim, out_dim) in linear_dict.items():
            self.input[ntype] = nn.Linear(in_dim, out_dim)

    def create_type_emb(self, num_nodes, emb_dim):
        emb = nn.Parameter(th.Tensor(num_nodes, emb_dim))
        nn.init.xavier_uniform_(emb, gain=nn.init.calculate_gain('relu'))
        return emb

    def forward(self, hg, feats):

        if type(hg) != list:
            # full graph training,
            for layer_id, layer in enumerate(self.layers):
                # Do convolution
                feats = layer(hg, feats)

                # Apply activation function
                feats = {k: self.activation(v) for k, v in feats.items()}

        else:
            # minibatch training, block
            for layer, block in zip(self.layers, hg):

                # Apply convolution
                feats = layer(block, feats)

                # Apply activation function
                feats = {k: self.activation(v) for k, v in feats.items()}

        # If we need to do something special, we do it here
        if self.loss_func == 'hmcn_out':
            # Perform multiple output calculations,
            # One for each hierarchical level
            for name, layer in self.flat_hml.items():
                feats[name] = self.sigmoid(layer(feats['S']))

        # linear output layer
        feats['S'] = self.sigmoid(self.out_layer(feats['S']))

        return feats

    def freeze_parameters(self):
        for type in self.ntypes:
            self.input_feature.embed_dict[type].requires_grad = False


class SAGEConvLayer(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            etypes,
            dropout):
        super(SAGEConvLayer, self).__init__()

        # Definition of HeteroGraphConv SAGE Layer
        self.layer = HeteroGraphConv(
                {rel: SAGEConv(
                    in_feats=in_dim,
                    out_feats=out_dim,
                    feat_drop=dropout,
                    aggregator_type='mean'
                ) for rel in etypes}
            )

    def forward(self, mfg, h):
        return self.layer(mfg, h)
