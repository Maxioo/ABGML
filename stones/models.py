import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import OrderedDict
from torch.nn import Parameter, BatchNorm1d
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform, glorot, zeros
from torch_geometric.utils import add_remaining_self_loops
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes

from .utils import init_weights


# Original Model
class GCN(nn.Module):
    def __init__(self, layer_sizes, batchnorm=False, batchnorm_mm=0.01, layernorm=False, weight_standardization=False):
        super().__init__()

        assert batchnorm != layernorm
        assert len(layer_sizes) >= 2
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.weight_standardization = weight_standardization

        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append((GCNConv(in_dim, out_dim), 'x, edge_index -> x'),)

            if batchnorm:
                layers.append(BatchNorm1d(out_dim, momentum=batchnorm_mm))
            else:
                layers.append(LayerNorm(out_dim))

            layers.append(nn.PReLU())

        self.model = Sequential('x, edge_index', layers)

    def forward(self, data):
        if self.weight_standardization:
            self.standardize_weights()
        return self.model(data.x, data.edge_index)

    def reset_parameters(self):
        self.model.reset_parameters()

    def standardize_weights(self):
        skipped_first_conv = False
        for m in self.model.modules():
            if isinstance(m, GCNConv):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                weight = m.lin.weight.data
                var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                m.lin.weight.data = weight

class GraphSAGE_GCN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super().__init__()

        self.convs = nn.ModuleList([
            SAGEConv(input_size, hidden_size, root_weight=True),
            SAGEConv(hidden_size, hidden_size, root_weight=True),
            SAGEConv(hidden_size, embedding_size, root_weight=True),
        ])

        self.skip_lins = nn.ModuleList([
            nn.Linear(input_size, hidden_size, bias=False),
            nn.Linear(input_size, hidden_size, bias=False),
            ])

        self.layer_norms = nn.ModuleList([
            LayerNorm(hidden_size),
            LayerNorm(hidden_size),
            LayerNorm(embedding_size),
        ])

        self.activations = nn.ModuleList([
            nn.PReLU(1),
            nn.PReLU(1),
            nn.PReLU(1),
        ])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        h1 = self.convs[0](x, edge_index)
        h1 = self.layer_norms[0](h1, batch)
        h1 = self.activations[0](h1)

        x_skip_1 = self.skip_lins[0](x)
        h2 = self.convs[1](h1 + x_skip_1, edge_index)
        h2 = self.layer_norms[1](h2, batch)
        h2 = self.activations[1](h2)

        x_skip_2 = self.skip_lins[1](x)
        ret = self.convs[2](h1 + h2 + x_skip_2, edge_index)
        ret = self.layer_norms[2](ret, batch)
        ret = self.activations[2](ret)
        return ret

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()
        for m in self.skip_lins:
            m.reset_parameters()
        for m in self.activations:
            m.weight.data.fill_(0.25)
        for m in self.layer_norms:
            m.reset_parameters()

class MetaCLASSBGRL(nn.Module):
    def __init__(self, encoder, predictor, classfier):
        super().__init__()
        # online network
        self.online_encoder = encoder
        self.predictor = predictor
        self.classfier = classfier
        # target network
        self.target_encoder = copy.deepcopy(encoder)

        # reinitialize weights
        self.target_encoder.reset_parameters()
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters()) + list(self.classfier.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        # if weights is None:
        #     weights = self.online_encoder.parameters()
        # else:
        #     weights = (param for name, param in weights if "online_encoder" in name)
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    @torch.no_grad()
    def update_target_network_meta(self, mm, weights=None):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        # assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        
        online_params = [param for name, param in weights.items() if "online_encoder" in name]
        target_params = [(name, param) for name, param in weights.items() if "target_encoder" in name]
        # print(len(online_params))
        # print(len(target_params))

        for i in range(len(online_params)):
            target_params[i][1].data.mul_(mm).add_(online_params[i].data, alpha=1. - mm)
        # for param_q, param_k in zip(online_params, target_params):
        #     param_k[1].data.mul_(mm).add_(param_q.data, alpha=1. - mm)

        for name, param in target_params:
            weights[name] = param
        return weights

    def forward(self, online_x, target_x, weights=None):
        if weights is None:
            weights = OrderedDict(self.named_parameters())
        # forward online network
        online_y = self.online_encoder(online_x, weights, 1)

        # prediction
        online_q = self.predictor(online_y, weights)

        # forward target network
        with torch.no_grad():
            target_y = self.target_encoder(target_x, weights, 0).detach()
        return online_y, online_q, target_y

    def linear(self, x, weights):
        if weights is None:
            weights = OrderedDict(self.named_parameters())
        x = F.linear(x, weights['classfier.weight'], weights['classfier.bias'])
        return x


class BGRL(torch.nn.Module):
    r"""BGRL architecture for Graph representation learning.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """
    def __init__(self, encoder, predictor):
        super().__init__()
        # online network
        self.online_encoder = encoder
        self.predictor = predictor

        # target network
        self.target_encoder = copy.deepcopy(encoder)

        # reinitialize weights
        # self.target_encoder.reset_parameters()
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, online_x, target_x):
        # forward online network
        online_y = self.online_encoder(online_x)

        # prediction
        online_q = self.predictor(online_y)

        # forward target network
        with torch.no_grad():
            target_y = self.target_encoder(target_x).detach()
        return online_q, target_y


# Meta Model
@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class MetaGCNConv(GCNConv):

    def forward(self, x, edge_index, edge_weight=None, edge_bias=None, weight=None):
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # print(weight.t().shape)
        # x = x @ weight.t()
        x = F.linear(x, weight)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if edge_bias is not None:
            out += edge_bias

        return out

class MetaGCN(nn.Module):
    def __init__(self, layer_sizes, batchnorm=False, batchnorm_mm=0.01):
        super(MetaGCN, self).__init__()

        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        # List of message-passing GNN convs
        self.conv1 = MetaGCNConv(in_channels=layer_sizes[0], out_channels=layer_sizes[1])
        # self.bn1 = BatchNorm(layer_sizes[1], momentum=batchnorm_mm)
        self.bn1 = BatchNorm1d(layer_sizes[1], momentum=batchnorm_mm)
        self.act1 = nn.PReLU()
        # self.conv2 = MetaGCNConv(in_channels=layer_sizes[1], out_channels=layer_sizes[2])
        # self.bn2 = BatchNorm(layer_sizes[2], momentum=batchnorm_mm)
        # self.bn2 = BatchNorm1d(layer_sizes[2], momentum=batchnorm_mm)
        # self.act2 = nn.PReLU()
        # self.reset_parameters()

    def forward(self, data, weights=None, on=True, training=True):
        if on:
            prefix = "online_encoder."
        else:
            prefix = "target_encoder."
        if weights is None:
            weights = OrderedDict(self.named_parameters())
            prefix = ""

        # 2 layers
        x = F.batch_norm(self.conv1(data.x, data.edge_index, data.edge_weight, weights[prefix + 'conv1.bias'], 
            weights[prefix + 'conv1.lin.weight']), self.bn1.running_mean, self.bn1.running_var, \
                                weights[prefix + 'bn1.weight'], weights[prefix+ 'bn1.bias'], training=training)
        x = F.prelu(x, weights[prefix + 'act1.weight'])

        # x = F.batch_norm(self.conv2(x, data.edge_index, data.edge_weight, weights[prefix + 'conv2.bias'],
        #     weights[prefix + 'conv2.lin.weight']), self.bn2.running_mean, self.bn2.running_var, \
        #                         weights[prefix + 'bn2.weight'], weights[prefix+ 'bn2.bias'], training=True)
        # x = F.prelu(x, weights[prefix + 'act2.weight'])

        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        # self.conv2.reset_parameters()

class MetaMLP_Predictor(nn.Module):
    r"""MLP used for predictor. The MLP has one hidden layer.

    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        self.ln1 = nn.Linear(input_size, hidden_size, bias=True)
        self.act = nn.PReLU()
        self.ln2 = nn.Linear(hidden_size, output_size, bias=True)

        self.reset_parameters()

    def forward(self, x, weight=None):
        if weight != None:
            x = F.linear(x, weight['predictor.ln1.weight'], weight['predictor.ln1.bias'])
            x = F.prelu(x, weight['predictor.act.weight'])
            x = F.linear(x, weight['predictor.ln2.weight'], weight['predictor.ln2.bias'])
        return x

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

class MetaBGRL(torch.nn.Module):
    r"""BGRL architecture for Graph representation learning.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """
    def __init__(self, encoder, predictor):
        super().__init__()
        # online network
        self.online_encoder = encoder
        self.predictor = predictor
        self.predictor.apply(init_weights)

        # target network
        self.target_encoder = copy.deepcopy(encoder)

        # reinitialize weights
        self.target_encoder.reset_parameters()
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        # if weights is None:
        #     weights = self.online_encoder.parameters()
        # else:
        #     weights = (param for name, param in weights if "online_encoder" in name)
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    @torch.no_grad()
    def update_target_network_meta(self, mm, weights=None):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        # assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        
        online_params = [param for name, param in weights.items() if "online_encoder" in name]
        target_params = [(name, param) for name, param in weights.items() if "target_encoder" in name]
        # print(len(online_params))
        # print(len(target_params))

        for i in range(len(online_params)):
            target_params[i][1].data.mul_(mm).add_(online_params[i].data, alpha=1. - mm)
        # for param_q, param_k in zip(online_params, target_params):
        #     param_k[1].data.mul_(mm).add_(param_q.data, alpha=1. - mm)

        for name, param in target_params:
            weights[name] = param
        return weights

    def forward(self, online_x, target_x, weights=None):
        if weights is None:
            weights = OrderedDict(self.named_parameters())
        # forward online network
        online_y = self.online_encoder(online_x, weights, 1)

        # prediction
        online_q = self.predictor(online_y, weights)

        # forward target network
        with torch.no_grad():
            target_y = self.target_encoder(target_x, weights, 0).detach()
        return online_q, target_y

def load_trained_encoder(encoder, ckpt_path, device):
    r"""Utility for loading the trained encoder."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(checkpoint['model'], strict=True)
    return encoder.to(device)


def compute_representations(net, dataset, device):
    r"""Pre-computes the representations for the entire dataset.

    Returns:
        [torch.Tensor, torch.Tensor]: Representations and labels.
    """
    net.eval()
    reps = []
    labels = []

    for data in dataset:
        # forward
        data = data.to(device)
        with torch.no_grad():
            reps.append(net(data, training=False))
            labels.append(data.y)

    reps = torch.cat(reps, dim=0)
    labels = torch.cat(labels, dim=0)
    return [reps, labels]