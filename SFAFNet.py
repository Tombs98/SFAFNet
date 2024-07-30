from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.arch_utils import LayerNorm2d, MySequential
# import dgl
import enum

# class LayerType(enum.Enum):
#     IMP1 = 0,
#     IMP2 = 1,
#     IMP3 = 2

# def get_layer_type(layer_type):
#     assert isinstance(layer_type, LayerType), f'Expected {LayerType} got {type(layer_type)}.'

#     if layer_type == LayerType.IMP3:
#         return GATLayerImp3
#     else:
#         raise Exception(f'Layer type {layer_type} not yet supported.')


# class GAT(torch.nn.Module):
#     """
#     I've added 3 GAT implementations - some are conceptually easier to understand some are more efficient.

#     The most interesting and hardest one to understand is implementation #3.
#     Imp1 and imp2 differ in subtle details but are basically the same thing.

#     Tip on how to approach this:
#         understand implementation 2 first, check out the differences it has with imp1, and finally tackle imp #3.

#     """

#     def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
#                  dropout=0.6, layer_type=LayerType.IMP3, log_attention_weights=False):
#         super().__init__()
#         assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

#         GATLayer = get_layer_type(layer_type)  # fetch one of 3 available implementations
#         num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below

#         gat_layers = []  # collect GAT layers
#         for i in range(num_of_layers):
#             layer = GATLayer(
#                 num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
#                 num_out_features=num_features_per_layer[i+1],
#                 num_of_heads=num_heads_per_layer[i+1],
#                 concat=True if i < num_of_layers - 1 else False,  # last GAT layer does mean avg, the others do concat
#                 activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
#                 dropout_prob=dropout,
#                 add_skip_connection=add_skip_connection,
#                 bias=bias,
#                 log_attention_weights=log_attention_weights
#             )
#             gat_layers.append(layer)

#         self.gat_net = nn.Sequential(
#             *gat_layers,
#         )

#     # data is just a (in_nodes_features, topology) tuple, I had to do it like this because of the nn.Sequential:
#     # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
#     def forward(self, data):
#         return self.gat_net(data)


# class GATLayer(torch.nn.Module):
#     """
#     Base class for all implementations as there is much code that would otherwise be copy/pasted.

#     """

#     head_dim = 1

#     def __init__(self, num_in_features, num_out_features, num_of_heads, layer_type, concat=True, activation=nn.ELU(),
#                  dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

#         super().__init__()

#         # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
#         self.num_of_heads = num_of_heads
#         self.num_out_features = num_out_features
#         self.concat = concat  # whether we should concatenate or average the attention heads
#         self.add_skip_connection = add_skip_connection

#         #
#         # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
#         # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
#         #

#         if layer_type == LayerType.IMP1:
#             # Experimenting with different options to see what is faster (tip: focus on 1 implementation at a time)
#             self.proj_param = nn.Parameter(torch.Tensor(num_of_heads, num_in_features, num_out_features))
#         else:
#             # You can treat this one matrix as num_of_heads independent W matrices
#             self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

#         # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
#         # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

#         # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
#         # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
#         self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
#         self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

#         if layer_type == LayerType.IMP1:  # simple reshape in the case of implementation 1
#             self.scoring_fn_target = nn.Parameter(self.scoring_fn_target.reshape(num_of_heads, num_out_features, 1))
#             self.scoring_fn_source = nn.Parameter(self.scoring_fn_source.reshape(num_of_heads, num_out_features, 1))

#         # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
#         if bias and concat:
#             self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
#         elif bias and not concat:
#             self.bias = nn.Parameter(torch.Tensor(num_out_features))
#         else:
#             self.register_parameter('bias', None)

#         if add_skip_connection:
#             self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
#         else:
#             self.register_parameter('skip_proj', None)

#         #
#         # End of trainable weights
#         #

#         self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
#         self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
#         self.activation = activation
#         # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
#         # and for attention coefficients. Functionality-wise it's the same as using independent modules.
#         self.dropout = nn.Dropout(p=dropout_prob)

#         self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
#         self.attention_weights = None  # for later visualization purposes, I cache the weights here

#         self.init_params(layer_type)

#     def init_params(self, layer_type):
#         """
#         The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
#             https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

#         The original repo was developed in TensorFlow (TF) and they used the default initialization.
#         Feel free to experiment - there may be better initializations depending on your problem.

#         """
#         nn.init.xavier_uniform_(self.proj_param if layer_type == LayerType.IMP1 else self.linear_proj.weight)
#         nn.init.xavier_uniform_(self.scoring_fn_target)
#         nn.init.xavier_uniform_(self.scoring_fn_source)

#         if self.bias is not None:
#             torch.nn.init.zeros_(self.bias)

#     def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
#         if self.log_attention_weights:  # potentially log for later visualization in playground.py
#             self.attention_weights = attention_coefficients

#         # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
#         # only imp1 will enter this one
#         if not out_nodes_features.is_contiguous():
#             out_nodes_features = out_nodes_features.contiguous()

#         if self.add_skip_connection:  # add skip or residual connection
#             if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
#                 # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
#                 # thus we're basically copying input vectors NH times and adding to processed vectors
#                 out_nodes_features += in_nodes_features.unsqueeze(1)
#             else:
#                 # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
#                 # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
#                 out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

#         if self.concat:
#             # shape = (N, NH, FOUT) -> (N, NH*FOUT)
#             out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
#         else:
#             # shape = (N, NH, FOUT) -> (N, FOUT)
#             out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

#         if self.bias is not None:
#             out_nodes_features += self.bias

#         return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


# class GATLayerImp3(GATLayer):
#     """
#     Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

#     But, it's hopefully much more readable! (and of similar performance)

#     It's suitable for both transductive and inductive settings. In the inductive setting we just merge the graphs
#     into a single graph with multiple components and this layer is agnostic to that fact! <3

#     """

#     src_nodes_dim = 0  # position of source nodes in edge index
#     trg_nodes_dim = 1  # position of target nodes in edge index

#     nodes_dim = 0      # node dimension/axis
#     head_dim = 1       # attention head dimension/axis

#     def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
#                  dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

#         # Delegate initialization to the base class
#         super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP3, concat, activation, dropout_prob,
#                       add_skip_connection, bias, log_attention_weights)

#     def forward(self, data):
#         #
#         # Step 1: Linear Projection + regularization
#         #

#         in_nodes_features, edge_index = data  # unpack data
#         num_of_nodes = in_nodes_features.shape[self.nodes_dim]
#         assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

#         # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
#         # We apply the dropout to all of the input node features (as mentioned in the paper)
#         # Note: for Cora features are already super sparse so it's questionable how much this actually helps
#         in_nodes_features = self.dropout(in_nodes_features)

#         # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
#         # We project the input node features into NH independent output features (one for each attention head)
#         nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

#         nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

#         #
#         # Step 2: Edge attention calculation
#         #

#         # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
#         # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
#         # Optimization note: torch.sum() is as performant as .sum() in my experiments
#         scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
#         scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

#         # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
#         # the possible combinations of scores we just prepare those that will actually be used and those are defined
#         # by the edge index.
#         # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
#         scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
#         scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

#         # shape = (E, NH, 1)
#         attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
#         # Add stochasticity to neighborhood aggregation
#         attentions_per_edge = self.dropout(attentions_per_edge)

#         #
#         # Step 3: Neighborhood aggregation
#         #

#         # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
#         # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
#         nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

#         # This part sums up weighted and projected neighborhood feature vectors for every target node
#         # shape = (N, NH, FOUT)
#         out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

#         #
#         # Step 4: Residual/skip connections, concat and bias
#         #

#         out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
#         return (out_nodes_features, edge_index)

#     #
#     # Helper functions (without comments there is very little code so don't be scared!)
#     #

#     def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
#         """
#         As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
#         Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
#         into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
#         in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
#         (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
#          i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

#         Note:
#         Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
#         and it's a fairly common "trick" used in pretty much every deep learning framework.
#         Check out this link for more details:

#         https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

#         """
#         # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
#         scores_per_edge = scores_per_edge - scores_per_edge.max()
#         exp_scores_per_edge = scores_per_edge.exp()  # softmax

#         # Calculate the denominator. shape = (E, NH)
#         neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

#         # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
#         # possibility of the computer rounding a very small number all the way to 0.
#         attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

#         # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
#         return attentions_per_edge.unsqueeze(-1)

#     def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
#         # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
#         trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

#         # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
#         size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
#         size[self.nodes_dim] = num_of_nodes
#         neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

#         # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
#         # target index)
#         neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

#         # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
#         # all the locations where the source nodes pointed to i (as dictated by the target index)
#         # shape = (N, NH) -> (E, NH)
#         return neighborhood_sums.index_select(self.nodes_dim, trg_index)

#     def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
#         size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
#         size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
#         out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

#         # shape = (E) -> (E, NH, FOUT)
#         trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
#         # aggregation step - we accumulate projected, weighted node features for all the attention heads
#         # shape = (E, NH, FOUT) -> (N, NH, FOUT)
#         out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

#         return out_nodes_features

#     def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
#         """
#         Lifts i.e. duplicates certain vectors depending on the edge index.
#         One of the tensor dims goes from N -> E (that's where the "lift" comes from).

#         """
#         src_nodes_index = edge_index[self.src_nodes_dim]
#         trg_nodes_index = edge_index[self.trg_nodes_dim]

#         # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
#         scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
#         scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
#         nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

#         return scores_source, scores_target, nodes_features_matrix_proj_lifted

#     def explicit_broadcast(self, this, other):
#         # Append singleton dimensions until this.dim() == other.dim()
#         for _ in range(this.dim(), other.dim()):
#             this = this.unsqueeze(-1)

#         # Explicitly expand so that shapes are the same
#         return this.expand_as(other)

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class FreG(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
       
        

    def forward(self, x):
        b,c,h,w = x.shape

       
        high_part = x - low_part
        
        return low_part, high_part




class Gate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Gate, self).__init__()
      

    def forward(self, x):

        # x = torch.cat([sx, f_l, f_h], dim=1)
        # x = self.conv1(x)

        ca_mean = self.avg_pool(x)
        ca_mean = self.conv_mean(ca_mean)

        B, C, H, W = x.size()
        x_dense = x.view(B, C, -1)
        ca_std = torch.std(x_dense, dim=2, keepdim=True)
        ca_std = ca_std.view(B, C, 1, 1)
        ca_var = self.conv_std(ca_std)
        gate = (ca_mean + ca_var)/2.0

        return gate * x



class CAM(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

      

    def forward(self, x_l, x_r):
        
        Q_l = self.l_proj1(self.norm_l(x_l))
        Q_r = self.r_proj1(self.norm_r(x_r))
        b, c, h, w = Q_r.shape
        Q_l = Q_l.permute(0, 2, 3, 1)
        Q_r_T = Q_r.permute(0, 2, 1, 3)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  #B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l) #B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l + x_r + F_l2r
        

    


class Fusion(nn.Module):
    def __init__(self, channel):
        

    def forward(self, sx_g, f_l_g, f_h_g):

        

        return out

class Fusion2(nn.Module):
    def __init__(self, in_channels, height=3,reduction=2,bias=False):
        super(Fusion2, self).__init__()
        
        
    def forward(self, sx, low, high):
       
        
        return self.out(feats_V)    


class BaseBlock(nn.Module):
    def __init__(self, c, num_blk, num_heads, bias):
        super().__init__()
        layers = [NAFBlock(c) for _ in range(num_blk)]
        self.naf = nn.Sequential(*layers)
        self.fg = FreG(c,num_heads,bias)
        # self.gatf = GAT(2,[8,1],[c*4,8,c],add_skip_connection=True,bias=bias,dropout=0.6)
        self.gate = Gate(c)
        self.fusion = Fusion(c)
    
    def forward(self, x):
        sx = self.naf(x)
        f_l, f_h = self.fg(sx + x)

        sx_g = self.gate(sx)
        f_l_g = self.gate(f_l)
        f_h_g = self.gate(f_h)

        out = self.fusion(sx_g, f_l_g, f_h_g)


        return out
  

       



class SFE(nn.Module):
    def __init__(self, out_plane):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_plane//2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=out_plane//2, out_channels=out_plane, kernel_size=3, padding=1, stride=1, groups=out_plane//2,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=out_plane // 2, out_channels=out_plane, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=out_plane//2, out_channels=out_plane//2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        return x

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = nn.Conv2d(channel*2, channel, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))
    
    
class GSFFNet(nn.Module):
    def __init__(self, base_channel = 64, num_res=15, num_heads=8,  bias = False):
        super(GSFFNet, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()


        self.intro = nn.Conv2d(in_channels=3, out_channels=base_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        chan = base_channel

        self.encoders = nn.ModuleList([
            BaseBlock(base_channel, num_res, num_heads, bias),
            BaseBlock(base_channel*2, num_res, num_heads, bias),
            BaseBlock(base_channel*4, num_res, num_heads, bias),
        ])
        for i in range(3):
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = BaseBlock(base_channel*8, num_res, num_heads, bias)

        self.decoders = nn.ModuleList([
            BaseBlock(base_channel*4, num_res, num_heads, bias),
            BaseBlock(base_channel*2, num_res, num_heads, bias),
            BaseBlock(base_channel, num_res, num_heads, bias),
        ])
        for i in range(3):
            self.ups.append(
                nn.Sequential(
                    DySample(chan,2),
                    nn.Conv2d(chan,chan//2,1)
                )
            )
            chan = chan // 2

        self.Convs = nn.ModuleList([
            nn.Conv2d(in_channels=base_channel*8, out_channels=base_channel * 4, kernel_size=1,  stride=1, groups=1,
                              bias=True),
            nn.Conv2d(in_channels=base_channel*4, out_channels=base_channel * 2, kernel_size=1,  stride=1, groups=1,
                              bias=True),
            nn.Conv2d(in_channels=base_channel*2, out_channels=base_channel , kernel_size=1, stride=1, groups=1,
                              bias=True)    
        ])

        self.ConvsOut = nn.ModuleList(
            [
                nn.Conv2d(in_channels=base_channel*4, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True),
                nn.Conv2d(in_channels=base_channel*2, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True),
                nn.Conv2d(in_channels=base_channel, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SFE(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SFE(base_channel * 2)

        self.FAM3 = FAM(base_channel * 2)


       

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        # 256
        x_ = self.intro(x)
        res1 = self.encoders[0](x_)



        z = self.downs[0](res1)
        # 128
        z = self.FAM2(z, z2)
        res2 = self.encoders[1](z)


        z = self.downs[1](res2)
        # 64
        z = self.FAM1(z, z4)
        res3 = self.encoders[2](z)


        z = self.downs[2](res3)
        #32
        z = self.middle_blks(z)


        z = self.ups[0](z)
        z = torch.cat([z, res3], dim=1)
        z = self.Convs[0](z)

        z = self.decoders[0](z)
        z_ = self.ConvsOut[0](z)
        # 128
        outputs.append(z_+x_4)

        z = self.ups[1](z)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[1](z)
        z = self.decoders[1](z)
        z_ = self.ConvsOut[1](z)
        # 256
        z = self.ups[2](z)
        outputs.append(z_+x_2)
      

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[2](z)
        z = self.decoders[2](z)
        z = self.ConvsOut[2](z)
     
        outputs.append(z+x)
        return outputs


class LocalAttention(FreG):
    def __init__(self, dim, num_heads, bias, base_size=None, kernel_size=None, fast_imp=False, train_size=None):
        super().__init__(dim, num_heads, bias)
        self.base_size = base_size
        self.kernel_size = kernel_size
        self.fast_imp = fast_imp
        self.train_size = train_size

    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c//3, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def _forward(self, qkv):
        q,k,v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        low_part = (attn @ v)
        
        
        return low_part

    def _pad(self, x):
        b,c,h,w = x.shape
        k1, k2 = self.kernel_size
        mod_pad_h = (k1- h % k1) % k1
        mod_pad_w = (k2 - w % k2) % k2
        pad = (mod_pad_w//2, mod_pad_w-mod_pad_w//2, mod_pad_h//2, mod_pad_h-mod_pad_h//2)
        x = F.pad(x, pad, 'reflect')
        return x, pad

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        
        if self.fast_imp:
            raise NotImplementedError
            # qkv, pad = self._pad(qkv)
            # b,C,H,W = qkv.shape
            # k1, k2 = self.kernel_size
            # qkv = qkv.reshape(b,C,H//k1, k1, W//k2, k2).permute(0,2,4,1,3,5).reshape(-1,C,k1,k2)
            # out = self._forward(qkv)
            # out = out.reshape(b,H//k1,W//k2,c,k1,k2).permute(0,3,1,4,2,5).reshape(b,c,H,W)
            # out = out[:,:,pad[-2]:pad[-2]+h, pad[0]:pad[0]+w]
        else:
            qkv = self.grids(qkv) # convert to local windows 
            out = self._forward(qkv)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=qkv.shape[-2], w=qkv.shape[-1])
            out = self.grids_inverse(out) # reverse
        
        high_part = x - out
        return out, high_part
       



class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size).cuda()

            if m.output_size == 1:
                setattr(model, n, pool)
            # assert m.output_size == 1
            

        if isinstance(m, FreG):
            attn = LocalAttention(dim=m.dim, num_heads=m.num_heads, bias=m.bias, base_size=base_size, fast_imp=False, train_size=train_size)
            setattr(model, n, attn)
        
    


class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)

class SFAFNetLocal(Local_Base, GSFFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        GSFFNet.__init__(self, *args, **kwargs)
        # self.cuda()

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
def build_net():
    return GSFFNet()

if __name__ == "__main__":
    import time
    # start = time.time()
    net = SFAFNetLocal()
    x = torch.randn((1, 3, 256, 256))
    print("Total number of param  is ", sum(i.numel() for i in net.parameters()))
    t=net(x)
    print(t[2].shape)
    inp_shape = (3, 256, 256)
    from ptflops import get_model_complexity_info
    FLOPS = 0

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)
    # # print(params)
    macs = float(macs[:-4]) + FLOPS / 10 ** 9



    print('mac', macs, params)

        
