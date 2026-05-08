import math
from typing import Dict, Union

import torch
import torch.nn.functional as F
import torch_geometric
from e3nn import nn, o3
from e3nn.io import CartesianTensor
from e3nn.math import soft_one_hot_linspace, soft_unit_step
from e3nn.nn import BatchNorm, FullyConnectedNet, Gate
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
from pymatgen.core.tensors import Tensor
from torch_geometric.data import Data
from torch_scatter import scatter

heads = 2
lmax = 3


def set_model_hparams(num_heads: int = 2, max_l: int = 3):
    global heads, lmax
    heads = int(num_heads)
    lmax = int(max_l)


class EquivariantLayerNormFast(torch.nn.Module):

    def __init__(self, irreps, eps=1e-5, affine=True, normalization='component'):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        if affine:
            self.affine_weight = torch.nn.Parameter(torch.ones(num_features))
            self.affine_bias = torch.nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

    def forward(self, node_input, **kwargs):
        '''
            Use torch layer norm for scalar features.
        '''

        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            if ir.l == 0 and ir.p == 1:
                weight = self.affine_weight[iw:(iw + mul)]
                bias = self.affine_bias[ib:(ib + mul)]
                iw += mul
                ib += mul
                field = F.layer_norm(field, tuple((mul,)), weight, bias, self.eps)
                fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]
                continue

            # For non-scalar features, use RMS value for std
            field = field.reshape(-1, mul, d)  # [batch * sample, mul, repr]

            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError("Invalid normalization option {}".format(self.normalization))
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)
            field_norm = 1.0 / ((field_norm + self.eps).sqrt())  # [batch * sample, mul]

            if self.affine:
                weight = self.affine_weight[None, iw:(iw + mul)]  # [1, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch * sample, mul]
            field = field * field_norm.reshape(-1, mul, 1)  # [batch * sample, mul, repr]

            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        assert ix == dim

        output = torch.cat(fields, dim=-1)
        return output






@compile_mode('script')
class Vec2AttnHeads(torch.nn.Module):
    '''
        Reshape vectors of shape [N, irreps_mid] to vectors of shape
        [N, num_heads, irreps_head].
    '''

    def __init__(self, irreps_head, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        N, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.mid_in_indices):
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, -1)
            out.append(temp)
        out = torch.cat(out, dim=2)
        return out

    def __repr__(self):
        return '{}(irreps_head={}, num_heads={})'.format(
            self.__class__.__name__, self.irreps_head, self.num_heads)

#from equiformer
@compile_mode('script')
class AttnHeads2Vec(torch.nn.Module):
    '''
        Convert vectors of shape [N, num_heads, irreps_head] into
        vectors of shape [N, irreps_head * num_heads].
    '''

    def __init__(self, irreps_head):
        super().__init__()
        self.irreps_head = irreps_head
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        N, _, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.head_indices):
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        return out

    def __repr__(self):
        return '{}(irreps_head={})'.format(self.__class__.__name__, self.irreps_head)


#from matten
class ShiftedSoftPlus(torch.nn.Module):
    """
    Shifted softplus as defined in SchNet, NeurIPS 2017.

    :param beta: value for the a more general softplus, default = 1
    :param threshold: values above are linear function, default = 20
    """

    _log2: float

    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self.softplus = torch.nn.Softplus(beta=beta, threshold=threshold)
        self._log2 = math.log(2.0)

    def forward(self, x):
        """
        Evaluate shifted softplus

        :param x: torch.Tensor, input
        :return: torch.Tensor, ssp(x)
        """
        return self.softplus(x) - self._log2


ACTIVATION = {
    # for even irreps
    "e": {
        "ssp": ShiftedSoftPlus(),
        "silu": torch.nn.functional.silu,
        "sigmoid": torch.sigmoid,
    },
    # for odd irreps
    "o": {
        "abs": torch.abs,
        "tanh": torch.tanh,
    },
}


def find_positions_in_tensor_fast(tensor):
    """
    Optimized function to find positions of each unique element in a PyTorch tensor
    using advanced indexing and broadcasting, keeping outputs as tensors.

    Parameters:
    tensor (torch.Tensor): The input tensor to analyze.

    Returns:
    dict: A dictionary where each key is a unique element from the tensor,
          and the value is a tensor of indices where this element appears.
    """
    unique_elements, inverse_indices = torch.unique(tensor, sorted=True, return_inverse=True)
    positions = {}
    for i, element in enumerate(unique_elements):
        # Directly store tensors of positions
        positions[element.item()] = torch.nonzero(inverse_indices == i, as_tuple=True)[0]

    return positions

find_positions_in_tensor_fast=torch.compile(find_positions_in_tensor_fast)


class Fromtensor(torch.nn.Module):
    def __init__(self, formula):
        super().__init__()
        self.tensor = CartesianTensor(formula)
    def forward(self, data):
        return self.tensor.from_cartesian(data)


class Totensor(torch.nn.Module):
    def __init__(self, formula):
        super().__init__()
        self.tensor = CartesianTensor(formula)

    def forward(self, data):
        return self.tensor.to_cartesian(data)





class TensorIrreps(torch.nn.Module):
    def __init__(self ,formula , conv_to_output_hidden_irreps_out):
        super().__init__()
        if formula is None:
            self.formula=formula
            self.irreps_in = conv_to_output_hidden_irreps_out
            # self.dropout=nn.Dropout(irreps=self.irreps_in,p=0.2)
            self.irreps_out = o3.Irreps('0e')
            self.extra_layers = o3.Linear(irreps_in=self.irreps_in, irreps_out=self.irreps_out)
        else:
            self.formula=formula
            self.irreps_in = conv_to_output_hidden_irreps_out
            # self.dropout = nn.Dropout(irreps=self.irreps_in, p=0.2)
            self.irreps_out = CartesianTensor(formula=self.formula)

            self.extra_layers = o3.Linear(irreps_in=self.irreps_in, irreps_out=self.irreps_out)

            # self.to_cartesian = Totensor(self.formula)

    def forward(self,data):
        # out=self.dropout(data)
        out=self.extra_layers(data)
        if self.formula is None:
            return out
        else:
            out = self.irreps_out.to_cartesian(out)
            return out


class UVUTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        node_attr:o3.Irreps,
        internal_and_share_weights: bool = False,
        # mlp_input_size: int = None,
        # mlp_hidden_size: int = 8,
        # mlp_num_hidden_layers: int = 1,
        # mlp_activation: Callable = ACTIVATION["e"]["ssp"],
    ):
        """
        UVU tensor product.

        Args:
            irreps_in1: irreps of first input, with available keys in `DataKey`
            irreps_in2: input of second input, with available keys in `DataKey`
            irreps_out: output irreps, with available keys in `DataKey`
            internal_and_share_weights: whether to create weights for the tensor
                product, if `True` all `mlp_*` params are ignored and if `False`,
                they should be provided to create an MLP to transform some data to be
                used as the weight of the tensor product.

        """

        super().__init__()

        self.out=irreps_out
        self.node_attr=node_attr
        # self.dropout = nn.Dropout(irreps=irreps_in1,p=0.3)

        # uvu instructions for tensor product
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in1) in enumerate(irreps_in1):
            for j, (_, ir_in2) in enumerate(irreps_in2):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in irreps_out or ir_out == o3.Irreps("0e"):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)

        assert irreps_mid.dim > 0, (
            f"irreps_in1={irreps_in1} times irreps_in2={irreps_in2} produces no "
            f"instructions in irreps_out={irreps_out}"
        )

        # sort irreps_mid to let irreps of the same type be adjacent to each other
        self.irreps_mid, permutation, _ = irreps_mid.sort()

        # sort instructions accordingly
        instructions = [
            (i_1, i_2, permutation[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        self.lin0=o3.FullyConnectedTensorProduct(irreps_in1, self.node_attr,irreps_in1)
        # self.dropout1=nn.Dropout(irreps=irreps_in1,p=0.2)
        self.tp = o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            self.irreps_mid,
            instructions,
            internal_weights=internal_and_share_weights,
            shared_weights=internal_and_share_weights,
        )
        self.dropout2 = nn.Dropout(irreps=self.irreps_mid, p=0.2)
        # self.lin=o3.Linear(irreps_in=self.irreps_mid,irreps_out=self.out)
        self.lin=o3.FullyConnectedTensorProduct(self.irreps_mid, self.node_attr,self.out)

        self.sc = o3.FullyConnectedTensorProduct(
            irreps_in1, self.node_attr, self.out
        )

    # def forward(
    #     self, data1: Tensor, data2: Tensor, data_weight: Tensor,data3:Tensor
    # ) -> Tensor:
    #     # if self.weight_nn is not None:
    #     #     assert data_weight is not None, "data for weight not provided"
    #     #     weight = self.weight_nn(data_weight)
    #     # else:
    #     #     weight = None
    #     x = self.tp(data1, data2, data_weight)
    #     x=self.lin(x)
    #
    #     return x


    def forward( self, data1: Tensor, data2: Tensor, data_weight: Tensor,data3:Tensor
    ) -> Tensor:
        node_feats = data1
        node_attrs = data3
        edge_attrs = data2
        # node_feats=self.dropout(node_feats)
        node_sc = self.sc(node_feats, node_attrs)
        # node_sc=self.dropout(node_sc)

        node_feats = self.lin0(node_feats, node_attrs)
        # node_feats=self.dropout1(node_feats)

        node_feats = self.tp(node_feats, edge_attrs, data_weight)
        node_feats=self.dropout2(node_feats)
        # node_feats=self.lin(node_feats,node_attrs)

        # update
        node_conv_out = self.lin(node_feats, node_attrs)
        # node_conv_out=self.dropout(node_conv_out)
        node_feats = node_sc + node_conv_out

        return node_feats

def tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Compose(torch.nn.Module):
    def __init__(self, first, second) -> None:
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)


def multiheadsplit(x):
    ll=[]
    for mul,ir in x:
        if mul % int(heads) != 0:
            raise ValueError(
                f"Irrep multiplicity {mul} for {ir} is not divisible by heads={heads}. "
                "Please set heads accordingly or increase multiplicities in irreps."
            )
        per_head = int(mul // int(heads))
        if per_head > 0:
            ll.append((per_head, ir))
    if len(ll) == 0:
        raise ValueError(
            f"multiheadsplit produced empty irreps for heads={heads}. "
            "Set heads=1 or use larger multiplicities in irreps."
        )
    return o3.Irreps(ll)


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way using PyTorch."""
    shiftx = x - torch.max(x)
    # exps = torch.exp(shiftx)
    return F.softmax(shiftx,dim=-1)
@compile_mode('script')
class Attention(torch.nn.Module):
    def __init__(self, node_attr,irreps_node_input, irreps_query, irreps_key, irreps_output, number_of_basis):
        super().__init__()
        # self.radial_cutoff = radial_cutoff
        self.heads=heads
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)


        self.norm = EquivariantLayerNormFast(irreps=irreps_output)


        self.number_of_basis = number_of_basis
        self.irreps_query = irreps_query
        self.node_attr=node_attr

        self.h_q = o3.FullyConnectedTensorProduct(irreps_node_input,self.node_attr,irreps_query)
        # self.h_q = o3.Linear(irreps_node_input,self.irreps_query)


        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_key=o3.Irreps(irreps_key)
        self.irreps_output=o3.Irreps(irreps_output)


        self.tp_k=UVUTensorProduct(self.irreps_node_input, self.irreps_sh, self.irreps_key,self.node_attr)
        # self.dropout1=nn.Dropout(irreps=self.irreps_node_input,p=0.2)
        self.fc_k = FullyConnectedNet(self.number_of_basis+ [self.tp_k.tp.weight_numel], act=torch.nn.functional.silu)


        self.tp_v=UVUTensorProduct(self.irreps_node_input, self.irreps_sh, self.irreps_output,self.node_attr)
        # self.dropout2=nn.Dropout(irreps=self.irreps_output,p=0.2)
        self.fc_v= FullyConnectedNet(self.number_of_basis+ [self.tp_v.tp.weight_numel], act=torch.nn.functional.silu)

        self.dot = torch.nn.ModuleList([o3.FullyConnectedTensorProduct(multiheadsplit(self.irreps_query).simplify(),multiheadsplit(self.irreps_key).simplify(), "0e") for _ in range(heads)])
        # self.lin=o3.Linear(2*self.irreps_output, self.irreps_output)

        self.vec2headsq = Vec2AttnHeads(multiheadsplit(self.irreps_query).simplify(),self.heads)
        self.vec2headsk=Vec2AttnHeads(multiheadsplit(self.irreps_key).simplify(),self.heads)
        self.vec2headsv=Vec2AttnHeads(multiheadsplit(self.irreps_output).simplify(),self.heads)


        # self.heads2vecq = AttnHeads2Vec(multiheadsplit(self.irreps_query).simplify())
        # self.heads2veck = AttnHeads2Vec(multiheadsplit(self.irreps_key).simplify())
        self.heads2vecv = AttnHeads2Vec(multiheadsplit(self.irreps_output).simplify())

        self.lin = o3.FullyConnectedTensorProduct(self.irreps_output,self.node_attr,self.irreps_output)
        self.sc = o3.FullyConnectedTensorProduct(
            irreps_node_input, node_attr, self.irreps_output
        )
        self.dim=self.irreps_key.dim
    def forward(self, node_attr,node_input,  edge_src, edge_dst, edge_attr, edge_scalars,edge_length,fpit) -> torch.Tensor:
        edge_length_embedded = edge_scalars
        # edge_length_embedded=self.dropout00(edge_length_embedded)
        edge_sh = edge_attr
        edge_weight_cutoff = edge_length
        # fpit = find_positions_in_tensor_fast(edge_dst)
        # node_input=self.dropout0(node_input)
        # print(node_input.shape)
        # print(self.dim)

        node_input_sc=self.sc(node_input,node_attr)

        # q = self.h_q0(node_input,node_attr)
        q = self.h_q(node_input,node_attr)

        weight0=self.fc_k(edge_length_embedded)

        k = self.tp_k(node_input[edge_src], edge_sh, weight0,node_attr[edge_src])


        weight1=self.fc_v(edge_length_embedded)

        v = self.tp_v(node_input[edge_src], edge_sh, weight1,node_attr[edge_src])


        q=self.vec2headsq(q)
        k=self.vec2headsk(k)
        v=self.vec2headsv(v)
        sca=[]

        for i in range(self.heads):
        # 假设 'edge_dst' 和 'at' 已经正确定义，并且 'at' 是要修改的张量
            at = self.dot[i](torch.split(q[edge_dst],1,dim=1)[i].squeeze(1),torch.split(k,1,dim=1)[i].squeeze(1))
            # at=self.dropoutdot(at)
            # for key, indices in fpit.items():  # 确保索引不为空
            #     max_val = torch.max(at[indices])  # 获取当前索引下的最大值
            #     at[indices] -= max_val  # 更新 at 张量中的对应值
            at=at / self.dim ** 0.5
            for key in fpit:
                at[fpit[key]]=stable_softmax(at[fpit[key]])
        # exp = edge_weight_cutoff[:, None] * ((at-max(at))/len(k)**0.5).exp()
        #     exp = edge_weight_cutoff[:, None] * (at / len(torch.split(k,1,dim=1)[i]) ** 0.5).exp()
        #     exp = edge_weight_cutoff[:, None] * (at / self.dim ** 0.5).exp()
            exp=edge_weight_cutoff[:, None]*at

            z = scatter(exp, edge_dst, dim=0, dim_size=len(node_input))

            z[z ==0] = 1
            alpha = exp / z[edge_dst]
            sca.append(scatter(alpha.relu().sqrt() * torch.split(v,1,dim=1)[i].squeeze(1), edge_dst, dim=0, dim_size=len(node_input)))

        sca=torch.stack(sca,dim=1)
        sca=self.heads2vecv(sca)
        sca_conv_out=self.lin(sca,node_attr)
        sca=sca_conv_out+node_input_sc
        # sca=self.norm(sca)

        return sca

class EquivariantAttention(torch.nn.Module):
    def __init__(
        self,
            node_attr,
        irreps_node_input,
            irreps_query,
            irreps_key,
        irreps_node_hidden,
        irreps_node_output,
        irreps_edge_attr,
        layers,
        fc_neurons,

    ) -> None:
        super().__init__()

        self.attr=o3.Irreps(node_attr)
        self.irreps_node_input = o3.Irreps(irreps_node_input)

        self.irreps_query=o3.Irreps(irreps_query)
        self.irreps_key=o3.Irreps(irreps_key)
        self.irreps_node_hidden = o3.Irreps(irreps_node_hidden)
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)



        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: ShiftedSoftPlus(),
            -1: torch.tanh,
        }

        self.layers = torch.nn.ModuleList()


        # self.layer.append(self.embed)
        for _ in range(layers):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_node_hidden
                    if ir.l == 0 and tp_path_exists(self.irreps_node_input, self.irreps_edge_attr, ir)
                ]
            ).simplify()

            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_node_hidden
                    if ir.l > 0 and tp_path_exists(self.irreps_node_input, self.irreps_edge_attr, ir)
                ]
            )
            # self.irreps_query1 = o3.Irreps(
            #     [(mul, ir) for mul, ir in o3.Irreps(self.irreps_query) if tp_path_exists(self.irreps_node_input, "0e", ir)])

            ir = "0e" if tp_path_exists(self.irreps_node_input, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )

            conv = Attention(self.attr,
                self.irreps_node_input,  self.irreps_query,self.irreps_key, gate.irreps_in, fc_neurons
            )
            self.irreps_node_input = gate.irreps_out
            self.norm=EquivariantLayerNormFast(self.irreps_node_input)
            self.layers.append(Compose(Compose(conv, gate),self.norm))
        self.layers.append(Attention(self.attr,
                self.irreps_node_input, self.irreps_query, self.irreps_key, self.irreps_node_output, fc_neurons
            )
        )

    def forward(self,node_attr, node_features,  edge_src, edge_dst, edge_attr, edge_scalars,edge_length) -> torch.Tensor:
        fpit = find_positions_in_tensor_fast(edge_dst)
        for lay in self.layers:
            node_features = lay(node_attr,node_features,  edge_src, edge_dst, edge_attr, edge_scalars,edge_length,fpit)
        return node_features


@compile_mode('script')
class Network(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
            embedding_dim,
            irreps_query,
            irreps_key,
        irreps_out,
            formula,
        max_radius,
            num_nodes,
        mul=32,
        layers=2,
            number_of_basis=10,
        lmax=lmax,
        pool_nodes=True,
        global_descriptor_dim=0,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_nodes = num_nodes
        self.pool_nodes = pool_nodes
        self.formula = formula
        self.global_descriptor_dim = int(global_descriptor_dim or 0)
        if self.global_descriptor_dim > 0 and not self.pool_nodes:
            raise ValueError("Global descriptors require pool_nodes=True.")

        self.irreps_in=irreps_in
        self.embeding_dim=embedding_dim

        irreps_node_hidden = o3.Irreps([(int(mul/2**(l)), (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_node_hidden = irreps_node_hidden.simplify()
        self.irreps_query = irreps_query
        self.irreps_key = irreps_key

        self.irreps_sh=o3.Irreps.spherical_harmonics(lmax)
        # self.dropout0 = nn.Dropout(irreps="{}x0e".format(self.embeding_dim),p=0.2)

        self.lin=o3.Linear(self.irreps_in,"{}x0e".format(self.embeding_dim))
        self.GAT=EquivariantAttention(
            node_attr=self.irreps_in,
        irreps_node_input="{}x0e".format(self.embeding_dim),
            irreps_query=irreps_query,
            irreps_key=irreps_key,
        irreps_node_hidden=self.irreps_node_hidden,
        irreps_node_output=irreps_out,
        irreps_edge_attr=self.irreps_sh,
        layers=layers,
        fc_neurons=[self.number_of_basis,100],
        )

        self.irreps_in = self.GAT.irreps_node_input
        self.irreps_out = self.GAT.irreps_node_output

        if self.global_descriptor_dim > 0:
            self.global_irreps = o3.Irreps(f"{self.global_descriptor_dim}x0e")
            self.irreps_with_global = (self.irreps_out + self.global_irreps).simplify()
        else:
            self.global_irreps = o3.Irreps("")
            self.irreps_with_global = self.irreps_out

        self.TI0=o3.Linear(self.irreps_with_global,self.irreps_out)
        self.TI1 = o3.Linear(self.irreps_out, self.irreps_out)
        self.TI = TensorIrreps(self.formula, self.irreps_out)

        self.dropout1 = nn.Dropout(irreps=self.irreps_out, p=0.2)

    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        edge_src = data['edge_index'][0]  # Edge source
        edge_dst = data['edge_index'][1]  # Edge destination

        # We need to compute this in the computation graph to backprop to positions
        # We are computing the relative distances + unit cell shifts from periodic boundaries
        edge_batch = batch[edge_src]
        edge_vec = (data['pos'][edge_dst]
                    - data['pos'][edge_src]
                    + torch.einsum('ni,nij->nj', data['edge_shift'], data['lattice'][edge_batch]))

        return batch, data['x'], edge_src, edge_dst, edge_vec

    def forward(self, data: Union[torch_geometric.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        global_attr = None
        if self.global_descriptor_dim > 0:
            if 'global_attr' not in data:
                raise ValueError("Model expects global_attr, but the batch does not contain it.")
            global_attr = data['global_attr']
        batch, node_inputs, edge_src, edge_dst, edge_vec = self.preprocess(data)
        del data
        node_attr=node_inputs
        edge_attr = o3.spherical_harmonics(self.irreps_sh, edge_vec, True, normalization="component")

        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis="smooth_finite",  # the cosine basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis ** 0.5)

        edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / self.max_radius))
        node_features=self.lin(node_inputs)
        # node_features=self.dropout0(node_features)
        node_outputs = self.GAT(node_attr,node_features, edge_src, edge_dst, edge_attr, edge_length_embedding,edge_weight_cutoff)
        # node_outputs=self.dropout1(node_outputs)
        if self.pool_nodes:
            node_outputs = scatter(node_outputs, batch, dim=0,reduce="mean").div(self.num_nodes ** 0.5)
        else:
            pass
        residual_outputs = node_outputs
        if self.global_descriptor_dim > 0:
            node_outputs = torch.cat([node_outputs, global_attr], dim=-1)
        node_outputs1=self.TI0(node_outputs)

        node_outputs2=self.TI1(node_outputs1)
        node_outputs=node_outputs2+residual_outputs
        node_outputs=self.TI(node_outputs)

        if torch.isnan(node_outputs).any():
            print('nan after TI')
        # node_outputs=self.dropout1(node_outputs)
        return node_outputs





def build_network(feature_dim, num_nodes, config, global_descriptor_dim=0):
    set_model_hparams(config.get("heads", 2), config.get("lmax", 3))
    return Network(
        irreps_in="{}x0e".format(feature_dim),
        embedding_dim=config.get("embedding_dim", 64),
        irreps_query=config.get(
            "irreps_query",
            "32x0e+32x0o+16x1e+16x1o+8x2e+8x2o+4x3e+4x3o+2x4e+2x4o",
        ),
        irreps_key=config.get(
            "irreps_key",
            "32x0e+32x0o+16x1e+16x1o+8x2e+8x2o+4x3e+4x3o+2x4e+2x4o",
        ),
        irreps_out=config.get("irreps_out", "2x0e+2x1o+2x2e"),
        formula=config.get("formula", "ij"),
        max_radius=config.get("max_radius", 7),
        num_nodes=num_nodes,
        mul=config.get("mul", 32),
        layers=config.get("layers", 2),
        number_of_basis=config.get("number_of_basis", 10),
        lmax=config.get("lmax", 3),
        pool_nodes=config.get("pool_nodes", True),
        global_descriptor_dim=global_descriptor_dim,
    )
