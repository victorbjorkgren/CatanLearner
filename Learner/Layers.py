import torch as T
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 final: bool = False,
                 feature_dim: int = -1,
                 residual: bool = False
                 ) -> None:
        super(MLP, self).__init__()
        self.final = final
        assert feature_dim in [-1, 1]
        self.feature_dim = feature_dim
        hidden = max(in_features, out_features)
        self.l1 = nn.Linear(in_features, hidden)
        self.l2 = nn.Linear(hidden, out_features)
        self.act = nn.LeakyReLU(.01)
        self.residual = residual

        if self.final:
            self.mlp = nn.Sequential(*[
                self.l1,
                self.act,
                self.l2
            ])
        else:
            self.mlp = nn.Sequential(*[
                self.l1,
                self.act,
                self.l2,
                self.act
            ])

    def forward(self, x: T.Tensor) -> T.Tensor:
        if self.feature_dim == 1:
            out = x.permute(0, 2, 1)
            out = self.mlp(out)
            out = out.permute(0, 2, 1)
        else:
            out = self.mlp(x)
        if self.residual:
            out = out + x
        return out


def norm(adj):
    # I = T.eye(adj.size(1)).to(adj.device)
    # A_hat = adj + I
    d_hat_diag = T.sum(adj, dim=2).pow(-0.5)
    d_hat = T.diag_embed(d_hat_diag.mean(-1)).unsqueeze(1)
    adj_normalized = (d_hat @ adj.permute(0, 3, 1, 2)) @ d_hat
    return adj_normalized


class PowerfulLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, adj_norm: T.Tensor) -> None:
        super(PowerfulLayer, self).__init__()
        self.adj_norm = adj_norm.unsqueeze(-1)
        self.mask = adj_norm > 0
        self.mask = self.mask.squeeze()

        self.m1 = MLP(in_features, out_features, feature_dim=-1)
        self.m2 = MLP(in_features, out_features, feature_dim=-1)
        self.g1 = MLP(2 * in_features, out_features)

    def forward(self, matrix: T.Tensor) -> T.Tensor:
        """
        matrix shape = (B, N, N, features)
        norm shape = (B, N, N)
        """
        # if matrix.device != self.adj_norm.device:
        #     self.adj_norm = self.adj_norm.to(matrix.device)

        # matrix = matrix.permute(0, 3, 1, 2)  # batch, out_feat, N, N
        # out1 = matrix.clone()
        # out2 = matrix.clone()
        #
        # # Feature Embedding
        # out1[:, :, self.mask] = self.m1(matrix[:, :, self.mask])
        # out2[:, :, self.mask] = self.m2(matrix[:, :, self.mask])

        # adj_norm = norm(matrix)

        out1 = self.m1(matrix).permute(0, 3, 1, 2)  # batch, out_feat, N, N
        out2 = self.m2(matrix).permute(0, 3, 1, 2)  # batch, out_feat, N, N

        out1 = out1 * self.mask
        out2 = out2 * self.mask

        # Message Propagation
        out = out1 @ out2
        del out1, out2

        # Add Gated Residual
        out = T.cat((out.permute(0, 2, 3, 1), matrix), dim=3)  # batch, N, N, out_feat
        del matrix

        out = self.g1(out)  # .permute(0, 3, 1, 2)

        # Reshape and Norm
        # out = (adj_norm @ out).permute(0, 2, 3, 1)
        return out

        # # Return Gated Residual
        # out = T.cat((out, matrix), dim=3)  # batch, N, N, out_feat
        # del matrix
        #
        # out[self.mask] = self.gate(out[self.mask])
        #
        # return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries):
        # Split the embedding into self.heads pieces
        n = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(n, value_len, self.heads, self.head_dim)
        keys = keys.reshape(n, key_len, self.heads, self.head_dim)
        queries = queries.reshape(n, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Attention mechanism
        energy = T.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = T.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            n, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out
