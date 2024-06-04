import torch
import torch as T
import torch.nn as nn
import torch_geometric as pyg

from .Utils import extract_attr, sparse_face_matrix, preprocess_adj
from .Layers import MLP, PowerfulLayer, MultiHeadAttention


class PlayerNet(nn.Module):
    def __init__(self, in_features, out_features, n_embed, n_heads):
        super(PlayerNet, self).__init__()
        self.player_embedding = MLP(in_features, n_embed)

        self.q = MLP(n_embed, n_embed // n_heads, final=True)
        self.k = MLP(n_embed, n_embed // n_heads, final=True)
        self.v = MLP(n_embed, n_embed // n_heads, final=True)

        self.multi_head_attention = MultiHeadAttention(n_embed, n_heads)

        self.output = nn.Sequential(
            MLP(n_embed, n_embed),
            MLP(n_embed, out_features, final=True)
        )

    def forward(self, player_states, board_embedding):
        player_states = self.player_embedding(player_states)
        board_embedding = board_embedding[T.argwhere(T.nonzero(board_embedding))]

        states = T.cat((player_states, board_embedding))
        del player_states, board_embedding

        q = self.q(states)
        k = self.k(states)
        v = self.v(states)

        # Multi-head attention
        attn_output = self.multi_head_attention(v, k, q)

        # Final output
        output = self.output(attn_output)
        return output


class GameNet(nn.Module):
    def __init__(self,
                 game,
                 n_embed,
                 n_output,
                 n_power_layers,
                 batch_size=1,
                 undirected_faces=True
                 ):
        super(GameNet, self).__init__()

        sparse_edge = game.board.state.edge_index
        sparse_face = game.board.state.face_index
        n_node_attr = game.board.state.num_node_features
        n_edge_attr = game.board.state.num_node_features
        n_face_attr = game.board.state.face_attr.shape[1]
        n_player_attr = game.players[0].state.shape[0]
        n_players = len(game.players)

        self.undirected_faces = undirected_faces
        self.sparse_edge = sparse_edge
        self.sparse_face = sparse_face_matrix(sparse_face, to_undirected=self.undirected_faces)
        self.sparse_full = T.cat((self.sparse_edge, self.sparse_face), dim=1)
        self.edge_adj = pyg.utils.to_dense_adj(self.sparse_edge)
        self.face_adj = pyg.utils.to_dense_adj(self.sparse_face)
        self.full_adj = pyg.utils.to_dense_adj(self.sparse_full)

        self.face_adj_norm = preprocess_adj(self.face_adj, batch_size)
        self.edge_adj_norm = preprocess_adj(self.edge_adj, batch_size)
        self.full_adj_norm = preprocess_adj(self.full_adj, batch_size)

        self.node_embed = MLP(n_node_attr + n_players * n_player_attr, n_embed)
        self.edge_embed = MLP(n_edge_attr + n_players * n_player_attr, n_embed)
        self.face_embed = MLP(n_face_attr, n_embed)

        self.output_embed = MLP(n_embed, n_output, final=True)

        self.power_layers = nn.Sequential(*[

            PowerfulLayer(n_embed, n_embed, self.full_adj_norm)
            for _ in range(n_power_layers)

        ])

    def forward(self, game):
        node_x, edge_x, face_x = extract_attr(game)

        # [N, Features]
        node_embedding = self.node_embed(node_x)
        edge_embedding = self.edge_embed(edge_x)
        face_embedding = self.face_embed(face_x)

        # TODO: Take batch into account
        node_embedding = T.cat((node_embedding, face_embedding))
        node_matrix = T.diag_embed(node_embedding.permute(1, 0)).permute(1, 2, 0)

        face_embedding = face_embedding.repeat_interleave(6, 0)
        if self.undirected_faces:
            face_embedding = torch.cat((face_embedding, face_embedding.flip(0)), dim=0)
        connection_embedding = T.cat((edge_embedding, face_embedding))
        connection_matrix = pyg.utils.to_dense_adj(self.sparse_full, edge_attr=connection_embedding)

        full_matrix = node_matrix + connection_matrix

        full_matrix = self.power_layers(full_matrix)

        return self.output_embed(full_matrix)
