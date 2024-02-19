import torch
import torch as T
import torch.nn as nn
import torch_geometric as pyg

from .Utils import extract_attr, sparse_face_matrix, preprocess_adj, sparse_misc_node
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

        self.n_output = n_output
        self.n_embed = n_embed
        self.undirected_faces = undirected_faces
        self.sparse_edge = sparse_edge
        self.sparse_face = sparse_face_matrix(sparse_face, to_undirected=self.undirected_faces)
        self.sparse_pass_node = sparse_misc_node(sparse_edge.max(), self.sparse_face.max() + 1)
        self.sparse_full = T.cat((self.sparse_edge, self.sparse_face, self.sparse_pass_node), dim=1)

        # TODO: Fix Magic Numbers
        self.node_adj = T.eye(74, dtype=T.long).unsqueeze(0)
        self.node_adj[:, 54:, 54:] = 0
        self.edge_adj = pyg.utils.to_dense_adj(self.sparse_edge, max_num_nodes=74)
        self.face_adj = pyg.utils.to_dense_adj(self.sparse_face, max_num_nodes=74)
        self.face_adj = self.face_adj + T.eye(self.face_adj.shape[-1])
        self.face_adj[0, self.node_adj[0]] = 0
        self.full_adj = pyg.utils.to_dense_adj(self.sparse_full)
        self.full_adj = self.full_adj + T.eye(self.full_adj.shape[-1])
        self.full_mask = self.full_adj > 0

        self.state_matrix = T.zeros((self.full_mask.shape[0], self.full_mask.shape[1], self.full_mask.shape[2], self.n_embed))
        self.out_matrix = T.zeros((self.state_matrix.shape[0], self.state_matrix.shape[1], self.state_matrix.shape[2], self.n_output))
        self.node_mask = self.node_adj > 0
        self.edge_mask = self.edge_adj > 0
        self.face_mask = self.face_adj > 0

        self.face_adj_norm = preprocess_adj(self.face_adj, batch_size)
        self.edge_adj_norm = preprocess_adj(self.edge_adj, batch_size)
        self.full_adj_norm = preprocess_adj(self.full_adj, batch_size)

        self.node_embed = MLP(n_node_attr + n_players * n_player_attr, n_embed)
        self.edge_embed = MLP(n_edge_attr + n_players * n_player_attr, n_embed)
        self.face_embed = MLP(n_face_attr, n_embed)

        self.action_value = MLP(n_embed, n_output, final=True)
        self.state_value = MLP(n_embed, n_output, final=True)

        self.power_layers = nn.Sequential(*[

            PowerfulLayer(n_embed, n_embed, self.full_adj_norm)
            for _ in range(n_power_layers)

        ])

    def forward(self, observation):
        obs_matrix = self.state_matrix.clone()

        obs_matrix[self.node_mask] = self.node_embed(observation[self.node_mask])
        obs_matrix[self.edge_mask] = self.edge_embed(observation[self.edge_mask])
        obs_matrix[self.face_mask] = self.face_embed(observation[self.face_mask][:, :6])

        obs_matrix = self.power_layers(obs_matrix)

        action_matrix = self.out_matrix.clone()
        state_matrix = self.out_matrix.clone()

        action_matrix[self.full_mask] = self.action_value(obs_matrix[self.full_mask])
        state_matrix[self.full_mask] = self.state_value(obs_matrix[self.full_mask])

        return action_matrix + state_matrix - action_matrix.mean(dim=-1, keepdim=True)

    def clone_state(self, other):
        self.load_state_dict(other.state_dict())

    def save(self):
        torch.save(self.state_dict(), f'./{self.name}_state.pth')

    def load(self):
        self.load_state_dict(torch.load(f'./{self.name}_state.pth'))

    def get_dense(self, game):
        node_x, edge_x, face_x = extract_attr(game)

        # Normalize-ish
        node_x /= 2
        face_x /= 12

        pass_x = T.zeros_like(node_x)
        # TODO: Make board size and n_player invariant
        face_x = T.cat((face_x, T.zeros(19, 8)), dim=1)
        node_x = T.cat((node_x, face_x, T.zeros((1, 14))))
        node_matrix = T.diag_embed(node_x.permute(1, 0)).permute(1, 2, 0).unsqueeze(0)
        face_x = face_x.repeat_interleave(6, 0)
        if self.undirected_faces:
            face_x = T.cat((face_x, face_x.flip(0)), dim=0)
        connection_x = T.cat((edge_x, face_x, pass_x))
        connection_matrix = pyg.utils.to_dense_adj(self.sparse_full, edge_attr=connection_x)

        full_matrix = node_matrix + connection_matrix
        return full_matrix
