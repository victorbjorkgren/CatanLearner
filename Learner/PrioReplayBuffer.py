from collections import deque

import numpy as np
import torch as T
import torch_geometric as pyg

from Learner.Utils import extract_attr, sparse_face_matrix, sparse_misc_node


def assert_capacity(n):
    # Check that n is an integer greater than 0 and n & (n - 1) == 0
    assert isinstance(n, int) and n > 0 and n & (n - 1) == 0, "Value is not a power of 2"


# class StateExtractor:
#     def __init__(self, game, undirected_faces=True):
#         self.undirected_faces = undirected_faces
#         sparse_edge = game.board.state.edge_index
#         sparse_face = game.board.state.face_index
#         self.sparse_edge = sparse_edge
#         self.sparse_face = sparse_face_matrix(sparse_face, to_undirected=self.undirected_faces)
#         self.sparse_pass_node = sparse_misc_node(sparse_edge.max(), self.sparse_face.max() + 1)
#         self.sparse_full = T.cat((self.sparse_edge, self.sparse_face, self.sparse_pass_node), dim=1)
#
#     def get_dense(self, game):
#         node_x, edge_x, face_x = extract_attr(game)
#
#         # Normalize-ish
#         node_x /= 2
#         face_x /= 12
#
#         pass_x = T.zeros_like(node_x)
#         # TODO: Make board size and n_player invariant
#         face_x = T.cat((face_x, T.zeros(19, 8)), dim=1)
#         node_x = T.cat((node_x, face_x, T.zeros((1, 14))))
#         node_matrix = T.diag_embed(node_x.permute(1, 0)).permute(1, 2, 0).unsqueeze(0)
#         face_x = face_x.repeat_interleave(6, 0)
#         if self.undirected_faces:
#             face_x = T.cat((face_x, face_x.flip(0)), dim=0)
#         connection_x = T.cat((edge_x, face_x, pass_x))
#         connection_matrix = pyg.utils.to_dense_adj(self.sparse_full, edge_attr=connection_x)
#
#         full_matrix = node_matrix + connection_matrix
#         return full_matrix
#
#     def get_sparse(self, game):
#         return extract_attr(game)


def reorder_players(tensor, i):
    return T.cat((tensor[..., [i]], tensor[..., :i], tensor[..., i + 1:]), dim=-1)


class PrioReplayBuffer:
    def __init__(self, capacity, alpha, beta):
        assert_capacity(capacity)

        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta

        # self.priority_sum = [0] * (2 * capacity)
        # self.priority_min = [float('inf')] * (2 * capacity)

        self.max_priority = 1.

        # TODO: Make board size and n_player invariant
        # TODO: Fix Magic Numbers
        self.data = {
            'state': T.zeros((capacity, 74, 74, 14)),
            'action': T.zeros((capacity, 2)),
            'new_state': T.zeros((capacity, 74, 74, 14)),
            'reward': T.zeros((capacity, 2)),
            'done': T.zeros((capacity,)),
            'prio': np.zeros((capacity,))
        }

        self.next_idx = 0
        self.size = 0

    def add(self, state, action, new_state, reward, done, i_am_player):
        idx = self.next_idx

        # TODO: Find root cause of this issue
        if len(action.shape) > 1:
            action = action.squeeze()

        self.data['state'][idx] = reorder_players(state, i_am_player)
        self.data['action'][idx] = action
        self.data['new_state'][idx] = reorder_players(new_state, i_am_player)
        self.data['reward'][idx] = reward[i_am_player]
        self.data['done'][idx] = done
        self.data['prio'][idx] = self.max_priority ** self.alpha

        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, n):
        prob = self.data['prio'] / self.data['prio'].sum()

        sample_inds = np.random.choice(self.capacity, n, p=prob)

        weights = (self.size * prob[sample_inds]) ** (-self.beta)
        weights /= weights.max()

        samples = {
            'inds': sample_inds,
            'weights': weights
        }
        for k, v in self.data.items():
            samples[k] = v[sample_inds]

        return samples

    def update_prio(self, ind: T.Tensor, prio: T.Tensor):
        self.data['prio'][ind] = prio.clamp_max(self.max_priority)

