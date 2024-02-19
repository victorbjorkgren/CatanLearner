import torch as T


def extract_attr(game):
    node_x = game.board.state.x
    edge_x = game.board.state.edge_attr
    face_x = game.board.state.face_attr

    player_states = T.cat([ps.state[None, :] for ps in game.players], dim=1)
    node_x = T.cat((node_x, player_states.repeat((node_x.shape[0], 1))), dim=1)
    edge_x = T.cat((edge_x, player_states.repeat((edge_x.shape[0], 1))), dim=1)

    return node_x, edge_x, face_x


def sparse_face_matrix(face_index, to_undirected):
    n = face_index.size(0)  # Number of faces
    num_nodes_per_face = face_index.size(1)  # Should be 6
    face_indices = T.arange(54, 54 + n).repeat_interleave(num_nodes_per_face)

    node_indices = face_index.flatten()

    # Create the [2, K] matrix by stacking face_indices and node_indices
    connections = T.stack([face_indices, node_indices], dim=0)

    if to_undirected:
        connections = T.cat((connections, connections.flip(0)), dim=1)
    return connections


def sparse_misc_node(node_n, misc_n):
    node_range = T.arange(node_n + 1)
    sparse = T.stack((node_range, T.full_like(node_range, misc_n)), dim=0)
    return sparse


def preprocess_adj(adj, batch_size):
    I = T.eye(adj.size(1)).to(adj.device)
    A_hat = adj[0] + I
    D_hat_diag = T.sum(A_hat, dim=1).pow(-0.5)
    D_hat = T.diag(D_hat_diag)
    adj_normalized = T.mm(T.mm(D_hat, A_hat), D_hat)
    return adj_normalized.repeat((batch_size, 1, 1))


def get_masks(game, i_am_player):
    road_mask = game.board.get_road_mask(i_am_player, game.players[i_am_player].hand, game.first_turn)
    village_mask = game.board.get_village_mask(i_am_player, game.players[i_am_player].hand, game.first_turn)
    return road_mask, village_mask
