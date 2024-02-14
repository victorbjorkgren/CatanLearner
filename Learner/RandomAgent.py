from random import randint

class RandomAgent:
    def __init__(self):
        pass

    def sample_action(self, board, players, i_am_player):
        action_type = randint(0, 2)

        if action_type == 1:
            index = randint(0, board.state.num_edges // 2 - 1)
        elif action_type == 2:
            index = randint(0, board.state.num_nodes - 1)
        else:
            index = 0

        return action_type, index