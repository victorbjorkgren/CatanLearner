from dataclasses import dataclass, field

from torch import Tensor

from Learner.Utility.ActionTypes import BaseAction, SparsePi
from Learner.Utility.Utils import Holders


@dataclass
class GameState(Holders):
    node_features: Tensor
    edge_features: Tensor
    face_features: Tensor
    game_features: Tensor

@dataclass
class PPOActionPack(Holders):
    action: BaseAction = field(metadata={'not_stackable': True})
    masks: SparsePi
    log_prob: Tensor
    value: Tensor
    lstm_h: Tensor
    lstm_c: Tensor


@dataclass
class PPORewardPack(Holders):
    reward: Tensor
    episode: Tensor
    done: Tensor
    player: Tensor


@dataclass
class PPOTransition(Holders):
    state: GameState
    seq_lens: Tensor
    action_pack: PPOActionPack
    reward_pack: PPORewardPack

    @property
    def as_net_input(self):
        state = self.state
        seq_lens = self.seq_lens
        lstm_h = self.action_pack.lstm_h.squeeze(2).permute(1, 0, 2)
        lstm_c = self.action_pack.lstm_c.squeeze(2).permute(1, 0, 2)
        return NetInput(state, seq_lens, lstm_h, lstm_c)


@dataclass
class NetInput(Holders):
    state: GameState
    seq_lens: Tensor
    lstm_h: Tensor
    lstm_c: Tensor


@dataclass
class NetOutput(Holders):
    pi: SparsePi
    state_value: Tensor
    hn: Tensor
    cn: Tensor
