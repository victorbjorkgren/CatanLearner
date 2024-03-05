import os
import random as r
from typing import Optional, List

from Learner.Agents import QAgent


class AgentTracker:
    def __init__(self, eps_min: float, eps_max: float, eps_zero: float, eps_one: float):
        self.eps_one = eps_one
        self.eps_zero = eps_zero
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.agent_list: Optional[List[QAgent]] = None
        self.contestants = None
        self.has_titan = False
        self.has_agents = False
        self.has_loaded = False

    def register_agents(self, agent_list):
        self.has_agents = True
        self.agent_list = agent_list

        self.assign_titan()
        self.load_contestants('random')

    def assign_titan(self):
        assert self.has_agents
        self.has_titan = True

        for agent in self.agent_list:
            agent.unset_titan()
        self.agent_list[0].set_titan()

    def load_contestants(self, method):
        assert method in ['random', 'weighted']
        assert self.has_agents
        assert self.has_titan

        self.has_loaded = True

        os.makedirs('./PastTitans/', exist_ok=True)
        self.contestants = os.listdir('./PastTitans/')
        if method == 'random':
            for agent in self.agent_list:
                if agent.is_titan:
                    champion = 'latest'
                elif len(self.contestants) == 0:
                    champion = 'latest'
                elif r.random() < .1:
                    champion = 'latest'
                else:
                    champion = r.choice(self.contestants)

                zero_one_p = r.random()
                if agent.is_titan | (r.random() < self.eps_one):
                    epsilon = 1.
                elif (r.random() < (self.eps_zero + self.eps_one)):
                    epsilon = 0.
                else:
                    epsilon = r.uniform(self.eps_min, self.eps_max)
                agent.load_state(champion, epsilon)
        else:
            raise NotImplementedError("Weighted Contestant Loading Not Implemented")

    def shuffle_agents(self):
        r.shuffle(self.agent_list)

    def update_elo_multiplayer(self, player_elos, player_ranks, k=32):
        """
        Update ELO ratings for a four-player game.

        Parameters:
        - player_elos: List of ELO ratings of the players before the match. Order corresponds to `player_ranks`.
        - player_ranks: List of player ranks after the match (1 for the winner, 4 for the last place, etc.).
                        Lower numbers indicate better performance.
        - k: The maximum change in rating.

        Returns:
        - updated_elos: New ELO ratings of the players.
        """
        n = len(player_elos)
        updated_elos = player_elos.copy()

        for i in range(n):
            actual_score = (n - player_ranks[i]) / (n - 1)  # Normalize rank to a score between 0 and 1
            for j in range(n):
                if i != j:
                    expected_score_i_vs_j = 1 / (1 + 10 ** ((player_elos[j] - player_elos[i]) / 400))
                    updated_elos[i] += k * (actual_score - expected_score_i_vs_j)

        return updated_elos

    def get_titan(self) -> QAgent:
        for agent in self.agent_list:
            if agent.is_titan:
                return agent
