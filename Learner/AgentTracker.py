import os
import pickle

import numpy as np
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

        self.checkpoint_elo = {}
        self.titan_elo = 1000
        self.random_elo = 1000

    def register_agents(self, agent_list):
        self.has_agents = True
        self.agent_list = agent_list

        for agent in agent_list:
            agent.tracker_instance = self

        self.set_titan()
        self.load_contestants('random')

    def set_titan(self):
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
        champions = []
        if method == 'random':
            for agent in self.agent_list:
                if agent.is_titan:
                    champions.append('latest')
                elif len(self.contestants) == 0:
                    champions.append('latest')
                elif r.random() < .1:
                    champions.append('latest')
                else:
                    champions.append(r.choice(self.contestants))

        else:
            champions = self.select_checkpoint(len(self.agent_list))

        epsilons = r.choices(
            [0., r.uniform(self.eps_min, self.eps_max), 1.],
            weights=[self.eps_zero, 1 - (self.eps_one + self.eps_zero), self.eps_one],
            k=len(self.agent_list)
        )


        # TODO: WAS HERE: GET ELO AND LOAD
        for i, agent in enumerate(self.agent_list):
        agent.load_state(champion, epsilon)

    def shuffle_agents(self):
        r.shuffle(self.agent_list)

    def update_elo(self, k=32):
        """
        Update ELO ratings for a four-player game.

        Parameters:
        - player_elos: List of ELO ratings of the players before the match. Order corresponds to `player_ranks`.
        - player_ranks: List of player scores after the match (1 for the winner, 4 for the last place, etc.).
                        Higher numbers indicate better performance.
        - k: The maximum change in rating.

        Returns:
        - updated_elos: New ELO ratings of the players.
        """
        n = len(self.agent_list)
        player_elo = []
        for agent in self.agent_list:
            if agent.my_name in ['latest', 'Titan']:
                player_elo.append(self.titan_elo)
            elif agent.epsilon == 0.:
                player_elo.append(self.random_elo)
            else:
                player_elo.append(self.checkpoint_elo[agent.my_name])
        updated_elo = player_elo.copy()

        for i in range(n):
            actual_score = self.agent_list[i].episode_score / 10  # Normalize rank to a score between 0 and 1
            for j in range(n):
                if i != j:
                    expected_score_i_vs_j = 1 / (1 + 10 ** ((player_elo[j] - player_elo[i]) / 400))
                    updated_elo[i] += k * (actual_score - expected_score_i_vs_j)

        for i, agent in enumerate(self.agent_list):
            if agent.my_name in ['latest', 'Titan']:
                self.titan_elo = updated_elo[i]
            else:
                self.checkpoint_elo[agent.my_name] = updated_elo[i]

        self.save_elo()

    def get_titan(self) -> QAgent:
        assert self.has_titan
        for agent in self.agent_list:
            if agent.is_titan:
                return agent

    @staticmethod
    def gaussian_pdf(x, mu, sigma):
        """Calculate Gaussian probability density function."""
        coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
        exponent = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return coefficient * exponent

    def calculate_weights(self, sigma=100):
        """
        Calculate normalized Gaussian weights for each checkpoint based on their ELO difference.

        Parameters:
        - checkpoint_elos: List of ELO ratings of available checkpoints.
        - agent_elo: ELO rating of the agent selecting a checkpoint.
        - sigma: Standard deviation of the Gaussian distribution (default: 100).

        Returns:
        - A list of weights corresponding to each checkpoint.
        """
        assert self.checkpoint_elo is not None

        elo_differences = np.array([self.titan_elo - elo for elo in self.checkpoint_elo.values()])
        weights = self.gaussian_pdf(elo_differences, mu=0, sigma=sigma)
        normalized_weights = weights / np.sum(weights)
        return normalized_weights

    def select_checkpoint(self, k, sigma=100):
        """
        Select a checkpoint based on Gaussian-weighted ELO differences.

        Parameters:
        - checkpoint_elos: List of ELO ratings of available checkpoints.
        - agent_elo: ELO rating of the agent selecting a checkpoint.
        - sigma: Standard deviation for the Gaussian distribution (adjusts the selection spread).

        Returns:
        - The ELO rating of the selected checkpoint.
        """
        assert self.checkpoint_elo is not None

        weights = self.calculate_weights(sigma)
        selected_checkpoints = r.choices(list(self.checkpoint_elo.keys()), weights=weights, k=k)
        return selected_checkpoints

    def save_elo(self):
        with open('./CheckPointELO.pkl', 'wb') as f:
            pickle.dump(self.checkpoint_elo, f)

    def load_elo(self):
        try:
            with open('./CheckPointELO.pkl', 'rb') as f:
                self.checkpoint_elo = pickle.load(f)
        except FileNotFoundError:
            self.checkpoint_elo = {}

        checkpoints = os.listdir('./PastTitans/')
        for e in checkpoints:
            if e not in self.checkpoint_elo:
                self.checkpoint_elo[e] = 1000



