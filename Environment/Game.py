import os
import random
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch as T
from torch_geometric.utils.convert import to_networkx

from Learner.Agents import BaseAgent, RandomAgent
from .Board import Board
from .Player import Player
from .constants import *


class Game:
    def __init__(self, n_players, max_turns=500, max_no_progress=50, start_episode=0):
        self.max_no_progress = max_no_progress
        self.n_players = n_players
        self.max_turns = max_turns
        self.episode = start_episode

        self.first_turn_village_switch = None
        self.first_turn = None
        self.current_player = None
        self.player_agents = None
        self.players = None
        self.board = None
        self.turn = None
        self.idle_turns = None

        # Placeholder inits
        self.board = Board(self.n_players)
        self.players = [Player(RandomAgent()) for _ in range(self.n_players)]

        self.zero_reward = T.zeros((self.n_players,))
        self.listeners = dict()

    @property
    def current_agent(self):
        return self.players[self.current_player].agent

    def reset(self):
        self.turn = 0
        self.idle_turns = 0
        self.episode += 1
        self.board = Board(self.n_players)
        self.players = [Player(agent) for agent in self.player_agents]
        self.current_player = 0
        self.first_turn = True
        self.first_turn_village_switch = True
        self.publish('reset')
        self.render()

    def set_agents(self, player_agents: [BaseAgent, BaseAgent]):
        assert len(player_agents) == self.n_players
        self.player_agents = player_agents

    def start(self, render=False):
        return self.game_loop(render)

    def step(self, action: tuple[int, int], render: bool = False) -> tuple[T.Tensor, bool, bool]:
        """Take one action step in the game. Returns reward, done, and the actual action."""
        succeeded = True
        if self.first_turn:
            self.first_turn_step(action)
            return self.zero_reward, False, succeeded

        if action[0] == 1:
            if not self.build_road(action[1], self.current_player):
                succeeded = False

        elif action[0] == 2:
            if not self.build_village(action[1], self.current_player):
                succeeded = False

        if (action[0] == 0) | (not succeeded):
            self.idle_turns += 1
            self.current_player = (self.current_player + 1) % self.n_players
            self.turn += 1 / self.n_players
            self.resource_step()
            if render:
                self.render()
        else:
            self.idle_turns = 0

        if self.game_on():
            return self.zero_reward, False, succeeded
        else:
            points = T.tensor([e.points for e in self.players])
            winner = points >= 10
            if winner.sum() >= 1:
                rewards = winner.float() * 2 - 1
            else:
                rewards = T.zeros_like(winner, dtype=T.float)
            return rewards, True, succeeded

    def first_turn_step(self, action) -> None:
        if self.first_turn_village_switch:
            build_succeeded = self.build_village(action[1], self.current_player, first_turn=True)
        else:
            build_succeeded = self.build_road(action[1], self.current_player, first_turn=True)
        if build_succeeded:
            # Check if first turn has ended
            if self.current_player == (self.n_players - 1):
                if self.players[self.current_player].n_roads == 2:
                    self.first_turn = False
                    # Give resources to all villages
                    face_hits = T.nonzero(self.board.state.face_attr)
                    for hit in face_hits:
                        self.give_resource(hit)
            self.current_player = (self.current_player + 1) % self.n_players
            if self.current_player == 0:
                self.first_turn_village_switch = not self.first_turn_village_switch

    def take_first_turn(self):
        for player in range(self.n_players):
            self.build_free_village(player)
            self.build_free_road(player)
        for player in range(self.n_players):
            self.build_free_village(player)
            self.build_free_road(player)

        # Give resources to all villages
        face_hits = T.nonzero(self.board.state.face_attr)
        for hit in face_hits:
            self.give_resource(hit)

    def build_free_village(self, player):
        village_built = False
        while not village_built:
            chosen_building = self.players[player].agent.sample_village(
                self,
                self.board.get_village_mask(player, self.players[player].hand, True),
                player)
            village_built = self.build_village(chosen_building, player, first_turn=True)

    def build_free_road(self, player):
        road_built = False
        while not road_built:
            chosen_road = self.players[player].agent.sample_road(
                self, self.board.get_road_mask(player, self.players[player].hand, True), player)
            road_built = self.build_road(chosen_road, player, first_turn=True)

    def game_loop(self, render=False):
        current_player = 0
        self.take_first_turn()
        while self.game_on():
            # Basic Game Loop
            self.resource_step()
            self.take_turn()

            # Update trackers
            self.current_player = (self.current_player + 1) % self.n_players
            self.turn += 1 / self.n_players

            # Render
            if render:
                self.render()

        points = T.tensor([e.points for e in self.players])
        winner = points.max() == points
        if winner.sum() > 1:
            rewards = T.zeros_like(winner, dtype=T.float)
        else:
            rewards = winner.float() * 2 - 1
        return rewards

    def take_turn(self):
        turn_completed = False
        while not turn_completed:
            action_type, action_index = self.players[self.current_player].sample_action(
                self,
                i_am_player=self.current_player
            )

            turn_completed = self.take_action(action_type, self.current_player, action_index)

    def game_on(self):
        for player in self.players:
            if player.points >= 10:
                return False

        if self.turn >= self.max_turns:
            return False

        # if self.turn == self.max_no_progress:
        #     progress = False
        #     for player in self.players:
        #         if player.points != 2:
        #             progress = True
        #             break
        #     if not progress:
        #         return False

        if (self.idle_turns / 2) > self.max_no_progress:
            return False

        return True

    def resource_step(self):
        dice = random.randint(1, 6) + random.randint(1, 6)

        # Rob if 7
        if dice == 7:
            for hand in self.players:
                hand.rob()

        # Find nodes attain resources
        face_hits = T.argwhere(self.board.state.face_attr == dice)
        for hit in face_hits:
            self.give_resource(hit)

    def give_resource(self, hit):
        node_hits = self.board.state.face_index[hit[0]].flatten()
        player_gains = self.board.state.x[node_hits].sum(0)
        resource = hit[1] - 1  # 0 is bandit for board only, not player
        if resource >= 0:
            for i, gain in enumerate(player_gains):
                self.players[i].add(resource.item(), gain.item())

    def take_action(self, act_type, player, index):
        # act_types
        # 0 = pass
        # 1 = road
        # 2 = village
        # TODO: trade, draw card
        if act_type == 0:
            return True
        if act_type == 1:
            self.build_road(index, player)
            return False
        if act_type == 2:
            self.build_village(index, player)
            return False

    def build_road(self, index, player, first_turn=False):
        # [Bricks, Grains, Ores, Lumbers, Wools]
        if self.board.can_build_road(index, player, first_turn):
            if (
                    (self.players[player].hand[0] > 0)
                    & (self.players[player].hand[3] > 0)
            ):
                self.players[player].sub(0, 1)
                self.players[player].sub(3, 1)
                self.players[player].n_roads += 1
                self.board.update_edges(index, player)
                return True
        return False

    def build_village(self, index, player, first_turn=False):
        can_build, size = self.board.can_build_village(index, player, first_turn)
        if can_build:
            if size == 0:
                # [Bricks, Grains, Ores, Lumbers, Wools]
                if (
                        (self.players[player].hand[0] > 0)
                        & (self.players[player].hand[1] > 0)
                        & (self.players[player].hand[3] > 0)
                        & (self.players[player].hand[4] > 0)
                ):
                    self.players[player].sub(0, 1)
                    self.players[player].sub(1, 1)
                    self.players[player].sub(3, 1)
                    self.players[player].sub(4, 1)
                    self.board.state.x[index, player] = size + 1
                    self.players[player].points += 1
                    self.players[player].n_villages += 1
                    return True
            if size == 1:
                # [Bricks, Grains, Ores, Lumbers, Wools]
                if (
                        (self.players[player].hand[1] > 2)
                        & (self.players[player].hand[2] > 3)
                ):
                    self.players[player].sub(1, 2)
                    self.players[player].sub(2, 3)
                    self.board.state.x[index, player] = size + 1
                    self.players[player].points += 1
                    return True
        return False

    def subscribe(self, event_type, listener):
        if event_type not in self.listeners:
            self.listeners[event_type] = [listener]
        else:
            self.listeners[event_type].append(listener)

    def publish(self, event_type: str, data: Any = None) -> None:
        if event_type in self.listeners:
            for listener in self.listeners[event_type]:
                listener(data)

    def render(self, training_img=False, debug=False):
        if self.n_players != 2:
            print("Rendering not implemented for more than two players.")
            return
        plt.figure(figsize=(FIG_X, FIG_Y))
        plt.axis('off')  # Hide axes
        plt.axis('equal')

        turn_appendix = int((self.turn - int(self.turn)) * self.n_players)

        ###
        # Make node colors
        ###
        node_colors = [''] * self.board.state.num_nodes
        for i in range(self.board.state.num_nodes):
            if self.board.state.x[i][0] == 1:
                node_colors[i] = '#ADBC9F'  # light green
            elif self.board.state.x[i][0] == 2:
                node_colors[i] = '#436850'  # dark green
            elif self.board.state.x[i][1] == 1:
                node_colors[i] = '#7FC7D9'  # light blue
            elif self.board.state.x[i][1] == 2:
                node_colors[i] = '#365486'  # dark blue
            else:
                node_colors[i] = '#FBFADA'  # neutral beige

        ###
        # Make edge weights
        ###
        edge_colors = [''] * self.board.state.num_edges
        for i in range(self.board.state.num_edges):
            if self.board.state.edge_attr[i][0] == 1:
                edge_colors[i] = '#265073'  # very deep green
            elif self.board.state.edge_attr[i][1] == 1:
                edge_colors[i] = '#D9917F'  # redish
            else:
                edge_colors[i] = '#ADC7C4'  # light teal

        ###
        # Draw text on faces
        ###
        for face_idx, node_indices in enumerate(self.board.state.face_index):
            # Extract positions for nodes in this face
            node_positions = np.array([self.board.state.pos[node] for node in node_indices])

            # Calculate centroid of the face
            centroid = node_positions.mean(axis=0)

            # Prepare the text based on face attribute (e.g., just converting attribute to string here)
            face_type = TILE_NAMES[self.board.state.face_attr[face_idx].argmax().item()]
            face_dice = str(int(self.board.state.face_attr[face_idx].max().item()))
            text = face_dice + '\n' + face_type
            if debug:
                text += '\n' + str(face_idx)

            # Annotate the graph
            plt.text(centroid[0], centroid[1], text, ha='center', va='center')

        ###
        # Player state texts
        ###
        text_top_right = "Player 1:\n" + str(self.players[0])
        text_bottom_right = "Player 2:\n" + str(self.players[1])
        text_top_left = f"Turn {int(self.turn + 1)} - {int(turn_appendix + 1)}"

        # Margins from the edges of the figure
        margin_x = 0.02  # X margin as a fraction of total width
        margin_y = 0.02  # Y margin as a fraction of total height

        # Calculate positions based on figure dimensions and margins
        bottom_left_pos = (margin_x * FIG_X, margin_y * FIG_Y)
        top_right_pos = ((1 - margin_x) * FIG_X, (1 - margin_y) * FIG_Y)
        bottom_right_pos = ((1 - margin_x) * FIG_X, margin_y * FIG_Y)
        top_left_pos = (margin_x * FIG_X, (1 - margin_y) * FIG_Y)

        # Adding custom text at specified positions
        # plt.text(bottom_left_pos[0], bottom_left_pos[1], text_bottom_left, ha='left', va='bottom')
        plt.text(top_right_pos[0], top_right_pos[1], text_top_right, ha='right', va='top')
        plt.text(bottom_right_pos[0], bottom_right_pos[1], text_bottom_right, ha='right', va='bottom')
        plt.text(top_left_pos[0], top_left_pos[1], text_top_left, ha='left', va='top')

        ###
        # Draw
        ###
        s = to_networkx(self.board.state, node_attrs=['pos', 'x'], edge_attrs=['edge_attr'])
        nx.draw_networkx_nodes(s, self.board.state.pos, node_color=node_colors)
        nx.draw_networkx_edges(s, self.board.state.pos, edge_color=edge_colors, width=2)
        if debug:
            nx.draw_networkx_labels(s, self.board.state.pos)

        ###
        # Save
        ###
        if training_img:
            folder = f"./Renders/Training/"
            filename = f"Ep {self.episode}-{int(self.turn)} -- {self.player_agents[0]} vs {self.player_agents[1]}.png"
        else:
            folder = f"./Renders/Test/Episode {int(self.episode)}/"
            filename = f"Turn {int(self.turn)}_{turn_appendix}.png"
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, filename))
        plt.close()
