import os
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils.convert import to_networkx

from .Board import Board
from .Player import Player
from .constants import *


class Game:
    def __init__(self, player_agents: [], max_turns=100, start_episode=0):
        assert len(player_agents) == 2
        self.players = None
        self.board = None
        self.turn = None

        self.player_agents = player_agents
        self.max_turns = max_turns
        self.episode = start_episode

        self.n_players = len(player_agents)

        self.reset()

    def reset(self):
        self.turn = 0
        self.episode += 1
        self.board = Board(self.n_players)
        self.players = [Player(agent) for agent in self.player_agents]

    def start(self, render=False):
        self.game_loop(render)

    def game_loop(self, render=False):
        current_player = 0
        while self.game_on():
            # Basic Game Loop
            self.resource_step()
            self.take_turn(current_player)

            # Update trackers
            current_player = (current_player + 1) % self.n_players
            self.turn += 1 / self.n_players

            # Render
            if render:
                self.render()

    def take_turn(self, current_player):
        turn_completed = False
        while not turn_completed:
            action_type, action_index = self.players[current_player].sample_action(
                self.board,
                self.players,
                i_am_player=current_player
            )

            turn_completed = self.take_action(action_type, current_player, action_index)

    def game_on(self):
        if self.turn >= self.max_turns:
            return False
        for hand in self.players:
            if hand.points >= 10:
                return False
        return True

    def resource_step(self):
        dice = random.randint(1, 6) + random.randint(1, 6)

        # Rob if 7
        if dice == 7:
            for hand in self.players:
                hand.rob()

        # Find nodes attain resources
        face_hits = torch.argwhere(self.board.state.face_attr == dice)
        for hit in face_hits:
            node_hits = self.board.state.face_index[hit[0]].flatten()
            player_gains = self.board.state.x[node_hits].sum(0)
            resource = hit[1] - 1  # 0 is bandit for board only, not player
            if resource > 0:
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
            self.build_village(index, 1, player)
            return False

    def build_road(self, index, player):
        # [Bricks, Grains, Ores, Lumbers, Wools]
        if self.board.can_build_road(index, player):
            if (
                    (self.players[player].hand[0] > 0)
                    & (self.players[player].hand[3] > 0)
            ):
                self.players[player].sub(0, 1)
                self.players[player].sub(3, 1)
                self.board.update_edges(index, player)
                return True
        return False

    def build_village(self, index, size, player):
        can_build, size = self.board.can_build_village(index, player)
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
                    self.board.state.x[index, player] = size
                    self.players[player].points += 1
                    return True
            if size == 1:
                # [Bricks, Grains, Ores, Lumbers, Wools]
                if (
                        (self.players[player].hand[1] > 2)
                        & (self.players[player].hand[2] > 3)
                ):
                    self.players[player].sub(1, 2)
                    self.players[player].sub(2, 3)
                    self.board.state.x[index, player] = size
                    self.players[player].points += 1
                    return True
        return False

    def render(self):
        if self.n_players != 2:
            print("Rendering not implemented for more than two players.")
        plt.figure(figsize=(FIG_X, FIG_Y))

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
            elif self.board.state.edge_attr[i][0] == 1:
                edge_colors[i] = '#1E3C50'
            else:
                edge_colors[i] = '#adc7c4'  # light teal

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
            text = face_dice + '\n' + face_type + '\n' + str(face_idx)

            # Annotate the graph
            plt.text(centroid[0], centroid[1], text, ha='center', va='center')

        ###
        # Player state texts
        ###
        text_top_right = "Player 1:\n" + str(self.players[0])
        text_bottom_right = "Player 2:\n" + str(self.players[1])
        text_top_left = f"Turn {self.turn + 1}"

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
        nx.draw_networkx_labels(s, self.board.state.pos)
        plt.axis('off')  # Hide axes
        plt.axis('equal')

        ###
        # Save
        ###
        turn_appendix = int((self.turn - int(self.turn)) * self.n_players)
        folder = f"./Renders/Episode {int(self.episode)}/"
        filename = f"Turn {int(self.turn)}_{turn_appendix}.png"
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, filename))
        plt.close()
