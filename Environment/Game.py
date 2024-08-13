import os
import random
from typing import Any, Tuple, Optional, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch as T
from PIL import Image
from torch import Tensor
from torch_geometric.utils.convert import to_networkx

from Learner.Agents import BaseAgent
from Learner.Utility.ActionTypes import BaseAction, BuildAction, TradeAction, NoopAction, SettlementAction, RoadAction, \
    FlatPi
from Learner.Utility.DataTypes import GameState
from Learner.Utility.Utils import TensorUtils, get_unique_filename
from Learner.constants import LATENT_REWARD
from .Board import Board
from .Player import Player
from .constants import *


# TODO: Add Cards and Robber mechanics

class Game:
    def __init__(self, n_players, max_turns=500, max_no_progress=100, start_episode=0):
        self.max_no_progress = max_no_progress
        self.n_players = n_players
        self.max_turns = max_turns
        self.episode = start_episode

        self.player_agents: Optional[List[BaseAgent]] = None
        self.first_turn_village_switch: Optional[bool] = None
        self.first_turn: Optional[bool] = None
        self.current_player: Optional[int] = None
        self.turn: Optional[int] = None
        self.idle_turns: Optional[int] = None
        self.build_cmap: List[str] = None
        self.dark_cmap: List[str] = None

        # Placeholder inits
        self.board = Board(self.n_players, self)
        self.players = [Player(None)] * self.n_players

        self.zero_reward = 0
        self.failed_action_penalty = -LATENT_REWARD
        self.listeners = dict()

    @property
    def current_agent(self):
        return self.players[self.current_player].agent

    def get_game_attr(self):
        return T.cat([player.state.clone()[None, :] for player in self.players], dim=1).float()

    @property
    def num_game_attr(self):
        return self.get_game_attr().shape[1]

    def extract_attributes(self):
        node_x = self.board.state.x.clone()
        edge_x = self.board.state.edge_attr.clone()
        face_x = self.board.state.face_attr.clone()
        game_x = self.get_game_attr()

        for i in range(self.n_players):
            road_mask = self.board.sparse_road_mask(i, self.players[i].hand, self.first_turn, self.first_turn_village_switch)
            village_mask = self.board.sparse_village_mask(i, self.players[i].hand, self.first_turn, self.first_turn_village_switch)

            # Pad with action possible mask
            node_x = torch.cat((node_x, torch.zeros((N_NODES, 1))), dim=1)
            edge_x = torch.cat((edge_x, torch.zeros((N_ROADS * 2, 1))), dim=1)
            node_x[village_mask, -1] = 1
            edge_x[road_mask, -1] = 1
        # node_x = T.cat((node_x, player_states.repeat((node_x.shape[0], 1))), dim=1)
        # edge_x = T.cat((edge_x, player_states.repeat((edge_x.shape[0], 1))), dim=1)
        
        node_x = TensorUtils.signed_log(node_x)
        edge_x = TensorUtils.signed_log(edge_x)
        face_x = TensorUtils.signed_log(face_x)
        game_x = TensorUtils.signed_log(game_x)

        return GameState(node_x[None, None, ...], edge_x[None, None, ...], face_x[None, None, ...], game_x[None, None, ...])

    def reset(self):
        self.turn = 0
        self.idle_turns = 0
        self.episode += 1
        self.board = Board(self.n_players, self)
        self.players = [Player(agent) for agent in self.player_agents]
        self.current_player = 0
        self.first_turn = True
        self.first_turn_village_switch = True
        self.publish('reset')

    def register_agents(self, player_agents: [BaseAgent, BaseAgent]):
        assert len(player_agents) == self.n_players
        self.player_agents = player_agents

    def decode_build_action(self, action: BuildAction):
        build_action = action.mat_index
        if build_action[0] != build_action[1]:
            bool_hit = (
                    (self.board.state.edge_index[0] == build_action[0])
                    & (self.board.state.edge_index[1] == build_action[1])
            )
            index = torch.argwhere(bool_hit).item()
            return torch.tensor((1, index))

        elif build_action[0] == build_action[1]:
            if build_action[0] >= 54:
                raise RuntimeError("Non-node index returned for building settlement")
            return torch.tensor((2, build_action[0]))
        else:
            raise RuntimeError("Invalid Build Action in QAgent")

    def step(self, action: BaseAction, render: bool = False) -> tuple[float, bool, bool]:
        """Take one action step in the game. Returns reward, done, and the actual action."""
        succeeded = True
        reward = 0.
        if self.first_turn:
            succeeded, reward = self.first_turn_step(action)
            if succeeded:
                return reward, False, True
            else:
                return self.failed_action_penalty + reward, False, True

        if isinstance(action, RoadAction):
            if not self.build_road(action.index.item(), self.current_player):
                succeeded = False
        elif isinstance(action, SettlementAction):
            if not self.build_village(action.index, self.current_player):
                succeeded = False
            else:
                reward += 1
        elif isinstance(action, TradeAction):
            succeeded = self.players[self.current_player].trade(give_ind=action.give, get_ind=action.get)

        if isinstance(action, NoopAction):
            self.idle_turns += 1
            self.current_player = (self.current_player + 1) % self.n_players
            self.turn += 1 / self.n_players
            self.resource_step()
        else:
            self.idle_turns = 0

        if self.game_on():
            return (not succeeded) * self.failed_action_penalty + reward, False, succeeded
        else:
            points = self.players[self.current_player].points
            winner = points >= 10
            if winner:
                self.player_agents[self.current_player].register_victory()
            return reward, True, succeeded

    def can_trade(self, player: int) -> Tensor:
        """
        Returns the inds that the player can trade at the asked rate
        :param player: index for the player to do the trade
        :param rate: rate for the trade
        :return: Tensor of tradeable inds
        """
        if self.first_turn:
            return torch.tensor([])
        return torch.argwhere(self.players[player].hand >= self.players[player].best_trade_rate)

    def can_no_op(self):
        return not self.first_turn

    def first_turn_step(self, action: BuildAction) -> Tuple[bool, float]:
        assert isinstance(action, RoadAction) | isinstance(action, SettlementAction)
        reward = 0.
        if self.first_turn_village_switch:
            if isinstance(action, SettlementAction):
                build_succeeded = self.build_village(action.index, self.current_player, first_turn=True)
            else:
                build_succeeded = False
        else:
            if isinstance(action, RoadAction):
                build_succeeded = self.build_road(action.index.item(), self.current_player, first_turn=True)
            else:
                build_succeeded = False
        if build_succeeded:
            if self.first_turn_village_switch:
                reward += 1

            # Give resources to all villages, done twice for simplicity of reward assignment
            if self.players[self.current_player].n_roads == 2:
                face_hits = torch.nonzero(self.board.state.face_attr)
                for hit in face_hits:
                    self.give_resource(hit)
                good_place_to_start = self.players[self.current_player].hand >= torch.tensor([1, 1, 0, 1, 1])
                if good_place_to_start.all():
                    reward += GOOD_START_REWARD

                # Check if first turn has ended
                if self.current_player == (self.n_players - 1):
                    self.first_turn = False

            # Switch active player and first turn switch
            self.current_player = (self.current_player + 1) % self.n_players
            if self.current_player == 0:
                self.first_turn_village_switch = not self.first_turn_village_switch
            return True, reward
        else:
            return False, reward

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
        face_hits = torch.argwhere(self.board.state.face_attr == dice)
        for hit in face_hits:
            self.give_resource(hit)

    def give_resource(self, hit):
        node_hits = self.board.state.face_index[hit[0]].flatten()
        player_gains = self.board.state.x[node_hits, :self.n_players].sum(0)
        resource = hit[1] - 1  # 0 is bandit for board only, not player
        if resource >= 0:
            for i, gain in enumerate(player_gains):
                self.players[i].add(resource.item(), gain.item())
                self.players[i].latent_reward += LATENT_REWARD

    def build_road(self, index, player, first_turn=False):
        # [Bricks, Grains, Ores, Lumbers, Wools]
        if self.board.can_build_road(index, player, first_turn):
            if (
                    (self.players[player].hand[0] > 0)
                    & (self.players[player].hand[3] > 0)
                    & (self.players[player].n_roads < 15)
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
                    self.players[player].n_settlements += 1
                    node_trade_rate = self.board.get_node_trade_rate(index)
                    self.players[player].update_best_trade_rate(node_trade_rate)
                    return True
            if size == 1:
                # [Bricks, Grains, Ores, Lumbers, Wools]
                if (
                        (self.players[player].hand[1] >= 2)
                        & (self.players[player].hand[2] >= 3)
                ):
                    self.players[player].sub(1, 2)
                    self.players[player].sub(2, 3)
                    self.board.state.x[index, player] = size + 1
                    self.players[player].points += 1
                    self.players[player].n_settlements -= 1
                    self.players[player].n_cities += 1
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

    def init_cmap(self, n, value_factor=.5):
        # Generate n distinct colors using matplotlib's built-in function
        colors = plt.cm.get_cmap('hsv', n)

        # Convert these colors to hex format and generate a darker version
        hex_colors = []
        darker_hex_colors = []
        for i in range(n):
            rgb = colors(i)[:3]  # Get the RGB values
            hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb)
            hex_colors.append(hex_color)

            # Generate a darker version by reducing the value
            darker_rgb = tuple(c * value_factor for c in rgb)
            darker_hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in darker_rgb)
            darker_hex_colors.append(darker_hex_color)

        self.build_cmap = hex_colors
        self.dark_cmap = darker_hex_colors

    def render(self, render_type='training', debug=False):
        assert render_type in ['training', 'testing', 'init']
        if self.build_cmap is None:
            self.init_cmap(self.n_players)

        plt.figure(figsize=(FIG_X, FIG_Y))
        plt.axis('off')  # Hide axes
        plt.axis('equal')

        turn_appendix = int((self.turn - int(self.turn)) * self.n_players)

        ###
        # Make node colors
        ###
        node_colors = ['#FBFADA'] * self.board.state.num_nodes  # neutral beige
        for i in range(self.board.state.num_nodes):
            for j in range(self.n_players):
                if self.board.state.x[i][j] == 1:
                    node_colors[i] = self.build_cmap[j]
                elif self.board.state.x[i][j] == 2:
                    node_colors[i] = self.dark_cmap[j]

        ###
        # Make edge weights
        ###
        edge_colors = ['#FBFADA'] * self.board.state.num_edges  # neutral beige
        for i in range(self.board.state.num_edges):
            for j in range(self.n_players):
                if self.board.state.edge_attr[i][0] == 1:
                    edge_colors[i] = self.build_cmap[j]

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
        text_top_left = f"Turn {int(self.turn + 1)} - {int(turn_appendix + 1)}"

        # Margins from the edges of the figure
        margin_x = 0.02  # X margin as a fraction of total width
        margin_y = 0.02  # Y margin as a fraction of total height

        # Calculate positions based on figure dimensions and margins
        top_left_pos = (margin_x * FIG_X, (1 - margin_y) * FIG_Y)

        # Adding custom text
        texts = [f"Player {i_}:\n" + str(self.players[i_]) for i_ in range(self.n_players)]

        # Position each text box
        y_pos = (1 - margin_y) * FIG_Y
        w = (FIG_Y - (self.n_players + 1) * margin_y) / self.n_players
        for i_, text_ in enumerate(texts):
            plt.text((1 - margin_x) * FIG_X, y_pos, text_, wrap=True, ha='right', va='top', fontsize=12,
                    bbox=dict(facecolor='white', edgecolor=self.build_cmap[i_], boxstyle='round,pad=0.5'))
            y_pos -= (w + margin_y)

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
        if render_type == 'training':
            folder = f"./Renders/Training/"
            filename = f"Ep {self.episode}-{int(self.turn)} -- Players : {str.join(', ', [str(a) for a in self.player_agents])}.png"
        elif render_type == 'testing':
            folder = f"./Renders/Test/Episode {int(self.episode)}/"
            filename = f"Turn {int(self.turn)}_{turn_appendix}.png"
        elif render_type == 'init':
            folder = f"./Renders/Initial/"
            filename = f"{self.episode}.png"
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, filename))
        plt.close()

    def render_probs(self, flat_pi: FlatPi, render_type: str):
        assert render_type in ['training', 'testing', 'init']

        node_p, edge_p, trade_p, noop_p = flat_pi.unstack_parts()
        b, t, _, _ = node_p.shape
        assert b == 1 and t == 1, "Expected batch and timesteps to be 1 in rendering"

        node_p = node_p[0, 0, :, 0]
        edge_p = edge_p[0, 0, :, 0]
        trade_p = trade_p[0, 0, :, :]
        noop_p = noop_p[0, 0, :, :]

        # if node_p[node_p == 0].all():
        #     node_p += 1e-10
        # if edge_p[edge_p == 0].all():
        #     edge_p += 1e-10
        # edge_p.clamp_min(1e-10)

        fig = plt.figure(figsize=(FIG_X, FIG_Y))
        fig.patch.set_facecolor('black')  # Set background to black
        plt.gca().set_facecolor('black')  # Ensure the axes background is also black
        plt.axis('off')
        plt.axis('equal')

        p_min=0
        p_max=flat_pi.index.max().item()

        plt.text(0.01, 0.99, f"Pmax = {p_max*100.:.2f} %", color='white', fontsize=12, ha='left', va='top',
                 transform=plt.gca().transAxes)

        turn_appendix = int((self.turn - int(self.turn)) * self.n_players)

        # Create a colormap for the heatmap
        cmap = plt.cm.gray  # Black to white colormap

        ###
        # Overlay heatmap for board
        ###
        s = to_networkx(self.board.state, node_attrs=['pos', 'x'], edge_attrs=['edge_attr'])
        nx.draw_networkx_nodes(s, self.board.state.pos, node_color=node_p.numpy(), cmap=cmap, vmin=p_min, vmax=p_max, alpha=1.0)
        nx.draw_networkx_edges(s, self.board.state.pos, edge_color=edge_p.numpy(), edge_cmap=cmap, width=2, edge_vmin=p_min, edge_vmax=p_max, alpha=1.0)

        # Define sub axes
        trade_ax = plt.axes((0.05, 0.05, 0.05*2, 0.05*5))  # Adjust the position and size as needed
        noop_ax = plt.axes((0.85, 0.05, 0.1, 0.1))  # Adjust position and size as needed

        # Add an inset for the trade matrix heatmap
        trade_ax.imshow(trade_p, cmap='gray', aspect='auto', interpolation='nearest')

        # Trade matrix heatmap
        trade_ax.set_title('Trade Matrix')
        trade_ax.title.set_color('white')
        trade_ax.set_xticks([0, 1])
        trade_ax.set_xticklabels(['Give', 'Get'], color='white')
        # trade_ax.xaxis.label.set_color('white')
        # trade_ax.yaxis.label.set_color('white')
        trade_ax.set_yticks(range(5))
        trade_ax.set_yticklabels(['brick', 'grain', 'ore', 'lumber', 'wool'], color='white')
        trade_ax.tick_params(axis='both', which='both', length=0, color='white')
        # trade_ax.spines['top'].set_color('white')
        # trade_ax.spines['right'].set_color('white')
        # trade_ax.spines['left'].set_color('white')
        # trade_ax.spines['bottom'].set_color('white')

        # Remove axes of the inset to keep it clean
        trade_ax.spines['top'].set_visible(False)
        trade_ax.spines['right'].set_visible(False)
        trade_ax.spines['left'].set_visible(False)
        trade_ax.spines['bottom'].set_visible(False)

        # Noop heat map
        noop_ax.imshow(noop_p, cmap='gray', vmin=p_min, vmax=p_max, aspect='auto')

        # Customize the inset for no-op
        noop_ax.set_title('No-op', fontsize=10, color='white')
        noop_ax.set_xticks([])
        noop_ax.set_yticks([])
        noop_ax.spines['top'].set_visible(False)
        noop_ax.spines['right'].set_visible(False)
        noop_ax.spines['left'].set_visible(False)
        noop_ax.spines['bottom'].set_visible(False)

        ###
        # Save or show as needed
        ###
        if render_type == 'training':
            folder = f"./Renders/Training/"
            filename = f"Ep {self.episode}-{int(self.turn)} -- Players : {str.join(', ', [str(a) for a in self.player_agents])}_heat.png"
        elif render_type == 'testing':
            folder = f"./Renders/Test/Episode {int(self.episode)}/"
            filename = f"Turn {int(self.turn)}_{turn_appendix}_heat.png"
        elif render_type == 'init':
            folder = f"./Renders/Initial/"
            filename = f"{self.episode}_heat.png"
        else:
            raise ValueError(f"Invalid render type: {render_type}")
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, filename))
        plt.close()

    def render_side_by_side(self, flat_pi: FlatPi, render_type: str):
        assert render_type in ['training', 'testing', 'init']

        turn_appendix = int((self.turn - int(self.turn)) * self.n_players)

        if render_type == 'training':
            folder = f"./Renders/Training/"
            filename = f"Ep {self.episode}-{int(self.turn)} -- Players : {str.join(', ', [str(a) for a in self.player_agents])}"
        elif render_type == 'testing':
            folder = f"./Renders/Test/Episode {int(self.episode)}/"
            filename = f"Turn {int(self.turn)}_{turn_appendix}"
        elif render_type == 'init':
            folder = f"./Renders/Initial/"
            filename = f"{self.episode}"
        else:
            raise ValueError(f"Invalid render type: {render_type}")

        self.render(render_type)
        self.render_probs(flat_pi, render_type)

        render_file = os.path.join(folder, filename+'.png')
        render_p_file = os.path.join(folder, filename + '_heat.png')
        render_comb_file = get_unique_filename(os.path.join(folder, filename + '_comb.png'))

        # Load the images using PIL
        image1 = Image.open(render_file)
        image2 = Image.open(render_p_file)

        # Get the width and height of the images
        width1, height1 = image1.size
        width2, height2 = image2.size

        # Create a new image with the width equal to the sum of the widths of both images
        # and the height equal to the maximum height of the two images
        combined_width = width1 + width2
        combined_height = max(height1, height2)

        # Create a new blank image with the appropriate size
        combined_image = Image.new("RGB", (combined_width, combined_height))

        # Paste the two images into the combined image
        combined_image.paste(image1, (0, 0))
        combined_image.paste(image2, (width1, 0))

        # Save the combined image as a new file
        combined_image.save(render_comb_file)

        # Close the image objects to release the file handles
        image1.close()
        image2.close()

        # Remove the original images
        os.remove(render_file)
        os.remove(render_p_file)


if __name__ == '__main__':
    game = Game(1)
    game.register_agents([1])
    game.reset()
    game.render(debug=True)
