import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils import *
from parameter import *
from local_node_manager_quadtree import Local_node_manager
from global_node_manager import Global_node_manager


class Agent:
    def __init__(self, policy_net, device='cpu', plot=False):
        self.device = device
        self.plot = plot
        self.policy_net = policy_net

        # location and global map
        self.location = None
        self.global_map_info = None
        self.local_center = None

        # local map related parameters
        self.cell_size = CELL_SIZE
        self.downsample_size = NODE_RESOLUTION  # cell
        self.downsampled_cell_size = self.cell_size * self.downsample_size  # meter

        # local map and extended local map
        self.local_map_info = None
        self.extended_local_map_info = None

        # local frontiers
        self.local_frontier = None

        # local  node managers
        self.local_node_manager = Local_node_manager(plot=self.plot)

        # local graph
        self.local_node_coords, self.utility, self.guidepost = None, None, None
        self.current_local_index, self.local_adjacent_matrix, self.local_neighbor_indices = None, None, None

    def update_global_map(self, global_map_info):
        # no need in training because of shallow copy
        self.global_map_info = global_map_info

    def update_local_map(self, location):
        self.local_map_info = self.global_map_info
        self.extended_local_map_info = self.global_map_info

    def update_location(self, location):
        self.location = location
        node = self.local_node_manager.local_nodes_dict.find((location[0], location[1]))
        if node:
            node.data.set_visited()

    def update_local_frontiers(self):
        self.local_frontier = get_frontier_in_map(self.extended_local_map_info)

    def update_planning_state(self, global_map_info, location):
        self.update_global_map(global_map_info)
        self.update_location(location)
        self.local_center = self.location
        self.update_local_map(self.local_center)
        self.update_local_frontiers()
        self.local_node_coords, self.utility, self.guidepost, self.local_adjacent_matrix, self.current_local_index, self.local_neighbor_indices = \
            self.local_node_manager.update_local_graph(self.location,
                                                       self.local_frontier,
                                                       self.local_map_info,
                                                       self.extended_local_map_info)

    def get_local_observation(self):
        local_node_coords = self.local_node_coords
        local_node_utility = self.utility.reshape(-1, 1)
        local_node_guidepost = self.guidepost.reshape(-1, 1)
        current_local_index = self.current_local_index
        local_edge_mask = self.local_adjacent_matrix
        current_local_edge = self.local_neighbor_indices
        n_local_node = local_node_coords.shape[0]

        current_local_node_coords = local_node_coords[self.current_local_index]
        local_node_coords = np.concatenate((local_node_coords[:, 0].reshape(-1, 1) - current_local_node_coords[0],
                                            local_node_coords[:, 1].reshape(-1, 1) - current_local_node_coords[1]),
                                           axis=-1) / (640 * 0.4)
        local_node_utility = local_node_utility / 200
        local_node_inputs = np.concatenate((local_node_coords, local_node_utility, local_node_guidepost), axis=1)
        local_node_inputs = torch.FloatTensor(local_node_inputs).unsqueeze(0).to(self.device)

        assert local_node_coords.shape[0] < LOCAL_NODE_PADDING_SIZE
        padding = torch.nn.ZeroPad2d((0, 0, 0, LOCAL_NODE_PADDING_SIZE - n_local_node))
        local_node_inputs = padding(local_node_inputs)

        local_node_padding_mask = torch.zeros((1, 1, n_local_node), dtype=torch.int16).to(self.device)
        local_node_padding = torch.ones((1, 1, LOCAL_NODE_PADDING_SIZE - n_local_node), dtype=torch.int16).to(
            self.device)
        local_node_padding_mask = torch.cat((local_node_padding_mask, local_node_padding), dim=-1)

        current_local_index = torch.tensor([current_local_index]).reshape(1, 1, 1).to(self.device)

        local_edge_mask = torch.tensor(local_edge_mask).unsqueeze(0).to(self.device)

        padding = torch.nn.ConstantPad2d(
            (0, LOCAL_NODE_PADDING_SIZE - n_local_node, 0, LOCAL_NODE_PADDING_SIZE - n_local_node), 1)
        local_edge_mask = padding(local_edge_mask)

        current_in_edge = np.argwhere(current_local_edge == self.current_local_index)[0][0]
        current_local_edge = torch.tensor(current_local_edge).unsqueeze(0)
        k_size = current_local_edge.size()[-1]
        padding = torch.nn.ConstantPad1d((0, LOCAL_K_SIZE - k_size), 0)
        current_local_edge = padding(current_local_edge)
        current_local_edge = current_local_edge.unsqueeze(-1)

        local_edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        local_edge_padding_mask[0, 0, current_in_edge] = 1
        padding = torch.nn.ConstantPad1d((0, LOCAL_K_SIZE - k_size), 1)
        local_edge_padding_mask = padding(local_edge_padding_mask)

        return [local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask]

    def select_next_waypoint(self, local_observation):
        _, _, _, _, current_local_edge, _ = local_observation
        with torch.no_grad():
            logp = self.policy_net(*local_observation)

        action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)
        next_node_index = current_local_edge[0, action_index.item(), 0].item()
        next_position = self.local_node_coords[next_node_index]

        return next_position, action_index

    def plot_local_env(self):
        plt.switch_backend('agg')
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 2)
        nodes = get_cell_position_from_coords(self.local_node_coords, self.local_map_info)
        frontiers = get_cell_position_from_coords(self.local_frontier, self.local_map_info)
        robot = get_cell_position_from_coords(self.location, self.local_map_info)
        plt.imshow(self.local_map_info.map, cmap='gray')
        plt.axis('off')
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.utility, zorder=2)
        plt.scatter(frontiers[:, 0], frontiers[:, 1], c='r')
        plt.plot(robot[0], robot[1], 'mo', markersize=16, zorder=5)
        for i in range(len(self.local_node_manager.x)):
           plt.plot((self.local_node_manager.x[i] - self.local_map_info.map_origin_x) / self.cell_size,
                    (self.local_node_manager.y[i] - self.local_map_info.map_origin_y) / self.cell_size, 'tan', zorder=1)