import copy
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import *
from parameter import *
from local_node_manager_quadtree import NodeManager


class Agent:
    def __init__(self, id, policy_net, node_manager, device='cpu', plot=False):
        self.id = id
        self.device = device
        self.plot = plot
        self.policy_net = policy_net

        # location and safe zone
        self.location = None
        self.global_map_info = None
        self.safe_zone_info = None

        # local map related parameters
        self.cell_size = CELL_SIZE
        self.downsample_size = NODE_RESOLUTION  # cell
        self.downsampled_cell_size = self.cell_size * self.downsample_size  # meter
        self.local_map_size = LOCAL_MAP_SIZE  # meter
        self.extended_local_map_size = EXTENDED_LOCAL_MAP_SIZE

        # local safe and extended local safe
        self.local_safe_zone_info = None
        self.extended_local_safe_zone_info = None
        self.local_map_info = None
        self.extended_local_map_info = None

        # local frontiers
        self.explore_frontier = None
        self.safe_frontier = None

        # local node managers
        self.node_manager = node_manager

        # local graph
        self.local_node_coords, self.explore_utility, self.safe_utility, self.guidepost, self.signal, self.occupancy = None, None, None, None, None, None
        self.current_local_index, self.local_adjacent_matrix, self.local_neighbor_indices = None, None, None

        self.travel_dist = 0

        self.episode_buffer = []
        for i in range(18):
            self.episode_buffer.append([])

        if self.plot:
            self.trajectory_x = []
            self.trajectory_y = []

    def update_global_map(self, global_map_info):
        self.global_map_info = global_map_info

    def update_global_safe_zone(self, global_safe_zone):
        self.safe_zone_info = global_safe_zone

    def update_local_safe_zone(self):
        self.local_safe_zone_info = self.get_local_map(self.location, self.safe_zone_info)
        self.extended_local_safe_zone_info = self.get_extended_local_map(self.location, self.safe_zone_info)

    def update_local_map(self):
        self.local_map_info = self.get_local_map(self.location, self.global_map_info)
        self.extended_local_map_info = self.get_extended_local_map(self.location, self.global_map_info)

    def update_location(self, location):
        if self.location is None:
            self.location = location

        dist = np.linalg.norm(self.location - location)
        self.travel_dist += dist

        self.location = location
        node = self.node_manager.local_nodes_dict.find((location[0], location[1]))
        if node:
            node.data.set_visited()
        if self.plot:
            self.trajectory_x.append(location[0])
            self.trajectory_y.append(location[1])

    def update_explore_frontiers(self):
        self.explore_frontier = get_explore_frontier(self.extended_local_map_info)

    def update_safe_frontiers(self):
        self.safe_frontier = get_safe_zone_frontier(self.extended_local_safe_zone_info, self.extended_local_map_info)

    def update_graph(self, map_info, location):
        self.update_global_map(map_info)
        self.update_location(location)
        self.update_local_map()
        self.update_explore_frontiers()
        self.node_manager.update_local_explore_graph(self.location, self.explore_frontier, self.local_map_info,
                                                     self.extended_local_map_info)

    def update_safe_graph(self, safe_zone_info):
        self.update_global_safe_zone(safe_zone_info)
        self.update_local_safe_zone()
        self.update_safe_frontiers()
        self.node_manager.update_local_safe_graph(self.location, self.safe_frontier, self.local_safe_zone_info,
                                                  self.extended_local_safe_zone_info, self.extended_local_map_info)

    def update_planning_state(self, robot_locations):
        (self.local_node_coords, self.explore_utility, self.safe_utility, self.guidepost, self.signal, self.occupancy, self.local_adjacent_matrix,
         self.current_local_index, self.local_neighbor_indices) = self.node_manager.get_all_node_graph(self.location, robot_locations)

    def get_local_observation(self):
        local_node_coords = self.local_node_coords
        local_node_explore_utility = self.explore_utility.reshape(-1, 1)
        local_node_safe_utility = self.safe_utility.reshape(-1, 1)
        local_node_guidepost = self.guidepost.reshape(-1, 1)
        local_node_occupancy = self.occupancy.reshape(-1, 1)
        local_node_signal = self.signal.reshape(-1, 1)
        current_local_index = self.current_local_index
        local_edge_mask = self.local_adjacent_matrix
        current_local_edge = self.local_neighbor_indices
        n_local_node = local_node_coords.shape[0]

        current_local_node_coords = local_node_coords[self.current_local_index]
        local_node_coords = np.concatenate((local_node_coords[:, 0].reshape(-1, 1) - current_local_node_coords[0],
                                            local_node_coords[:, 1].reshape(-1, 1) - current_local_node_coords[1]),
                                           axis=-1) / LOCAL_MAP_SIZE
        local_node_explore_utility = local_node_explore_utility / 30
        local_node_safe_utility = local_node_safe_utility / 30
        local_node_inputs = np.concatenate((local_node_coords, local_node_explore_utility, local_node_safe_utility,
                                            local_node_guidepost, local_node_signal, local_node_occupancy), axis=1)
        local_node_inputs = torch.FloatTensor(local_node_inputs).unsqueeze(0).to(self.device)

        assert local_node_coords.shape[0] < LOCAL_NODE_PADDING_SIZE, print(local_node_coords.shape[0])
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
        # local_edge_padding_mask[0, 0, current_in_edge] = 1
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

        return next_position, next_node_index, action_index

    def get_local_map(self, location, map_info):
        local_map_origin_x = (location[
                                  0] - self.local_map_size / 2) // self.downsampled_cell_size * self.downsampled_cell_size
        local_map_origin_y = (location[
                                  1] - self.local_map_size / 2) // self.downsampled_cell_size * self.downsampled_cell_size
        local_map_top_x = local_map_origin_x + self.local_map_size + NODE_RESOLUTION
        local_map_top_y = local_map_origin_y + self.local_map_size + NODE_RESOLUTION

        min_x = map_info.map_origin_x
        min_y = map_info.map_origin_y
        max_x = map_info.map_origin_x + self.cell_size * map_info.map.shape[1]
        max_y = map_info.map_origin_y + self.cell_size * map_info.map.shape[0]

        if local_map_origin_x < min_x:
            local_map_origin_x = min_x
        if local_map_origin_y < min_y:
            local_map_origin_y = min_y
        if local_map_top_x > max_x:
            local_map_top_x = max_x
        if local_map_top_y > max_y:
            local_map_top_y = max_y

        local_map_origin_x = np.around(local_map_origin_x, 1)
        local_map_origin_y = np.around(local_map_origin_y, 1)
        local_map_top_x = np.around(local_map_top_x, 1)
        local_map_top_y = np.around(local_map_top_y, 1)

        local_map_origin = np.array([local_map_origin_x, local_map_origin_y])
        local_map_origin_in_global_map = get_cell_position_from_coords(local_map_origin, map_info)

        local_map_top = np.array([local_map_top_x, local_map_top_y])
        local_map_top_in_global_map = get_cell_position_from_coords(local_map_top, map_info)

        local_map = map_info.map[
                    local_map_origin_in_global_map[1]:local_map_top_in_global_map[1],
                    local_map_origin_in_global_map[0]:local_map_top_in_global_map[0]]

        local_map_info = Map_info(local_map, local_map_origin_x, local_map_origin_y, self.cell_size)

        return local_map_info

    def get_extended_local_map(self, location, map_info):
        # expanding local map to involve all related frontiers
        local_map_origin_x = (location[
                                  0] - self.extended_local_map_size / 2) // self.downsampled_cell_size * self.downsampled_cell_size
        local_map_origin_y = (location[
                                  1] - self.extended_local_map_size / 2) // self.downsampled_cell_size * self.downsampled_cell_size
        local_map_top_x = local_map_origin_x + self.extended_local_map_size + 2 * NODE_RESOLUTION
        local_map_top_y = local_map_origin_y + self.extended_local_map_size + 2 * NODE_RESOLUTION

        min_x = map_info.map_origin_x
        min_y = map_info.map_origin_y
        max_x = map_info.map_origin_x + self.cell_size * map_info.map.shape[1]
        max_y = map_info.map_origin_y + self.cell_size * map_info.map.shape[0]

        if local_map_origin_x < min_x:
            local_map_origin_x = min_x
        if local_map_origin_y < min_y:
            local_map_origin_y = min_y
        if local_map_top_x > max_x:
            local_map_top_x = max_x
        if local_map_top_y > max_y:
            local_map_top_y = max_y

        local_map_origin_x = np.around(local_map_origin_x, 1)
        local_map_origin_y = np.around(local_map_origin_y, 1)
        local_map_top_x = np.around(local_map_top_x, 1)
        local_map_top_y = np.around(local_map_top_y, 1)

        local_map_origin = np.array([local_map_origin_x, local_map_origin_y])
        local_map_origin_in_global_map = get_cell_position_from_coords(local_map_origin, map_info)

        local_map_top = np.array([local_map_top_x, local_map_top_y])
        local_map_top_in_global_map = get_cell_position_from_coords(local_map_top, map_info)

        local_map = map_info.map[
                    local_map_origin_in_global_map[1]:local_map_top_in_global_map[1],
                    local_map_origin_in_global_map[0]:local_map_top_in_global_map[0]]

        local_map_info = Map_info(local_map, local_map_origin_x, local_map_origin_y, self.cell_size)

        return local_map_info

    def save_observation(self, local_observation):
        local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask = local_observation
        self.episode_buffer[0] += local_node_inputs
        self.episode_buffer[1] += local_node_padding_mask.bool()
        self.episode_buffer[2] += local_edge_mask.bool()
        self.episode_buffer[3] += current_local_index
        self.episode_buffer[4] += current_local_edge
        self.episode_buffer[5] += local_edge_padding_mask.bool()

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.reshape(1, 1, 1)

    def save_reward(self, reward):
        self.episode_buffer[7] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)

    def save_done(self, done):
        self.episode_buffer[8] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)

    def save_all_indices(self, all_agent_curr_indices):
        self.episode_buffer[15] += torch.tensor(all_agent_curr_indices).reshape(1, -1, 1).to(self.device)

    def save_next_observations(self, local_observation, next_node_index_list):
        self.episode_buffer[9] = copy.deepcopy(self.episode_buffer[0])[1:]
        self.episode_buffer[10] = copy.deepcopy(self.episode_buffer[1])[1:]
        self.episode_buffer[11] = copy.deepcopy(self.episode_buffer[2])[1:]
        self.episode_buffer[12] = copy.deepcopy(self.episode_buffer[3])[1:]
        self.episode_buffer[13] = copy.deepcopy(self.episode_buffer[4])[1:]
        self.episode_buffer[14] = copy.deepcopy(self.episode_buffer[5])[1:]
        self.episode_buffer[16] = copy.deepcopy(self.episode_buffer[15])[1:]

        local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask = local_observation
        self.episode_buffer[9] += local_node_inputs
        self.episode_buffer[10] += local_node_padding_mask.bool()
        self.episode_buffer[11] += local_edge_mask.bool()
        self.episode_buffer[12] += current_local_index
        self.episode_buffer[13] += current_local_edge
        self.episode_buffer[14] += local_edge_padding_mask.bool()
        self.episode_buffer[16] += torch.tensor(next_node_index_list).reshape(1, -1, 1).to(self.device)
        self.episode_buffer[17] = copy.deepcopy(self.episode_buffer[16])[1:]
        self.episode_buffer[17] += copy.deepcopy(self.episode_buffer[16])[:-1]

