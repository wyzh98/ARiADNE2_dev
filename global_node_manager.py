import numpy as np

import quads
import time

from parameter import *
from utils import *


class Global_node_manager:
    def __init__(self, local_node_manager, plot=False):
        self.global_nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.local_node_manager = local_node_manager
        self.plot = plot
        if self.plot:
            self.x = []
            self.y = []

    def check_valid_global_node(self, coords, local_map_info):

        neighbor_boundary = self.get_min_neighbor_boundary(coords)
        global_nodes_in_neighbor_range = self.global_nodes_dict.within_bb(neighbor_boundary)

        local_node_coords, local_free_connected_map = get_local_node_coords(coords, local_map_info)

        if len(global_nodes_in_neighbor_range) == 0:
            return True
        else:
            for global_node in global_nodes_in_neighbor_range:
                cell = get_cell_position_from_coords(global_node.data.coords, local_map_info)
                if local_free_connected_map[cell[1], cell[0]] == 1:
                    return False
            return True

    def add_global_node(self, coords, local_map_info, neighbor_boundary):
        local_node_coords, _ = get_local_node_coords(coords, local_map_info)

        key = (coords[0], coords[1])
        associated_utility_list = []
        for local_node in local_node_coords:
            node = self.local_node_manager.local_nodes_dict.find((local_node[0], local_node[1]))
            associated_utility_list.append(node.data.utility_share)

        global_node = Global_node(coords, associated_utility_list)
        self.global_nodes_dict.insert(point=key, data=global_node)

        global_node_in_neighbor_boundary = self.global_nodes_dict.within_bb(neighbor_boundary)
        global_neighbors, global_neighbor_dist = self.get_global_neighbors_and_dist(coords, global_node_in_neighbor_boundary, local_map_info)
        global_node.update_global_neighbors(global_neighbors, global_neighbor_dist)

    def update_global_node_utility(self, global_node, global_map_info):
        coords = global_node.coords
        local_map_info = get_partial_map_from_center(global_map_info, coords, LOCAL_MAP_SIZE)

        local_node_coords, _ = get_local_node_coords(coords, local_map_info)

        associated_utility_list = []
        for local_node in local_node_coords:
            node = self.local_node_manager.local_nodes_dict.find((local_node[0], local_node[1]))
            if node is not None:
                associated_utility_list.append(node.data.utility_share)
        global_node.associated_utility_list = associated_utility_list

    def update_global_node_sign(self, global_node):
        global_node.informative_global_node = global_node.check_local_utility()

    def get_min_neighbor_boundary(self, coords):
        min_x = coords[0] - MIN_NEIGHBOR_SIZE / 2
        min_y = coords[1] - MIN_NEIGHBOR_SIZE / 2
        max_x = coords[0] + MIN_NEIGHBOR_SIZE / 2
        max_y = coords[1] + MIN_NEIGHBOR_SIZE / 2
        min_x = np.round(min_x, 1)
        min_y = np.round(min_y, 1)
        max_x = np.round(max_x, 1)
        max_y = np.round(max_y, 1)

        neighbor_boundary = quads.BoundingBox(min_x, min_y, max_x, max_y)
        return neighbor_boundary

    def get_max_neighbor_boundary(self, coords):
        min_x = coords[0] - LOCAL_MAP_SIZE / 2 + 1
        min_y = coords[1] - LOCAL_MAP_SIZE / 2 + 1
        max_x = coords[0] + LOCAL_MAP_SIZE / 2 - 1
        max_y = coords[1] + LOCAL_MAP_SIZE / 2 - 1
        min_x = np.round(min_x, 1)
        min_y = np.round(min_y, 1)
        max_x = np.round(max_x, 1)
        max_y = np.round(max_y, 1)

        neighbor_boundary = quads.BoundingBox(min_x, min_y, max_x, max_y)
        return neighbor_boundary

    def get_global_neighbors_and_dist(self, coords, global_node_in_neighbor_boundary, local_map_info):
        global_neighbors = []
        global_neighbors_dist = []
        _, local_connected_map = get_local_node_coords(coords, local_map_info)
        for node in global_node_in_neighbor_boundary:
            node_coords = node.data.coords
            cell = get_cell_position_from_coords(node_coords, local_map_info)
            if local_connected_map[cell[1], cell[0]] == 1:
                global_neighbors.append(node.data.coords)
                path, dist = self.local_node_manager.a_star(coords, node_coords)
                global_neighbors_dist.append(dist)

                if node_coords[0] != coords[0] or node_coords[1] != coords[1]:
                    node.data.global_neighbors.append(coords)
                    node.data.global_neighbor_dist.append(dist)
                if self.plot:
                    self.x.append([coords[0], node.data.coords[0]])
                    self.y.append([coords[1], node.data.coords[1]])

        return global_neighbors, global_neighbors_dist

    def update_global_graph(self, location, local_map_info, global_map_info):
        valid = self.check_valid_global_node(location, local_map_info)

        neighbor_boundary = self.get_max_neighbor_boundary(location)
        global_node_in_neighbor_boundary = self.global_nodes_dict.within_bb(neighbor_boundary)
        #for node in global_node_in_neighbor_boundary:
        #    self.update_global_node_utility(node.data, global_map_info)
        for node in self.global_nodes_dict.__iter__():
            self.update_global_node_sign(node.data)
        if valid:
            self.add_global_node(location, local_map_info, neighbor_boundary)

        global_node_coords = []
        informative_sign = []
        for node in self.global_nodes_dict.__iter__():
            global_node_coords.append(node.data.coords)
            informative_sign.append(node.data.informative_global_node)
        global_node_coords = np.array(global_node_coords)
        informative_sign = np.array(informative_sign)

        n_node = self.global_nodes_dict.__len__()
        adjacent_matrix = np.ones((n_node, n_node)).astype(int)
        global_node_coords_to_check = global_node_coords[:, 0] + global_node_coords[:, 1] * 1j
        for i, coords in enumerate(global_node_coords):
            node = self.global_nodes_dict.find((coords[0], coords[1])).data
            for neighbor in node.global_neighbors:
                index = np.argwhere(global_node_coords_to_check == neighbor[0] + neighbor[1] * 1j)[0][0]
                adjacent_matrix[i, index] = 0

        current_index = np.argmin(np.linalg.norm(global_node_coords - location.reshape(1, 2), axis=-1))
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)

        return global_node_coords, informative_sign, adjacent_matrix, current_index, neighbor_indices


class Global_node:
    def __init__(self, coords, associated_utility_list):
        self.coords = coords
        self.associated_utility_list = associated_utility_list
        self.informative_global_node = self.check_local_utility()
        self.global_neighbors = []
        self.global_neighbor_dist = []

    def check_local_utility(self):
        return 1 if sum([utility[0] for utility in self.associated_utility_list]) > 0 else 0

    def update_global_neighbors(self, global_neighbors, global_neighbor_dist):
        self.global_neighbors = global_neighbors
        self.global_neighbor_dist = global_neighbor_dist



