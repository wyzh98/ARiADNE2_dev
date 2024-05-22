import time

import numpy as np
import networkx as nx
import itertools
from utils import *
from parameter import *
import quads


class NodeManager:
    def __init__(self, ground_truth=None, ground_truth_info=None, explore=False, plot=False):
        self.local_nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        if ground_truth is not None:
            self.ground_truth_nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
            self.init_ground_truth_nodes(ground_truth, ground_truth_info, explore)
        else:
            if explore:
                raise ValueError("Ground truth is needed for exploration.")
        self.plot = plot
        if self.plot:
            self.x = []
            self.y = []

    def init_ground_truth_nodes(self, ground_truth, ground_truth_info, explore):
        for coords in ground_truth:
            key = (coords[0], coords[1])
            node = Node(coords, np.array([]), ground_truth_info)
            self.ground_truth_nodes_dict.insert(point=key, data=node)
            if not explore:
                self.local_nodes_dict.insert(point=key, data=node)
        for coords in ground_truth:
            node = self.ground_truth_nodes_dict.find((coords[0], coords[1])).data
            node.update_neighbor_explored_nodes(ground_truth_info, self.ground_truth_nodes_dict)
            node.update_visible_neighbor_explored_nodes(ground_truth_info, self.ground_truth_nodes_dict)

    def check_node_exist_in_dict(self, coords):
        key = (coords[0], coords[1])
        exist = self.local_nodes_dict.find(key)
        return exist

    def add_node_to_dict(self, coords, local_frontiers, extended_local_map_info):
        key = (coords[0], coords[1])
        node = Node(coords, local_frontiers, extended_local_map_info)
        self.local_nodes_dict.insert(point=key, data=node)
        return self.check_node_exist_in_dict(coords)

    def update_local_explore_graph(self, robot_location, local_frontiers, local_map_info, extended_local_map_info):
        extended_local_node_coords, _ = get_local_node_coords(robot_location, extended_local_map_info)
        for coords in extended_local_node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is not None:
                node = node.data
                if (node.explore_utility > 0) and (np.linalg.norm(node.coords - robot_location) <= 2 * SENSOR_RANGE):
                    node.update_observable_explore_frontiers(local_frontiers, extended_local_map_info)

        local_node_coords, _ = get_local_node_coords(robot_location, local_map_info)

        for coords in local_node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is None:
                self.add_node_to_dict(coords, local_frontiers, extended_local_map_info)

        for coords in local_node_coords:
            plot_x = self.x if self.plot else None
            plot_y = self.y if self.plot else None
            node = self.local_nodes_dict.find((coords[0], coords[1])).data
            node.update_neighbor_explored_nodes(extended_local_map_info, self.local_nodes_dict, plot_x, plot_y)
            node.update_visible_neighbor_explored_nodes(extended_local_map_info, self.local_nodes_dict)

    def update_local_safe_graph(self, robot_location, safe_frontiers, extended_safe_zone_info, extended_local_map_info):
        extended_explore_node_coords, _ = get_local_node_coords(robot_location, extended_local_map_info)
        extended_safe_node_coords, _ = get_local_node_coords(robot_location, extended_safe_zone_info, connected=False)
        extended_node_coords = np.unique(np.concatenate((extended_explore_node_coords, extended_safe_node_coords)), axis=0)

        for coords in extended_node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is not None:
                node = node.data
                if np.any(np.all(coords == extended_safe_node_coords, axis=1)):
                    node.set_safe()
                    node.update_observable_safe_frontiers(safe_frontiers, extended_safe_zone_info)
                else:
                    node.set_unsafe()
            else:
                print("Warning: Node should be added in exploration graph first")

    def get_all_node_graph(self, robot_location, robot_locations):
        all_node_coords = []
        for node in self.local_nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)

        explore_utility = []
        safe_utility = []
        signal = []

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        local_node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_node_coords):
            node = self.local_nodes_dict.find((coords[0], coords[1])).data
            explore_utility.append(node.explore_utility)
            safe_utility.append(node.safe_utility)
            signal.append(node.safe)
            for neighbor in node.neighbor_list:
                index = np.argwhere(local_node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                if index or index == [[0]]:
                    index = index[0][0]
                    adjacent_matrix[i, index] = 0

        visible_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        for i, coords in enumerate(all_node_coords):
            node = self.ground_truth_nodes_dict.find((coords[0], coords[1])).data
            for neighbor in node.visible_neighbor_list:
                index = np.argwhere(local_node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                if index or index == [[0]]:
                    index = index[0][0]
                    visible_matrix[i, index] = 0

        explore_utility = np.array(explore_utility)
        safe_utility = np.array(safe_utility)
        signal = np.array(signal)

        robot_in_graph = self.local_nodes_dict.nearest_neighbors(robot_location.tolist(), 1)[0].data.coords
        current_index = np.argwhere(local_node_coords_to_check == robot_in_graph[0] + robot_in_graph[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)

        occupancy = np.zeros((n_nodes, 1))
        for location in robot_locations:
            location_in_graph = self.local_nodes_dict.find((location[0], location[1])).data.coords
            index = np.argwhere(local_node_coords_to_check == location_in_graph[0] + location_in_graph[1] * 1j)[0][0]
            if index != current_index:
                occupancy[index] = 1

        guidepost = np.zeros_like(explore_utility)
        for other_location in robot_locations:
            if other_location[0] != robot_location[0] or other_location[1] != robot_location[1]:
                path_coords, dist = self.a_star(robot_location, other_location)
                for coords in path_coords:
                    if coords[0] != robot_location[0] or coords[1] != robot_location[1]:
                        index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j)[0]
                        guidepost[index] = 1

        return (all_node_coords, explore_utility, safe_utility, guidepost, signal, occupancy, adjacent_matrix, visible_matrix,
                current_index, neighbor_indices)

    def get_underlying_node_graph(self, all_node_coords):
        ground_truth_coords = copy.deepcopy(all_node_coords).tolist()

        for node in self.ground_truth_nodes_dict.__iter__():
            coords = node.data.coords
            if not (coords == all_node_coords).all(1).any(0):
                ground_truth_coords.append(coords)

        ground_truth_coords = np.array(ground_truth_coords).reshape(-1, 2)

        n_nodes = ground_truth_coords.shape[0]
        ground_truth_adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = ground_truth_coords[:, 0] + ground_truth_coords[:, 1] * 1j

        for i, coords in enumerate(ground_truth_coords):
            node = self.ground_truth_nodes_dict.find((coords[0], coords[1])).data
            for neighbor in node.neighbor_list:
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                if index or index == [[0]]:
                    index = index[0][0]
                    ground_truth_adjacent_matrix[i, index] = 0

        ground_truth_visible_matrix = np.ones((n_nodes, n_nodes)).astype(int)

        for i, coords in enumerate(ground_truth_coords):
            node = self.ground_truth_nodes_dict.find((coords[0], coords[1])).data
            for neighbor in node.visible_neighbor_list:
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                if index or index == [[0]]:
                    index = index[0][0]
                    ground_truth_visible_matrix[i, index] = 0

        return ground_truth_coords, ground_truth_adjacent_matrix, ground_truth_visible_matrix

    def get_global_node_graph(self, visible_matrix, ground_truth=False, max_hop=1):
        visible_matrix = 1 - visible_matrix
        node_coords = []
        for node in self.local_nodes_dict.__iter__():
            node_coords.append(node.data.coords)
        node_coords = np.array(node_coords).reshape(-1, 2)

        if ground_truth:
            node_coords = node_coords.tolist()
            for node in self.ground_truth_nodes_dict.__iter__():
                coords = node.data.coords
                if not (coords == node_coords).all(1).any(0):
                    node_coords.append(coords)
            node_coords = np.array(node_coords).reshape(-1, 2)

        cliques = self.find_cliques(node_coords, visible_matrix)
        center_indices = self.calc_clique_center(node_coords, cliques)
        global_node_coords = node_coords[center_indices]

        G = nx.from_numpy_array(visible_matrix)
        global_adj_matrix = np.zeros((len(center_indices), len(center_indices)))
        center_combs = list(itertools.combinations(range(len(center_indices)), r=2))

        for center1, center2 in center_combs:
            path = nx.shortest_path(G, center_indices[center1], center_indices[center2])
            for p in path[1: -1]:  # check if in the same clique
                if p in cliques[center1] or p in cliques[center2]:
                    path.pop(1)
            if len(path) - 2 < max_hop:
                global_adj_matrix[center1, center2] = 1
                global_adj_matrix[center2, center1] = 1

        global_adjacent_matrix_all_nodes = np.zeros_like(visible_matrix)
        indices = np.where(global_adj_matrix == 1)
        new_indices = [np.array(center_indices)[i] for i in indices]
        global_adjacent_matrix_all_nodes[new_indices[0], new_indices[1]] = 1

        return global_node_coords, 1 - global_adjacent_matrix_all_nodes, cliques

    @staticmethod
    def find_cliques(all_node_coords, visible_matrix, min_clique_node=4):
        cardinals = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]]) * NODE_RESOLUTION
        G = nx.from_numpy_array(visible_matrix)
        cliques = []
        while len(G.nodes) > 0:
            max_clique = set()
            # max_clique = max(nx.find_cliques(G), key=len)  # too slow
            nodes = set(G.nodes)
            while nodes:
                v = max(nodes, key=lambda x: len(set(G.neighbors(x)) & nodes))  # node with the max number of neighbors
                max_clique.add(v)
                nodes.remove(v)
                nodes &= set(G.neighbors(v))
            if len(max_clique) >= min_clique_node:
                cliques.append(list(max_clique))
            else:
                for node in all_node_coords[list(max_clique)]:
                    indices = [np.where((coords == all_node_coords).all(1))[0] for coords in node + cardinals]
                    for idx in indices:
                        if idx.size > 0:  # valid index
                            clique_found = next((clique for clique in cliques if idx[0] in clique), None)
                            if clique_found:
                                clique_found.extend(list(max_clique))
                                break
                    if clique_found:
                        break
                if not clique_found:
                    print(f"Warning: clique size smaller than {min_clique_node}")
                    cliques.append(list(max_clique))
            G.remove_nodes_from(max_clique)
        return cliques

    @staticmethod
    def calc_clique_center(all_node_coords, cliques):
        center_indices = []
        for clique in cliques:
            clique_coords = all_node_coords[clique]
            clique_centroid = np.mean(clique_coords, axis=0)
            distances = np.linalg.norm(clique_coords - clique_centroid, axis=1)
            center_index = np.argmin(distances)
            center_indices.append(clique[center_index])
        return center_indices

    @staticmethod
    def h(coords_1, coords_2):
        h = ((coords_1[0] - coords_2[0]) ** 2 + (coords_1[1] - coords_2[1]) ** 2) ** (1 / 2)
        h = np.round(h, 2)
        return h

    def a_star(self, start, destination, max_dist=1e8):
        # the path does not include the start
        if not self.check_node_exist_in_dict(start):
            Warning("start position is not in node dict")
            return [], 1e8
        if not self.check_node_exist_in_dict(destination):
            Warning("end position is not in node dict")
            return [], 1e8

        if start[0] == destination[0] and start[1] == destination[1]:
            return [destination], 0

        open_list = {(start[0], start[1])}
        closed_list = set()
        g = {(start[0], start[1]): 0}
        parents = {(start[0], start[1]): (start[0], start[1])}

        while len(open_list) > 0:
            n = None
            h_n = 1e8

            for v in open_list:
                h_v = self.h(v, destination)
                if n is not None:
                    node = self.local_nodes_dict.find(n).data
                    n_coords = node.coords
                    h_n = self.h(n_coords, destination)
                if n is None or g[v] + h_v < g[n] + h_n:
                    n = v
                    node = self.local_nodes_dict.find(n).data
                    n_coords = node.coords

            # if g[n] > max_dist:
            #     return [], 1e8

            if n_coords[0] == destination[0] and n_coords[1] == destination[1]:
                path = []
                length = g[n]
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.reverse()
                return path, np.round(length, 2)

            for neighbor_node_coords in node.neighbor_list:
                cost = ((neighbor_node_coords[0] - n_coords[0]) ** 2 + (
                            neighbor_node_coords[1] - n_coords[1]) ** 2) ** (1 / 2)
                cost = np.round(cost, 2)
                m = (neighbor_node_coords[0], neighbor_node_coords[1])
                if g[n] + cost > max_dist:
                    continue
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + cost
                else:
                    if g[m] > g[n] + cost:
                        g[m] = g[n] + cost
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)
            open_list.remove(n)
            closed_list.add(n)
        print('Path does not exist!')

        return [], 1e8

    def Dijkstra(self, start):
        q = set()
        dist_dict = {}
        prev_dict = {}

        for node in self.local_nodes_dict.__iter__():
            coords = node.data.coords
            key = (coords[0], coords[1])
            dist_dict[key] = 1e8
            prev_dict[key] = None
            q.add(key)

        dist_dict[(start[0], start[1])] = 0

        while len(q) > 0:

            u = None
            for coords in q:
                if u is None:
                    u = coords
                elif dist_dict[coords] < dist_dict[u]:
                    u = coords

            q.remove(u)

            node = self.local_nodes_dict.find(u).data
            for neighbor_node_coords in node.neighbor_list:
                v = (neighbor_node_coords[0], neighbor_node_coords[1])
                if v in q:
                    cost = ((neighbor_node_coords[0] - u[0]) ** 2 + (
                            neighbor_node_coords[1] - u[1]) ** 2) ** (1 / 2)
                    cost = np.round(cost, 2)
                    alt = dist_dict[u] + cost
                    if alt < dist_dict[v]:
                        dist_dict[v] = alt
                        prev_dict[v] = u

        return dist_dict, prev_dict

    def get_Dijkstra_path_and_dist(self, dist_dict, prev_dict, end):
        dist = dist_dict[(end[0], end[1])]

        path = [(end[0], end[1])]
        prev_node = prev_dict[(end[0], end[1])]
        while prev_node is not None:
            path.append(prev_node)
            temp = prev_node
            prev_node = prev_dict[temp]

        path.reverse()
        return path[1:], np.round(dist, 2)


class Node:
    def __init__(self, coords, local_frontiers, extended_local_map_info):
        self.coords = coords
        self.utility_range = UTILITY_RANGE
        self.observable_explore_frontiers = self.init_observable_explore_frontiers(local_frontiers, extended_local_map_info)
        self.observable_safe_frontiers = None
        self.explore_utility = self.observable_explore_frontiers.shape[0] if self.observable_explore_frontiers.shape[0] > MIN_UTILITY else 0
        self.safe_utility = 0
        self.visited = 0
        self.safe = 0

        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_list = []
        self.neighbor_matrix[2, 2] = 1
        self.neighbor_list.append(self.coords)

        self.visible_neighbor_matrix = -np.ones((11, 11))
        self.visible_neighbor_list = []
        self.visible_neighbor_list.append(self.coords)
        self.vision_mask = self.generate_circle(11)

    @staticmethod
    def generate_circle(n):
        Y, X = np.ogrid[:n, :n]
        center = n // 2
        mask = (X - center) ** 2 + (Y - center) ** 2 <= (n // 2) ** 2
        return mask.astype(int)

    def init_observable_explore_frontiers(self, local_frontiers, extended_local_map_info):
        if local_frontiers.shape[0] == 0:
            self.explore_utility = 0
            return local_frontiers
        else:
            observable_explore_frontiers = []
            dist_list = np.linalg.norm(local_frontiers - self.coords, axis=-1)
            frontiers_in_range = local_frontiers[dist_list < self.utility_range]
            for point in frontiers_in_range:
                collision = check_collision(self.coords, point, extended_local_map_info)
                if not collision:
                    observable_explore_frontiers.append(point)
            observable_explore_frontiers = np.array(observable_explore_frontiers)
            return observable_explore_frontiers

    def update_observable_explore_frontiers(self, local_frontiers, extended_local_map_info):
        if local_frontiers.shape[0] == 0:
            self.explore_utility = 0
            self.observable_explore_frontiers = local_frontiers
            return
        local_frontiers = local_frontiers.reshape(-1, 2)
        old_frontier_to_check = self.observable_explore_frontiers[:, 0] + self.observable_explore_frontiers[:, 1] * 1j
        local_frontiers_to_check = local_frontiers[:, 0] + local_frontiers[:, 1] * 1j
        to_observe_index = np.where(
            np.isin(old_frontier_to_check, local_frontiers_to_check, assume_unique=True) == True)
        new_frontier_index = np.where(
            np.isin(local_frontiers_to_check, old_frontier_to_check, assume_unique=True) == False)
        self.observable_explore_frontiers = self.observable_explore_frontiers[to_observe_index]
        new_frontiers = local_frontiers[new_frontier_index]

        # add new frontiers in the observable frontiers
        if new_frontiers.shape[0] > 0:
            dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
            new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]
            for point in new_frontiers_in_range:
                collision = check_collision(self.coords, point, extended_local_map_info)
                if not collision:
                    self.observable_explore_frontiers = np.concatenate((self.observable_explore_frontiers, point.reshape(1, 2)), axis=0)
        self.explore_utility = self.observable_explore_frontiers.shape[0] if self.observable_explore_frontiers.shape[0] > MIN_UTILITY else 0

    def update_observable_safe_frontiers(self, safe_frontiers, extended_safe_zone_info):
        if not self.safe:
            self.safe_utility = 0
            return
        if safe_frontiers.shape[0] == 0:
            self.safe_utility = 0
        else:
            observable_safe_frontiers = []
            dist_list = np.linalg.norm(safe_frontiers - self.coords, axis=-1)
            frontiers_in_range = safe_frontiers[dist_list < self.utility_range]
            for point in frontiers_in_range:
                collision = check_collision(self.coords, point, extended_safe_zone_info)
                if not collision:
                    observable_safe_frontiers.append(point)
            self.observable_safe_frontiers = np.array(observable_safe_frontiers)
            self.safe_utility = self.observable_safe_frontiers.shape[0] if self.observable_safe_frontiers.shape[0] > MIN_UTILITY else 0

    def update_neighbor_explored_nodes(self, extended_local_map_info, nodes_dict, plot_x=None, plot_y=None):
        center_index = self.neighbor_matrix.shape[0] // 2
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                else:
                    if i == center_index and j == center_index:
                        self.neighbor_matrix[i, j] = 1
                        continue

                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                          self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        cell = get_cell_position_from_coords(neighbor_coords, extended_local_map_info)
                        if cell[0] < extended_local_map_info.map.shape[1] and cell[1] < extended_local_map_info.map.shape[0]:
                            if extended_local_map_info.map[cell[1], cell[0]] == 1:
                                self.neighbor_matrix[i, j] = 1
                            continue
                    else:
                        neighbor_node = neighbor_node.data
                        collision = check_collision(self.coords, neighbor_coords, extended_local_map_info)
                        neighbor_matrix_x = center_index + (center_index - i)
                        neighbor_matrix_y = center_index + (center_index - j)
                        if not collision:
                            self.neighbor_matrix[i, j] = 1
                            self.neighbor_list.append(neighbor_coords)

                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_list.append(self.coords)

                            if plot_x is not None and plot_y is not None:
                                plot_x.append([self.coords[0], neighbor_coords[0]])
                                plot_y.append([self.coords[1], neighbor_coords[1]])

    def update_visible_neighbor_explored_nodes(self, extended_local_map_info, nodes_dict):
        center_index = self.visible_neighbor_matrix.shape[0] // 2
        for i in range(self.visible_neighbor_matrix.shape[0]):
            for j in range(self.visible_neighbor_matrix.shape[1]):
                if (self.visible_neighbor_matrix[i, j] != -1) or (self.vision_mask[i, j] == 0):
                    continue
                else:
                    if i == center_index and j == center_index:
                        self.visible_neighbor_matrix[i, j] = 1
                        continue

                    visible_neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                                  self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                    visible_neighbor_node = nodes_dict.find((visible_neighbor_coords[0], visible_neighbor_coords[1]))
                    if visible_neighbor_node is None:
                        cell = get_cell_position_from_coords(visible_neighbor_coords, extended_local_map_info)
                        if cell[0] < extended_local_map_info.map.shape[1] and cell[1] < extended_local_map_info.map.shape[0]:
                            if extended_local_map_info.map[cell[1], cell[0]] == 1:
                                self.visible_neighbor_matrix[i, j] = 1
                            continue
                    else:
                        visible_neighbor_node = visible_neighbor_node.data
                        collision = check_collision(self.coords, visible_neighbor_coords, extended_local_map_info)
                        visible_neighbor_matrix_x = center_index + (center_index - i)
                        visible_neighbor_matrix_y = center_index + (center_index - j)
                        if not collision:
                            self.visible_neighbor_matrix[i, j] = 1
                            self.visible_neighbor_list.append(visible_neighbor_coords)

                            visible_neighbor_node.visible_neighbor_matrix[visible_neighbor_matrix_x, visible_neighbor_matrix_y] = 1
                            visible_neighbor_node.visible_neighbor_list.append(self.coords)

    def set_safe(self):
        self.safe = 1

    def set_unsafe(self):
        self.safe = 0
        self.observable_safe_frontiers = np.array([])
        self.safe_utility = 0

    def set_visited(self):
        self.visited = 1
        self.observable_explore_frontiers = np.array([])
        self.explore_utility = 0
