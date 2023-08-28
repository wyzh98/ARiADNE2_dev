import time

import numpy as np
from utils import *
from parameter import *
import quads


class NodeManager:
    def __init__(self, free=None, plot=False):
        self.local_nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.plot = plot
        if free is not None:
            self.add_free_nodes(free)
        if self.plot:
            self.x = []
            self.y = []

    def add_free_nodes(self, free):
        for coords in free:
            key = (coords[0], coords[1])
            node = LocalNode(coords)
            self.local_nodes_dict.insert(point=key, data=node)

    def check_node_exist_in_dict(self, coords):
        key = (coords[0], coords[1])
        exist = self.local_nodes_dict.find(key)
        return exist

    def add_node_to_dict(self, coords):
        key = (coords[0], coords[1])
        node = LocalNode(coords)
        self.local_nodes_dict.insert(point=key, data=node)
        return self.check_node_exist_in_dict(coords)

    def update_local_explore_graph(self, robot_location, local_frontiers, local_map_info, extended_local_map_info):
        extended_local_node_coords, _ = get_local_node_coords(robot_location, extended_local_map_info)
        # for coords in extended_local_node_coords:
        #     node = self.check_node_exist_in_dict(coords)
        #     if node is not None:
        #         node = node.data
        #         if (node.safe_utility != 0) and (np.linalg.norm(node.coords - robot_location) <= 2 * SENSOR_RANGE):
        #             pass  # TODO: update exploration frontier

        local_node_coords, _ = get_local_node_coords(robot_location, local_map_info)

        for coords in local_node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is None:
                node = self.add_node_to_dict(coords)

        for coords in local_node_coords:
            plot_x = self.x if self.plot else None
            plot_y = self.y if self.plot else None
            node = self.local_nodes_dict.find((coords[0], coords[1])).data
            node.update_neighbor_explored_nodes(extended_local_map_info, self.local_nodes_dict, plot_x, plot_y)

    def update_local_safe_graph(self, robot_location, safe_frontiers, local_safe_zone_info, extended_safe_zone_info, extended_local_map_info):
        extended_explore_node_coords, _ = get_local_node_coords(robot_location, extended_local_map_info)
        extended_safe_node_coords, _ = get_local_node_coords(robot_location, extended_safe_zone_info)

        for coords in extended_explore_node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is not None:
                node = node.data
                if np.any(np.all(coords == extended_safe_node_coords, axis=1)):
                    node.set_safe()
                    if (node.safe_utility != 0) and (np.linalg.norm(node.coords - robot_location) <= 2 * SENSOR_RANGE):
                        node.update_observable_safe_frontiers(safe_frontiers, extended_safe_zone_info)
                else:
                    node.set_unsafe()

        local_node_coords, _ = get_local_node_coords(robot_location, local_safe_zone_info)

        for coords in local_node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is None:
                node = self.add_node_to_dict(coords)
            node = node.data
            node.initialize_observable_safe_frontiers(safe_frontiers, extended_safe_zone_info)

    def get_all_node_graph(self, robot_location, robot_locations):
        all_node_coords = []
        for node in self.local_nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = []
        signal = []

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        local_node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_node_coords):
            node = self.local_nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.safe_utility)
            signal.append(node.safe)
            for neighbor in node.explored_neighbor_list:
                index = np.argwhere(local_node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                if index or index == [[0]]:
                    index = index[0][0]
                    adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        signal = np.array(signal)

        indices = np.argwhere(utility > 0).reshape(-1)
        utility_node_coords = all_node_coords[indices]
        dist_dict, prev_dict = self.Dijkstra(robot_location)
        nearest_utility_coords = robot_location
        nearest_dist = 1e8
        for coords in utility_node_coords:
            if coords[0] != robot_location[0] and coords[1] != robot_location[1]:
                dist = dist_dict[(coords[0], coords[1])]
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_utility_coords = coords
                # print(nearest_dist, coords, nearest_utility_coords, robot_location)
        path_coords, dist = self.a_star(robot_location, nearest_utility_coords)
        guidepost = np.zeros_like(utility)
        for coords in path_coords:
            if coords[0] != robot_location[0] and coords[1] != robot_location[1]:
                index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j)[0]
                guidepost[index] = 1

        robot_in_graph = self.local_nodes_dict.nearest_neighbors(robot_location.tolist(), 1)[0].data.coords
        current_index = np.argwhere(local_node_coords_to_check == robot_in_graph[0] + robot_in_graph[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)

        occupancy = np.zeros((n_nodes, 1))
        for location in robot_locations:
            location_in_graph = self.local_nodes_dict.find((location[0], location[1])).data.coords
            index = np.argwhere(local_node_coords_to_check == location_in_graph[0] + location_in_graph[1] * 1j)[0][0]
            if index == current_index:
                occupancy[index] = -1
            else:
                occupancy[index] = 1
        assert sum(occupancy) == N_AGENTS-2, print(robot_locations)
        return all_node_coords, utility, guidepost, signal, occupancy, adjacent_matrix, current_index, neighbor_indices

    def h(self, coords_1, coords_2):
        # h = abs(coords_1[0] - coords_2[0]) + abs(coords_1[1] - coords_2[1])
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

            for neighbor_node_coords in node.explored_neighbor_list:
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
            for neighbor_node_coords in node.explored_neighbor_list:
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


class LocalNode:
    def __init__(self, coords):
        self.coords = coords
        self.utility_range = UTILITY_RANGE
        self.observable_explore_frontiers = None
        self.observable_safe_frontiers = None
        self.safe_utility = 0
        self.visited = 0
        self.safe = 0

        self.explored_neighbor_matrix = -np.ones((5, 5))
        self.explored_neighbor_list = []
        self.explored_neighbor_matrix[2, 2] = 1
        self.explored_neighbor_list.append(self.coords)

    def initialize_observable_safe_frontiers(self, safe_frontiers, extended_safe_zone_info):
        if not self.safe:
            self.safe_utility = 0
            return None
        elif safe_frontiers.shape[0] == 0:
            self.safe_utility = 0
            return safe_frontiers
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

    def update_observable_safe_frontiers(self, safe_frontiers, extended_safe_zone_info):
        if not self.safe:
            self.safe_utility = 0
            self.observable_safe_frontiers = np.array([])
            return
        elif safe_frontiers.shape[0] == 0:
            self.safe_utility = 0
            self.observable_safe_frontiers = safe_frontiers
            return
        else:
            safe_frontiers = safe_frontiers.reshape(-1, 2)
            old_frontier_to_check = self.observable_safe_frontiers[:, 0] + self.observable_safe_frontiers[:, 1] * 1j
            local_frontiers_to_check = safe_frontiers[:, 0] + safe_frontiers[:, 1] * 1j
            to_observe_index = np.where(
                np.isin(old_frontier_to_check, local_frontiers_to_check, assume_unique=True) == True)
            new_frontier_index = np.where(
                np.isin(local_frontiers_to_check, old_frontier_to_check, assume_unique=True) == False)
            self.observable_safe_frontiers = self.observable_safe_frontiers[to_observe_index]
            new_frontiers = safe_frontiers[new_frontier_index]

            # add new frontiers in the observable frontiers
            if new_frontiers.shape[0] > 0:
                dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
                new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]
                for point in new_frontiers_in_range:
                    collision = check_collision(self.coords, point, extended_safe_zone_info)
                    if not collision:
                        self.observable_safe_frontiers = np.concatenate((self.observable_safe_frontiers, point.reshape(1, 2)), axis=0)
            self.safe_utility = self.observable_safe_frontiers.shape[0]
            if self.safe_utility <= MIN_UTILITY:
                self.safe_utility = 0

    def update_neighbor_explored_nodes(self, extended_local_map_info, nodes_dict, plot_x=None, plot_y=None):
        for i in range(self.explored_neighbor_matrix.shape[0]):
            for j in range(self.explored_neighbor_matrix.shape[1]):
                if self.explored_neighbor_matrix[i, j] != -1:
                    continue
                else:
                    center_index = self.explored_neighbor_matrix.shape[0] // 2
                    if i == center_index and j == center_index:
                        self.explored_neighbor_matrix[i, j] = 1
                        # self.explored_neighbor_list.append(self.coords)
                        continue

                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                          self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        cell = get_cell_position_from_coords(neighbor_coords, extended_local_map_info)
                        if cell[0] < extended_local_map_info.map.shape[1] and cell[1] < extended_local_map_info.map.shape[0]:
                            if extended_local_map_info.map[cell[1], cell[0]] == 1:
                                self.explored_neighbor_matrix[i, j] = 1
                            continue
                    else:
                        neighbor_node = neighbor_node.data
                        collision = check_collision(self.coords, neighbor_coords, extended_local_map_info)
                        neighbor_matrix_x = center_index + (center_index - i)
                        neighbor_matrix_y = center_index + (center_index - j)
                        if not collision:
                            self.explored_neighbor_matrix[i, j] = 1
                            self.explored_neighbor_list.append(neighbor_coords)

                            neighbor_node.explored_neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.explored_neighbor_list.append(self.coords)

                            if plot_x is not None and plot_y is not None:
                                plot_x.append([self.coords[0], neighbor_coords[0]])
                                plot_y.append([self.coords[1], neighbor_coords[1]])

    def set_safe(self):
        self.safe = 1

    def set_unsafe(self):
        self.safe = 0
        self.observable_safe_frontiers = np.array([])
        self.safe_utility = 0

    def set_visited(self):
        self.visited = 1
        # self.observable_safe_frontiers = np.array([])
        # self.safe_utility = 0