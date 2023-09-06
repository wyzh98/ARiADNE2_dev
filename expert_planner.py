from copy import deepcopy

from utils import *
from ortools_solver import solve_vrp


class Expert_planner:
    def __init__(self, local_node_manager):
        self.max_iteration_step = 5
        self.local_node_manager = local_node_manager
        self.last_viewpoints = None

    def plan_coverage_paths(self, robot_locations):
        all_node_coords = []
        utility = []
        for node in self.local_node_manager.local_nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
            utility.append(node.data.utility)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = np.array(utility)
        
        best_paths = None
        c_best = 1e10
        q_indices = np.where(utility > 0)[0]
        q_array = all_node_coords[q_indices]

        if self.last_viewpoints:
            nodes_dict = deepcopy(self.local_node_manager.local_nodes_dict)
            q_array_prime = deepcopy(q_array)
            v_list = [location for location in robot_locations]

            for viewpoints in self.last_viewpoints:
                last_node = nodes_dict.find(viewpoints.tolist()).data
                if last_node.utility > 0:
                    v_list.append(last_node.coords)
                    observable_frontiers = np.array(last_node.observable_frontiers)
                    index = np.argwhere(
                        last_node.coords[0] + last_node.coords[1] * 1j == q_array_prime[:, 0] + q_array_prime[:,
                                                                                                1] * 1j)[0][0]
                    q_array_prime = np.delete(q_array_prime, index, axis=0)
                    for coords in q_array_prime:
                        node = nodes_dict.find(coords.tolist()).data
                        if node.utility > 0 and np.linalg.norm(coords - last_node.coords) < 2 * SENSOR_RANGE:
                            node.delete_observed_frontiers(observable_frontiers)

            q_utility = []
            for coords in q_array_prime:
                node = nodes_dict.find(coords.tolist()).data
                q_utility.append(node.utility)
            q_utility = np.array(q_utility)

            while q_array_prime.shape[0] > 0 and q_utility.sum() > 0:
                indices = np.array(range(q_array_prime.shape[0]))
                weights = q_utility / q_utility.sum()
                sample = np.random.choice(indices, size=1, replace=False, p=weights)[0]
                viewpoint_coords = q_array_prime[sample]
                # assert viewpoint_coords[0] + viewpoint_coords[1] * 1j not in v_list[:][0] + v_list[:][1] * 1j
                v_list.append(viewpoint_coords)
                viewpoint = nodes_dict.find(viewpoint_coords.tolist()).data
                observable_frontiers = np.array(viewpoint.observable_frontiers)
                q_array_prime = np.delete(q_array_prime, sample, axis=0)
                q_utility = []
                for coords in q_array_prime:
                    node = nodes_dict.find(coords.tolist()).data
                    if node.utility > 0 and np.linalg.norm(coords - viewpoint_coords) < 2 * SENSOR_RANGE:
                        node.delete_observed_frontiers(observable_frontiers)
                    q_utility.append(node.utility)
                q_utility = np.array(q_utility)

            paths, dist = self.find_paths(v_list, robot_locations, all_node_coords, utility)
            best_paths = paths
            c_best = dist

        for i in range(self.max_iteration_step):
            nodes_dict = deepcopy(self.local_node_manager.local_nodes_dict)
            q_array_prime = deepcopy(q_array)
            v_list = [location for location in robot_locations]
            q_utility = utility[q_indices]
            while q_array_prime.shape[0] > 0 and q_utility.sum() > 0:
                indices = np.array(range(q_array_prime.shape[0]))
                weights = q_utility / q_utility.sum()
                sample = np.random.choice(indices, size=1, replace=False, p=weights)[0]
                viewpoint_coords = q_array_prime[sample]
                # assert viewpoint_coords[0] + viewpoint_coords[1] * 1j not in v_list[:][0] + v_list[:][1] * 1j
                v_list.append(viewpoint_coords)
                node = nodes_dict.find(viewpoint_coords.tolist()).data
                observable_frontiers = np.array(node.observable_frontiers)
                q_array_prime = np.delete(q_array_prime, sample, axis=0)
                q_utility = []
                for coords in q_array_prime:
                    node = nodes_dict.find(coords.tolist()).data
                    if node.utility > 0:
                        node.delete_observed_frontiers(observable_frontiers)
                    q_utility.append(node.utility)
                q_utility = np.array(q_utility)

            paths, dist = self.find_paths(v_list, robot_locations, all_node_coords, utility)
            if dist < c_best:
                best_paths = paths
                c_best = dist
                self.last_viewpoints = v_list[len(robot_locations):]

        return best_paths

    def find_paths(self, viewpoints, robot_locations, all_node_coords, utility):
        size = len(viewpoints)
        path_matrix = []
        distance_matrix = np.ones((size, size), dtype=int) * 1000
        for i in range(size):
            path_matrix.append([])
            for j in range(size):
                path_matrix[i].append([])

        # for i in range(size):
        #     for j in range(size)[i:]:
        #         if i == j:
        #             pass
        #         else:
        #             path, dist = self.local_node_manager.a_star(viewpoints[i], viewpoints[j])
        #             dist = np.round(dist).astype(int)
        #             distance_matrix[i][j] = dist
        #             distance_matrix[j][i] = dist
        #
        #             path_matrix[i][j] = path
        #             path_reverse = [(viewpoints[i][0], viewpoints[i][1])]
        #             path_reverse += path
        #             path_reverse = path_reverse[::-1][1:]
        #             path_matrix[j][i] = path_reverse

        for i in range(size):
            dist_dict, prev_dict = self.local_node_manager.Dijkstra(viewpoints[i])
            for j in range(size):
                path, dist = self.local_node_manager.get_Dijkstra_path_and_dist(dist_dict, prev_dict,
                                                                                       viewpoints[j])
                dist = dist.astype(int)
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist

                path_matrix[i][j] = path
                path_reverse = [(viewpoints[i][0], viewpoints[i][1])]
                path_reverse += path
                path_reverse = path_reverse[::-1][1:]
                path_matrix[j][i] = path_reverse

        robot_indices = [i for i in range(len(robot_locations))]
        for i in range(size):
            for j in robot_indices:
                distance_matrix[i][j] = 0

        paths, dist = solve_vrp(distance_matrix, robot_indices)

        paths_coords = []
        for path, robot_location in zip(paths, robot_locations):
            path_coords = []
            for index1, index2 in zip(path[:-1], path[1:]):
                path_coords += path_matrix[index1][index2]
            if len(path_coords) == 0:
                indices = np.argwhere(utility > 0).reshape(-1)
                node_coords = all_node_coords[indices]
                dist_dict, prev_dict = self.local_node_manager.Dijkstra(robot_location)
                nearest_utility_coords = robot_location
                nearest_dist = 1e8
                for coords in node_coords:
                    dist = dist_dict[(coords[0], coords[1])]
                    if 0 < dist < nearest_dist:
                        nearest_dist = dist
                        nearest_utility_coords = coords
                    if nearest_dist == 1e8 and dist != 0:
                        print(dist, robot_location, coords)
                        # print(nearest_dist, coords, nearest_utility_coords, robot_location)

                path_coords, dist = self.local_node_manager.a_star(robot_location, nearest_utility_coords)
                if len(path_coords) == 0:
                    print("nearest", nearest_utility_coords, robot_location, node_coords.shape, nearest_dist)

            paths_coords.append(path_coords)
        return paths_coords, dist