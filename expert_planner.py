from copy import deepcopy

from utils import *
from ortools_solver import solve_vrp


class Expert_planner:
    def __init__(self, local_node_manager):
        self.max_iteration_step = 10
        self.local_node_manager = local_node_manager
        
    def plan_coverage_paths(self, robot_locations):
        for coords in robot_locations:
            node = self.local_node_manager.local_nodes_dict.find(coords.tolist())
            node.data.set_visited()

        all_node_coords = []
        for node in self.local_node_manager.local_nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)

        utility = []
        for i, coords in enumerate(all_node_coords):
            node = self.local_node_manager.local_nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
        utility = np.array(utility)

        best_paths = None
        c_best = 1e10
        q_indices = np.where(utility > 0)[0]
        q_array = all_node_coords[q_indices]

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

            paths, dist = self.find_paths(v_list, robot_locations)
            if dist < c_best:
                best_paths = paths
                c_best = dist

        return best_paths

    def find_paths(self, viewpoints, robot_locations):
        size = len(viewpoints)
        path_matrix = []
        distance_matrix = np.ones((size, size), dtype=int) * 1000
        for i in range(size):
            path_matrix.append([])
            for j in range(size):
                path_matrix[i].append([])
        for i in range(size):
            for j in range(size)[i:]:
                if i == j:
                    pass
                else:
                    path, dist = self.local_node_manager.a_star(viewpoints[i], viewpoints[j])
                    dist = np.round(dist).astype(int)
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
                max_utility = 0
                max_utility_coords = robot_location
                for coords in viewpoints[len(robot_locations):]:
                    node = self.local_node_manager.ground_truth_nodes_dict.find(coords.tolist())
                    utility = node.data.utility
                    if utility > max_utility:
                        max_utility = utility
                        max_utility_coords = node.data.coords

                path_coords, dist = self.local_node_manager.a_star(robot_location, max_utility_coords)

            paths_coords.append(path_coords)
        return paths_coords, dist