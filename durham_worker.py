import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from env import Env
from agent import Agent
from utils import *
from local_node_manager_quadtree import NodeManager
from sensor import exploration_sensor, coverage_sensor


TEST_N_AGENTS = 2
EXPLORATION = True
# GROUP_START: change in test_parameter.py

MAX_EPISODE_STEP = 35
SENSOR_RANGE = 20
UTILITY_RANGE = 0.8 * SENSOR_RANGE
NODE_RESOLUTION = 4.0
CELL_SIZE = 0.4
gifs_path = 'results/gifs'

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class DurhamWorker:
    def __init__(self, meta_agent_id, global_step, save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        np.random.seed(123)
        self.env = Env(global_step, n_agent=TEST_N_AGENTS, explore=EXPLORATION, plot=self.save_image, test=True)
        self.node_manager = NodeManager(self.env.ground_truth_coords, self.env.ground_truth_info, explore=EXPLORATION, plot=self.save_image)
        self.robot_list = [Agent(i, None, self.node_manager, 'cpu', self.save_image) for i in range(self.env.n_agent)]
        self.utility = None
        self.agent_status = [0] * self.env.n_agent  # 0: follower, 1: frontier guard
        self.agent_next_location = [None] * self.env.n_agent

    def assign_frontier_guard(self, best_locations):
        frontier_guards = []
        for location in best_locations:
            min_dist = 1e6
            for robot in self.robot_list:
                if robot.id in frontier_guards:
                    continue
                path_coords, dist = self.node_manager.a_star(robot.location, location)
                if dist < min_dist:
                    min_dist = dist
                    frontier_guard = robot.id
                    best_path = path_coords
            frontier_guards.append(frontier_guard)
            self.agent_status[frontier_guard] = 1
            self.agent_next_location[frontier_guard] = best_path[0]
        assert len(frontier_guards) > 0
        return frontier_guards

    def assign_follower(self, frontier_guards, best_locations):
        for robot in self.robot_list:
            if robot.id not in frontier_guards:
                min_dist = 1e6
                for guard_coords in best_locations:
                    path_coords, dist = self.node_manager.a_star(robot.location, guard_coords)
                    if dist < min_dist:
                        min_dist = dist
                        best_path = path_coords
                self.agent_next_location[robot.id] = best_path[0]

    def update_imaginary_safe_zone(self, robot_cell, imaginary_safe_zone):
        padded_robot_belief = deepcopy(self.env.robot_belief)
        padded_robot_belief[padded_robot_belief == 127] = 255
        new_imaginary_safe_zone = coverage_sensor(robot_cell, round(SENSOR_RANGE / CELL_SIZE), deepcopy(imaginary_safe_zone), padded_robot_belief)
        safe_increase = np.sum(new_imaginary_safe_zone == 255) - np.sum(imaginary_safe_zone == 255)
        return new_imaginary_safe_zone, safe_increase

    def find_next_best_views(self):
        all_node_coords = self.robot_list[0].local_node_coords
        self.utility = self.robot_list[0].safe_utility
        non_zero_utility_node_indices = np.argwhere(self.utility > 0)[:, 0].tolist()
        assert len(non_zero_utility_node_indices) > 0
        candidate_node_coords = all_node_coords[non_zero_utility_node_indices]
        imaginary_safe_zone = self.env.safe_zone
        max_increase = 0
        best_locations = []
        filtered_indices = []

        while True:
            for i, coords in enumerate(candidate_node_coords):
                if i in filtered_indices:
                    continue
                candidate_flag = True
                for location in best_locations:
                    dist_to_candidate = np.linalg.norm(location - coords)
                    if (dist_to_candidate < UTILITY_RANGE) and (not check_collision(location, coords, self.env.belief_info)):
                        candidate_flag = False
                        break
                if candidate_flag:
                    robot_cell = get_cell_position_from_coords(coords, self.env.belief_info)
                    new_imaginary_safe_zone, safe_increase = self.update_imaginary_safe_zone(robot_cell, imaginary_safe_zone)
                    if safe_increase > max_increase:
                        max_increase = safe_increase
                        best_coords = coords
                        best_imaginary_safe_zone = new_imaginary_safe_zone
                else:
                    filtered_indices.append(i)
            if len(best_locations) > 0:
                if np.all(best_coords == best_locations[-1]):
                    print(best_locations)
                    return best_locations
            imaginary_safe_zone = best_imaginary_safe_zone
            best_locations.append(best_coords)
            max_increase = 0
        # return best_locations[:-1]


    def run_episode(self):
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
        for robot in self.robot_list:
            robot.update_safe_graph(self.env.safe_info, self.env.uncovered_safe_frontiers, self.env.counter_safe_info)
        for robot in self.robot_list:
            robot.update_planning_state(self.env.robot_locations)
        if self.save_image:
            self.plot_local_env(-1)

        max_travel_dist = 0
        for i in range(MAX_EPISODE_STEP):
            self.agent_status = [0] * self.env.n_agent
            best_locations = self.find_next_best_views()
            frontier_guards = self.assign_frontier_guard(best_locations)
            self.assign_follower(frontier_guards, best_locations)

            selected_locations = []
            dist_list = []
            for robot in self.robot_list:
                next_location = self.agent_next_location[robot.id]
                selected_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))

            selected_locations = np.array(selected_locations).reshape(-1, 2)
            arriving_sequence = np.argsort(np.array(dist_list))
            selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]
            for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
                solved_locations = selected_locations_in_arriving_sequence[:j]
                while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                    id = arriving_sequence[j]
                    nearby_nodes = self.robot_list[id].node_manager.local_nodes_dict.nearest_neighbors(
                        selected_location.tolist(), 25)
                    for node in nearby_nodes:
                        coords = node.data.coords
                        if coords[0] + coords[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                            continue
                        selected_location = coords
                        break
                    selected_locations_in_arriving_sequence[j] = selected_location
                    selected_locations[id] = selected_location

            self.env.decrease_safety(selected_locations)
            # self.env.safe_zone_frontiers = get_safe_zone_frontier(self.env.safe_info, self.env.belief_info)

            self.env.step(selected_locations)

            self.env.classify_safe_frontier(selected_locations)

            for robot in self.robot_list:
                robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
            for robot in self.robot_list:
                robot.update_safe_graph(self.env.safe_info, self.env.uncovered_safe_frontiers,
                                        self.env.counter_safe_info)
            for robot in self.robot_list:
                robot.update_planning_state(self.env.robot_locations)

            max_travel_dist += np.max(dist_list)

            done = self.env.check_done()

            if self.save_image:
                self.plot_local_env(i)

            if done:
                break

            if max_travel_dist >= 1000:
                max_travel_dist = 1000
                break

        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def plot_local_env(self, step, planned_paths=None):
        plt.switch_backend('agg')
        plt.figure(figsize=(9, 4))
        plt.subplot(1, 2, 2)
        plt.imshow(self.env.robot_belief, cmap='gray', vmin=0, alpha=0)
        plt.axis('off')
        color_list = ['r', 'b', 'g', 'y', 'm', 'c', 'k', 'w', (1,0.5,0.5), (0.2,0.5,0.7)]
        robot = self.robot_list[0]
        nodes = get_cell_position_from_coords(robot.local_node_coords, robot.safe_zone_info)
        plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.safe_utility, s=5, zorder=2)  # 5, 20
        for i in range(nodes.shape[0]):
            for j in range(i+1, nodes.shape[0]):
                if robot.local_adjacent_matrix[i, j] == 0:
                    plt.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], c=(0.988, 0.557, 0.675), linewidth=1.5, zorder=1)  # 0.5, 1.5

        plt.subplot(1, 2, 1)
        plt.imshow(self.env.robot_belief, cmap='gray')

        self.env.classify_safe_frontier(self.env.robot_locations)
        covered_safe_frontier_cells = get_cell_position_from_coords(self.env.covered_safe_frontiers, self.env.safe_info).reshape(-1, 2)
        uncovered_safe_frontier_cells = get_cell_position_from_coords(self.env.uncovered_safe_frontiers, self.env.safe_info).reshape(-1, 2)
        if covered_safe_frontier_cells.shape[0] != 0:
            plt.scatter(covered_safe_frontier_cells[:, 0], covered_safe_frontier_cells[:, 1], c='g', s=1, zorder=6)  # 0.4, 1
        if uncovered_safe_frontier_cells.shape[0] != 0:
            plt.scatter(uncovered_safe_frontier_cells[:, 0], uncovered_safe_frontier_cells[:, 1], c='r', s=1, zorder=6)  # 0.4, 1

        n_segments = len(self.robot_list[0].trajectory_x) - 1
        alpha_values = np.linspace(0.3, 1, n_segments)
        for robot in self.robot_list:
            c = color_list[robot.id]
            if robot.id == 0:
                alpha_mask = robot.safe_zone_info.map / 255 / 3
                plt.imshow(robot.safe_zone_info.map, cmap='Greens', alpha=alpha_mask)
                plt.axis('off')

            robot_cell = get_cell_position_from_coords(robot.location, robot.safe_zone_info)
            plt.plot(robot_cell[0], robot_cell[1], c=c, marker='o', markersize=10, zorder=5)  # 5,10

            for i in range(n_segments):
                plt.plot((np.array(robot.trajectory_x[i:i+2]) - robot.global_map_info.map_origin_x) / robot.cell_size,
                         (np.array(robot.trajectory_y[i:i+2]) - robot.global_map_info.map_origin_y) / robot.cell_size, c,
                         linewidth=2, alpha=alpha_values[i], zorder=3)  # 1,2

        plt.axis('off')
        plt.suptitle('Explored rate: {:.4g} | Cleared rate: {:.4g} | Trajectory length: {:.4g}'.format(self.env.explored_rate,
                                                                                                self.env.safe_rate,
                                                                                                max([robot.travel_dist for robot in self.robot_list])))
        plt.tight_layout()
        # plt.show()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step))
        plt.close()
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)
        self.env.frame_files.append(frame)


if __name__ == '__main__':
    worker = DurhamWorker(0, 0, True)
    worker.run_episode()