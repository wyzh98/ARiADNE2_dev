import copy
import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import block_reduce
from scipy import ndimage
from copy import deepcopy
import numpy as np

from sensor import exploration_sensor, coverage_sensor, decrease_safety_by_frontier
from parameter import *
from utils import *


class Env:
    def __init__(self, episode_index, n_agent=N_AGENTS, plot=False, test=False):
        self.episode_index = episode_index
        self.plot = plot
        self.test = test
        self.n_agent = n_agent

        self.ground_truth, initial_cell = self.import_ground_truth(episode_index)
        self.cell_size = CELL_SIZE  # meter
        self.sensor_range = SENSOR_RANGE  # meter
        self.safety_range = SAFETY_RANGE  # meter
        self.belief_origin_x = -np.round(initial_cell[0] * self.cell_size, 1)  # meter
        self.belief_origin_y = -np.round(initial_cell[1] * self.cell_size, 1)  # meter

        self.explored_rate = 0
        self.safe_rate = 0
        self.done = False

        self.ground_truth_info = Map_info(self.ground_truth, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.ground_truth_coords, _ = get_local_node_coords(np.array([0.0, 0.0]), self.ground_truth_info)

        self.robot_belief = np.ones_like(self.ground_truth) * 127
        self.update_robot_belief(initial_cell)
        self.belief_info = Map_info(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        self.safe_zone = np.zeros_like(self.ground_truth)
        self.update_safe_zone(initial_cell)
        self.safe_info = Map_info(self.safe_zone, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        free, _ = get_local_node_coords(np.array([0.0, 0.0]), self.belief_info)
        choice = np.random.choice(free.shape[0], self.n_agent, replace=False)
        start_loc = free[choice]
        self.robot_locations = np.array(start_loc)

        robot_cells = get_cell_position_from_coords(self.robot_locations, self.belief_info)
        for robot_cell in robot_cells:
            self.update_robot_belief(robot_cell)
        for robot_cell in robot_cells:
            self.update_safe_zone(robot_cell)

        self.old_safe_zone = deepcopy(self.safe_zone)
        self.explore_frontiers = get_explore_frontier(self.belief_info)
        self.safe_zone_frontiers = get_safe_zone_frontier(self.safe_info, self.belief_info)
        self.covered_safe_frontiers = deepcopy(self.safe_zone_frontiers)
        self.uncovered_safe_frontiers = []

        if self.plot:
            self.frame_files = []


    def import_ground_truth(self, episode_index):
        if self.test:
            map_dir = f'maps_test'
        else:
            map_dir = f'maps_medium'
        map_list = os.listdir(map_dir)
        map_index = episode_index % np.size(map_list)

        ground_truth = (io.imread(map_dir + '/' + map_list[map_index], 1)).astype(int)  # 127: obstacle, 195: free, 208: start
        ground_truth = block_reduce(ground_truth, 2, np.min)
        robot_cell = np.array(np.nonzero(ground_truth == 208))
        robot_cell = np.array([robot_cell[1, 10], robot_cell[0, 10]])

        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1

        return ground_truth, robot_cell

    def update_robot_belief(self, robot_cell):
        self.robot_belief = exploration_sensor(robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief, self.ground_truth)

    def update_safe_zone(self, robot_cell):
        self.safe_zone = coverage_sensor(robot_cell, round(self.sensor_range / self.cell_size), self.safe_zone, self.ground_truth)

    def get_intersect_area(self, locations_togo):
        robot_cells = get_cell_position_from_coords(self.robot_locations, self.belief_info)
        robot_cells_togo = get_cell_position_from_coords(locations_togo, self.belief_info)
        curr_coverage = np.zeros_like(self.robot_belief)
        next_coverage = np.zeros_like(self.robot_belief)
        for robot_cell in robot_cells:
            curr_coverage = coverage_sensor(robot_cell, round(self.sensor_range / self.cell_size), curr_coverage, self.robot_belief)
        for robot_cell in robot_cells_togo:
            next_coverage = coverage_sensor(robot_cell, round(self.sensor_range / self.cell_size), next_coverage, self.robot_belief)
        intersection = curr_coverage * next_coverage
        intersection[intersection > curr_coverage.max()] = curr_coverage.max()
        return intersection

    def decrease_safety(self, locations_togo):
        cells_frontiers = get_cell_position_from_coords(self.safe_zone_frontiers, self.safe_info).reshape(-1, 2)
        cells_togo = get_cell_position_from_coords(locations_togo, self.safe_info).reshape(-1, 2)
        sensor_cell_range = round(self.sensor_range / self.cell_size)
        safety_cell_range = round(self.safety_range / self.cell_size)
        intersect_area = self.get_intersect_area(locations_togo)
        for frontier_loc, frontier_cell in zip(self.safe_zone_frontiers, cells_frontiers):
            nearby_agent_indices = np.argwhere(np.linalg.norm(frontier_cell - cells_togo, axis=1) <= sensor_cell_range)
            nearby_agent_locations = locations_togo[nearby_agent_indices]
            uncovered = True

            for loc in nearby_agent_locations:
                if not check_collision(frontier_loc, loc, self.belief_info, max_collision=3):
                    uncovered = False
            if uncovered:
                cell_center = [safety_cell_range, safety_cell_range]
                x_lower, x_upper = frontier_cell[0] - safety_cell_range, frontier_cell[0] + safety_cell_range + 1
                y_lower, y_upper = frontier_cell[1] - safety_cell_range, frontier_cell[1] + safety_cell_range + 1
                if x_lower < 0:
                    cell_center[0] += x_lower
                    x_lower = 0
                if x_upper > self.safe_zone.shape[1]:
                    x_upper = self.safe_zone.shape[1]
                if y_lower < 0:
                    cell_center[1] += y_lower
                    y_lower = 0
                if y_upper > self.safe_zone.shape[0]:
                    y_upper = self.safe_zone.shape[0]
                sub_safe_zone = self.safe_zone[y_lower: y_upper, x_lower: x_upper]
                sub_belief = self.robot_belief[y_lower: y_upper, x_lower: x_upper]
                sub_intersection = intersect_area[y_lower: y_upper, x_lower: x_upper]
                decrease_safety_by_frontier(cell_center, safety_cell_range, sub_safe_zone, sub_belief, sub_intersection)

    def classify_safe_frontier(self, robot_locations):
        self.uncovered_safe_frontiers, self.covered_safe_frontiers = [], []
        cells_frontiers = get_cell_position_from_coords(self.safe_zone_frontiers, self.safe_info).reshape(-1, 2)
        cells_togo = get_cell_position_from_coords(robot_locations, self.safe_info).reshape(-1, 2)
        sensor_cell_range = round(self.sensor_range / self.cell_size)

        for frontier_loc, frontier_cell in zip(self.safe_zone_frontiers, cells_frontiers):
            nearby_agent_indices = np.argwhere(np.linalg.norm(frontier_cell - cells_togo, axis=1) <= sensor_cell_range)
            nearby_agent_locations = robot_locations[nearby_agent_indices]
            uncovered = True
            for loc in nearby_agent_locations:
                if not check_cumulative_collision(frontier_loc, loc, self.belief_info, max_collision=8):
                    uncovered = False
            if uncovered:
                self.uncovered_safe_frontiers.append(frontier_loc)
            else:
                self.covered_safe_frontiers.append(frontier_loc)
        self.uncovered_safe_frontiers = np.array(self.uncovered_safe_frontiers).reshape(-1, 2)
        self.covered_safe_frontiers = np.array(self.covered_safe_frontiers).reshape(-1, 2)

    @staticmethod
    def get_positive_cluster_info(diff_map):
        diff_map = deepcopy(diff_map)
        diff_map[diff_map < 0] = 0
        clusters, n_clusters = ndimage.label(diff_map, structure=np.ones((3, 3)))
        cluster_centers = []
        cluster_size = []
        for i in range(n_clusters):
            cluster = np.argwhere(clusters == i + 1)
            cluster_centers.append(np.mean(cluster, axis=0))
            cluster_size.append(cluster.shape[0])
        return cluster_centers, cluster_size

    def calculate_safety_change_clusters(self):
        # Separate safety increase and decrease to avoid structual connection
        inc_centers, inc_sizes = self.get_positive_cluster_info(self.safe_zone - self.old_safe_zone)
        dec_centers, dec_sizes = self.get_positive_cluster_info(self.old_safe_zone - self.safe_zone)
        dec_sizes = [-s for s in dec_sizes]
        cluster_centers = inc_centers + dec_centers
        cluster_sizes = inc_sizes + dec_sizes
        return np.asarray(cluster_centers), np.asarray(cluster_sizes)

    def calculate_reward(self):
        safety_increase_flag = np.sum(self.safe_zone == 255) - np.sum(self.old_safe_zone == 255)
        reward_list = np.zeros(self.n_agent)
        cluster_centers, cluster_sizes = self.calculate_safety_change_clusters()
        robot_cells = get_cell_position_from_coords(self.robot_locations, self.belief_info)
        for center, cluster_size in zip(cluster_centers, cluster_sizes):
            inverse_dist = 1 / (np.linalg.norm(robot_cells - center, axis=1) + 1)
            weights = inverse_dist / np.sum(inverse_dist)
            reward_list += weights * cluster_size / 1000

        self.old_safe_zone = deepcopy(self.safe_zone)

        return reward_list * self.n_agent, safety_increase_flag

    def check_done(self):
        assert self.explored_rate >= self.safe_rate
        if self.explored_rate > 0.999 and self.safe_rate >= 0.999:
            self.done = True
        return self.done

    def evaluate_exploration_rate(self):
        self.explored_rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)

    def evaluate_safe_zone_rate(self):
        self.safe_rate = np.sum(self.safe_zone > 0) / np.sum(self.ground_truth == 255)

    def step(self, next_waypoints):
        self.robot_locations = next_waypoints
        next_cells = get_cell_position_from_coords(next_waypoints, self.belief_info)
        for cell in next_cells:
            self.update_robot_belief(cell)
            self.update_safe_zone(cell)
        self.explore_frontiers = get_explore_frontier(self.belief_info)
        self.safe_zone_frontiers = get_safe_zone_frontier(self.safe_info, self.belief_info)
        self.evaluate_exploration_rate()
        self.evaluate_safe_zone_rate()
