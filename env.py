import copy
import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import block_reduce
from copy import deepcopy
import numpy as np

from sensor import exploration_sensor, coverage_sensor
from parameter import *
from utils import *


class Env:
    def __init__(self, episode_index, plot=False):
        self.episode_index = episode_index
        self.plot = plot
        self.ground_truth, initial_cell = self.import_ground_truth(episode_index)
        self.cell_size = CELL_SIZE  # meter

        self.robot_belief = copy.deepcopy(self.ground_truth)  # full knowledge of the environment
        self.belief_origin_x = -np.round(initial_cell[0] * self.cell_size, 1)   # meter
        self.belief_origin_y = -np.round(initial_cell[1] * self.cell_size, 1)  # meter
        self.belief_info = Map_info(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        self.sensor_range = SENSOR_RANGE  # meter
        self.explored_rate = 0
        self.safe_rate = 0
        self.done = False

        self.free_locations, _ = get_local_node_coords(np.array([0.0, 0.0]), self.belief_info)
        start_loc_idx = np.argsort(np.linalg.norm(self.free_locations, axis=1))[:N_AGENTS]
        start_loc = self.free_locations[start_loc_idx]
        self.robot_locations = np.array(start_loc)

        self.safe_zone = np.zeros_like(self.ground_truth)
        self.safe_info = Map_info(self.safe_zone, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        robot_cells = get_cell_position_from_coords(self.robot_locations, self.belief_info)
        for robot_cell in robot_cells:
            self.update_safe_zone(robot_cell)

        self.old_safe_zone = deepcopy(self.safe_zone)
        self.safe_zone_frontiers = get_safe_zone_frontier(self.safe_info, self.belief_info)

        if self.plot:
            self.frame_files = []


    def import_ground_truth(self, episode_index):
        map_dir = 'maps_simple'
        map_list = os.listdir(map_dir)
        map_index = episode_index % np.size(map_list)

        if map_dir == 'maps_simple':
            ground_truth = (io.imread(map_dir + '/' + map_list[map_index], 1) * 255).astype(int)
        elif map_dir == 'maps_medium':
            ground_truth = (io.imread(map_dir + '/' + map_list[map_index], 1)).astype(int)
        else:
            raise NotImplementedError

        ground_truth = block_reduce(ground_truth, 2, np.min)
        robot_cell = np.array(np.nonzero(ground_truth == 208))

        if map_dir == 'maps_simple':
            robot_cell = np.array([robot_cell[1, 10], robot_cell[0, 10]])
        elif map_dir == 'maps_medium':
            robot_cell = np.array([robot_cell[1, 64], robot_cell[0, 64]])
        else:
            raise NotImplementedError

        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1

        return ground_truth, robot_cell

    def update_robot_belief(self, robot_cell):
        self.robot_belief = exploration_sensor(robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief,
                                               self.ground_truth)

    def update_safe_zone(self, robot_cell):
        self.safe_zone = coverage_sensor(robot_cell, round(self.sensor_range / self.cell_size), self.safe_zone,
                                         self.ground_truth)

    def calculate_reward(self):
        reward = 0

        safe_zone_frontiers = get_safe_zone_frontier(self.safe_info, self.belief_info)
        if safe_zone_frontiers.shape[0] == 0:
            delta_num = self.safe_zone_frontiers.shape[0]
        else:
            safe_zone_frontiers = safe_zone_frontiers.reshape(-1, 2)
            frontiers_to_check = safe_zone_frontiers[:, 0] + safe_zone_frontiers[:, 1] * 1j
            pre_frontiers_to_check = self.safe_zone_frontiers[:, 0] + self.safe_zone_frontiers[:, 1] * 1j
            frontiers_num = np.intersect1d(frontiers_to_check, pre_frontiers_to_check).shape[0]
            pre_frontiers_num = pre_frontiers_to_check.shape[0]
            delta_num = pre_frontiers_num - frontiers_num

        reward += delta_num / 50

        new_area = np.sum(self.safe_zone == 255) - np.sum(self.old_safe_zone == 255)
        # reward += np.clip(new_area / 1000, 0.1, 0.5)

        self.safe_zone_frontiers = safe_zone_frontiers
        self.old_safe_zone = deepcopy(self.safe_zone)

        return reward

    def check_done(self):
        if np.sum(self.ground_truth == 255) - np.sum(self.safe_zone == 255) <= 250:
            self.done = True

    def evaluate_exploration_rate(self):
        self.explored_rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)

    def evaluate_safe_zone_rate(self):
        self.safe_rate = np.sum(self.safe_zone > 0) / np.sum(self.ground_truth == 255)

    def step(self, next_waypoint, agent_id):
        self.evaluate_safe_zone_rate()
        self.robot_locations[agent_id] = next_waypoint
        reward = 0
        cell = get_cell_position_from_coords(next_waypoint, self.belief_info)
        self.update_safe_zone(cell)
        # reward = self.calculate_reward(dist)

        return reward


