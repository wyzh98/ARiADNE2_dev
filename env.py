import copy
import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import block_reduce
from copy import deepcopy
import numpy as np

from sensor import exploration_sensor, coverage_sensor, decrease_safety_by_frontier
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
        self.safety_range = SENSOR_RANGE  # meter
        self.explored_rate = 0
        self.safe_rate = 0
        self.done = False

        self.safe_zone = np.zeros_like(self.ground_truth)
        self.safe_info = Map_info(self.safe_zone, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.update_safe_zone(initial_cell)
        safe, _ = get_local_node_coords(np.array([0.0, 0.0]), self.safe_info)
        choice = np.random.choice(safe.shape[0], N_AGENTS, replace=False)

        self.free_locations, _ = get_local_node_coords(np.array([0.0, 0.0]), self.belief_info)
        # start_loc_idx = np.argsort(np.linalg.norm(self.free_locations, axis=1))[:N_AGENTS]
        # start_loc = self.free_locations[start_loc_idx]
        start_loc = safe[choice]
        self.robot_locations = np.array(start_loc)

        robot_cells = get_cell_position_from_coords(self.robot_locations, self.belief_info)
        for robot_cell in robot_cells:
            self.update_safe_zone(robot_cell)

        self.old_safe_zone = deepcopy(self.safe_zone)
        self.safe_zone_frontiers = get_safe_zone_frontier(self.safe_info, self.belief_info)

        if self.plot:
            self.frame_files = []


    def import_ground_truth(self, episode_index):
        map_dir = 'maps_medium'
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
        robot_cell = np.array([robot_cell[1, 10], robot_cell[0, 10]])

        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1

        return ground_truth, robot_cell

    def update_robot_belief(self, robot_cell):
        self.robot_belief = exploration_sensor(robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief,
                                               self.ground_truth)

    def update_safe_zone(self, robot_cell):
        self.safe_zone = coverage_sensor(robot_cell, round(self.sensor_range / self.cell_size), self.safe_zone,
                                         self.ground_truth)

    def decrease_safety(self, cells_togo):
        cells_frontiers = get_cell_position_from_coords(self.safe_zone_frontiers, self.safe_info).reshape(-1, 2)
        for frontier in cells_frontiers:
            nearby_agent_indices = np.argwhere(np.linalg.norm(frontier - cells_togo, axis=1) < round(self.sensor_range / self.cell_size) + 1)
            nearby_agent_cells = cells_togo[nearby_agent_indices]
            uncovered = True

            for cell in nearby_agent_cells:
                if not check_collision(frontier, cell, self.belief_info):
                    uncovered = False
            if uncovered:
                sub_safe_zone = self.safe_zone[frontier[1] - self.safety_range: frontier[1] + self.safety_range + 1,
                                               frontier[0] - self.safety_range: frontier[0] + self.safety_range + 1]
                sub_belief = self.robot_belief[frontier[1] - self.safety_range: frontier[1] + self.safety_range + 1,
                                               frontier[0] - self.safety_range: frontier[0] + self.safety_range + 1]
                decrease_safety_by_frontier(self.safety_range, sub_safe_zone, sub_belief)

    def calculate_reward(self):
        reward = 0

        new_area = np.sum(self.safe_zone == 255) - np.sum(self.old_safe_zone == 255)
        reward += new_area / 1000

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
        next_cell = get_cell_position_from_coords(next_waypoint, self.belief_info)
        self.robot_locations[agent_id] = next_waypoint
        self.update_safe_zone(next_cell)
        self.safe_zone_frontiers = get_safe_zone_frontier(self.safe_info, self.belief_info)
        self.evaluate_safe_zone_rate()
