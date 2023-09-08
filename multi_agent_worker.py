import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

from env import Env
from agent import Agent
from parameter import *
from utils import *
from local_node_manager_quadtree import Local_node_manager
from expert_planner import Expert_planner
from ground_truth_planner import Ground_truth_planner

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class Multi_agent_worker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image)
        self.local_node_manager = Local_node_manager(plot=self.save_image)

        self.robot_list = [Agent(i, policy_net, self.local_node_manager, self.device, self.save_image) for i in
                           range(self.env.n_agent)]

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(15):
            self.episode_buffer.append([])

    def run_episode(self):
        done = False
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
        for robot in self.robot_list:    
            robot.update_planning_state(self.env.robot_locations)

        if EXPERT == 'tare':
            self.env.expert_planner = Expert_planner(self.local_node_manager)
            paths = self.env.get_expert_paths()
        if EXPERT == 'ground_truth':
            self.env.ground_truth_planner = Ground_truth_planner(self.env.ground_truth_info, self.local_node_manager)
            paths = self.env.get_ground_truth_paths()
        expert_locations = []
        for path in paths:
            expert_locations.append(np.array(path[0]))
        for i in range(MAX_EPISODE_STEP):
            selected_locations = []
            dist_list = []
            next_node_index_list = []
            for robot in self.robot_list:
                local_observation = robot.get_local_observation()
                robot.save_observation(local_observation)

                next_location, next_node_index, action_index = robot.select_next_waypoint(local_observation)
                robot.save_action(action_index)

                node = robot.local_node_manager.local_nodes_dict.find((robot.location[0], robot.location[1]))
                check = np.array(node.data.neighbor_list)
                # assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j, print(next_location,
                                                                                                         robot.location,
                                                                                                         node.data.neighbor_list)
                # assert next_location[0] != robot.location[0] or next_location[1] != robot.location[1]

                selected_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))
                next_node_index_list.append(next_node_index)

            # selected_locations = []
            # for path in paths:
            #    if path:
            #        selected_locations.append(np.array(path[0]))

            reward_list = []
            for selected_location, expert_location in zip(selected_locations, expert_locations):
                if expert_location is not None:
                    reward = np.linalg.norm(selected_location - expert_location) / (4 * NODE_RESOLUTION * 1.41)
                    reward = np.round((-np.exp(reward) + np.exp(0)) / (np.exp(1) - np.exp(0)), 3)
                    #reward = -np.round(np.linalg.norm(selected_location - expert_location) / (4 * NODE_RESOLUTION * 1.41), 3)
                    reward_list.append(reward)
            # print(reward_list)

            selected_locations = np.array(selected_locations).reshape(-1, 2)
            arriving_sequence = np.argsort(np.array(dist_list))
            selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]

            for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
                solved_locations = selected_locations_in_arriving_sequence[:j]
                while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                    id = arriving_sequence[j]
                    nearby_nodes = self.robot_list[id].local_node_manager.local_nodes_dict.nearest_neighbors(
                        selected_location.tolist(), 25)
                    for node in nearby_nodes:
                        coords = node.data.coords
                        if coords[0] + coords[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                            continue
                        selected_location = coords
                        break

                    selected_locations_in_arriving_sequence[j] = selected_location
                    selected_locations[id] = selected_location

            # reward_list = []
            for robot, next_location, next_node_index in zip(self.robot_list, selected_locations, next_node_index_list):
                self.env.step(next_location, robot.id)
                # individual_reward = robot.utility[next_node_index] / 50
                # reward_list.append(individual_reward)

                robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))

            for robot in self.robot_list:
                robot.update_planning_state(self.env.robot_locations)

            # if self.save_image:
            #     self.plot_local_env(i, paths)

            if self.robot_list[0].utility.sum() == 0:
                done = True

            exception = False
            if not done:
                if EXPERT == 'tare':
                    paths = self.env.get_expert_paths()
                if EXPERT == 'ground_truth':
                    paths = self.env.get_ground_truth_paths()

                if self.save_image:
                    self.plot_local_env(i, paths)
                expert_locations = []
                for path in paths:
                    if path == []:
                        exception = True
                        print(self.robot_list[0].utility.sum())
                        print(self.env.ground_truth_planner.ground_truth_node_manager.utility.sum())
                        break
                    expert_locations.append(np.array(path[0]))

            # team_reward = self.env.calculate_reward() - 0.5
            # if done:
            #     team_reward += 10

            for robot, reward in zip(self.robot_list, reward_list):
                robot.save_reward(reward)
                robot.save_done(done)

            if done:
                if self.save_image:
                    self.plot_local_env(i + 1)
                break

            if exception:
                break

        # save metrics
        self.perf_metrics['travel_dist'] = max([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        # save episode buffer
        for robot in self.robot_list:
            local_observation = robot.get_local_observation()
            robot.save_next_observations(local_observation)
            for i in range(len(self.episode_buffer)):
                self.episode_buffer[i] += robot.episode_buffer[i]

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def plot_local_env(self, step, planned_paths=None):
        self.env.global_frontiers = get_frontier_in_map(self.env.belief_info)
        plt.switch_backend('agg')
        color_list = ['r', 'b', 'g', 'y']
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 3)
        plt.imshow(self.env.ground_truth_info.map, cmap='gray')
        # nodes = get_cell_position_from_coords(self.env.ground_truth_planner.ground_truth_node_manager.ground_truth_node_coords, self.env.ground_truth_info)
        plt.axis('off')
        # plt.scatter(nodes[:, 0], nodes[:, 1], c=self.env.ground_truth_planner.ground_truth_node_manager.utility, zorder=2)
        # frontiers = get_cell_position_from_coords(self.env.ground_truth_planner.ground_truth_node_manager.ground_truth_frontiers, self.env.belief_info)
        # frontiers = frontiers.reshape(-1, 2)
        # plt.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=1, zorder=10)

        if planned_paths:
            robot = self.robot_list[0]
            for i, path in enumerate(planned_paths):
                if path != []:
                    c = color_list[i]
                    plt.plot((np.array(path)[:, 0] - robot.global_map_info.map_origin_x) / robot.cell_size,
                            (np.array(path)[:, 1] - robot.global_map_info.map_origin_y) / robot.cell_size, c,
                            linewidth=2, zorder=1)
                    robot = self.robot_list[i]
                    robot_cell = get_cell_position_from_coords(robot.location, robot.global_map_info)
                    plt.plot(robot_cell[0], robot_cell[1], c + 'o', markersize=16, zorder=5)

        plt.subplot(1, 3, 2)
        plt.imshow(self.env.robot_belief, cmap='gray')
        plt.axis('off')
        frontiers = get_cell_position_from_coords(self.env.global_frontiers, self.env.belief_info)
        frontiers = frontiers.reshape(-1, 2)
        plt.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=1)
        for robot in self.robot_list:
            c = color_list[robot.id]
            robot_cell = get_cell_position_from_coords(robot.location, robot.global_map_info)
            plt.plot(robot_cell[0], robot_cell[1], c+'o', markersize=16, zorder=5)
            plt.plot((np.array(robot.trajectory_x) - robot.global_map_info.map_origin_x) / robot.cell_size,
                     (np.array(robot.trajectory_y) - robot.global_map_info.map_origin_y) / robot.cell_size, c,
                     linewidth=2, zorder=1)
            # for i in range(len(self.local_node_manager.x)):
            #   plt.plot((self.local_node_manager.x[i] - self.local_map_info.map_origin_x) / self.cell_size,
            #            (self.local_node_manager.y[i] - self.local_map_info.map_origin_y) / self.cell_size, 'tan', zorder=1)

        plt.subplot(1, 3, 1)
        plt.imshow(self.env.robot_belief, cmap='gray')
        frontiers = get_cell_position_from_coords(self.env.global_frontiers, self.env.belief_info)
        frontiers = frontiers.reshape(-1, 2)
        plt.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=1)
        for robot in self.robot_list:
            c = color_list[robot.id]
            if robot.id == 0:
                nodes = get_cell_position_from_coords(robot.local_node_coords, robot.global_map_info)
                plt.imshow(robot.global_map_info.map, cmap='gray')
                plt.axis('off')
                plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.utility, zorder=2)

            robot_cell = get_cell_position_from_coords(robot.location, robot.global_map_info)
            plt.plot(robot_cell[0], robot_cell[1], c+'o', markersize=16, zorder=5)

        plt.axis('off')
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(self.env.explored_rate,
                                                                              max([robot.travel_dist for robot in
                                                                                   self.robot_list])))
        plt.tight_layout()
        # plt.show()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)
        self.env.frame_files.append(frame)


if __name__ == '__main__':
    from model import PolicyNet
    policy_net = PolicyNet(LOCAL_NODE_INPUT_DIM, EMBEDDING_DIM)
    worker = Multi_agent_worker(0, policy_net, 0, 'cpu', True)
    worker.run_episode()
