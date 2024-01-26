import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

from env import Env
from agent import Agent
from parameter import *
from utils import *
from model import PolicyNet
from local_node_manager_quadtree import NodeManager

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class Multi_agent_worker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image)
        self.n_agent = N_AGENTS
        self.node_manager = NodeManager(self.env.ground_truth_coords, self.env.ground_truth_info, explore=EXPLORATION, plot=self.save_image)

        self.robot_list = [Agent(i, policy_net, self.node_manager, self.device, self.save_image) for i in range(self.n_agent)]

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(24):
            self.episode_buffer.append([])

    def run_episode(self):
        done = False
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
        for robot in self.robot_list:
            robot.update_safe_graph(self.env.safe_info, self.env.uncovered_safe_frontiers)
        for robot in self.robot_list:
            robot.update_planning_state(self.env.robot_locations)
            robot.update_underlying_state()

        safe_increase_log = []
        max_travel_dist = 0
        for i in range(MAX_EPISODE_STEP):
            selected_locations = []
            dist_list = []
            next_node_index_list = []
            for robot in self.robot_list:
                local_observation = robot.get_local_observation()
                state = robot.get_state()
                robot.save_observation(local_observation)
                robot.save_state(state)

                next_location, next_node_index, action_index = robot.select_next_waypoint(local_observation)
                robot.save_action(action_index)

                node = robot.node_manager.local_nodes_dict.find((robot.location[0], robot.location[1]))
                check = np.array(node.data.neighbor_list)
                assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j, print(next_location,
                                                                                                         robot.location,
                                                                                                         node.data.neighbor_list)
                selected_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))
                next_node_index_list.append(next_node_index)

            selected_locations = np.array(selected_locations).reshape(-1, 2)
            arriving_sequence = np.argsort(np.array(dist_list))
            selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]

            for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
                solved_locations = selected_locations_in_arriving_sequence[:j]
                while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                    id = arriving_sequence[j]
                    nearby_nodes = self.robot_list[id].node_manager.local_nodes_dict.nearest_neighbors(selected_location.tolist(), 25)
                    for node in nearby_nodes:
                        coords = node.data.coords
                        if coords[0] + coords[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                            continue
                        selected_location = coords
                        break

                    selected_locations_in_arriving_sequence[j] = selected_location
                    selected_locations[id] = selected_location

            curr_node_indices = np.array([robot.current_local_index for robot in self.robot_list])

            self.env.decrease_safety(selected_locations)

            self.env.step(selected_locations)

            self.env.classify_safe_frontier(selected_locations)

            for robot in self.robot_list:
                robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
            for robot in self.robot_list:
                robot.update_safe_graph(self.env.safe_info, self.env.uncovered_safe_frontiers)

            done = self.env.check_done()

            reward_list, safety_increase_flag = self.env.calculate_reward()

            team_reward = - np.mean(dist_list) / 30

            max_travel_dist += np.max(dist_list)
            if safety_increase_flag > 0:
                safe_increase_log.append(1)
            else:
                safe_increase_log.append(0)

            if done:
                team_reward += 30

            for robot, reward in zip(self.robot_list, reward_list):
                robot.save_all_indices(np.array(curr_node_indices))
                robot.save_reward(reward + team_reward)
                robot.save_done(done)
                robot.update_planning_state(self.env.robot_locations)
                robot.update_underlying_state()

            if self.save_image:
                self.plot_local_env(i)

            if done:
                break

        # save metrics
        self.perf_metrics['travel_dist'] = max([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['max_travel_dist'] = max_travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['safe_rate'] = self.env.safe_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['safe_increase_rate'] = np.mean(safe_increase_log)

        # save episode buffer
        for robot in self.robot_list:
            local_observation = robot.get_local_observation()
            state = robot.get_state()
            robot.save_next_observations(local_observation, next_node_index_list)
            robot.save_next_state(state)

            for i in range(len(self.episode_buffer)):
                self.episode_buffer[i] += robot.episode_buffer[i]

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.safe_rate)

    def plot_local_env(self, step):
        plt.switch_backend('agg')
        plt.figure(figsize=(11, 5))
        plt.subplot(1, 2, 2)
        plt.imshow(self.env.robot_belief, cmap='gray', vmin=0)
        plt.axis('off')
        color_list = ['r', 'b', 'g', 'y']
        for robot in self.robot_list:
            c = color_list[robot.id]
            robot_cell = get_cell_position_from_coords(robot.location, robot.global_map_info)
            plt.plot(robot_cell[0], robot_cell[1], c+'o', markersize=13, zorder=5)
            plt.plot((np.array(robot.trajectory_x) - robot.global_map_info.map_origin_x) / robot.cell_size,
                     (np.array(robot.trajectory_y) - robot.global_map_info.map_origin_y) / robot.cell_size, c,
                     linewidth=2, zorder=3)
            # guidepost = robot.local_node_coords[np.where(robot.guidepost == 1)[0]]
            # guidepost_cell = get_cell_position_from_coords(guidepost, robot.global_map_info).reshape(-1, 2)
            # plt.scatter(guidepost_cell[:, 0], guidepost_cell[:, 1], c=c, marker='*', s=11, zorder=7)
            if robot.id == 0:
                nodes = get_cell_position_from_coords(robot.local_node_coords, robot.safe_zone_info)
                plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.explore_utility, zorder=2)

        if self.env.explore_frontiers.shape[0] != 0:
            explore_frontier_cells = get_cell_position_from_coords(self.env.explore_frontiers, self.env.belief_info).reshape(-1, 2)
            plt.scatter(explore_frontier_cells[:, 0], explore_frontier_cells[:, 1], c='b', s=1, zorder=6)

        plt.subplot(1, 2, 1)
        plt.imshow(self.env.robot_belief, cmap='gray')

        self.env.classify_safe_frontier(self.env.robot_locations)
        covered_safe_frontier_cells = get_cell_position_from_coords(self.env.covered_safe_frontiers, self.env.safe_info).reshape(-1, 2)
        uncovered_safe_frontier_cells = get_cell_position_from_coords(self.env.uncovered_safe_frontiers, self.env.safe_info).reshape(-1, 2)
        if covered_safe_frontier_cells.shape[0] != 0:
            plt.scatter(covered_safe_frontier_cells[:, 0], covered_safe_frontier_cells[:, 1], c='g', s=1, zorder=6)
        if uncovered_safe_frontier_cells.shape[0] != 0:
            plt.scatter(uncovered_safe_frontier_cells[:, 0], uncovered_safe_frontier_cells[:, 1], c='r', s=1, zorder=6)

        for robot in self.robot_list:
            c = color_list[robot.id]
            if robot.id == 0:
                nodes = get_cell_position_from_coords(robot.local_node_coords, robot.safe_zone_info)
                alpha_mask = robot.safe_zone_info.map / 255 / 3
                plt.imshow(robot.safe_zone_info.map, cmap='Greens', alpha=alpha_mask)
                plt.axis('off')
                plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.safe_utility, zorder=2)
                # for i, (x, y) in enumerate(nodes):
                #     plt.text(x, y, f"{robot.safe_utility[i]}", fontsize=5, ha='center', va='center')
                # signal = robot.local_node_coords[np.where(robot.signal == 1)[0]]
                # signal_cell = get_cell_position_from_coords(signal, robot.global_map_info).reshape(-1, 2)
                # plt.scatter(signal_cell[:, 0], signal_cell[:, 1], c='w', marker='.', s=2, zorder=3, alpha=0.5)

            robot_cell = get_cell_position_from_coords(robot.location, robot.safe_zone_info)
            plt.plot(robot_cell[0], robot_cell[1], c+'o', markersize=13, zorder=5)

        plt.axis('off')
        plt.suptitle('Explored ratio: {:.4g} | Safe ratio: {:.4g} | Travel distance: {:.4g}'.format(self.env.explored_rate,
                                                                                                self.env.safe_rate,
                                                                                                max([robot.travel_dist for robot in self.robot_list])))
        plt.tight_layout()
        # plt.show()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step), dpi=150)
        plt.close()
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)
        self.env.frame_files.append(frame)


if __name__ == '__main__':
    from parameter import *
    policy_net = PolicyNet(LOCAL_NODE_INPUT_DIM, EMBEDDING_DIM)
    ckp = torch.load('model/advsearch_10/checkpoint.pth', map_location='cpu')
    policy_net.load_state_dict(ckp['policy_model'])
    worker = Multi_agent_worker(0, policy_net, 0, 'cpu', True)
    worker.run_episode()
