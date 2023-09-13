import numpy as np
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
        self.node_manager = NodeManager(self.env.free_locations, plot=self.save_image)

        self.robot_list = [Agent(i, policy_net, self.node_manager, self.device, self.save_image) for i in
                           range(N_AGENTS)]

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(18):
            self.episode_buffer.append([])

    def run_episode(self):
        done = False
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, self.env.safe_info, deepcopy(self.env.robot_locations[robot.id]))
        for robot in self.robot_list:    
            robot.update_planning_state(self.env.robot_locations)

        safe_increase_log = []
        for i in range(MAX_EPISODE_STEP):
            selected_locations = []
            dist_list = []
            next_node_index_list = []
            for robot in self.robot_list:
                local_observation = robot.get_local_observation()
                robot.save_observation(local_observation)

                next_location, next_node_index, action_index = robot.select_next_waypoint(local_observation)
                robot.save_action(action_index)

                node = robot.node_manager.local_nodes_dict.find((robot.location[0], robot.location[1]))
                check = np.array(node.data.explored_neighbor_list)
                assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j, print(next_location,
                                                                                                         robot.location,
                                                                                                         node.data.explored_neighbor_list)

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

            reward_list = [0] * self.n_agent
            curr_node_indices = []
            next_node_real_indices = []
            # for robot in self.robot_list:
            #     num_dangerous_frontiers = robot.get_num_dangerous_frontiers(selected_locations)
            #     reward_list[robot.id] += -num_dangerous_frontiers / 100
            # old_safe_frontier = copy.deepcopy(self.env.safe_zone_frontiers)

            for robot, next_location in zip(self.robot_list, selected_locations):
                # dist = np.linalg.norm(next_location - robot.location)
                next_node_real_index = np.where((next_location == robot.local_node_coords).all(axis=1))
                next_node_real_indices.append(next_node_real_index)
                curr_node_indices.append(robot.current_local_index)

                self.env.step(next_location, robot.id)
                robot.update_graph(self.env.belief_info, self.env.safe_info, deepcopy(self.env.robot_locations[robot.id]))

            # for robot in self.robot_list:
            #     num_new_frontiers = robot.get_num_new_safe_frontiers(old_safe_frontier)
            #     reward_list[robot.id] += num_new_frontiers / 100

            self.env.decrease_safety(selected_locations)

            for robot, next_location in zip(self.robot_list, selected_locations):
                self.env.step(next_location, robot.id)
                robot.update_graph(self.env.belief_info, self.env.safe_info, deepcopy(self.env.robot_locations[robot.id]))

            if (self.robot_list[0].signal == 0).sum() == 0:  # no unsafe node
                done = True

            team_reward = self.env.calculate_reward() - 0.3

            if team_reward + 0.3 > 0:
                safe_increase_log.append(1)
            else:
                safe_increase_log.append(0)

            if done:
                team_reward += 30

            for robot, reward in zip(self.robot_list, reward_list):
                robot.save_all_indices(np.array(curr_node_indices), np.array(next_node_real_indices))
                robot.save_reward(reward + team_reward)
                robot.save_done(done)
                robot.update_planning_state(self.env.robot_locations)

            if self.save_image:
                self.plot_local_env(i)

            if done:
                if self.save_image:
                    self.plot_local_env(i + 1)
                break

        # save metrics
        self.perf_metrics['travel_dist'] = max([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['explored_rate'] = self.env.safe_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['safe_increase_rate'] = np.mean(safe_increase_log)

        # save episode buffer
        for robot in self.robot_list:
            local_observation = robot.get_local_observation()
            robot.save_next_observations(local_observation)
            for i in range(len(self.episode_buffer)):
                self.episode_buffer[i] += robot.episode_buffer[i]

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.safe_rate)

    def plot_local_env(self, step):
        plt.switch_backend('agg')
        plt.figure(figsize=(11, 5))
        plt.subplot(1, 2, 2)
        plt.imshow(self.env.robot_belief, cmap='gray', vmin=-255)
        plt.axis('off')
        color_list = ['r', 'b', 'g', 'y']
        frontiers = get_safe_zone_frontier(self.env.safe_info, self.env.belief_info)
        frontiers = get_cell_position_from_coords(frontiers, self.env.belief_info).reshape(-1, 2)
        for robot in self.robot_list:
            c = color_list[robot.id]
            robot_cell = get_cell_position_from_coords(robot.location, robot.global_map_info)
            plt.plot(robot_cell[0], robot_cell[1], c+'o', markersize=16, zorder=5)
            plt.plot((np.array(robot.trajectory_x) - robot.global_map_info.map_origin_x) / robot.cell_size,
                     (np.array(robot.trajectory_y) - robot.global_map_info.map_origin_y) / robot.cell_size, c,
                     linewidth=2, zorder=1)

        plt.subplot(1, 2, 1)
        plt.imshow(self.env.robot_belief, cmap='gray')
        plt.scatter(frontiers[:, 0], frontiers[:, 1], c='g', s=1, zorder=6)
        for robot in self.robot_list:
            c = color_list[robot.id]
            if robot.id == 0:
                nodes = get_cell_position_from_coords(robot.local_node_coords, robot.safe_zone_info)
                plt.imshow(robot.safe_zone_info.map, cmap='Greens', alpha=0.5)
                plt.axis('off')
                plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.utility, zorder=2)
                # guidepost = robot.local_node_coords[np.where(robot.guidepost == 1)[0]]
                # guidepost_cell = get_cell_position_from_coords(guidepost, robot.global_map_info).reshape(-1, 2)
                # plt.scatter(guidepost_cell[:, 0], guidepost_cell[:, 1], c=c, marker='*', s=10, zorder=7)
                signal = robot.local_node_coords[np.where(robot.signal == 1)[0]]
                signal_cell = get_cell_position_from_coords(signal, robot.global_map_info).reshape(-1, 2)
                plt.scatter(signal_cell[:, 0], signal_cell[:, 1], c='w', marker='.', s=2, zorder=3, alpha=0.5)

            robot_cell = get_cell_position_from_coords(robot.location, robot.safe_zone_info)
            plt.plot(robot_cell[0], robot_cell[1], c+'o', markersize=16, zorder=5)

        plt.axis('off')
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(self.env.safe_rate,
                                                                              max([robot.travel_dist for robot in
                                                                                   self.robot_list])))
        plt.tight_layout()
        # plt.show()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)
        self.env.frame_files.append(frame)


if __name__ == '__main__':
    from parameter import *
    policy_net = PolicyNet(LOCAL_NODE_INPUT_DIM, EMBEDDING_DIM)
    ckp = torch.load('model/checkpoint.pth', map_location='cpu')
    policy_net.load_state_dict(ckp['policy_model'])
    worker = Multi_agent_worker(0, policy_net, 0, 'cpu', False)
    worker.run_episode()
