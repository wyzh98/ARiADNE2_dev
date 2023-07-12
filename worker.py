from time import time

import numpy as np
import torch

from env import Env
from agent import Agent
from parameter import *
from utils import *
from model import PolicyNet

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class Worker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image)
        self.robot = Agent(policy_net, self.device, self.save_image)

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(15):
            self.episode_buffer.append([])

    def run_episode(self):
        done = False
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        local_observation = self.robot.get_local_observation()
        for i in range(MAX_EPISODE_STEP):
            self.save_observation(local_observation)

            next_location, action_index = self.robot.select_next_waypoint(local_observation)
            self.save_action(action_index)

            if self.save_image:
                self.robot.plot_local_env()
                self.env.plot_env(i)

            node = self.robot.local_node_manager.local_nodes_dict.find((self.robot.location[0], self.robot.location[1]))
            check = np.array(node.data.neighbor_list)
            assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j, print(next_location, self.robot.location, node.data.neighbor_list)
            assert next_location[0] != self.robot.location[0] or next_location[1] != self.robot.location[1]

            reward, _ = self.env.step(next_location)

            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
            if self.robot.utility.sum() == 0:
                done = True
                reward += 20
            self.save_reward_done(reward, done)

            local_observation = self.robot.get_local_observation()
            self.save_next_observations(local_observation)

            if done:
                if self.save_image:
                    self.robot.plot_local_env()
                    self.env.plot_env(i)
                break

        # save metrics
        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def save_observation(self, local_observation):
        local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask = local_observation
        self.episode_buffer[0] += local_node_inputs
        self.episode_buffer[1] += local_node_padding_mask
        self.episode_buffer[2] += local_edge_mask
        self.episode_buffer[3] += current_local_index
        self.episode_buffer[4] += current_local_edge
        self.episode_buffer[5] += local_edge_padding_mask

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.reshape(1, 1, 1)

    def save_reward_done(self, reward, done):
        self.episode_buffer[7] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)
        self.episode_buffer[8] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)

    def save_next_observations(self, local_observation):
        local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask = local_observation
        self.episode_buffer[9] += local_node_inputs
        self.episode_buffer[10] += local_node_padding_mask
        self.episode_buffer[11] += local_edge_mask
        self.episode_buffer[12] += current_local_index
        self.episode_buffer[13] += current_local_edge
        self.episode_buffer[14] += local_edge_padding_mask


if __name__ == "__main__":
    model = PolicyNet(LOCAL_NODE_INPUT_DIM, EMBEDDING_DIM)
    worker = Worker(0, model, 0, save_image=True)
    worker.run_episode()
