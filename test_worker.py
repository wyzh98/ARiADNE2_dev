import matplotlib.pyplot as plt

from env import Env
from agent import Agent
from utils import *
from local_node_manager_quadtree import Local_node_manager
from expert_planner import Expert_planner
from ground_truth_planner import Ground_truth_planner
from test_parameter import *

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class TestWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False, greedy=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.greedy = greedy

        self.env = Env(global_step, n_agent=TEST_N_AGENTS, plot=self.save_image, test=True)
        self.local_node_manager = Local_node_manager(plot=self.save_image)

        self.robot_list = [Agent(i, policy_net, self.local_node_manager, self.device, self.save_image) for i in
                           range(self.env.n_agent)]

        self.perf_metrics = dict()

    def run_episode(self):
        done = False
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
        for robot in self.robot_list:
            robot.update_planning_state(self.env.robot_locations)

        if TEST_METHOD == 'tare':
            self.env.expert_planner = Expert_planner(self.local_node_manager)
        if TEST_METHOD == 'ground_truth':
            self.env.ground_truth_planner = Ground_truth_planner(self.env.ground_truth_info, self.local_node_manager)

        for i in range(MAX_EPISODE_STEP):
            selected_locations = []
            dist_list = []

            if TEST_METHOD == 'rl':
                for robot in self.robot_list:
                    local_observation = robot.get_no_padding_observation()
                    next_location, _, _ = robot.select_next_waypoint(local_observation, self.greedy)
                    selected_locations.append(next_location)
                    dist_list.append(np.linalg.norm(next_location - robot.location))

            if TEST_METHOD == 'tare':
                paths = self.env.get_expert_paths()
                for path in paths:
                    selected_locations.append(np.array(path[0]))
                dist_list = [k for k in range(self.env.n_agent)]

            if TEST_METHOD == 'ground_truth':
                paths = self.env.get_ground_truth_paths()
                for path in paths:
                    selected_locations.append(np.array(path[0]))
                dist_list = [k for k in range(self.env.n_agent)]

            selected_locations = np.array(selected_locations).reshape(-1, 2)
            arriving_sequence = np.argsort(np.array(dist_list))
            selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]

            for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
                solved_locations = selected_locations_in_arriving_sequence[:j]
                while selected_location[0] + selected_location[1] * 1j in solved_locations[:,
                                                                          0] + solved_locations[:, 1] * 1j:
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

            for robot, next_location in zip(self.robot_list, selected_locations):
                self.env.step(next_location, robot.id)

                robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))

            for robot in self.robot_list:
                robot.update_planning_state(self.env.robot_locations)

            if self.robot_list[0].utility.sum() == 0:
                done = True

            if self.save_image:
                self.plot_local_env(i)

            if done:
                break

        # save metrics
        self.perf_metrics['travel_dist'] = max([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['success_rate'] = done


        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def plot_local_env(self, step, planned_paths=None):
        self.env.global_frontiers = get_frontier_in_map(self.env.belief_info)
        plt.switch_backend('agg')
        color_list = ['r', 'b', 'g', 'y']
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 2)
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
            #   plt.plot((self.local_node_manager.x[i] - self.env.belief_info.map_origin_x) / self.env.cell_size,
            #            (self.local_node_manager.y[i] - self.env.belief_info.map_origin_y) / self.env.cell_size, 'tan', zorder=1)

        plt.subplot(1, 2, 1)
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
