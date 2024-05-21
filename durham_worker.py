import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import ray

from env import Env
from agent import Agent
from utils import *
from node_manager_quadtree import NodeManager
from sensor import coverage_sensor


TEST_N_AGENTS = 4
EXPLORATION = True
# GROUP_START: change in test_parameter.py
# MIN_UTILITY = 0  # !! change in test_parameter.py
RAY_META_AGENT = 10
NUM_TEST = 100

MAX_EPISODE_STEP = 128
SENSOR_RANGE = 20
CELL_SIZE = 0.4
SAVE_IMG = False
gifs_path = 'results/gifs'

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class DurhamWorker:
    def __init__(self, meta_agent_id, global_step, save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        np.random.seed(123)
        self.env = Env(self.global_step, n_agent=TEST_N_AGENTS, explore=EXPLORATION, plot=self.save_image, test=True)
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

    def assign_follower(self):
        for robot in self.robot_list:
            if self.agent_status[robot.id] == 0:  # follower
                min_dist = 1e6
                for status, guard_next_coords in zip(self.agent_status, self.agent_next_location):
                    if status == 1:  # frontier guard
                        path_coords, dist = self.node_manager.a_star(robot.location, guard_next_coords)
                        if dist < min_dist:
                            min_dist = dist
                            best_path = path_coords
                self.agent_next_location[robot.id] = best_path[0]

    def check_frontier_coverage(self, selected_locations):
        selected_locations = np.array(selected_locations).reshape(-1, 2)
        selected_frontier_uncoverage = np.zeros(selected_locations.shape[0])
        for frontier_loc in self.env.safe_zone_frontiers:
            nearby_selected_indices = np.argwhere(np.linalg.norm(selected_locations - frontier_loc, axis=1) < SENSOR_RANGE)
            nearby_selected_locations = selected_locations[nearby_selected_indices]
            uncovered = True
            for loc in nearby_selected_locations:
                if not check_collision(frontier_loc, loc, self.env.belief_info, max_collision=3):
                    uncovered = False
            if uncovered:
                nearest_index = np.argmin(np.linalg.norm(self.env.robot_locations - frontier_loc, axis=1))
                selected_frontier_uncoverage[nearest_index] += 1
        for i, loc in enumerate(selected_locations):
            if selected_frontier_uncoverage[i] > 1:
                selected_locations[i] = self.env.robot_locations[i]
        return selected_locations, selected_frontier_uncoverage

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
        max_increase = 2
        best_locations = []
        filtered_indices = []

        while True:
            for i, coords in enumerate(candidate_node_coords):
                if i in filtered_indices:
                    continue
                candidate_flag = True
                for location in best_locations:
                    dist_to_candidate = np.linalg.norm(location - coords)
                    if (dist_to_candidate < SENSOR_RANGE) and (not check_collision(location, coords, self.env.belief_info)):
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

            if len(best_locations) > 0 and np.all(best_coords == best_locations[-1]):
                return best_locations

            imaginary_safe_zone = best_imaginary_safe_zone
            best_locations.append(best_coords)
            max_increase = 2

    def run_episode(self):
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
        for robot in self.robot_list:
            robot.update_safe_graph(self.env.safe_info)
        for robot in self.robot_list:
            robot.update_planning_state(self.env.robot_locations)
        if self.save_image:
            self.plot_local_env(-1)

        max_travel_dist = 0
        for i in range(MAX_EPISODE_STEP):
            self.agent_status = [0] * self.env.n_agent
            best_locations = self.find_next_best_views()
            self.assign_frontier_guard(best_locations)
            self.assign_follower()

            next_locations = []
            dist_list = []
            for robot in self.robot_list:
                next_location = self.agent_next_location[robot.id]
                next_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))

            selected_locations, _ = self.check_frontier_coverage(next_locations)

            if NUM_TEST == 1:
                frontier_guards = [robot.id for robot in self.robot_list if self.agent_status[robot.id] == 1]
                followers = [robot.id for robot in self.robot_list if self.agent_status[robot.id] == 0]
                print(f"Step {i}\tFrontier guards: {frontier_guards}\tFollowers: {followers}")

            if self.save_image:
                self.plot_local_env(i, best_locations)

            selected_locations = self.solve_path_confict(selected_locations, dist_list)

            self.env.decrease_safety(selected_locations)
            # self.env.safe_zone_frontiers = get_safe_zone_frontier(self.env.safe_info, self.env.belief_info)

            self.env.step(selected_locations)

            for robot in self.robot_list:
                robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
            for robot in self.robot_list:
                robot.update_safe_graph(self.env.safe_info)
            for robot in self.robot_list:
                robot.update_planning_state(self.env.robot_locations)

            max_travel_dist += np.max(dist_list)

            done = self.env.check_done()

            if max_travel_dist >= 1000:
                max_travel_dist = 1000
                break

            if done:
                if self.save_image:
                    self.plot_local_env(i+1, best_locations)
                break

        if NUM_TEST == 1:
            print(f"{TEST_N_AGENTS} agents, max travelled distance: {max_travel_dist}, explored rate: {self.env.explored_rate}, cleared rate: {self.env.safe_rate}")

        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

        perf_metrics = [max_travel_dist, self.env.explored_rate, self.env.safe_rate, done]

        return perf_metrics


    def solve_path_confict(self, selected_locations, dist_list):
        selected_locations = np.array(selected_locations).reshape(-1, 2)
        arriving_sequence = np.argsort(np.array(dist_list))
        selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]
        for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
            solved_locations = selected_locations_in_arriving_sequence[:j]
            while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:,
                                                                                               1] * 1j:
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

        return selected_locations

    def plot_local_env(self, step, best_locations=None):
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

        safe_frontier_cells = get_cell_position_from_coords(self.env.safe_zone_frontiers, self.env.safe_info).reshape(-1, 2)
        if safe_frontier_cells.shape[0] != 0:
            plt.scatter(safe_frontier_cells[:, 0], safe_frontier_cells[:, 1], c='g', s=1, zorder=6)  # 0.4, 1

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

        if best_locations is not None:
            cells = get_cell_position_from_coords(np.array(best_locations), self.env.belief_info).reshape(-1, 2)
            for i, cell in enumerate(cells):
                plt.scatter(cell[0], cell[1], c='k', s=1, zorder=10)
                plt.text(cell[0], cell[1], str(i+1), fontsize=8, color='k', zorder=10)

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

@ray.remote(num_cpus=1, num_gpus=0)
class DurhamWorkerParallel:
    def __init__(self, meta_agent_id, save_image=False):
        self.meta_agent_id = meta_agent_id
        self.save_image = save_image

    def job(self, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        worker = DurhamWorker(self.meta_agent_id, episode_number, self.save_image)
        metrics = worker.run_episode()
        return metrics, self.meta_agent_id


if __name__ == '__main__':
    curr_test = 0
    max_dist_history = []
    explored_rate_history = []
    safe_rate_history = []
    success_rate_history = []

    if NUM_TEST == 1:
        worker = DurhamWorker(0, curr_test, SAVE_IMG)
        metrics = worker.run_episode()
        max_dist_history.append(metrics[0])
        explored_rate_history.append(metrics[1])
        safe_rate_history.append(metrics[2])
        success_rate_history.append(metrics[3])

    else:
        ray.init()
        meta_agents = [DurhamWorkerParallel.remote(i, SAVE_IMG) for i in range(RAY_META_AGENT)]
        run_list = []
        for meta_agent in meta_agents:
            run_list.append(meta_agent.job.remote(curr_test))
            curr_test += 1

        try:
            while len(max_dist_history) < curr_test:
                done_id, run_list = ray.wait(run_list)
                done_runs = ray.get(done_id)

                for res in done_runs:
                    metrics, meta_id = res
                    max_dist_history.append(metrics[0])
                    explored_rate_history.append(metrics[1])
                    safe_rate_history.append(metrics[2])
                    success_rate_history.append(metrics[3])

                    if curr_test < NUM_TEST:
                        run_list.append(meta_agents[meta_id].job.remote(curr_test))
                        curr_test += 1

        except KeyboardInterrupt:
            print("CTRL_C pressed. Killing remote workers")
            for a in meta_agents:
                ray.kill(a)

    print('=====================================')
    print('|#Test:', FOLDER_NAME)
    print('|#Total test:', NUM_TEST)
    print('|#Average max length:', np.array(max_dist_history).mean())
    print('|#Std max length:', np.array(max_dist_history).std())
    print('|#Average explored rate:', np.array(explored_rate_history).mean())
    print('|#Average safe rate:', np.array(safe_rate_history).mean())
    print('|#Average success rate:', np.array(success_rate_history).mean())
