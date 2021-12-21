import numpy as np
from contextual_environments.Env import Environment
from numba import jit
import matplotlib.pyplot as plt


class PlanarRobotObstacle_2dim(Environment):
    def __init__(self, action_dim, ctxt_dim, target_x, target_y, n_obstacles, obstacles_x, obstacles_y,
                 dist_reward_punishment_fac,
                 obstacle_reward_punishment_fac, action_regularizer, ctxt_out_of_range_punishment, context_range_bounds):
        super().__init__(action_dim, ctxt_dim)
        self.target_x = target_x
        self.target_y = target_y
        self.n_obstacles = n_obstacles
        self.obstacles_x = obstacles_x
        self.obstacles_y = obstacles_y
        """
        only rectangles whcih are not turned, i.e. no parallelograms are supported
        #[[x_11, x_12, x_21, x_22], [x_11, x_12, x_21, x_22]], e.g. 2 rectangles  -> self.obstacles_x
        #[[y_11, y_12, y_21, y_22], [y_11, y_12, y_21, y_22]]. e.g. 2 rectangels  -> self.obstacles_y  
        #   x_11: left bottom x-value; x_21: left upper x-value; x_21: right bottom x-value; x_22: right upper x-value
        #   y_11: left bottom y-value; y_21: left upper y-value; y_21: right bottom y-value; y_22: right upper y-value
        """
        assert len(
            self.obstacles_x) == self.n_obstacles, "The number of obstacles and their x-Value specifications are " \
                                                   "not consistent. You have to specify n_obstacles lists with " \
                                                   "4 x-Values for every obstacle."
        assert len(
            self.obstacles_y) == self.n_obstacles, "The number of obstacles and their y-Value specifications are " \
                                                   "not consistent. You have to specify n_obstacles lists with " \
                                                   "4 y-Values for every obstacle."
        self.obs_a_points, self.obs_b_points = self._get_line_equations_obstacles()
        self.dist_reward_punishment_fac = dist_reward_punishment_fac
        self.obstacle_reward_punishment_fac = obstacle_reward_punishment_fac
        self.action_regularizer = action_regularizer
        self.ctxt_out_of_range_punishment = ctxt_out_of_range_punishment
        self.context_range_bounds = context_range_bounds
        self.states = None

    def angle_normalize(self, x):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)

    def _fwd_kin(self, actions):
        angles_ = np.cumsum(actions, axis=1)
        pos0 = np.cos(angles_)
        pos1 = np.sin(angles_)
        return pos0, pos1

    def _eval(self, actions):
        return self._fwd_kin(actions)

    def _get_line_equations_obstacles(self):
        a_points = []
        b_points = []

        for i in range(self.n_obstacles):
            cur_obstacle_a = []
            x_points = self.obstacles_x[i]
            y_points = self.obstacles_y[i]

            a_vert_left = np.array([x_points[0], y_points[0]])  # starting point of equation -> a
            a_hor_bottom = np.array([x_points[0], y_points[0]])
            a_hor_upper = np.array([x_points[1], y_points[1]])
            a_vert_right = np.array([x_points[2], y_points[2]])
            cur_obstacle_a.append(a_vert_left)
            cur_obstacle_a.append(a_hor_bottom)
            cur_obstacle_a.append(a_hor_upper)
            cur_obstacle_a.append(a_vert_right)

            cur_obstacle_b = []
            b_vert_left = a_hor_upper - a_vert_left  # difference of points -> b
            b_hor_bottom = a_vert_right - a_hor_bottom
            b_hor_upper = np.array([x_points[3], y_points[3]]) - a_hor_upper
            b_vert_right = np.array([x_points[3], y_points[3]]) - a_vert_right
            cur_obstacle_b.append(b_vert_left)
            cur_obstacle_b.append(b_hor_bottom)
            cur_obstacle_b.append(b_hor_upper)
            cur_obstacle_b.append(b_vert_right)

            a_points.append(cur_obstacle_a)
            b_points.append(cur_obstacle_b)
        return np.array(a_points), np.array(b_points)

    @staticmethod
    @jit(nopython=True)
    def _check_line_crossings(a, b, n_obstacles, obs_a_points, obs_b_points):
        for k in range(n_obstacles):
            # a_vert_left = obs_a_points[k][0]
            a_vert_left = obs_a_points[k, 0]
            # a_hor_bottom = obs_a_points[k][1]
            a_hor_bottom = obs_a_points[k, 1]
            # a_hor_upper = obs_a_points[k][2]
            a_hor_upper = obs_a_points[k, 2]
            # a_vert_right = obs_a_points[k][3]
            a_vert_right = obs_a_points[k, 3]

            # b_vert_left = obs_b_points[k][0]
            b_vert_left = obs_b_points[k, 0]
            # b_hor_bottom = obs_b_points[k][1]
            b_hor_bottom = obs_b_points[k, 1]
            # b_hor_upper = obs_b_points[k][2]
            b_hor_upper = obs_b_points[k, 2]
            # b_vert_right = obs_b_points[k][3]
            b_vert_right = obs_b_points[k, 3]

            # check collision with vertical left line
            mu = (a_vert_left[0] - a[0]) / (b[0] + 1e-15)
            lamb = (a[1] - a_vert_left[1] + mu * b[1]) / (b_vert_left[1] + 1e-15)
            if 1 >= mu >= 0 and 1 >= lamb >= 0:  # condition for detection
                return 1

            # check collision with vertical right line
            mu = (a_vert_right[0] - a[0]) / (b[0] + 1e-15)
            lamb = (a[1] - a_vert_right[1] + mu * b[1]) / (b_vert_right[1] + 1e-15)
            if 1 >= mu >= 0 and 1 >= lamb >= 0:  # condition for detection
                return 1

            # check collision with horizontal bottom line
            mu = (a_hor_bottom[1] - a[1]) / (b[1] + 1e-15)
            lamb = (a[0] - a_hor_bottom[0] + mu * b[0]) / (b_hor_bottom[0] + 1e-15)
            if 1 >= mu >= 0 and 1 >= lamb >= 0:  # condition for detection
                return 1

            # check collision with horizontal upper line
            mu = (a_hor_upper[1] - a[1]) / (b[1] + 1e-15)
            lamb = (a[0] - a_hor_upper[0] + mu * b[0]) / (b_hor_upper[0] + 1e-15)
            if 1 >= mu >= 0 and 1 >= lamb >= 0:  # condition for detection
                return 1
        return 0

    def prepare_for_check_collision(self, pos_x_in, pos_y_in):

        pos_x = np.cumsum(pos_x_in, axis=1)
        pos_y = np.cumsum(pos_y_in, axis=1)

        all_points_x = np.column_stack((np.zeros(pos_x.shape[0]), pos_x))
        all_points_y = np.column_stack((np.zeros(pos_y.shape[0]), pos_y))

        all_diff_x = np.diff(all_points_x, axis=1)
        all_diff_y = np.diff(all_points_y, axis=1)

        return pos_x, pos_y, all_points_x, all_points_y, all_diff_x, all_diff_y


    @staticmethod
    @jit(nopython=True)
    def _check_collision(all_points_x, all_points_y, all_diff_x, all_diff_y, n_obstacles, obs_a_points,  obs_b_points):
    # def _check_collision(pos_x_in, pos_y_in, check_line_crossings):

        collision_detections = np.zeros(all_points_x.shape[0])

        # first get the difference points of each link (for creating the line equation)

        for i in range(all_points_x.shape[0]):
            # build up line equation:
            for j in range(all_points_x.shape[1] - 1):
                a = np.array([all_points_x[i, j], all_points_y[i, j]])
                b = np.array([all_diff_x[i, j], all_diff_y[i, j]])
                col_detected = False
                for k in range(n_obstacles):
                    # a_vert_left = obs_a_points[k][0]
                    a_vert_left = obs_a_points[k, 0]
                    # a_hor_bottom = obs_a_points[k][1]
                    a_hor_bottom = obs_a_points[k, 1]
                    # a_hor_upper = obs_a_points[k][2]
                    a_hor_upper = obs_a_points[k, 2]
                    # a_vert_right = obs_a_points[k][3]
                    a_vert_right = obs_a_points[k, 3]

                    # b_vert_left = obs_b_points[k][0]
                    b_vert_left = obs_b_points[k, 0]
                    # b_hor_bottom = obs_b_points[k][1]
                    b_hor_bottom = obs_b_points[k, 1]
                    # b_hor_upper = obs_b_points[k][2]
                    b_hor_upper = obs_b_points[k, 2]
                    # b_vert_right = obs_b_points[k][3]
                    b_vert_right = obs_b_points[k, 3]

                    # check collision with vertical left line
                    mu = (a_vert_left[0] - a[0]) / (b[0] + 1e-15)
                    lamb = (a[1] - a_vert_left[1] + mu * b[1]) / (b_vert_left[1] + 1e-15)
                    if 1 >= mu >= 0 and 1 >= lamb >= 0:  # condition for detection
                        col_detected = True

                    # check collision with vertical right line
                    mu = (a_vert_right[0] - a[0]) / (b[0] + 1e-15)
                    lamb = (a[1] - a_vert_right[1] + mu * b[1]) / (b_vert_right[1] + 1e-15)
                    if 1 >= mu >= 0 and 1 >= lamb >= 0:  # condition for detection
                        col_detected = True

                    # check collision with horizontal bottom line
                    mu = (a_hor_bottom[1] - a[1]) / (b[1] + 1e-15)
                    lamb = (a[0] - a_hor_bottom[0] + mu * b[0]) / (b_hor_bottom[0] + 1e-15)
                    if 1 >= mu >= 0 and 1 >= lamb >= 0:  # condition for detection
                        col_detected = True

                    # check collision with horizontal upper line
                    mu = (a_hor_upper[1] - a[1]) / (b[1] + 1e-15)
                    lamb = (a[0] - a_hor_upper[0] + mu * b[0]) / (b_hor_upper[0] + 1e-15)
                    if 1 >= mu >= 0 and 1 >= lamb >= 0:  # condition for detection
                        col_detected = True

                    if col_detected == True:
                        break
                if col_detected:
                    collision_detections[i] = 1
                    break

        return collision_detections

    def _reward(self, actions, contexts):
        pos_x, pos_y = self._eval(actions)
        end_eff_x = np.sum(pos_x, axis=1)
        end_eff_y = np.sum(pos_y, axis=1)
        # collisions_detected = self._check_collision(pos_x, pos_y)
        pos_x, pos_y, all_points_x, all_points_y, all_diff_x, all_diff_y = self.prepare_for_check_collision(pos_x, pos_y)

        collisions_detected = PlanarRobotObstacle_2dim._check_collision(all_points_x, all_points_y, all_diff_x,
                                                                        all_diff_y, self.n_obstacles, self.obs_a_points,
                                                                        self.obs_b_points)
        target_x = contexts[:, 0]
        target_y = contexts[:, 1]

        points_target = np.column_stack((target_x, target_y))
        points = np.column_stack((end_eff_x, end_eff_y))
        dif = points_target - points

        # prepare punishment for sampling contexts out of the context range (only important if we sample locally)
        idx_max, idx_min, reordered_ctxts, quad_dists_to_val_region = self.check_where_invalid(contexts.copy(), context_range_bounds=self.context_range_bounds,
                                                   set_to_valid_region=False)

        # when using L2 norm on punishing outliers
        # diff_ctxts = reordered_ctxts - contexts
        # ctxt_out_of_range_reward = -self.ctxt_out_of_range_punishment*np.linalg.norm(diff_ctxts, axis=1) ** 2

        # when using dirac punishments
        ctxt_out_of_range_reward = np.zeros(contexts.shape[0])
        for c_dim in range(len(idx_min)):
            ctxt_out_of_range_reward[idx_max[c_dim]] = -self.ctxt_out_of_range_punishment*1
            ctxt_out_of_range_reward[idx_min[c_dim]] = -self.ctxt_out_of_range_punishment*1

        ctxt_out_of_range_reward -= 0 * quad_dists_to_val_region

        normalized_actions = self.angle_normalize(actions)
        smoothnes_reward = - self.action_regularizer * np.linalg.norm(normalized_actions[:, 1:], axis=1) ** 2
        distance_reward = - self.dist_reward_punishment_fac * np.linalg.norm(dif, axis=1) ** 2
        rewards = distance_reward + smoothnes_reward + ctxt_out_of_range_reward
                  # - self.action_regularizer*np.linalg.norm(actions[:, 1:], axis=1)**2
        collisions_detected *= -self.obstacle_reward_punishment_fac
        collision_reward = collisions_detected
        total_rewards = collision_reward + rewards
        return total_rewards, points, smoothnes_reward, distance_reward, collision_reward

    def step(self, actions, contexts):
        """"
        This function will record internally states. If this is not whished, use eval function.
        """
        rewards, self.states, smoothnes_reward, distance_reward, collision_reward = self._reward(actions, contexts)
        return rewards, self.states, smoothnes_reward, distance_reward, collision_reward

    def check_if_contexts_in_obstacles(self, contexts):

        in_obstacle = np.zeros((contexts.shape[0]))

        for k in range(self.n_obstacles):
            current_x = self.obstacles_x[k]
            candidates_left_vert = np.where((contexts[:, 0] >= current_x[0] - 0.3))[0]       # check x values for left line of rectangle
            candidates_right_vert = np.where((contexts[:, 0] <= current_x[2] + 0.3))[0]

            current_y = self.obstacles_y[k]
            candidates_bottom = np.where((contexts[:, 1] >= current_y[0] - 0.3))[0]
            candidates_upper = np.where((contexts[:, 1] <= current_y[1] + 0.3))[0]

            elements_in_obstacles_vert = np.isin(candidates_left_vert, candidates_right_vert)    # check if indices of lef in right
            elements_in_obstacles_vert = candidates_left_vert[elements_in_obstacles_vert]             # extract from mask

            elements_in_obstacles_hor = np.isin(candidates_bottom, candidates_upper)            # check if indices of bottom in upper
            elements_in_obstacles_hor = candidates_bottom[elements_in_obstacles_hor]        # extract from mask

            elements_in_obstacle = np.isin(elements_in_obstacles_vert, elements_in_obstacles_hor)   # now check if they are in obstacle
            elements_in_obstacle = elements_in_obstacles_vert[elements_in_obstacle]
            in_obstacle[elements_in_obstacle] = 1

        in_obstacle_indices = np.where((in_obstacle == 1))[0]
        return in_obstacle, in_obstacle_indices

    def visualize(self, actions, config, fig, cond_gating, contexts, all_contexts, show=True, color=None,  sparsity_thresh=None):


        if fig is None:
            fig_was_none = True
            fig = plt.figure()
        else:
            fig_was_none = False
        # prepare everything
        if cond_gating is not None:
            if sparsity_thresh is None:
                sparsity_thresh = config.action_neglect_importance_weight_thresh

        pos_x_in, pos_y_in = self._eval(actions)
        starting_point_x = 0
        starting_point_y = 0
        pos_x = np.cumsum(pos_x_in, axis=1)
        pos_y = np.cumsum(pos_y_in, axis=1)
        all_points_x = np.column_stack((np.ones(pos_x.shape[0]) * starting_point_x, pos_x + starting_point_x))
        all_points_y = np.column_stack((np.ones(pos_y.shape[0]) * starting_point_y, pos_y + starting_point_y))

        if color is not None:
            use_color = color
        else:
            use_color = 'black'
        plt.figure(fig.number)
        for i in range(all_points_x.shape[0]):
            if cond_gating is not None:
                # if cond_gating[i] > sparsity_thresh:
                if cond_gating > sparsity_thresh:
                    plt.plot(all_points_x[i, :], all_points_y[i, :], 'o', color= use_color, alpha=0.5)
                    plt.plot(all_points_x[i, :], all_points_y[i, :], '-', color=use_color, linewidth=2, alpha=0.1)
                    plt.plot(all_points_x[i, -1], all_points_y[i, -1], 'go',  linewidth=2, alpha=0.8)
            else:
                plt.plot(all_points_x[i, :], all_points_y[i, :], 'o',color=use_color, alpha=0.5)
                plt.plot(all_points_x[i, :], all_points_y[i, :], '-', color=use_color, linewidth=2, alpha=0.8)
                plt.plot(all_points_x[i, -1], all_points_y[i, -1], 'go',  linewidth=2, alpha=0.8)

        if all_contexts is not None:
            plt.plot(all_contexts[:, 0], all_contexts[:, 1], 'ro', alpha=1)
        if contexts is not None:
            plt.plot(contexts[:, 0], contexts[:, 1], 'ro',  alpha=1)

        if fig_was_none:
            self.visualize_env(fig, show)

        return fig

    def visualize_env(self, fig, show):

        plt.figure(fig.number)
        for k in range(self.n_obstacles):
            obstacle_x = self.obstacles_x[k]
            obstacle_y = self.obstacles_y[k]

            l1_x = [obstacle_x[0], obstacle_x[1]]
            l1_y = [obstacle_y[0], obstacle_y[1]]

            l2_x = [obstacle_x[1], obstacle_x[3]]
            l2_y = [obstacle_y[1], obstacle_y[3]]

            l3_x = [obstacle_x[0], obstacle_x[2]]
            l3_y = [obstacle_y[0], obstacle_y[2]]

            l4_x = [obstacle_x[2], obstacle_x[3]]
            l4_y = [obstacle_y[2], obstacle_y[3]]
            plt.plot(l1_x, l1_y, 'r-', linewidth=2)
            plt.plot(l2_x, l2_y, 'r-', linewidth=2)
            plt.plot(l3_x, l3_y, 'r-', linewidth=2)
            plt.plot(l4_x, l4_y, 'r-', linewidth=2)
        if show:
            plt.show()

    def visualize_contexts_in_obstacle(self, contexts, in_obstacle_contexts):

        import matplotlib.pyplot as plt
        fig = plt.figure()
        for i in range(contexts.shape[0]):
            c_ctxt = contexts[i]
            if in_obstacle_contexts[i] == 1:
                c = 'c'
            else:
                c = 'r'
            plt.plot(c_ctxt[0], c_ctxt[1], c+'o')

        self.visualize_env(fig, show=True)


    def check_collision_visualizer(self, actions, collisions, contexts=None, means=None):

        pos_x_in, pos_y_in = self._eval(actions)
        starting_point_x = 0
        starting_point_y = 0
        if means is not None:
            means_x_in, means_y_in = self._eval(means)
            pos_x_mean = np.cumsum(means_x_in, axis=1)
            pos_y_mean = np.cumsum(means_y_in, axis=1)
            all_means_x = np.column_stack((np.ones(pos_x_mean.shape[0]) * starting_point_x, pos_x_mean + starting_point_x))
            all_means_y = np.column_stack((np.ones(pos_y_mean.shape[0]) * starting_point_y, pos_y_mean + starting_point_y))

        pos_x = np.cumsum(pos_x_in, axis=1)
        pos_y = np.cumsum(pos_y_in, axis=1)
        all_points_x = np.column_stack((np.ones(pos_x.shape[0]) * starting_point_x, pos_x + starting_point_x))
        all_points_y = np.column_stack((np.ones(pos_y.shape[0]) * starting_point_y, pos_y + starting_point_y))

        fig = plt.figure()
        for i in range(all_points_x.shape[0]):
            if collisions[i] != 1:
                if i == 0:
                    plt.plot(all_points_x[i, :], all_points_y[i, :], 'bo')
                    plt.plot(all_points_x[i, :], all_points_y[i, :], 'g-', linewidth=2)
                    plt.plot(all_points_x[i, -1], all_points_y[i, -1], 'go', linewidth=2)
                else:
                    plt.plot(all_points_x[i, :], all_points_y[i, :], 'bo')
                    plt.plot(all_points_x[i, :], all_points_y[i, :], 'k-', linewidth=2)
                    plt.plot(all_points_x[i, -1], all_points_y[i, -1], 'go', linewidth=2)
            else:
                plt.plot(all_points_x[i, :], all_points_y[i, :], 'co')
                plt.plot(all_points_x[i, :], all_points_y[i, :], 'c-', linewidth=2)
                plt.plot(all_points_x[i, -1], all_points_y[i, -1], 'co', linewidth=2)

        if means is not None:
            for k in range(all_means_x.shape[0]):
                plt.plot(all_means_x[k, :], all_means_y[k, :], 'bo')
                plt.plot(all_means_x[k, :], all_means_y[k, :], 'r-', linewidth=2)
                plt.plot(all_means_x[k, -1], all_means_y[k, -1], 'ro', linewidth=2)

        self.visualize_env(fig, show=True)

        if contexts is not None:
            plt.plot(contexts[:, 0], contexts[:, 1], 'ro')

    @staticmethod
    def create_env():
        action_dim = 10  # number of links

        ctxt_dim = 2
        target_x = [4.5, 7]
        target_y = [-6, 6]
        n_obstacles = 3

        # obstacles
        obstacles_x = [[2.5, 2.5, 3.3, 3.3], [2.7, 2.7, 3.5, 3.5], [3.0, 3.0, 3.8, 3.8]]
        obstacles_y = [[-0.5, 0.5, -0.5, 0.5], [-3.5, -2.5, -3.5, -2.5], [2.5, 3.5, 2.5, 3.5]]

        dist_reward_punishment_fac = 2
        obstacle_reward_punishment_fac = 3
        action_regularizer = 1
        ctxt_out_of_range_punishment = 10

        context_range_x = target_x
        context_range_y = target_y
        context_range = [context_range_x, context_range_y]
        max_ctxt_coords = np.array([target_x[1], target_y[1]])
        min_ctxts_coords = np.array([target_x[0], target_y[0]])
        context_range_bounds = [min_ctxts_coords, max_ctxt_coords]

        env = PlanarRobotObstacle_2dim(action_dim=action_dim, ctxt_dim=ctxt_dim, target_x=target_x, target_y=target_y,
                                       n_obstacles=n_obstacles, obstacles_x=obstacles_x, obstacles_y=obstacles_y,
                                       dist_reward_punishment_fac=dist_reward_punishment_fac,
                                       obstacle_reward_punishment_fac=obstacle_reward_punishment_fac,
                                       action_regularizer=action_regularizer,
                                       ctxt_out_of_range_punishment=ctxt_out_of_range_punishment,
                                       context_range_bounds=context_range_bounds)
        return env