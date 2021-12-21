import numpy as np


class HierarchicalRewardTableTennis(object):
    """Class for hierarchical reward function for table tennis experiment.

    Return Highest Reward.
    Reward = 0

    Step 0 Context Range: [-∞, 0]
                    if valid:
                        self.temp_reward += 2
                    else:
                        self.temp_reward += 0
                        break

    Step 1 action_durations_valid_upper_lower_bound_valid: [-∞, 0]
                Upper Bound 0
                If hit_lower_bound < hit_duration <hit_lower_bound:
                    Reward = 0
                    continue
                else:
                    Reward += -1 * |hit_lower_bound -  hit_duration| * |hit_duration < hit_lower_bound| * 3
                    Reward += -1 * |hit_duration - hit_upper_bound| * |hit_upper_bound< hit_duration| * 3
                    break

    Step 2 joint_motion_mean_minus_penalty: [-∞, 0]
                motion_mean_reward = -1 * (error_high_mean + error_low_mean)
                if motion_mean_reward<=:
                    reward += motion_mean_reward
                    break
                else:
                    continue

    Step 3: hitting. [0, 2]
                if hitting:
                    [0, 2]
                    Reward = 2 * (1 - tanh(|shortest_hitting_dist|))
                    continue
                if not hitting:
                    [0, 0.2]
                    Reward = 0.2 * (1 - tanh(|shortest_hitting_dist|))
                    break
    Step 4: target_achievement: [2, 7]
                if table_contact_detector:
                    Reward += 1
                    Reward += (1 - tanh(|shortest_hitting_dist|)) * 4

                    if contact_coordinate[0] < 0: # landing on left table
                        Reward += 1

                    else: # landing on right table
                        Reward += 0

                elif:   # not landing on right table
                    Reward += (1 - tanh(|shortest_hitting_dist|))

                continue

    """

    def __init__(self, ctxt_dim, context_range_bounds):
        self.reward = 0
        self.goal_achievement = False
        self.temp_reward = 0
        self.shortest_hitting_dist = 1000
        self.highest_reward = -10000000
        self.table_contact_detector = False
        self.floor_contact_detector = False
        self.hit_contact_detector = False
        self.target_flag = False
        self.dist_target_virtual = 100
        self.ball_z_pos_lowest = 100
        self.hitting_flag = False
        self.ctxt_dim = ctxt_dim
        self.counter_after_htting = 0

        self.context_range_bounds = context_range_bounds

    @classmethod
    def goal_distance(cls, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    @classmethod
    def relu(cls, x):
        return np.maximum(0, x)

    @classmethod
    def contact_detection(cls, env, goal_contact):
        for i in range(env.sim.data.ncon):
            contact = env.sim.data.contact[i]
            achieved_geom1_name = env.sim.model.geom_id2name(contact.geom1)
            achieved_geom2_name = env.sim.model.geom_id2name(contact.geom2)
            if np.all([(achieved_geom1_name in goal_contact), (achieved_geom2_name in goal_contact)]):
                return True
            else:
                return False

    def check_valid(self, ctxt, context_range_bounds):
        valid = True
        valid_count = 0
        for dim in range(self.ctxt_dim):
            min_dim = context_range_bounds[0][dim]
            max_dim = context_range_bounds[1][dim]
            if ctxt[dim] > max_dim or ctxt[dim] < min_dim:
                valid = False
                valid_count += np.abs(ctxt[dim] - max_dim)* (ctxt[dim] > max_dim) + \
                               np.abs(min_dim - ctxt[dim]) * (ctxt[dim] < min_dim)
        return valid,valid_count

    def refresh_highest_reward(self):
        if self.temp_reward >= self.highest_reward:
            self.highest_reward = self.temp_reward

    def huge_value_unstable(self):
        self.temp_reward += -10
        self.highest_reward = -1

    def context_valid(self, context):
        valid, _ = self.check_valid(context.copy(), context_range_bounds=self.context_range_bounds)
        # when using dirac punishments
        if valid:
            self.temp_reward += 2
        else:
            self.temp_reward += 0
        self.refresh_highest_reward()

    def context_invalid_punishment(self, context):
        valid, invalid_account = self.check_valid(context.copy(), context_range_bounds=self.context_range_bounds)
        if not valid:
            self.highest_reward = -1 * invalid_account * 20
            self.goal_achievement = False
        else:
            self.goal_achievement = True

    def action_durations_valid_upper_lower_bound_valid(self, duration=None):
        hit_duration = duration
        hit_lower_bound, hit_upper_bound = [0.1, 5]
        self.goal_achievement = (hit_duration >= hit_lower_bound and hit_duration <= hit_upper_bound)
        if self.goal_achievement:
            self.temp_reward = 0
            self.goal_achievement = True
        else:
            if hit_duration < hit_lower_bound:
                self.temp_reward = -1 * ((np.abs(hit_lower_bound - hit_duration) * 3))
            elif hit_duration > hit_upper_bound:
                self.temp_reward = -1 * ((np.abs(hit_duration - hit_upper_bound) * 3))
            self.temp_reward += -1
            self.goal_achievement = False
        self.refresh_highest_reward()




    def joint_motion_mean_minus_penalty(self, trj):
        whole_trj = trj.whole_trj
        # config = trj.config
        action_space_low = np.array([-2.6, -2.0, -2.8, -0.9, -4.8, -1.6, -2.2])
        action_space_high = np.array([2.6, 2.0, 2.8, 3.1, 1.3, 1.6, 2.2])
        action_space_low = np.expand_dims(np.array(action_space_low), axis=1)
        action_space_high = np.expand_dims(np.array(action_space_high), axis=1)
        higher_than_upper_bound_indices = np.where(whole_trj > action_space_high)
        lower_than_lower_bound_indices = np.where(whole_trj < action_space_low)
        error_high = whole_trj - action_space_high
        error_low = action_space_low - whole_trj

        if error_high[higher_than_upper_bound_indices].shape[0] > 0:
            error_high_mean = np.mean(np.abs(error_high[higher_than_upper_bound_indices]))
        else:
            error_high_mean = 0
        if error_low[lower_than_lower_bound_indices].shape[0] > 0:
            error_low_mean = np.mean(np.abs(error_low[lower_than_lower_bound_indices]))
        else:
            error_low_mean = 0

        motion_mean_reward = -1 * (error_high_mean + error_low_mean)

        if motion_mean_reward < 0:
            self.highest_reward = motion_mean_reward
            self.goal_achievement = False
        else:
            self.goal_achievement = True

    def hitting(self, env):
        self.temp_reward = 0
        hit_contact_obj = ["target_ball", "bat"]
        target_ball_pos = env.target_ball_pos
        racket_center_pos = env.racket_center_pos
        # hit contact detection
        # Record the hitting history
        self.hitting_flag = False
        if not self.hit_contact_detector:
            self.hit_contact_detector = self.contact_detection(env, hit_contact_obj)
            if self.hit_contact_detector:
                self.hitting_flag = True
        if self.hit_contact_detector:

            dist = self.goal_distance(target_ball_pos, racket_center_pos)
            if dist < 0:
                dist = 0
            if dist <= self.shortest_hitting_dist:
                self.shortest_hitting_dist = dist
            # Keep the shortest hitting distance.
            dist_reward = 2 * (1 - np.tanh(np.abs(self.shortest_hitting_dist)))

            self.temp_reward += dist_reward
            self.goal_achievement = True
            self.reward = dist_reward
        else:
            dist = self.goal_distance(target_ball_pos, racket_center_pos)
            if dist <= self.shortest_hitting_dist:
                self.shortest_hitting_dist = dist
            dist_reward = 1 - np.tanh(self.shortest_hitting_dist)
            reward = 0.2 * dist_reward  # because it does not hit the ball, so multiply 0.2
            self.temp_reward += reward
            self.goal_achievement = False
            self.reward = 1 - np.tanh(dist)
        self.refresh_highest_reward()
        # return np.copy(self.highest_reward)

    def target_achievement(self, env):
        target_coordinate = env.target_pos.copy()
        table_contact_obj = ["target_ball", "table_tennis_table"]
        floor_contact_obj = ["target_ball", "floor"]

        if 0.78 < env.target_ball_pos[2] < 0.8:
            dist_target_virtual = np.abs(np.linalg.norm(env.target_ball_pos[:2] - target_coordinate[:2]))
            if self.dist_target_virtual > dist_target_virtual:
                self.dist_target_virtual = dist_target_virtual
        if -0.07 < env.target_ball_pos[0] < 0.07 and env.sim.data.get_joint_qvel('tar:x') < 0:
            if self.ball_z_pos_lowest > env.target_ball_pos[2]:
                self.ball_z_pos_lowest = env.target_ball_pos[2].copy()
        if not self.table_contact_detector:
            self.table_contact_detector = self.contact_detection(env, table_contact_obj)
        if not self.floor_contact_detector:
            self.floor_contact_detector = self.contact_detection(env, floor_contact_obj)
        if not self.target_flag:
            # Table Contact Reward.
            if self.table_contact_detector:
                self.target_flag = True
                env.sim.model.body_pos[3] = env.target_ball_pos
                contact_coordinate = env.target_ball_pos[:2].copy()
                dist_target = np.abs(np.linalg.norm(contact_coordinate - target_coordinate))
                self.temp_reward += (1 - np.tanh(dist_target)) * 4
                # Net Contact Reward. Precondition: Table Contact exits.
                if contact_coordinate[0] < 0:

                    self.goal_achievement = True

                    self.temp_reward += 1
                else:

                    self.goal_achievement = False

            # Floor Contact Reward. Precondition: Table Contact exits.
            elif self.floor_contact_detector:
                self.temp_reward += (1 - np.tanh(self.dist_target_virtual))
                self.target_flag = True
                self.goal_achievement = False
            # No Contact of Floor or Table, flying
            else:
                pass
        self.refresh_highest_reward()