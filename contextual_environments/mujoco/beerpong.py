from util.det_promp import DeterministicProMP
from contextual_environments.mujoco.abstractenv import AbstractEnvironment
from scipy.interpolate import make_interp_spline
from mujoco_py import MjSimState
from mujoco_py.generated import const

import os
import numpy as np
import mujoco_py

ctxt_dim = 2
context_range_x = [ -0.32, 0.32 ]
context_range_y = [ -2.2, -1.2 ]
context_range = [context_range_x, context_range_y]
max_ctxt_coords = np.array([context_range_x[1], context_range_y[1]])
min_ctxts_coords = np.array([context_range_x[0], context_range_y[0]])
CONTEXT_RANGE_BOUNDS = [min_ctxts_coords, max_ctxt_coords]
ACTION_DIM=15
CTXT_DIM=2

def get_env(name, n_cores, env_bounds, render = False, promp_time=True):
    return BeerPong(name=name, n_cores=n_cores, env_bounds=env_bounds, render=render, promp_time=promp_time)

def get_beerpong_cost_func(env_bounds, promp_time=True):
    return BeerPongCostFunction(env_bounds, promp_time=promp_time)

class BeerPongCostFunction:

    def __init__(self, env_bounds=None, name=None, promp_time=False):
        self.name = name
        self.env_bounds = env_bounds
        self.promp_time = promp_time
        self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xml",
                                     "beerpong_wo_cup" + ".xml")
        self.p_gains = 1 * np.array([200, 300, 100, 100, 10, 10, 2.5])
        self.d_gains = 0.5 * np.array([7, 15, 5, 2.5, 0.3, 0.3, 0.05])  # for sim step size 0.005
        self.max_ctrl = np.array([150., 125., 40., 60., 5., 5., 2.])
        self.min_ctrl = -self.max_ctrl

        self.j_min = np.array([-2.6, -1.985, -2.8, -0.9, -4.55, -1.5707, -2.7])
        self.j_max = np.array([2.6, 1.985, 2.8, 3.14159, 1.25, 1.5707, 2.7])

        self.sim = None
        self.viewer = None
        self.init_sim_state = None

        self.robot_collision_objects = ["wrist_palm_link_convex_geom",
                                        "wrist_pitch_link_convex_decomposition_p1_geom",
                                        "wrist_pitch_link_convex_decomposition_p2_geom",
                                        "wrist_pitch_link_convex_decomposition_p3_geom",
                                        "wrist_yaw_link_convex_decomposition_p1_geom",
                                        "wrist_yaw_link_convex_decomposition_p2_geom",
                                        "forearm_link_convex_decomposition_p1_geom",
                                        "forearm_link_convex_decomposition_p2_geom",
                                        "upper_arm_link_convex_decomposition_p1_geom",
                                        "upper_arm_link_convex_decomposition_p2_geom",
                                        "shoulder_link_convex_decomposition_p1_geom",
                                        "shoulder_link_convex_decomposition_p2_geom",
                                        "shoulder_link_convex_decomposition_p3_geom",
                                        "base_link_convex_geom", "table_contact_geom"]
        self.robot_collision_ids = None
        self.cup_robot_id = None
        self.cup_robot_collision_id = None

        self.cup_collision_objects = ["cup_geom_table3", "cup_geom_table4", "cup_geom_table5", "cup_geom_table6",
                                      "cup_geom_table7", "cup_geom_table8", "cup_geom_table9", "cup_geom_table10",
                                      "cup_base_table", "cup_base_table_contact", "cup_geom_table15",
                                      "cup_geom_table16",
                                      "cup_geom_table17", "cup_geom1_table8", "cup_base_table_contact",
                                      "cup_base_table"]
        self.cup_collision_ids = None
        self.ball_id = None
        self.ball_collision_id = None
        self.table_collision_id = None
        self.wall_collision_id = None
        self.goal_id = None
        self.goal_final_id = None
        self.cup_table_id = None
        self.cup_table_collision_id = None
        self.init_ball_pos_site_id = None
        self.ground_collision_id = None

        self.ball_traj = None
        self.joint_traj = None
        self.first_bounce = False
        self.n_first_table_bounces = 0
        self.n_first_wall_bounces = 0
        self.n_first_floor_bounces = 0

        self.n_ep_samples_executed = 0
        self.n_env_interacts = 0

    def __call__(self, cs, xs, env_bounds, render=True):
        xs = np.copy(xs)
        if self.promp_time:
            xs[:-1] *= np.sqrt(0.02)  # * np.copy(xs)
            if xs[-1] < 0.1:
                return -30 - 10 * (xs[-1] - 0.1) ** 2, 0
            elif xs[-1] > 1.3:
                return -30 - 10 * (xs[-1] - 1.3) ** 2, 0
        else:
            xs *= np.sqrt(0.02)
        cost, success, cons_violation = self._run_experiment(cs, xs, env_bounds, render=render)

        if cons_violation:
            return -30 - cost, 0

        reward = -1 * cost
        return reward, success

    @staticmethod
    def _check_collision_with_set_of_objects(sim, id_1, id_list):
        for coni in range(0, sim.data.ncon):
            con = sim.data.contact[coni]

            collision = con.geom1 in id_list and con.geom2 == id_1
            collision_trans = con.geom1 == id_1 and con.geom2 in id_list

            if collision or collision_trans:
                return True
        return False

    @staticmethod
    def _check_collision_single_objects(sim, id_1, id_2):
        for coni in range(0, sim.data.ncon):
            con = sim.data.contact[coni]

            collision = con.geom1 == id_1 and con.geom2 == id_2
            collision_trans = con.geom1 == id_2 and con.geom2 == id_1

            if collision or collision_trans:
                return True
        return False

    def _check_collision_with_itself(self, sim, collision_ids):
        col_1, col_2 = False, False
        for j, id in enumerate(collision_ids):
            col_1 = self._check_collision_with_set_of_objects(sim, id, collision_ids[:j])
            if j != len(collision_ids) - 1:
                col_2 = self._check_collision_with_set_of_objects(sim, id, collision_ids[j + 1:])
            else:
                col_2 = False
        collision = True if col_1 or col_2 else False
        return collision

    def _load_xml(self):
        try:
            sim = mujoco_py.MjSim(mujoco_py.load_model_from_path(self.xml_path), nsubsteps=1)
            return sim
        except:
            print('Failed to load XML-Model')

    def _run_experiment(self, context, theta, env_bounds, render=True):

        invalid, dist = self._check_where_invalid(context, env_bounds)
        if invalid:
            return dist, 0, 1

        if self.sim is None:
            self.sim = self._load_xml()
            self.init_sim_state = MjSimState(time=0.0, qpos=self.sim.data.qpos.copy(), qvel=self.sim.data.qvel.copy(), act=None,
                                         udd_state={})
            self.ball_id = self.sim.model._body_name2id["ball"]
            self.ball_collision_id = self.sim.model._geom_name2id["ball_geom"]
            # self.cup_robot_id = self.sim.model._site_name2id["cup_robot_final"]
            self.goal_id = self.sim.model._site_name2id["cup_goal_table"]
            self.goal_final_id = self.sim.model._site_name2id["cup_goal_final_table"]
            self.robot_collision_ids = [self.sim.model._geom_name2id[name] for name in self.robot_collision_objects]
            self.cup_collision_ids = [self.sim.model._geom_name2id[name] for name in self.cup_collision_objects]
            self.cup_table_id = self.sim.model._body_name2id["cup_table"]
            self.table_collision_id = self.sim.model._geom_name2id["table_contact_geom"]
            self.wall_collision_id = self.sim.model._geom_name2id["wall"]
            self.cup_table_collision_id = self.sim.model._geom_name2id["cup_base_table_contact"]
            self.init_ball_pos_site_id = self.sim.model._site_name2id["init_ball_pos_site"]
            self.ground_collision_id = self.sim.model._geom_name2id["ground"]

            if render:
                self.viewer = mujoco_py.MjViewer(self.sim)
        else:
            self.sim.set_state(self.init_sim_state)
            if render and self.viewer is None:
                self.viewer = mujoco_py.MjViewer(self.sim)

        if context.shape[0] == 2:
            ctxt = np.array([context[0], context[1], 0.840])
        elif context.shape[0] == 1:
            ctxt = np.array([0, context[0], 0.840])
        else:
            raise ValueError("Wrong context dimension")

        init_pos = self.sim.data.qpos.copy()
        init_vel = np.zeros(init_pos.shape[0])
        start_pos = np.array([0.0, 1.35, 0.0, 1.18, 0.0, -0.786, -1.59])

        weights = DeterministicProMP.shape_weights(theta, self.promp_time)
        if self.promp_time:
            dur = theta[-1]
            f = dur/ (0.005*3.5)+1
        else:
            f = 15

        n_steps = weights.shape[0]
        if n_steps == 1:
            width = 0.02
        elif n_steps == 2:
            width = 0.01
        elif n_steps == 3:  # for 7 joints !!
            width = 0.008
        elif n_steps == 4:  # for 7 joints !!
            width = 0.005
        else:
            width = 0.0035  # default
        pmp = DeterministicProMP(n_basis=n_steps + 2, width=width, off=0.01)
        if weights.shape[1] == 7:
            weights = np.concatenate((weights[:, 0][:, None], weights[:, 1][:, None], weights[:, 2][:, None],
                                      weights[:, 3][:, None], weights[:, 4][:, None], weights[:, 5][:, None],
                                      weights[:, 6][:, None]), axis=1)
        elif weights.shape[1] == 6:
            weights = np.concatenate((weights[:, 0][:, None], weights[:, 1][:, None], weights[:, 2][:, None],
                                      weights[:, 3][:, None], weights[:, 4][:, None], weights[:, 5][:, None],
                                      np.zeros(weights[:, 0][:, None].shape)), axis=1)
        else:
            weights = np.concatenate(
                (np.zeros((weights[:, 0].shape[0], 1)), weights[:, 0][:, None], np.zeros((weights[:, 0].shape[0], 1)),
                 weights[:, 1][:, None], np.zeros((weights[:, 0].shape[0], 1)), weights[:, 2][:, None],
                 np.zeros((weights[:, 0].shape[0], 1))), axis=1)

        pmp.set_weights(3.5, np.concatenate((np.zeros((2, weights.shape[1])), weights), axis=0))
        des_pos, des_vel = pmp.compute_trajectory(f, 1.)[1:3]  # corresponds to 0.2625 s if step-size to 0.005
        des_pos += start_pos[None, :]

        cons_violation = self._check_traj_in_joint_limits(des_pos)
        if cons_violation:
            print('cons violation of promp')
            return 0, 0, 1

        # Reset the system
        self.sim.data.qpos[:] = init_pos
        self.sim.data.qvel[:] = init_vel
        self.sim.data.qpos[0:7] = start_pos.copy()
        self.sim.step()
        self.sim.data.qpos[7::] = self.sim.data.site_xpos[self.init_ball_pos_site_id, :].copy()
        # only translational vel is sufficient, slide joints do not have rotational vel....
        self.sim.data.qvel[7::] = self.sim.data.site_xvelp[self.init_ball_pos_site_id, :].copy()
        self.sim.model.body_pos[self.cup_table_id] = ctxt.copy()
        self.sim.step()
        dists = []
        dists_final = []
        k = 0
        torques = []
        ball_in_cup = False
        sim_time = des_pos.shape[0] + 500
        joint_positions = np.zeros((sim_time, 7))
        ball_traj = np.zeros((sim_time, 3))
        ball_table_bounce = None
        ball_cup_table_cont = None
        ball_wall_cont = None
        first_con_with_table_wall_cup = False
        while k < sim_time:
            self._check_first_bounce()
            if self._check_collision_single_objects(self.sim, self.ball_collision_id, self.table_collision_id):
                ball_table_bounce = self.sim.data.body_xpos[self.ball_id].copy()
            if self._check_collision_with_set_of_objects(self.sim, self.ball_collision_id, self.cup_collision_ids):
                ball_cup_table_cont = self.sim.data.body_xpos[self.ball_id].copy()
            if self._check_collision_single_objects(self.sim, self.ball_collision_id, self.wall_collision_id):
                ball_wall_cont = self.sim.data.body_xpos[self.ball_id].copy()
            if self._check_collision_single_objects(self.sim, self.ball_collision_id, self.cup_table_collision_id):
                ball_in_cup=True
            if not first_con_with_table_wall_cup:
                first_con_with_table_wall_cup = True if (ball_table_bounce is not None or ball_wall_cont is not None
                                                         or ball_cup_table_cont is not None) else False
            goal_pos = self.sim.data.site_xpos[self.goal_id]
            ball_pos = self.sim.data.body_xpos[self.ball_id]
            goal_final_pos = self.sim.data.site_xpos[self.goal_final_id]
            dists.append(np.linalg.norm(goal_pos - ball_pos))
            dists_final.append(np.linalg.norm(goal_final_pos - ball_pos))
            ball_traj[k, :] = ball_pos
            # Compute the controls
            cur_pos = self.sim.data.qpos[0:7].copy()
            cur_vel = self.sim.data.qvel[0:7].copy()
            k_actual = np.minimum(des_pos.shape[0] - 1, k)
            trq = self.p_gains * (des_pos[k_actual, :] - cur_pos) + self.d_gains * (des_vel[k_actual, :] - cur_vel)
            joint_positions[k, :] = cur_pos
            # Advance the simulation
            self.sim.data.qfrc_applied[0:7] = trq + self.sim.data.qfrc_bias[:7].copy()
            torques.append(trq + self.sim.data.qfrc_bias[:7].copy())
            try:
                self.sim.step()
                self.n_env_interacts += 1
                if k == 0:
                    self.n_ep_samples_executed += 1
                if k<=des_pos.shape[0]:
                    self.sim.data.qpos[7::] = self.sim.data.site_xpos[self.init_ball_pos_site_id, :].copy()
                    # only translational vel is sufficient, slide joints do not have rotational vel....
                    self.sim.data.qvel[7::] = self.sim.data.site_xvelp[self.init_ball_pos_site_id, :].copy()
            except mujoco_py.MujocoException as e:
                return 0,0,1 # cost, success, cons_violation
            k += 1
            # # Check for a collision - in which case we end the simulation
            if self._check_collision_with_itself(self.sim, self.robot_collision_ids):
                return 500, 0, 0

            if render:
                self.viewer.cam.fixedcamid = self.sim.model.camera_name2id('visualization')
                self.viewer.cam.type = const.CAMERA_FIXED
                self.viewer.render()
        min_dist = np.min(dists)
        torques = np.stack(torques)

        torques_mean = np.mean(np.sum(np.square(torques), axis=1), axis=0)

        # before:
        if not ball_in_cup:
            cost_offset = 2
            if ball_cup_table_cont is None and ball_table_bounce is None and ball_wall_cont is None:
                cost_offset += 2
            cost = cost_offset + min_dist ** 2 + 0.5 * dists_final[-1] ** 2 + 1e-4 * torques_mean
        else:
            cost = dists_final[-1] ** 2 + 1.5 * torques_mean * 1e-4

        self.ball_traj = ball_traj
        self.joint_traj = joint_positions

        # Phase Based cost
        ball_in_cup_final = self._check_collision_single_objects(self.sim, self.ball_collision_id,
                                                                 self.cup_table_collision_id)
        self.first_bounce = False
        return cost, 1. if ball_in_cup_final else 0., 0

    def _check_traj_in_joint_limits(self, traj):

        for i in range(traj.shape[1]):
            c_joint_lim_max = self.j_max[i]
            c_joint_lim_min = self.j_min[i]
            max_joint_violation = np.where(traj[:, i]>=c_joint_lim_max)[0]
            min_joint_violation = np.where(traj[:, i]<=c_joint_lim_min)[0]
            if max_joint_violation.shape[0] != 0:
                return True
            if min_joint_violation.shape[0] != 0:
                return True
        return False

    def get_tot_n_samples(self):
        n_ep_samples_executed = np.copy(self.n_ep_samples_executed)
        n_env_interacts = np.copy(self.n_env_interacts)
        self.n_ep_samples_executed = 0
        self.n_env_interacts = 0
        return n_ep_samples_executed, n_env_interacts

    def _check_first_bounce(self):

        if not self.first_bounce:
            if self._check_collision_single_objects(self.sim, self.ball_collision_id, self.table_collision_id):
                self.first_bounce = True
                self.n_first_table_bounces += 1
            elif self._check_collision_single_objects(self.sim, self.ball_collision_id, self.wall_collision_id):
                self.first_bounce = True
                self.n_first_wall_bounces += 1
            elif self._check_collision_single_objects(self.sim, self.ball_collision_id, self.ground_collision_id):
                self.first_bounce = True
                self.n_first_floor_bounces += 1

    def _check_where_invalid(self, single_context, context_range_bounds, set_to_valid_region=False):

        ctxt_dim = single_context.shape[0]  # in this environment we do not have a 2 dim array...
        violated = False
        dist_quad = 0
        for dim in range(ctxt_dim):
            min_dim = context_range_bounds[0][dim]
            max_dim = context_range_bounds[1][dim]
            idx_max_c = np.where(single_context[dim] > max_dim)[0]
            idx_min_c = np.where(single_context[dim] < min_dim)[0]
            if set_to_valid_region:
                if idx_max_c.shape[0] != 0:
                    single_context[dim] = max_dim
                if idx_min_c.shape[0] != 0:
                    single_context[dim] = min_dim
            if idx_max_c.shape[0] != 0 or idx_min_c.shape[0] != 0:
                violated = True
            if idx_max_c.shape[0] != 0:
                dist_quad += (context_range_bounds[1][dim] - single_context[dim]) ** 2
            if idx_min_c.shape[0] != 0:
                dist_quad += (context_range_bounds[0][dim] - single_context[dim]) ** 2
        if violated:
            return 1, dist_quad
        else:
            return 0, 0

    def sample_contexts(self, n_samples, context_range_bounds):
        ctxt_dim = context_range_bounds[0].shape[0]
        ctxt_samples = np.random.uniform(context_range_bounds[0], context_range_bounds[1], size=(n_samples, ctxt_dim))
        return ctxt_samples

    def step(self, xs, cs, render=False):
        rewards = []
        successes = []
        for i in range(cs.shape[0]):
            r, s = self(cs[i, :], xs[i, :], self.env_bounds, render=render)
            rewards.append(r)
            successes.append(s)
        return np.stack(rewards), np.stack(successes)

class BeerPong(AbstractEnvironment):

    def __init__(self, name, n_cores, env_bounds, render=False, promp_time=False):
        super(BeerPong, self).__init__(name, n_cores, BeerPongCostFunction, env_bounds=env_bounds, render=render,
                                       promp_time=promp_time)
