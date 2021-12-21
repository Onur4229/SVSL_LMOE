import mujoco_py


class TableTennisSimulation(object):
    def __init__(self, render=False):
        self.time_step = 0.002
        self.render = render

    def simulation(self, trajectory=None, env=None):
        n_env_interacts = 0
        right_table_contact_detector = False
        ball_gun_target_flag = False
        trj_shape = trajectory.whole_trj.shape
        trj_time = trajectory.whole_time.shape
        env.reset()
        for t in range(trj_time[0]):
            # real_time = t
            assert trj_time[0] == trj_shape[1]
            if self.render:
                env.render()
            action = trajectory.whole_trj[:, t]
            try:
                observation, reward, done, info = env.step(action)
                n_env_interacts += 1
                # Visualize the right contact position
                if not right_table_contact_detector:
                    table_contact_obj = ["target_ball", "table_tennis_table"]
                    right_table_contact_detector = env.reward_obj.contact_detection(env, table_contact_obj)
                if right_table_contact_detector:
                    if not ball_gun_target_flag:
                        ball_gun_target_flag = True
                        env.sim.model.body_pos[4] = env.target_ball_pos
                if done:
                    env.reset()
                    break
            except mujoco_py.builder.MujocoException as e:
                # Give a huge negative reward
                env.reward_obj.huge_value_unstable()
                # Directly stop the experiment
                break
        return n_env_interacts