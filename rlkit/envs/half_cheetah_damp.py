import numpy as np

from . import register_env
from .half_cheetah import HalfCheetahEnv


@register_env('cheetah-non-stationary-damp')
class HalfCheetahDampEnv(HalfCheetahEnv):
    """Half-cheetah environment with target velocity, as described in [1]. The
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand.py
    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for
        model-based control", 2012
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """
    def __init__(self, task={}, n_train_tasks=2, n_eval_tasks=2, use_normalized_env=True, scripted_policy=False, state_reconstruction_clip=8):
        self.meta_mode = 'train'
        self._task = task
        self.train_tasks = n_train_tasks
        self.eval_tasks = n_eval_tasks
        n_tasks = self.train_tasks + self.eval_tasks
        self.tasks = self.sample_tasks(n_tasks)

        self.damping_scale_set=[0.85, 0.9, 0.95, 1.0]
        self.damp_scale = 1.0

        self.steps = 0
        self.change_prob = 1.0
        self.change_interval_base = 60
        self.has_change_interval = False

        super(HalfCheetahDampEnv, self).__init__()

        self.original_damping = np.copy(self.model.dof_damping)

    def step(self, action): 
        self.steps += 1
        self.check_env_change()  
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run

        observation = self._get_obs()
        done = False
        infos = dict(reward_forward=reward_run,
            reward_ctrl=reward_ctrl, task=self._task)
        return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks):
        self.num_tasks = num_tasks
        if self.meta_mode == 'train':
            dampings = np.random.uniform(0.5, 1.5, size=(self.train_tasks,))
        elif self.meta_mode == 'test':
            dampings = np.random.uniform(0.5, 1.5, size=(self.eval_tasks,))
        tasks = [{'damp': damp} for damp in dampings]

        return tasks

    def get_all_task_idx(self):
        return range(len(self.tasks))
    
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._damping = self._task['damp']
        self.model.dof_damping[:] = self._damping
        self.reset()

    def get_change_interval(self):
        self.change_interval = max(
            int(np.random.normal(self.change_interval_base, 20)), 10)
        self.has_change_interval = True
    
    def check_env_change(self):
        # with some probability change reward function
        prob = np.random.uniform(0, 1)
        if not self.has_change_interval:
            self.get_change_interval()
        if prob < self.change_prob and self.steps > self.change_interval and self.steps > 0:
            self.change_damping()
            self.steps = 0
            # print("Env has changed!")

    def change_damping(self):
        random_index = self.np_random.randint(len(self.damping_scale_set))
        self.damping_scale = self.damping_scale_set[random_index]

        damping = np.copy(self.original_damping)
        damping[2:5] *= self.damping_scale
        damping[5:8] *= self.damping_scale
        damping[8:11] *= 1.0/self.damping_scale
        damping[11:14] *= 1.0/self.damping_scale
        self.model.dof_damping[:] = damping
        
    def set_meta_mode(self, mode):
        self.meta_mode = mode
       