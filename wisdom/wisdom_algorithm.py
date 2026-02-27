import numpy as np
import torch
from collections import OrderedDict
from rlkit.core import logger
import gtimer as gt
import pickle
import os
import ray
import gc

import rlkit.torch.pytorch_util as ptu


class WISDOMAlgorithm:
    def __init__(self,
                 replay_buffer,
                 rollout_coordinator,
                 reconstruction_trainer,
                 policy_trainer,
                 agent,
                 networks,
                 train_tasks,
                 test_tasks,

                 num_epochs,
                 num_reconstruction_steps,
                 num_policy_steps,
                 num_train_tasks_per_episode,
                 num_transitions_initial,
                 num_transistions_per_episode,
                 num_eval_trajectories,
                 experiment_log_dir,
                 latent_dim
                 ):
        self.replay_buffer = replay_buffer
        self.rollout_coordinator = rollout_coordinator
        self.reconstruction_trainer = reconstruction_trainer
        self.policy_trainer = policy_trainer
        self.agent = agent
        self.networks = networks

        self.train_tasks = train_tasks
        self.test_tasks = test_tasks

        self.num_epochs = num_epochs
        self.num_reconstruction_steps = num_reconstruction_steps
        self.num_policy_steps = num_policy_steps
        self.num_transitions_initial = num_transitions_initial
        self.num_train_tasks_per_episode = num_train_tasks_per_episode
        self.num_transitions_per_episode = num_transistions_per_episode
        self.num_eval_trajectories = num_eval_trajectories
        self.experiment_log_dir = experiment_log_dir
        self.latent_dim = latent_dim

        self._n_env_steps_total = 0

    def train(self):
        params = self.get_epoch_snapshot()
        logger.save_itr_params(-1, params)
        previous_epoch_end = 0

        print("Collecting initial samples ...")
        if self.num_transitions_initial > 0:
            self._n_env_steps_total += self.rollout_coordinator.collect_replay_data(self.train_tasks, max_samples=self.num_transitions_initial)

        for epoch in gt.timed_for(range(self.num_epochs), save_itrs=True):
            tabular_statistics = OrderedDict()

            # 1. collect data with rollout coordinator
            print("Collecting samples ...")
            data_collection_tasks = np.random.permutation(self.train_tasks)[:self.num_train_tasks_per_episode]
            self._n_env_steps_total += self.rollout_coordinator.collect_replay_data(data_collection_tasks, max_samples=self.num_transitions_per_episode)
            tabular_statistics['n_env_steps_total'] = self._n_env_steps_total
            gt.stamp('data_collection')

            # replay buffer stats
            self.replay_buffer.stats_dict = self.replay_buffer.get_stats()

            print("Reconstruction Trainer ...")
            self.reconstruction_trainer.train(self.num_reconstruction_steps)
            gt.stamp('reconstruction_trainer')

            # 4. train policy via SAC with data from the replay buffer
            print("Policy Trainer ...")
            temp, sac_stats = self.policy_trainer.train(self.num_policy_steps)
            tabular_statistics.update(sac_stats)
            gt.stamp('policy_trainer')

            # 5. Evaluation
            print("Evaluation ...")
            eval_output = self.rollout_coordinator.evaluate('train', data_collection_tasks, self.num_eval_trajectories, deterministic=True, animated=False, log=True)
            average_test_reward, std_test_reward, max_test_reward, min_test_reward, eval_stats = eval_output
            tabular_statistics.update(eval_stats)
            eval_output = self.rollout_coordinator.evaluate('test', self.test_tasks, self.num_eval_trajectories, deterministic=False, animated=False, log=True)
            average_test_reward, std_test_reward, max_test_reward, min_test_reward, eval_stats = eval_output
            tabular_statistics.update(eval_stats)
            gt.stamp('evaluation')

            # 8. Time
            times_itrs = gt.get_times().stamps.itrs
            tabular_statistics['time_data_collection'] = times_itrs['data_collection'][-1]
            # tabular_statistics['time_reconstruction_trainer'] = times_itrs['reconstruction_trainer'][-1]
            tabular_statistics['time_policy_trainer'] = times_itrs['policy_trainer'][-1]
            tabular_statistics['time_evaluation'] = times_itrs['evaluation'][-1]
            total_time = gt.get_times().total
            epoch_time = total_time - previous_epoch_end
            previous_epoch_end = total_time
            tabular_statistics['time_epoch'] = epoch_time
            tabular_statistics['time_total'] = total_time

            # other
            tabular_statistics['n_env_steps_total'] = self._n_env_steps_total
            tabular_statistics['epoch'] = epoch

            for key, value in tabular_statistics.items():
                logger.record_tabular(key, value)

            logger.dump_tabular(with_prefix=False, with_timestamp=False)

        ray.shutdown()

    def get_epoch_snapshot(self):
        snapshot = OrderedDict()
        for name, net in self.networks.items():
            snapshot[name] = net.state_dict()
        return snapshot

    def to(self, device=None):
        if device is None:
            device = ptu.device
        for net in self.networks:
            self.networks[net].to(device)
        self.agent.to(device)
