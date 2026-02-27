import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu


class StackedReplayBuffer:
    def __init__(self, max_replay_buffer_size, time_steps, observation_dim, action_dim, task_indicator_dim, data_usage_reconstruction, data_usage_sac, num_last_samples, permute_samples):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._task_indicator_dim = task_indicator_dim
        self._max_replay_buffer_size = max_replay_buffer_size

        self._observations = np.zeros((max_replay_buffer_size, observation_dim), dtype=np.float32)
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim), dtype=np.float32)
        self._actions = np.zeros((max_replay_buffer_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros((max_replay_buffer_size, 1), dtype=np.float32)
        # task indicator computed through encoder
        self._base_task_indicators = np.zeros(max_replay_buffer_size, dtype=np.float32)
        self._task_indicators = np.zeros((max_replay_buffer_size, task_indicator_dim), dtype=np.float32)
        self._next_task_indicators = np.zeros((max_replay_buffer_size, task_indicator_dim), dtype=np.float32)

        self._sparse_rewards = np.zeros((max_replay_buffer_size, 1), dtype=np.float32)
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')

        self.time_steps = time_steps
        self._top = 0
        self._size = 0
        self._episode_starts = []

        # allowed points specify locations in the buffer, that, alone or together with the <self.time_step> last entries
        # can be sampled
        self._allowed_points = []
        self._train_indices = []
        self._val_indices = []
        self.stats_dict = None

        self.data_usage_reconstruction = data_usage_reconstruction
        self.data_usage_sac = data_usage_sac
        self.num_last_samples = num_last_samples
        self.permute_samples = permute_samples

        self.add_zero_elements()
        self._cur_episode_start = self._top

    def add_zero_elements(self):
        # TODO: as already spawned as zeros, actually not zero writing needed, could only advance
        for t in range(self.time_steps):
            self.add_sample(
                np.zeros(self._observation_dim),
                np.zeros(self._action_dim),
                np.zeros(1),
                np.zeros(1, dtype='uint8'),
                np.zeros(self._observation_dim),
                np.zeros(self._task_indicator_dim),
                np.zeros(self._task_indicator_dim)
            )


    def add_episode(self, episode):
        # Assume all array are same length (as they come from same rollout)
        length = episode['observations'].shape[0]

        # check, if whole episode fits into buffer
        if length >= self._max_replay_buffer_size:
            error_string =\
                "-------------------------------------------------------------------------------------------\n\n" \
                "ATTENTION:\n" \
                "The current episode was longer than the replay buffer and could not be fitted in.\n" \
                "Please consider decreasing the maximum episode length or increasing the task buffer size.\n\n" \
                "-------------------------------------------------------------------------------------------"
            print(error_string)
            return
        if self._size + length >= self._max_replay_buffer_size:
            # A bit space is not used, but assuming a big buffer it does not matter so much
            # TODO: additional 0 samples must be added
            self._top = 0

        low = self._top
        high = self._top + length
        self._observations[low:high] = episode['observations']
        self._next_obs[low:high] = episode['next_observations']
        self._actions[low:high] = episode['actions']
        self._rewards[low:high] = episode['rewards']
        self._task_indicators[low:high] = episode['task_indicators']
        self._next_task_indicators[low:high] = episode['next_task_indicators']
        self._terminals[low:high] = episode['terminals']

        self._advance_multi(length)
        self.terminate_episode()

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, task_indicator, next_task_indicator, **kwargs):
        self._observations[self._top] = observation
        self._next_obs[self._top] = next_observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._task_indicators[self._top] = task_indicator
        self._next_task_indicators[self._top] = next_task_indicator
        self._terminals[self._top] = terminal

        self._advance()

    def terminate_episode(self):
        # store the episode beginning once the episode is over
        # n.b. allows last episode to loop but whatever
        self._episode_starts.append(self._cur_episode_start)
        # TODO: allowed points must be "reset" at buffer overflow
        self._allowed_points += list(range(self._cur_episode_start, self._top))
        self.add_zero_elements()
        self._cur_episode_start = self._top

    def size(self):
        return self._size

    def get_allowed_points(self):
        return self._allowed_points

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def _advance_multi(self, length):
        self._top = (self._top + length) % self._max_replay_buffer_size
        if self._size + length <= self._max_replay_buffer_size:
            self._size += length
        else:
            self._size = self._max_replay_buffer_size

    def sample_data(self, indices):
        return dict(
            observations=self._observations[indices],
            next_observations=self._next_obs[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            task_indicators=self._task_indicators[indices],
            next_task_indicators=self._next_task_indicators[indices],
            sparse_rewards=self._sparse_rewards[indices],
            terminals=self._terminals[indices]
        )

    def get_indices(self, points, batch_size, prio=None):
        if prio == 'linear':
            # prioritized version: later samples get more weight
            weights = np.linspace(0.1, 0.9, points.shape[0])
            weights = weights / np.sum(weights)
            indices = np.random.choice(points, batch_size, replace=True if batch_size > points.shape[0] else False, p=weights)
        elif prio == 'cut':
            indices = np.random.choice(points[-self.num_last_samples:], batch_size, replace=True if batch_size > points[-self.num_last_samples:].shape[0] else False)
        elif prio == 'tree_sampling':
            # instead of using 'np.random.choice' directly on the whole 'points' array, which is O(n)
            # and highly inefficient for big replay buffers, we subdivide 'points' in buckets, which we apply
            # 'np.random.choice' to.
            # 'points' needs to be shuffled already, to ensure i.i.d assumption
            root = int(np.sqrt(points.shape[0]))
            if root < batch_size:
                indices = np.random.choice(points, batch_size, replace=True if batch_size > points.shape[0] else False)
            else:
                partition = int(points.shape[0] / root)
                division = np.random.randint(0, root)  # sample a sub-bucket
                points_division = points[partition * division: partition * (division + 1)]
                replace = True if batch_size > points_division.shape[0] else False
                indices = np.random.choice(points_division, batch_size, replace=replace)
        else:
            indices = np.random.choice(points, batch_size, replace=True if batch_size > points.shape[0] else False)
        return indices

    # Single transition sample functions
    def random_batch(self, indices, batch_size, prio='tree_sampling'):
        ''' batch of unordered transitions '''
        indices = self.get_indices(indices, batch_size, prio=prio)
        return self.sample_data(indices)

    def sample_sac_data_batch(self, indices, batch_size):
        return self.random_batch(indices, batch_size, prio=self.data_usage_sac)

    # Sequence sample functions

    def sample_few_step_batch(self, points, batch_size, normalize=True):
        # the points in time together with their <time_step> many entries from before are sampled
        all_indices = []
        for ind in points:
            all_indices += list(range(ind - self.time_steps, ind + 1))

        data = self.sample_data(all_indices)
        if normalize:
            data = self.normalize_data(data)
        for key in data:
            data[key] = np.reshape(data[key], (batch_size, self.time_steps + 1, -1))

        return data

    def sample_random_few_step_batch(self, points, batch_size, normalize=True):
        ''' batch of unordered small sequences of transitions '''
        indices = self.get_indices(points, batch_size, prio=self.data_usage_reconstruction)
        return self.sample_few_step_batch(indices, batch_size, normalize=normalize)

    def get_train_val_indices(self, train_val_percent):
        # Split all data from replay buffer into training and validation set
        # not very efficient but hopefully readable code in this function
        points = np.array(self.get_allowed_points())
        train_indices = np.array(self._train_indices)
        val_indices = np.array(self._val_indices)
        points = points[np.isin(points, train_indices, invert=True)]
        points = points[np.isin(points, val_indices, invert=True)]
        points = np.random.permutation(points)
        splitter = int(points.shape[0] * train_val_percent)
        new_train_indices = points[:splitter]
        new_val_indices = points[splitter:]
        self._train_indices += new_train_indices.tolist()
        self._val_indices += new_val_indices.tolist()
        self._train_indices.sort()
        self._val_indices.sort()

        return np.array(self._train_indices), np.array(self._val_indices)


    def make_encoder_data(self, data, batch_size, mode='multiply'):
        # MLP encoder input: state of last timestep + state, action, reward of all timesteps before
        # input is in form [[t-N], ... [t-1], [t]]
        # therefore set action and reward of last timestep = 0
        # Returns: [batch_size, timesteps, obs+action+reward dim]
        # assumes, that a flat encoder flattens the data itself

        observations = torch.from_numpy(data['observations'])
        actions = torch.from_numpy(data['actions'])
        rewards = torch.from_numpy(data['rewards'])
        next_observations = torch.from_numpy((data['next_observations']))
        task_indicators = torch.from_numpy((data['task_indicators']))

        observations_encoder_input = observations.clone().detach()[:, :-1, :]
        actions_encoder_input = actions.clone().detach()[:, :-1, :]
        rewards_encoder_input = rewards.clone().detach()[:, :-1, :]
        next_observations_encoder_input = next_observations.clone().detach()[:, :-1, :]
        task_indicators_z_model_input = task_indicators.clone().detach()[:, :-1, :]

        # size: [batch_size, time_steps, obs+action+reward]
        encoder_input = torch.cat([observations_encoder_input, actions_encoder_input, rewards_encoder_input, next_observations_encoder_input], dim=-1)

        if self.permute_samples:
            perm = torch.randperm(encoder_input.shape[1]).long()
            encoder_input = encoder_input[:, perm]
        pass
    
        return encoder_input.to(ptu.device), task_indicators_z_model_input.to(ptu.device)

    def get_stats(self):
        data = self.sample_data(self.get_allowed_points())
        stats_dict = dict(
                          observations={},
                          next_observations={},
                          actions={},
                          rewards={},
                          )
        for key in stats_dict:
            stats_dict[key]["max"] = data[key].max(axis=0)
            stats_dict[key]["min"] = data[key].min(axis=0)
            stats_dict[key]["mean"] = data[key].mean(axis=0)
            stats_dict[key]["std"] = data[key].std(axis=0)
        return stats_dict

    def normalize_data(self, data):
        stats_dict = self.stats_dict
        for key in stats_dict:
            data[key] = (data[key] - stats_dict[key]["mean"]) / (stats_dict[key]["std"] + 1e-9)
        return data