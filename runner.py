import os
import numpy as np
import click
import json
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv, CameraWrapper
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config
from wisdom.encoder_decoder_networks import MlpEncoder
from wisdom.networks import Mlp, FlattenMlp, Y_Network
from wisdom.sac import PolicyTrainer
from wisdom.stacked_replay_buffer import StackedReplayBuffer
from wisdom.reconstruction_trainer import ReconstructionTrainer
from wisdom.rollout_worker import RolloutCoordinator
from wisdom.agent import WISDOMAgent, ScriptedPolicyAgent
from wisdom.wisdom_algorithm import WISDOMAlgorithm

import multiprocessing as mp
from itertools import product
from metaworld import ML1


def global_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

def experiment(variant, seed=None):
    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])

    experiment_log_dir = setup_logger(
        variant['env_name'],
        variant=variant,
        exp_id=None,
        base_log_dir=variant['util_params']['base_log_dir'],
        seed=seed
    )

    if variant['env_name'] != 'metaworld':
        env = ENVS[variant['env_name']](**variant['env_params'])
        if variant['env_params']['use_normalized_env']:
            env = NormalizedBoxEnv(env)
        if variant['train_or_showcase'] == 'showcase':
            env = CameraWrapper(env)
        tasks = list(range(len(env.tasks)))
    else:
        print(ML1.ENV_NAMES)
        name = ML1.ENV_NAMES[variant['env_params']['task_id']]
        ml1 = ML1(name)  # Construct the Meta-World benchmark, sampling tasks

        env = ml1.train_classes[name]() 
        tasks = ml1.train_tasks + ml1.test_tasks
        env = NormalizedBoxEnv(env)
        print(env)
        env.tasks_pool = tasks
        tasks = list(range(variant['env_params']['n_tasks']))

    if seed is not None:
        global_seed(seed)
        env.seed(seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1
    train_tasks = list(tasks[:variant['env_params']['n_train_tasks']])
    test_tasks = list(tasks[-variant['env_params']['n_eval_tasks']:])

    # instantiate networks
    latent_dim = variant['algo_params']['latent_size']
    time_steps = variant['algo_params']['time_steps']
    encoder_input_dim = 2 * obs_dim + action_dim + reward_dim
    encoder_output_dim = latent_dim * 2
    decompose_times = variant['wavelet_params']['depth']
    filter_size = variant['wavelet_params']['filter_size']
    dimension = variant['wavelet_params']['dimension']

    encoder = MlpEncoder(
        hidden_sizes=[200, 200, 200],
        input_size=encoder_input_dim,
        output_size=encoder_output_dim,
    )
    z_model = Y_Network(
        d_model=dimension,    
        kernel_size=filter_size,    # filter size
        depth=decompose_times,    # depth of each layer
        wavelet_init=None,  
        dropout=0.2,
        indep_res_init=False, 
    )
    target_z_model = Y_Network(
        d_model=dimension,    
        kernel_size=filter_size,    # filter size
        depth=decompose_times,    # depth of each layer
        wavelet_init=None,    
        seq_len=None,
        dropout=0.2,
        indep_res_init=False,    
    )

    M = variant['algo_params']['sac_layer_size']
    qf1 = FlattenMlp(
        input_size=(obs_dim + latent_dim) + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    qf2 = FlattenMlp(
        input_size=(obs_dim + latent_dim) + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=(obs_dim + latent_dim) + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=(obs_dim + latent_dim) + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=(obs_dim + latent_dim),
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_sizes=[M, M, M],
    )
    alpha_net = Mlp(
        hidden_sizes=[latent_dim * 10],
        input_size=latent_dim,
        output_size=1
    )

    networks = {'encoder': encoder,
                'z_model': z_model,
                'target_z_model': target_z_model,
                'qf1': qf1,
                'qf2': qf2,
                'target_qf1': target_qf1,
                'target_qf2': target_qf2,
                'policy': policy,
                'alpha_net': alpha_net}

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        itr = variant['showcase_itr']
        path = variant['path_to_weights']
        for name, net in networks.items():
            net.load_state_dict(torch.load(os.path.join(path, name + '_itr_' + str(itr) + '.pth'), map_location='cpu'))

    replay_buffer = StackedReplayBuffer(
        variant['algo_params']['max_replay_buffer_size'],
        time_steps,
        obs_dim,
        action_dim,
        latent_dim,
        variant['algo_params']['data_usage_reconstruction'],
        variant['algo_params']['data_usage_sac'],
        variant['algo_params']['num_last_samples'],
        variant['algo_params']['permute_samples']
    )

    agent_class = ScriptedPolicyAgent if variant['env_params']['scripted_policy'] else WISDOMAgent
    agent = agent_class(
        encoder,
        latent_dim,
        policy
    )

    rollout_coordinator = RolloutCoordinator(
        env,
        variant['env_name'],
        variant['env_params'],
        tasks,
        train_tasks,
        test_tasks,
        variant['train_or_showcase'],
        agent,
        replay_buffer,
        time_steps,
        variant['algo_params']['max_path_length'],
        variant['algo_params']['permute_samples'],
        variant['util_params']['use_multiprocessing'],
        variant['algo_params']['use_data_normalization'],
        variant['util_params']['num_workers'],
        variant['util_params']['gpu_id'],
        variant['env_params']['scripted_policy']
        )

    reconstruction_trainer = ReconstructionTrainer(
        encoder,
        z_model,
        target_z_model,
        agent,
        replay_buffer,
        variant['algo_params']['batch_size_reconstruction'],
        variant['algo_params']['td_loss_coefficient'],
        latent_dim,
        time_steps,
        encoder_output_dim,
        variant['reconstruction_params']['lr_encoder'],
        variant['reconstruction_params']['train_val_percent'],
        experiment_log_dir,
        variant['algo_params']['data_usage_reconstruction'],
    )

    policy_trainer = PolicyTrainer(
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        alpha_net,
        z_model,
        agent,
        replay_buffer,
        variant['algo_params']['batch_size_policy'],
        action_dim,
        variant['algo_params']['data_usage_sac'],
        use_parametrized_alpha=variant['algo_params']['use_parametrized_alpha'],
        target_entropy_factor=variant['algo_params']['target_entropy_factor'],
        alpha=variant['algo_params']['sac_alpha']
    )

    algorithm =  WISDOMAlgorithm(
        replay_buffer,
        rollout_coordinator,
        reconstruction_trainer,
        policy_trainer,
        agent,
        networks,
        train_tasks,
        test_tasks,
        variant['algo_params']['num_train_epochs'],
        variant['algo_params']['num_reconstruction_steps'],
        variant['algo_params']['num_policy_steps'],
        variant['algo_params']['num_train_tasks_per_episode'],
        variant['algo_params']['num_transitions_initial'],
        variant['algo_params']['num_transitions_per_episode'],
        variant['algo_params']['num_eval_trajectories'],
        experiment_log_dir,
        latent_dim
        )

    if ptu.gpu_enabled():
        algorithm.to()

    DEBUG = variant['util_params']['debug']
    PLOT = variant['util_params']['plot']
    os.environ['DEBUG'] = str(int(DEBUG))
    os.environ['PLOT'] = str(int(PLOT))

    # run the algorithm
    algorithm.train()


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=2)
@click.option('--num_workers', default=4)
@click.option('--use_mp', is_flag=True, default=False)

def main(config, gpu, use_mp, num_workers):
    variant = default_config

    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu
    variant['util_params']['use_multiprocessing'] = use_mp
    variant['util_params']['num_workers'] = num_workers

    # multi-processing
    p = mp.Pool(4)
    if len(variant['seed_list']) > 0:
        p.starmap(experiment, product([variant], variant['seed_list']))
    else:
        experiment(variant)

if __name__ == "__main__":
    main()
