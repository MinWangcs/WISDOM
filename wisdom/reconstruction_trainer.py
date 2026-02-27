import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributions.kl as kl
from wisdom.utils import generate_gaussian
import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
import matplotlib.pyplot as plt
import torch.nn.functional as F


class ReconstructionTrainer(nn.Module):
    def __init__(self,
                 encoder,
                 z_model,
                 target_z_model,
                 agent,
                 replay_buffer,
                 batch_size,
                 td_loss_coefficient,
                 latent_dim,
                 timesteps,
                 encoder_output_dim,
                 lr_encoder,
                 train_val_percent,
                 experiment_log_dir,
                 data_usage_reconstruction,
                 optimizer_class=optim.Adam,
                 ):
        super(ReconstructionTrainer, self).__init__()
        self.encoder = encoder
        self.z_model = z_model
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.td_loss_coefficient = td_loss_coefficient
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.encoder_output_dim = encoder_output_dim
        self.lr_encoder = lr_encoder
        self.train_val_percent = train_val_percent
        self.experiment_log_dir = experiment_log_dir
        self.data_usage_reconstruction = data_usage_reconstruction

        self.lowest_loss = np.inf
        self.lowest_loss_epoch = 0

        self.optimizer_class = optimizer_class

        self.optimizer_encoder = self.optimizer_class(
            self.encoder.parameters(),
            lr=self.lr_encoder,
        )
        self.z_optimizer = optimizer_class(
            self.z_model.parameters(),
            lr=self.lr_encoder,
        )
        self.soft_target_tau = 5e-3
        self.gamma = 0.99
        self.target_z_model = target_z_model

    def train(self, epochs):
        train_indices, val_indices = self.replay_buffer.get_train_val_indices(self.train_val_percent)

        if self.data_usage_reconstruction == "tree_sampling":
            train_indices = np.random.permutation(train_indices)
            val_indices = np.random.permutation(val_indices)

        train_overall_losses = []
        self.lowest_loss_epoch = 0
        self.lowest_loss = np.inf

        for epoch in range(epochs):
            overall_loss = self.training_step(train_indices, epoch)
            train_overall_losses.append(overall_loss)
            
        logger.record_tabular("Reconstruction_epochs", epoch + 1)


    def training_step(self, indices, step):
        '''
        Computes a forward pass to encoder and decoder with sampling at the encoder.
        '''

        # get data from replay buffer and prepare for usage in encoder       
        batch = self.replay_buffer.sample_sac_data_batch(indices, self.batch_size)
        rewards = ptu.from_numpy(batch['rewards'])
        obs = ptu.from_numpy(batch['observations'])
        actions = ptu.from_numpy(batch['actions'])
        next_obs = ptu.from_numpy(batch['next_observations'])
        task_z = ptu.from_numpy(batch['task_indicators'])
        new_task_z = ptu.from_numpy(batch['next_task_indicators'])
        encoder_input = torch.cat((obs, actions, rewards, next_obs), dim=-1)
        # Forward pass through encoder
        _, z_means, z_vars = self.agent.infer_posterior(encoder_input)
        kl_div = self.agent.compute_kl_div(z_means, z_vars)
        kl_loss = 0.1 * kl_div

        task_z = task_z.view(-1, 1, task_z.shape[-1])
        pred_z, res_lo = self.z_model(task_z)
        data_shape = (task_z.shape[0], task_z.shape[1], task_z.shape[2])
        data_shape = list(data_shape)
        task_z = task_z.reshape([-1] + data_shape)
        task_z = task_z.reshape(task_z.shape[1], task_z.shape[0], -1, task_z.shape[-1])
        pred_z = pred_z.reshape([-1, pred_z.shape[1]] + data_shape[-2:])
        n_task_z = new_task_z.view(-1, 1, new_task_z.shape[-1])
        _, next_res_lo = self.target_z_model(n_task_z)
        target_res_lo = task_z + self.gamma * next_res_lo
        td_loss = torch.sqrt(torch.mean((res_lo - target_res_lo.detach()) ** 2))
        ptu.soft_update_from_to(
                self.z_model, self.target_z_model, self.soft_target_tau
            )

        self.z_optimizer.zero_grad()
        pred_loss = F.mse_loss(pred_z, task_z)
        z_loss = pred_loss + self.td_loss_coefficient*td_loss
        z_loss.backward()
        self.z_optimizer.step()

        self.optimizer_encoder.zero_grad()
        kl_loss.backward()
        self.optimizer_encoder.step()

        logger.record_tabular("td loss", np.mean(ptu.get_numpy(td_loss)))
        logger.record_tabular("pred loss", np.mean(ptu.get_numpy(pred_loss)))

        return ptu.get_numpy(z_loss)/self.batch_size