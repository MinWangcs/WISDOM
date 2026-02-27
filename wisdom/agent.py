import torch
import torch.nn as nn
import numpy as np
from rlkit.torch.core import np_ify
import rlkit.torch.pytorch_util as ptu
from wisdom.scripted_policies import policies
from torch.nn import functional as F


class WISDOMAgent(nn.Module):
    def __init__(self,
                 encoder,
                 latent_dim,
                 policy
                 ):
        super(WISDOMAgent, self).__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.policy = policy
    
    def _product_of_gaussians(self, mus, sigmas_squared):
        '''
        compute mu, sigma of product of gaussians
        '''
        sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
        sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
        mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
        return mu, sigma_squared

    def compute_kl_div(self, z_means, z_vars):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(z_means), torch.unbind(z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        
        return kl_div_sum

    def infer_posterior(self, encoder_input):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.encoder(encoder_input)
        params = params.view(encoder_input.size(0), -1, self.latent_dim*2)
        # predict mean and variance of q(z | c)
        mu = params[..., :self.latent_dim]
        sigma_squared = F.softplus(params[..., self.latent_dim:])
        z_params = [self._product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
        z_means = torch.stack([p[0] for p in z_params])
        z_vars = torch.stack([p[1] for p in z_params])
        posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
        z = [d.rsample() for d in posteriors]
        z = torch.stack(z)
        return z, z_means, z_vars

    def get_action(self, encoder_input, state, deterministic=False, z_debug=None, env=None):
        a = encoder_input.shape[-1]
        encoder_input = encoder_input.view(1, -1, a)
        state = ptu.from_numpy(state).view(1, -1)
        task_z, _, _ = self.infer_posterior(encoder_input)
        if z_debug is not None:
            task_z = z_debug
        policy_input = torch.cat([state, task_z], dim=1)
        return self.policy.get_action(policy_input, deterministic=deterministic), np_ify(task_z.clone().detach())[0, :]

    def forward(self, context, state, deterministic=False):
        ''' given context, get statistics under the current policy of a set of observations '''
        task_z, _, _ = self.infer_posterior(context)
        # run policy, get log probs and new actions
        in_ = torch.cat([state, task_z.detach()], dim=1)
        policy_outputs = self.policy(in_, reparameterize=True, deterministic=deterministic, return_log_prob=True)

        return policy_outputs, task_z

class ScriptedPolicyAgent(nn.Module):
    def __init__(self,
                 encoder,
                 policy
                 ):
        super(ScriptedPolicyAgent, self).__init__()
        self.encoder = encoder
        self.policy = policy
        self.latent_dim = encoder.latent_dim

    def get_action(self, encoder_input, state, deterministic=False, z_debug=None, env=None):
        env_name = env.active_env_name
        oracle_policy = policies[env_name]()
        action = oracle_policy.get_action(state)
        return (action.astype('float32'), {}), np.zeros(self.latent_dim, dtype='float32')
