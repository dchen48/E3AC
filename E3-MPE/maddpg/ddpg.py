import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from models import model_factory

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def adjust_lr(optimizer, init_lr, episode_i, num_episode, start_episode):
    if episode_i < start_episode:
        return init_lr
    lr = init_lr * (1 - (episode_i - start_episode) / (num_episode - start_episode))
    #lr = init_lr * (0.1 ** episode_i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


nn.LayerNorm = LayerNorm


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        # self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.ln2 = nn.LayerNorm(hidden_size)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        mu = self.mu(x)
        return mu


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs, num_agents, critic_type='mlp', agent_id=0):
        super(Critic, self).__init__()

        self.num_agents = num_agents
        self.critic_type = critic_type
        sa_dim = int((num_inputs + num_outputs) / num_agents)
        self.agent_id = agent_id
        self.net_fn = model_factory.get_model_fn(critic_type)
        self.net = self.net_fn(sa_dim, num_agents, hidden_size)

    def forward(self, inputs, actions):
        bz = inputs.size()[0]
        #s_n = inputs.view(bz, -1, self.num_agents)
        #a_n = actions.view(bz, -1, self.num_agents)
        s_n = inputs.view(bz, self.num_agents, -1)
        a_n = actions.view(bz, self.num_agents, -1)
        x = torch.cat((s_n, a_n), dim=2)
        #x = torch.transpose(x, 1, 2)
        V = self.net(x)
        return V


class DDPG(object):
    def __init__(self, gamma, tau, hidden_size, obs_dim, n_action, n_agent, obs_dims, agent_id, actor_lr, critic_lr,
                 fixed_lr, critic_type, train_noise, num_episodes, num_steps,
                 critic_dec_cen, target_update_mode='soft', device='cpu'):
        self.device = device
        self.obs_dim = obs_dim
        self.n_agent = n_agent
        self.n_action = n_action
        self.actor = Actor(hidden_size, obs_dim, n_action).to(self.device)
        self.actor_target = Actor(hidden_size, obs_dim, n_action).to(self.device)
        self.actor_perturbed = Actor(hidden_size, obs_dim, n_action)
        self.actor_optim = Adam(self.actor.parameters(),
                                lr=actor_lr, weight_decay=0)

        if critic_dec_cen == 'decen':
            self.critic = Critic(hidden_size, obs_dims[agent_id + 1], n_action, 1, critic_type, agent_id).to(self.device)
            self.critic_target = Critic(hidden_size, obs_dims[agent_id + 1], n_action, 1, critic_type, agent_id).to(self.device)
        else:
            self.critic = Critic(hidden_size, np.sum(obs_dims),
                                 n_action * n_agent, n_agent, critic_type, agent_id).to(self.device)
            self.critic_target = Critic(hidden_size, np.sum(
                obs_dims), n_action * n_agent, n_agent, critic_type, agent_id).to(self.device)
        critic_n_params = sum(p.numel() for p in self.critic.parameters())
        print('# of critic params', critic_n_params)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
        self.fixed_lr = fixed_lr
        self.init_act_lr = actor_lr
        self.init_critic_lr = critic_lr
        self.num_episodes = num_episodes
        #self.start_episode = num_episodes / 2
        self.start_episode = 0
        self.num_steps = num_steps
        self.actor_scheduler = LambdaLR(self.actor_optim, lr_lambda=self.lambda1)
        self.critic_scheduler = LambdaLR(self.critic_optim, lr_lambda=self.lambda1)
        self.gamma = gamma
        self.tau = tau
        self.train_noise = train_noise
        self.obs_dims_cumsum = np.cumsum(obs_dims)
        self.critic_dec_cen = critic_dec_cen
        self.agent_id = agent_id
        self.debug = False
        self.target_update_mode = target_update_mode
        # Make sure target is with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)


    def adjust_lr(self, i_episode):
        adjust_lr(self.actor_optim, self.init_act_lr, i_episode, self.num_episodes, self.start_episode)
        adjust_lr(self.critic_optim, self.init_critic_lr, i_episode, self.num_episodes, self.start_episode)

    def lambda1(self, step):
        start_decrease_step = ((self.num_episodes / 2)
                               * self.num_steps) / 100
        max_step = (self.num_episodes * self.num_steps) / 100
        return 1 - ((step - start_decrease_step) / (
                max_step - start_decrease_step)) if step > start_decrease_step else 1

    def select_action(self, state, action_noise=None, param_noise=False, grad=False):
        self.actor.eval()
        if param_noise:
            mu = self.actor_perturbed((Variable(state)))
        else:
            mu = self.actor((Variable(state)))

        self.actor.train()
        if not grad:
            mu = mu.data

        if action_noise:
            noise = np.log(-np.log(np.random.uniform(0, 1, mu.size())))
            try:
                mu -= torch.Tensor(noise).to(self.device)
            except (AttributeError, AssertionError):
                mu -= torch.Tensor(noise)

        action = F.softmax(mu, dim=1)
        if not grad:
            return action
        else:
            return action, mu

    def update_critic_parameters(self, batch, agent_id, shuffle=None, eval=False):
        state_batch = Variable(torch.cat(batch.state)).to(self.device)
        action_batch = Variable(torch.cat(batch.action)).to(self.device)
        reward_batch = Variable(torch.cat(batch.reward)).to(self.device)
        mask_batch = Variable(torch.cat(batch.mask)).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        if shuffle == 'shuffle':
            rand_idx = np.random.permutation(self.n_agent)
            #shuffle_others_loc(list(rand_idx), state_batch)
            #shuffle_others_loc(list(rand_idx), next_state_batch)
            new_state_batch = state_batch.view(-1, self.n_agent, self.obs_dim)
            state_batch = new_state_batch[:, rand_idx, :].view(-1, self.obs_dim * self.n_agent)
            new_next_state_batch = next_state_batch.view(-1, self.n_agent, self.obs_dim)
            next_state_batch = new_next_state_batch[:, rand_idx, :].view(-1, self.obs_dim * self.n_agent)
            new_action_batch = action_batch.view(-1, self.n_agent, self.n_action)
            action_batch = new_action_batch[:, rand_idx, :].view(-1, self.n_action * self.n_agent)

        next_actions = []
        for k in range(self.n_agent):
            next_obs_batch = next_state_batch[:,
                             self.obs_dims_cumsum[k]: self.obs_dims_cumsum[k + 1]]
            next_actions.append(self.select_action(
                next_obs_batch, action_noise=self.train_noise))
        next_action_batch = torch.cat(next_actions, dim=1)
        if self.critic_dec_cen == 'decen':
            next_state_action_values = self.critic_target(
                next_state_batch[:, self.obs_dims_cumsum[self.agent_id]: self.obs_dims_cumsum[self.agent_id + 1]],
                next_action_batch[:, self.agent_id * self.n_action: (self.agent_id + 1) * self.n_action])
        else:
            next_state_action_values = self.critic_target(
                next_state_batch, next_action_batch)

        reward_batch = reward_batch[:, agent_id].unsqueeze(1)
        mask_batch = mask_batch[:, agent_id].unsqueeze(1)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)
        self.critic_optim.zero_grad()
        if self.critic_dec_cen == 'decen':
            state_action_batch = self.critic(
                (state_batch[:, self.obs_dims_cumsum[self.agent_id]: self.obs_dims_cumsum[self.agent_id + 1]]),
                (action_batch[:, self.agent_id * self.n_action: (self.agent_id + 1) * self.n_action]))
        else:
            state_action_batch = self.critic(state_batch, action_batch)
        perturb_out = 0
        if self.debug:
            critic_clone = self.critic
            state_action_batch = critic_clone(state_batch, action_batch)
            #perturbed_state_batch = torch.Tensor(np.random.normal(state_batch, 0.1))
            #perturbed_action_batch = torch.Tensor(np.random.normal(action_batch, 0.1))
            state_action_batch_perturb = critic_clone(1.01 * state_batch, action_batch)
            perturb_out = F.mse_loss(state_action_batch, state_action_batch_perturb).item()
            print('ori - perturb ', perturb_out)
        value_loss = (
                (state_action_batch - expected_state_action_batch) ** 2).mean()
        if eval:
            return value_loss.item(), perturb_out
        value_loss.backward()
        unclipped_norm = clip_grad_norm_(self.critic.parameters(), 0.5)

        self.critic_optim.step()
        if self.target_update_mode == 'soft':
            soft_update(self.critic_target, self.critic, self.tau)
        elif self.target_update_mode == 'hard':
            hard_update(self.critic_target, self.critic)
        return value_loss.item(), perturb_out, unclipped_norm

    def update_actor_parameters(self, batch, agent_id, shuffle=None):
        state_batch = Variable(torch.cat(batch.state)).to(self.device)
        action_batch = Variable(torch.cat(batch.action)).to(self.device)
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        if shuffle == 'shuffle':
            rand_idx = np.random.permutation(self.n_agent)
            #shuffle_others_loc(list(rand_idx), state_batch)
            #shuffle_others_loc(list(rand_idx), next_state_batch)
            new_state_batch = state_batch.view(-1, self.n_agent, self.obs_dim)
            state_batch = new_state_batch[:, rand_idx, :].view(-1, self.obs_dim * self.n_agent)




        #next_actions = []
        #for k in range(self.n_agent):
        #    next_obs_batch = next_state_batch[:,
        #                     self.obs_dims_cumsum[k]: self.obs_dims_cumsum[k + 1]]
        #    next_actions.append(self.select_action(
        #        next_obs_batch, action_noise=self.train_noise))

        self.actor_optim.zero_grad()
        actions = []
        for k in range(self.n_agent):
            obs_batch = state_batch[:, self.obs_dims_cumsum[k]: self.obs_dims_cumsum[k + 1]]
            if k == agent_id:
                action, logit = self.select_action(
                    obs_batch, action_noise=self.train_noise, grad=True)
            else:
                action = self.select_action(
                    obs_batch, action_noise=self.train_noise, grad=False)
            actions.append(action)
        action_batch_n = torch.cat(actions, dim=1)
        if self.critic_dec_cen == 'decen':
            policy_loss = -self.critic(
                state_batch[:, self.obs_dims_cumsum[self.agent_id]: self.obs_dims_cumsum[self.agent_id + 1]],
                action_batch_n[:, self.agent_id * self.n_action: (self.agent_id + 1) * self.n_action])
        else:
            policy_loss = -self.critic(state_batch, action_batch_n)
        policy_loss = policy_loss.mean() + 1e-3 * (logit ** 2).mean()
        policy_loss.backward()
        clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)


        return policy_loss.item()


    def update_parameters(self, batch, agent_id, shuffle=None):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = torch.cat(batch.next_state)
        if shuffle == 'shuffle':
            rand_idx = np.random.permutation(self.n_agent)
            #shuffle_others_loc(list(rand_idx), state_batch)
            #shuffle_others_loc(list(rand_idx), next_state_batch)
            new_state_batch = state_batch.view(-1, self.n_agent, self.obs_dim)
            state_batch = new_state_batch[:, rand_idx, :].view(-1, self.obs_dim * self.n_agent)
            new_next_state_batch = next_state_batch.view(-1, self.n_agent, self.obs_dim)
            next_state_batch = new_next_state_batch[:, rand_idx, :].view(-1, self.obs_dim * self.n_agent)
            new_action_batch = action_batch.view(-1, self.n_agent, self.n_action)
            action_batch = new_action_batch[:, rand_idx, :].view(-1, self.n_action * self.n_agent)
        #elif shuffle == 'sort':
        #    idxs = sort_state_batch(state_batch)
        #    for i in range(state_batch.size()[0]):
        #        shuffle_others_loc(list(idxs[i]), next_state_batch[i].view(1, -1))
        #        next_state_batch[i] = next_state_batch[i].view(self.n_agent, self.obs_dim)[idxs[i]].view(-1)
        #        action_batch[i] = action_batch[i].view(self.n_agent, self.n_action)[idxs[i]].view(-1)

        next_actions = []
        for k in range(self.n_agent):
            next_obs_batch = next_state_batch[:,
                             self.obs_dims_cumsum[k]: self.obs_dims_cumsum[k + 1]]
            next_actions.append(self.select_action(
                next_obs_batch, action_noise=self.train_noise))
        next_action_batch = torch.cat(next_actions, dim=1)
        next_state_action_values = self.critic_target(
            next_state_batch, next_action_batch)

        reward_batch = reward_batch[:, agent_id].unsqueeze(1)
        mask_batch = mask_batch[:, agent_id].unsqueeze(1)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)
        self.critic_optim.zero_grad()
        state_action_batch = self.critic((state_batch), (action_batch))

        value_loss = (
                (state_action_batch - expected_state_action_batch) ** 2).mean()
        value_loss.backward()
        clip_grad_norm_(self.critic.parameters(), 0.5)

        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actions = []
        for k in range(self.n_agent):
            obs_batch = state_batch[:, self.obs_dims_cumsum[k]: self.obs_dims_cumsum[k + 1]]
            if k == agent_id:
                action, logit = self.select_action(
                    obs_batch, action_noise=self.train_noise, grad=True)
            else:
                action = self.select_action(
                    obs_batch, action_noise=self.train_noise, grad=False)
            actions.append(action)
        action_batch_n = torch.cat(actions, dim=1)
        policy_loss = -self.critic(state_batch, action_batch_n)

        policy_loss = policy_loss.mean() + 1e-3 * (logit ** 2).mean()
        policy_loss.backward()
        clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)


        return value_loss.item(), policy_loss.item()

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            param += torch.randn(param.shape) * param_noise.current_stddev

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/ddpg_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
