import hydra
import copy
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core import DeterministicActor, DDPGCritic
import utils.utils as utils

#for segnn
from e3nn.o3 import Irreps
#from models.e3nn_nn_models import GraphNetWithAttributes
from models.segnn import SEGNN
from models.balanced_irreps import BalancedIrreps, WeightBalancedIrreps

class E3Actor(torch.nn.Module):
    def __init__(self,
                graph_info,
                hidden_features = 64, hidden_lmax = 2, num_layers = 3,
                subspace_type = "balanced",
    ):
        super(E3Actor, self).__init__()

        self.graph_info = graph_info
        if subspace_type == "weightbalanced":
            hidden_irreps = WeightBalancedIrreps(
                Irreps("{}x0e".format(hidden_features)), graph_info['irreps_node_attr'], sh=True, lmax=hidden_lmax)
        elif subspace_type == "balanced":
            hidden_irreps = BalancedIrreps(hidden_lmax, hidden_features, True)
        else:
            raise Exception("Subspace type not found")
        self.net = SEGNN(
            input_irreps = graph_info['irreps_node_input'],
            hidden_irreps = hidden_irreps,
            output_irreps = Irreps("1x0e"), # 3D continous action
            edge_attr_irreps = graph_info['irreps_edge_attr'],
            node_attr_irreps = graph_info['irreps_node_attr'],
            num_layers = num_layers,
            norm = None,
            pool = "avg",
            task = "graph",
            additional_message_irreps=graph_info['irreps_additional_message'],
        )
        
    
    def normalize_u(self,u):
        # ToDo
        # Step 1. Compute l_p=2 norm of u, u_norm and the directional vector of u, u_dir
        # Step 2. Compute the action range within the l_p=1 box along u_dir, range
        # Step 3. Normalize u as range * sigmoid(u_norm) * u_dir
        
        return torch.tanh(u)
    
    def forward(self, obs):
        obs_graph =  utils.gen_obs_graph(
        batch_s = obs.cpu().numpy(),
        lmax_attr = 3, node_input_type = '', gen_graph_info = False, device = 'cuda'
        )
        #u_3D = self.net(obs_graph) # 3D action
        #u = u_3D[...,:1]           # 2D action
        u = self.net(obs_graph)
        return self.normalize_u(u)
        
class E3Critic(torch.nn.Module):
    def __init__(self, graph_info,
                hidden_features = 64, hidden_lmax = 2, num_layers = 3,
                subspace_type = "balanced",
    ):
        super(E3Critic, self).__init__()

        self.graph_info = graph_info
        
        if subspace_type == "weightbalanced":
            hidden_irreps = WeightBalancedIrreps(
                Irreps("{}x0e".format(hidden_features)), graph_info['irreps_node_attr'], sh=True, lmax=hidden_lmax)
        elif subspace_type == "balanced":
            hidden_irreps = BalancedIrreps(hidden_lmax, hidden_features, True)
        else:
            raise Exception("Subspace type not found")
        self.net = SEGNN(
            input_irreps = graph_info['irreps_node_input'],
            hidden_irreps = hidden_irreps,
            output_irreps = Irreps("1x0e"),
            edge_attr_irreps = graph_info['irreps_edge_attr'],
            node_attr_irreps = graph_info['irreps_node_attr'],
            num_layers = num_layers,
            norm = None,
            pool = "avg",
            task = "graph",
            additional_message_irreps=graph_info['irreps_additional_message'],
        )
        
    def forward(self, obs, action):

        #obs_action_n_graph
        obs_action_n_graph = utils.gen_obs_action_n_graph(
        batch_s = obs.cpu().numpy(), batch_action_n = action,
        lmax_attr = 3, node_input_type = '', gen_graph_info = False, device = 'cuda')
        self.net(obs_action_n_graph)
        return self.net(obs_action_n_graph)


class DDPGAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, linear_approx, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip,
                 clipped_noise, critic_type, actor_type, obs_graph_info, obs_action_n_graph_info):

        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.clipped_noise = clipped_noise
        self.stddev_clip = stddev_clip
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        #critic_type, actor_type
        self.critic_type = critic_type
        self.actor_type = actor_type
        # models
        
        if self.actor_type == 'mlp':
            self.actor = DeterministicActor(obs_shape, 1,
                                        hidden_dim, linear_approx).to(self.device)
        else:
            self.obs_graph_info = self.to_irreps(obs_graph_info)
            self.actor = E3Actor(
                self.obs_graph_info,
                hidden_features = 64, hidden_lmax = 2, num_layers = 3, #16, 2, 2
                subspace_type = "balanced",
            ).to(self.device)

        self.actor_target = copy.deepcopy(self.actor)

        if self.critic_type == 'mlp':
            self.critic = DDPGCritic(30, action_shape[0],
                                 hidden_dim, linear_approx).to(self.device)
        elif self.critic_type == 'segnn':
            #change graph_info from str to irreps
            self.obs_action_n_graph_info = self.to_irreps(obs_action_n_graph_info)
            self.critic = E3Critic(
                self.obs_action_n_graph_info,
                hidden_features = 64, hidden_lmax = 2, num_layers = 3, #16, 2, 2
                subspace_type = "balanced",
                ).to(self.device)
        else:
            raise NotImplementedError
            
        self.critic_target = copy.deepcopy(self.critic)
      
        
        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.actor_target.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
    
    def to_irreps(self, graph_info):
        graph_info['irreps_node_input'] = Irreps(graph_info['irreps_node_input'])
        graph_info['irreps_node_attr'] =  Irreps(graph_info['irreps_node_attr'])
        graph_info['irreps_edge_attr'] = Irreps(graph_info['irreps_edge_attr'])
        graph_info['irreps_additional_message'] = Irreps(graph_info['irreps_additional_message'])
        return graph_info

    def act(self, obs, step, eval_mode):
        obs = utils.obs_with_id(obs)
        obs = torch.as_tensor(obs, device=self.device)
        stddev = utils.schedule(self.stddev_schedule, step)
        #action = self.actor(obs.float().unsqueeze(0))
        action = self.actor(obs.float()).view((1,2))

        if eval_mode:
            action = action.cpu().numpy()[0]
        else:
            action = action.cpu().numpy()[0] + np.random.normal(0, stddev, size=self.action_dim)
            if step < self.num_expl_steps:
                action = np.random.uniform(-1.0, 1.0, size=self.action_dim)
        return action.astype(np.float32)

    def observe(self, obs, action):
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        action = torch.as_tensor(action, device=self.device).float().unsqueeze(0)

        q = self.critic(obs, action)

        return {
            'state': obs.cpu().numpy()[0],
            'value': q.cpu().numpy()[0]
        }

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()
        
        next_obs_id = utils.obs_with_id(next_obs)
        next_obs_id = next_obs_id.reshape((-1, next_obs_id.shape[-1]))
        next_obs_id =torch.as_tensor(next_obs_id, device=self.device).float()
        
        with torch.no_grad():
            if self.clipped_noise:
                # Select action according to policy and add clipped noise
                stddev = utils.schedule(self.stddev_schedule, step)
                noise = (torch.randn_like(action) * stddev).clamp(-self.stddev_clip, self.stddev_clip)
                next_action = (self.actor_target(next_obs_id).view(action.shape) + noise).clamp(-1.0, 1.0)
            else:
                next_action = self.actor_target(next_obs_id).view(action.shape)
            
            next_action = next_action.view((-1,2))
            
            # Compute the target Q value
            target_Q = self.critic_target(next_obs, next_action)
            target_Q = reward + discount * target_Q

        # Get current Q estimates
        current_Q = self.critic(obs, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q'] = current_Q.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # Optimize the critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()
        
        obs_id = utils.obs_with_id(obs)
        obs_id = obs_id.reshape((-1, obs_id.shape[-1]))
        obs_id = torch.as_tensor(obs_id, device=self.device).float()
        
        # Compute actor loss
        action = self.actor(obs_id).view((-1,2))
        actor_loss = -self.critic(obs, action).mean() + 1e-3 * (action ** 2).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        metrics['actor_loss'] = actor_loss.item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, _, point_cloud_obs, next_point_cloud_obs = utils.to_torch(
            batch, self.device)

        obs = obs.float()
        next_obs = next_obs.float()
        
        point_cloud_obs = point_cloud_obs.float()
        next_point_cloud_obs = next_point_cloud_obs.float()

        metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(self.update_critic(point_cloud_obs, action, reward, discount, next_point_cloud_obs, step))

        # update actor (delayed)
        if step % self.update_every_steps == 0:
            metrics.update(self.update_actor(point_cloud_obs.detach(), step))

            # update target networks
            utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
            utils.soft_update_params(self.actor, self.actor_target, self.critic_target_tau)
        return metrics

    def save(self, model_dir, step):
        model_save_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')
        model_save_dir.mkdir(exist_ok=True, parents=True)

        torch.save(self.actor.state_dict(), f'{model_save_dir}/actor.pt')
        torch.save(self.critic.state_dict(), f'{model_save_dir}/critic.pt')

    def load(self, model_dir, step):
        print(f"Loading the model from {model_dir}, step: {step}")
        model_load_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')

        self.actor.load_state_dict(
            torch.load(f'{model_load_dir}/actor.pt', map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f'{model_load_dir}/critic.pt', map_location=self.device)
        )
