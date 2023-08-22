import numpy as np
import torch
from ddpg_vec import soft_update, hard_update, adjust_lr, Actor, Critic

from e3nn.o3 import Irreps
from models.e3nn_nn_models import GraphNetWithAttributes
from models.segnn import SEGNN 
from models.balanced_irreps import BalancedIrreps, WeightBalancedIrreps

class E3Actor(torch.nn.Module):
    def __init__(self, 
                net_type, graph_info, 
                hidden_features = 64, hidden_lmax = 2, num_layers = 3,
                number_of_basis = 10,
                subspace_type = "balanced",
    ):
        super(E3Actor, self).__init__() 

        self.net_type = net_type
        self.graph_info = graph_info

        if net_type == 'e3nn_nn_models':
            self.net_fn = GraphNetWithAttributes
            self.net = self.net_fn(
                irreps_node_input = graph_info['irreps_node_input'].__str__(),
                irreps_node_attr = graph_info['irreps_node_attr'].__str__(),
                irreps_edge_attr = graph_info['irreps_edge_attr'].__str__(),
                irreps_node_output = '1x1o', # 3D continous action
                max_radius = graph_info['max_radius'], number_of_basis = number_of_basis,  # Edge length embedding
                num_neighbors = graph_info['num_neighbors'],                  # Scaling constant for MessagePassing
                num_nodes = graph_info['num_nodes'], pool_nodes=True,         # Scaling constant for node pooling
                mul=hidden_features, layers=num_layers, lmax=hidden_lmax
            )
        elif net_type == 'segnn':
            self.net_fn = SEGNN
            if subspace_type == "weightbalanced":
                hidden_irreps = WeightBalancedIrreps(
                    Irreps("{}x0e".format(hidden_features)), graph_info['irreps_node_attr'], sh=True, lmax=hidden_lmax)
            elif subspace_type == "balanced":
                hidden_irreps = BalancedIrreps(hidden_lmax, hidden_features, True)
            else:
                raise Exception("Subspace type not found")
            self.net = self.net_fn(
                input_irreps = graph_info['irreps_node_input'],
                hidden_irreps = hidden_irreps,
                output_irreps = Irreps("1x1o"), # 3D continous action
                edge_attr_irreps = graph_info['irreps_edge_attr'],
                node_attr_irreps = graph_info['irreps_node_attr'],
                num_layers = num_layers,
                norm = None,
                pool = "avg",
                task = "graph",
                additional_message_irreps=graph_info['irreps_additional_message'],
            )
        else:
            raise NotImplementedError
    
    def normalize_u(self,u):
        # ToDo
        # Step 1. Compute l_p=2 norm of u, u_norm and the directional vector of u, u_dir
        # Step 2. Compute the action range within the l_p=1 box along u_dir, range
        # Step 3. Normalize u as range * sigmoid(u_norm) * u_dir
        
        return u
    
    def forward(self, obs_graph):
        u_3D = self.net(obs_graph) # 3D action
        u = u_3D[...,:2]           # 2D action

        return self.normalize_u(u)

class E3Critic(torch.nn.Module):
    def __init__(self, 
                net_type, graph_info, 
                hidden_features = 64, hidden_lmax = 2, num_layers = 3,
                number_of_basis = 10,
                subspace_type = "balanced",
    ):
        super(E3Critic, self).__init__() 

        self.net_type = net_type
        self.graph_info = graph_info    
        
        if net_type == 'e3nn_nn_models':
            self.net_fn = GraphNetWithAttributes
            self.net = self.net_fn(
                irreps_node_input = graph_info['irreps_node_input'].__str__(),
                irreps_node_attr = graph_info['irreps_node_attr'].__str__(),
                irreps_edge_attr = graph_info['irreps_edge_attr'].__str__(),
                irreps_node_output = '1x0e',
                max_radius = graph_info['max_radius'], number_of_basis = number_of_basis,  # Edge length embedding
                num_neighbors = graph_info['num_neighbors'],                  # Scaling constant for MessagePassing
                num_nodes = graph_info['num_nodes'], pool_nodes=True,         # Scaling constant for node pooling
                mul=hidden_features, layers=num_layers, lmax=hidden_lmax
            )
        elif net_type == 'segnn':
            self.net_fn = SEGNN
            if subspace_type == "weightbalanced":
                hidden_irreps = WeightBalancedIrreps(
                    Irreps("{}x0e".format(hidden_features)), graph_info['irreps_node_attr'], sh=True, lmax=hidden_lmax)
            elif subspace_type == "balanced":
                hidden_irreps = BalancedIrreps(hidden_lmax, hidden_features, True)
            else:
                raise Exception("Subspace type not found")
            self.net = self.net_fn(
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
        else:
            raise NotImplementedError

    def forward(self, state_action_n_graph):
        return self.net(state_action_n_graph)

class E3DDPG(object):
    def __init__(self,
        gamma, continuous, obs_dim, n_action, n_agent, obs_dims, action_range, 
        actor_graph_info, critic_graph_info,
        actor_type, critic_type, actor_hidden_size, critic_hidden_size, 
        lmax_attr, node_input_type,
        actor_lr, critic_lr, fixed_lr, num_episodes, 
        train_noise,
        target_update_mode,  tau,
        actor_clip_grad_norm = 0.5,
        device='cpu',
        scenario = None, world = None,
    ):
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.tau = tau
        self.continuous = continuous
        self.action_range = action_range
        if not self.continuous:
            raise NotImplementedError
        self.n_agent = n_agent
        self.n_action = n_action
        self.train_noise = train_noise
        self.target_update_mode = target_update_mode
        self.device = device

        self.actor_type = actor_type
        self.critic_type = critic_type

        self.scenario = scenario
        self.world = world
        self.lmax_attr = lmax_attr
        self.node_input_type = node_input_type
        
        # actor, actor_optim
        if actor_type in ['mlp',]:
            self.actor = Actor(actor_hidden_size, obs_dim, n_action, action_range).to(self.device)
            self.actor_target = Actor(actor_hidden_size, obs_dim, n_action, action_range).to(self.device)
        elif actor_type in ['segnn',]:
            self.actor = E3Actor(
                actor_type, actor_graph_info, 
                hidden_features = 64, hidden_lmax = 2, num_layers = 3,
                number_of_basis = 10,
                subspace_type = "balanced",
            ).to(self.device)
            self.actor_target = E3Actor(
                actor_type, actor_graph_info, 
                hidden_features = 64, hidden_lmax = 2, num_layers = 3,
                number_of_basis = 10,
                subspace_type = "balanced",
            ).to(self.device)
        else:
            raise NotImplementedError
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # centralized critic, critic_optim
        if critic_type in ['mlp', 'gcn_max']:
            self.critic = Critic(critic_hidden_size, np.sum(obs_dims),
                                n_action * n_agent, n_agent, critic_type).to(self.device)
            self.critic_target = Critic(critic_hidden_size, np.sum(obs_dims), 
                                n_action * n_agent, n_agent, critic_type).to(self.device)
        elif critic_type in ['segnn']:
            self.critic = E3Critic(
                critic_type, critic_graph_info, 
                hidden_features = 64, hidden_lmax = 2, num_layers = 3,
                number_of_basis = 10,
                subspace_type = "balanced",
            ).to(self.device)
            self.critic_target = E3Critic(
                critic_type, critic_graph_info, 
                hidden_features = 64, hidden_lmax = 2, num_layers = 3,
                number_of_basis = 10,
                subspace_type = "balanced",
            ).to(self.device)
        else:
            raise NotImplementedError
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Inititalize target 
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Exploration noise
        if self.continuous:
            from ounoise import OUNoise
            self.exploration_noise_n = [OUNoise(n_action) for _ in range(n_agent)]
        
        # Adjust lr
        self.fixed_lr = fixed_lr
        self.init_act_lr = actor_lr
        self.init_critic_lr = critic_lr
        self.actor_clip_grad_norm = actor_clip_grad_norm
        self.num_episodes = num_episodes
        self.start_episode = 0

    def reset_noise(self):
        if self.continuous:
            for exploration_noise in self.exploration_noise_n:
                exploration_noise.reset() 

    def scale_noise(self, scale):
        if self.continuous:
            for exploration_noise in self.exploration_noise_n:
                exploration_noise.scale = scale
    
    def adjust_lr(self, i_episode):
        adjust_lr(self.actor_optim, self.init_act_lr, i_episode, self.num_episodes, self.start_episode)
        adjust_lr(self.critic_optim, self.init_critic_lr, i_episode, self.num_episodes, self.start_episode)

    def gen_obs_graph(self, batch_obs, device = None):
        return self.scenario.gen_obs_graph(
            batch_obs, 
            self.lmax_attr, node_input_type = self.node_input_type,
            world = self.world, gen_graph_info = False, device = device,
        ) 
    def gen_state_action_n_graph(self, batch_s, batch_action_n, device = None):
        return self.scenario.gen_state_action_n_graph(
            batch_s, batch_action_n, 
            self.lmax_attr, node_input_type = self.node_input_type,
            world = self.world, gen_graph_info = False, device = device,
        )
    
    def select_action(self, batch_obs_n, action_noise=False, param_noise=False, grad=False):
        # batch_obs_n: (batch_size, n_agent, obs_dim) or (n_agent, obs_dim)
        if self.actor_type in ['mlp']:
            actor_in = torch.tensor(batch_obs_n, dtype=torch.get_default_dtype(), device= self.device)
            actor_in = actor_in.reshape((-1, self.obs_dim))
        elif self.actor_type in ['segnn']:
            actor_in = self.gen_obs_graph(batch_obs_n, device = self.device)
        else:
            raise NotImplementedError


        self.actor.eval()
        if param_noise:
            raise NotImplementedError
        else:
            mu = self.actor(actor_in)
        action_dim = mu.shape[-1]
        mu = torch.reshape(mu, (-1, self.n_agent, action_dim)) #(batch_size, n_agent, action_dim)
        self.actor.train()
        
        if not grad:
            mu = mu.data
        # action_noise for exploration: 
        # https://github.com/shariqiqbal2810/maddpg-pytorch/blob/40388d7c18e4662cf23c826d97e209df9003d86c/utils/agents.py#L55
        if not self.continuous:
            raise NotImplementedError
        else:
            action = mu
            if action_noise:
                noise = [exploration_noise.noise() for exploration_noise in self.exploration_noise_n]
                noise = torch.tensor(np.array(noise), dtype=torch.get_default_dtype(), device=self.device)
                noise = torch.unsqueeze(noise, dim = 0)  # (n_agent, action_dim) -> (1, n_agent, action_dim)
                action = action + noise                  # (batch_size, n_agent, action_dim)
            action = action.clamp(-self.action_range, self.action_range) # (-1, 1) for MPE

        if not grad:
            return action
        else:
            return action, mu
    
    def update_critic_parameters(self, batch, agent_id = 0, eval=False):
        
        if self.critic_type in ['mlp', 'gcn_max']:
            batch_next_obs_n = torch.tensor(batch.next_obs_n, dtype=torch.get_default_dtype(), device=self.device)
            critic_obs_in = batch_next_obs_n.reshape((-1, self.obs_dim * self.n_agent))
            batch_next_action_n = self.select_action(batch.next_obs_n, action_noise=self.train_noise)
            critic_action_in = batch_next_action_n.view(-1, self.n_action * self.n_agent)
            next_state_action_values = self.critic_target(
                critic_obs_in, critic_action_in)
        elif self.critic_type in ['segnn']:
            batch_next_action_n = self.select_action(
                batch.next_obs_n, action_noise=self.train_noise)
            batch_next_state_next_action_n_graph = self.gen_state_action_n_graph(
                batch.next_state, batch_next_action_n,
                device = self.device
            )
            next_state_action_values = self.critic_target(batch_next_state_next_action_n_graph)
        else:
            raise NotImplementedError

        reward_n_batch = torch.tensor(batch.reward_n, dtype=torch.get_default_dtype(), device = self.device)
        mask_n_batch = torch.tensor(batch.mask_n, dtype=torch.get_default_dtype(), device = self.device)
        reward_batch = reward_n_batch[:, agent_id].unsqueeze(1)
        mask_batch = mask_n_batch[:, agent_id].unsqueeze(1)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)
        
        if self.critic_type in ['mlp', 'gcn_max']:
            batch_obs_n = torch.tensor(batch.obs_n, dtype=torch.get_default_dtype(), device=self.device)
            critic_obs_in = batch_obs_n.reshape((-1, self.obs_dim * self.n_agent))
            batch_action_n = torch.tensor(batch.action_n, dtype=torch.get_default_dtype(), device=self.device)
            critic_action_in = batch_action_n.view(-1, self.n_action * self.n_agent)
            state_action_batch = self.critic(critic_obs_in, critic_action_in)
        elif self.critic_type in ['segnn']:
            batch_state_action_n_graph = self.gen_state_action_n_graph(
                batch.state, batch.action_n,
                device = self.device
            )
            state_action_batch = self.critic(batch_state_action_n_graph)
        else:
            raise NotImplementedError
        value_loss = ((state_action_batch - expected_state_action_batch) ** 2).mean()
        if eval:
            # return value_loss.item(), perturb_out
            return value_loss.item()
        
        self.critic_optim.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()
        if self.target_update_mode == 'soft':
            soft_update(self.critic_target, self.critic, self.tau)
        elif self.target_update_mode == 'hard':
            hard_update(self.critic_target, self.critic)
        # return value_loss.item(), perturb_out, unclipped_norm
        return value_loss.item()
    
    def update_actor_parameters(self, batch):

        # https://github.com/shariqiqbal2810/maddpg-pytorch/blob/40388d7c18e4662cf23c826d97e209df9003d86c/algorithms/maddpg.py#L134
        self.actor_optim.zero_grad()
        if not self.continuous:
            raise NotImplementedError
        else:
            batch_action_n, mu = self.select_action(batch.obs_n, action_noise=self.train_noise, grad=True)
        if self.critic_type in ['mlp', 'gcn_max']:
            critic_action_in = batch_action_n.reshape(-1, self.n_action * self.n_agent)
            batch_obs_n = torch.tensor(batch.obs_n, dtype=torch.get_default_dtype(), device=self.device)
            critic_obs_in = batch_obs_n.reshape((-1, self.obs_dim * self.n_agent))
            policy_loss = -self.critic(critic_obs_in, critic_action_in).mean()
        elif self.critic_type in ['segnn']:
            batch_state_action_n_graph = self.gen_state_action_n_graph(
                batch.state, batch_action_n,
                device = self.device
            )
            policy_loss = -self.critic(batch_state_action_n_graph).mean()
        else:
            raise NotImplementedError            

        if not self.continuous:
            raise NotImplementedError
        else:
            policy_loss = policy_loss + 1e-3 * (mu ** 2).mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_clip_grad_norm)
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return policy_loss.item()
    
    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            # if 'ln' in name: # ?
            #     pass
            param = params[name]
            param = param + torch.randn(param.shape) * param_noise.current_stddev
