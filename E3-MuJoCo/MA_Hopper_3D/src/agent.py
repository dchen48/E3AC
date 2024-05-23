from numpy.lib.arraysetops import isin
import torch
import torch.nn.functional as F
import os
from common.agents import BaseAgent
from common.networks import MLPNetwork, PolicyNetworkFactory, get_optimizer
import numpy as np
from common import util, functional
from operator import itemgetter

from ModularActor import *
from ModularCritic import *
from StructureActor import StructurePolicy
from StructureCritic import CriticStructurePolicy
from SEActor import SEPolicy
from SECritic import SECritic
from MLPActor import MlpPolicy
from MLPCritic import MlpCritic
import utils

class Agent(BaseAgent):
    def __init__(self, args, obs_action_n_graph_info, obs_graph_info):
        super(Agent, self).__init__()
        #save parameters
        self.args = args
        '''
        if args.actor_type == 'set':
            actor = SEPolicy
        elif args.actor_type == 'swat':
            actor = StructurePolicy
        elif args.actor_type == 'smp':
            actor = ActorGraphPolicy
        elif args.actor_type == 'mlp':
            actor = MlpPolicy
        elif args.actor_type == 'mlp_v':
            actor = ActorVanilla
        elif args.actor_type == 'segnn':
            actor = E3Actor
        else:
            raise NotImplementedError
        '''
        print("args.limb_obs_size,args.limb_action_size",args.limb_obs_size,args.limb_action_size)
        
        '''
        self.actor = actor(
            args.limb_obs_size,
            args.limb_action_size,
            args.msg_dim,
            args.batch_size,
            args.max_action,
            args.max_children,
            args.disable_fold,
            args.td,
            args.bu,
            args,
        ).to(util.device)
        self.actor_target = actor(
            args.limb_obs_size,
            args.limb_action_size,
            args.msg_dim,
            args.batch_size,
            args.max_action,
            args.max_children,
            args.disable_fold,
            args.td,
            args.bu,
            args,
        ).to(util.device)
        '''
        #print(args)

        name = self.args.envs_train_names[0]
        state_dim = args.observation_space[name].shape[0]
        action_dim = args.action_space[name].shape[0] + 3
        #print(name, state_dim, action_dim)
        
        
        if args.actor_type == 'mlp_v':
            self.actor = DeterministicActor(
                state_dim+2,
                3,
                args.max_action,
            ).to(util.device)
            self.actor_target = DeterministicActor(
                state_dim+2,
                3,
                args.max_action,
            ).to(util.device)
        elif args.actor_type == 'segnn':
            self.actor = E3Actor(obs_graph_info, args.max_action).to(util.device)
            self.actor_target = E3Actor(obs_graph_info, args.max_action).to(util.device)
        else:
            raise NotImplementedError
        
        '''
        if args.critic_type == 'set':
            critic = SECritic
        elif args.critic_type == 'swat':
            critic = CriticStructurePolicy
        elif args.critic_type == 'smp':
            critic = CriticGraphPolicy
        elif args.critic_type == 'mlp':
            critic = MlpCritic
        elif args.critic_type == 'mlp_v':
            critic = CriticVanilla
        elif args.critic_type == 'segnn':
            critic = E3Critic
        else:
            raise NotImplementedError
        '''

        '''
        self.critic = critic(
            args.limb_obs_size,
            args.limb_action_size,
            args.msg_dim,
            args.batch_size,
            args.max_children,
            args.disable_fold,
            args.td,
            args.bu,
            args,
        ).to(util.device)
        self.critic_target = critic(
            args.limb_obs_size,
            args.limb_action_size,
            args.msg_dim,
            args.batch_size,
            args.max_children,
            args.disable_fold,
            args.td,
            args.bu,
            args,
        ).to(util.device)
        '''
        if args.critic_type == 'mlp_v':
            self.critic = DDPGCritic(
                state_dim,
                action_dim,
            ).to(util.device)
            self.critic_target = DDPGCritic(
                state_dim,
                action_dim,
            ).to(util.device)
        elif args.critic_type == 'segnn':
            self.critic = E3Critic(obs_action_n_graph_info).to(util.device)
            self.critic_target = E3Critic(obs_action_n_graph_info).to(util.device)
        else:
            raise NotImplementedError

        
        #sync network parameters
        functional.soft_update_network(self.actor, self.actor_target, 1.0)
        functional.soft_update_network(self.critic, self.critic_target, 1.0)

        #initialize optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        # self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
        # self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

        self.models2eval()

        #hyper-parameters
        self.tot_update_count = 0 
        self.target_smoothing_tau = args.agent.target_smoothing_tau
        self.reward_scale = args.agent.reward_scale


    def update(self, data_batch, it):
        obs_batch = data_batch['obs']
        action_batch = data_batch['action'] 
        next_obs_batch = data_batch['next_obs'] 
        reward_batch = data_batch['reward']
        done_batch = data_batch['done']
        
        batch_size = len(obs_batch)
        
        reward_batch = reward_batch * self.reward_scale

        # select action according to policy and add clipped noise
        with torch.no_grad():
            #print('action_batch: ', action_batch.shape)
            noise = torch.zeros_like(action_batch).data.normal_(0, self.args.policy_noise).to(util.device)
            noise = noise.clamp(-self.args.noise_clip, self.args.noise_clip)
            '''
            if self.args.actor_type == 'mlp_v':
                next_action = self.actor_target(next_obs_batch)# + noise
            elif self.args.actor_type == 'segnn':
                next_obs_batch_id = utils.obs_with_id(next_obs_batch)
                next_action = self.actor_target(next_obs_batch_id) #+ noise[:,3:9] #.cpu().numpy()
            else:
                raise NotImplementedError
            '''
            
            next_obs_batch_id = utils.obs_with_id(next_obs_batch)
            if self.args.actor_type == 'mlp_v':
                next_obs_batch_id = torch.FloatTensor(next_obs_batch_id).to(util.device)
            next_action = self.actor_target(next_obs_batch_id)
            
            next_action = next_action.view(batch_size, -1)
            next_dummy_action = torch.zeros((len(next_action),3)).to(next_action.device)
            next_action = torch.cat((next_dummy_action, next_action), dim = -1)
                        
            next_action = next_action + noise
            next_action = next_action.clamp(
                -self.args.max_action, self.args.max_action
            )

            # Qtarget = reward + discount * min_i(Qi(next_state, pi(next_state)))
            target_Q = self.critic_target(next_obs_batch, next_action)
            #target_Q = torch.min(target_Q1, target_Q2)
            # print(reward_batch.shape,done_batch.shape,target_Q.shape)
            target_Q = reward_batch + ((1.0-done_batch) * self.args.discount * target_Q)

        # get current Q estimates
        current_Q = self.critic(obs_batch, action_batch)
        # print(current_Q1.shape,target_Q.shape)
        # compute critic loss

        #critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
        #    current_Q2, target_Q
        #)
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.args.grad_clipping_value > 0:
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.args.grad_clipping_value
            )
        self.critic_optimizer.step()

        loss_dict = {
            "loss/critic_loss": critic_loss, 
            "misc/train_reward_mean": torch.mean(reward_batch).item(),
            "misc/train_reward_var": torch.var(reward_batch).item()
        }

        # delayed policy updates
        if it % self.args.policy_freq == 0:
            # compute actor loss
            
            '''
            if self.args.actor_type == 'mlp_v':
                cur_action = self.actor_target(obs_batch)
            elif self.args.actor_type == 'segnn':
                obs_batch_id = utils.obs_with_id(obs_batch)
                cur_action = self.actor(obs_batch_id)#.cpu().numpy()
            else:
                raise NotImplementedError
            '''
            
            obs_batch_id = utils.obs_with_id(obs_batch)
            if self.args.actor_type == 'mlp_v':
                obs_batch_id = torch.FloatTensor(obs_batch_id).to(util.device)
            cur_action = self.actor(obs_batch_id)#.cpu().numpy()
        
            cur_action = cur_action.view(batch_size, -1)
            cur_dummy_action = torch.zeros((len(cur_action),3)).to(cur_action.device)
            cur_action = torch.cat((cur_dummy_action, cur_action), dim = -1)

            actor_loss = -self.critic(obs_batch, cur_action).mean()

            # optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.args.grad_clipping_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.args.grad_clipping_value
                )
            self.actor_optimizer.step()

            self.try_update_target_network()

            loss_dict.update({"loss/actor_loss": actor_loss})
        

        return loss_dict

    def try_update_target_network(self):
        functional.soft_update_network(self.critic, self.critic_target, self.target_smoothing_tau)
        functional.soft_update_network(self.actor, self.actor_target, self.target_smoothing_tau)

    @torch.no_grad()  
    def select_action(self, obs, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None,]
    
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(util.device)
        
        batch_size = len(obs)
        
        obs = utils.obs_with_id(obs)
        if self.args.actor_type == 'mlp_v':
            obs = torch.FloatTensor(obs).to(util.device)
        action = self.actor(obs)#.cpu().numpy()
        action = action.view(batch_size, -1)
        dummy_action = torch.zeros((len(action),3)).to(action.device)
        action = torch.cat((dummy_action, action), dim = -1)
        action = action.cpu().numpy()

        return action


    def change_morphology(self, graph):
        self.actor.change_morphology(graph)
        self.actor_target.change_morphology(graph)
        self.critic.change_morphology(graph)
        self.critic_target.change_morphology(graph)

    def models2eval(self):
        self.actor = self.actor.eval()
        self.actor_target = self.actor_target.eval()
        self.critic = self.critic.eval()
        self.critic_target = self.critic_target.eval()

    def models2train(self):
        self.actor = self.actor.train()
        self.actor_target = self.actor_target.train()
        self.critic = self.critic.train()
        self.critic_target = self.critic_target.train()

        



