import argparse
import random

import numpy as np
import torch

import os
import time
import torch.multiprocessing as mp
from multiprocessing import Queue
from multiprocessing.sharedctypes import Value

from utils import make_env, sample_action_n, gen_n_actions, gen_action_range
from utils import copy_actor_policy
from eval import eval_model_q

from e3_ddpg_vec import hard_update


if __name__ == '__main__':
    # env, seed
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='simple_spread_n6',
                        help='name of the environment to run')
    parser.add_argument('--continuous', default=False, action='store_true')
    parser.add_argument('--gamma', type=float, default=0.95, 
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--num_steps', type=int, default=25,
                        help='max episode length')
    parser.add_argument('--seed', type=int, default=1, )

    # actor net (parameter sharing)
    parser.add_argument('--actor_type', type=str, default='mlp', help="Supports [mlp, segnn]")
    # actor net: actor_type in [mlp,]
    parser.add_argument('--actor_hidden_size', type=int, default=128, )

    # critic net (centralized)
    parser.add_argument('--critic_type', type=str, default='mlp', help="Supports [mlp, gcn_max, segnn]")
    # critic net: critic_type in [mlp, gcn_max]
    parser.add_argument('--critic_hidden_size', type=int, default=128, )
    # actor/critic_net_type in [segnn]
    parser.add_argument('--lmax_attr', type=int, default=3,
                        help='lmax_attr for edge_attr')
    parser.add_argument("--node_input_type", type=str, default='',
                        help="Supports ['', 'pos']")

    # actor-crtic optimization
    parser.add_argument('--train_noise', default=False, action='store_true')
    parser.add_argument('--target_update_mode', default='soft', help='soft | hard | episodic')
    parser.add_argument('--tau', type=float, default=0.01, 
                        help='tau for soft target update')
    parser.add_argument('--num_episodes', type=int, default=62000,
                        help='total number of training episodes')
    parser.add_argument('--replay_size', type=int, default=1000000, )
    parser.add_argument('--batch_size', type=int, default=1024, )
    parser.add_argument('--steps_per_actor_update',  type=int, default=100)
    parser.add_argument('--steps_per_critic_update', type=int, default=100)
    parser.add_argument('--actor_updates_per_step',  type=int, default=8)
    parser.add_argument('--critic_updates_per_step', type=int, default=8)
    # actor-crtic optimization: lr
    parser.add_argument('--actor_lr',  type=float, default=1e-2)
    parser.add_argument('--actor_clip_grad_norm',  type=float, default=0.5)
    parser.add_argument('--critic_lr', type=float, default=1e-2)
    parser.add_argument('--fixed_lr', default=False, action='store_true')
    # actor-crtic optimization: training exploration noise
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)

    # eval
    parser.add_argument('--eval_freq',     type=int, default=1000)
    parser.add_argument('--num_eval_runs', type=int, default=200, help='number of runs per evaluation (default: 5)')

    # log
    parser.add_argument("--save_dir", type=str, default="./ckpt_plot",
                        help="directory in which training state and model should be saved")

    # cuda
    parser.add_argument('--cuda', default=False, action='store_true')

    args = parser.parse_args()
    
    args.exp_name =  args.scenario
    args.exp_name += '_' + ('continuous' if args.continuous else 'discrete') 
    args.exp_name += '_actor_'+ args.actor_type + '_lr_' +str(args.actor_lr)
    args.exp_name += '_critic_'+ args.critic_type + '_lr_' +str(args.critic_lr)
    args.exp_name += '_fixed_lr' if args.fixed_lr else ''
    args.exp_name += '_batch_size_'+ str(args.batch_size)
    args.exp_name += '_actor_clip_grad_norm_'+ str(args.actor_clip_grad_norm)
    args.exp_name += '_seed' + str(args.seed)
    
    print("=================Arguments==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    # env
    env = make_env(args.scenario, args.continuous, arglist = None)
    scenario, world = env.scenario, env.world
    env.n_agent = n_agent  = env.n # n_agent = 6
    env.n_actions = n_actions = gen_n_actions(env.action_space) # [2] * n_agent if continuous, [5] * n_agent if discrete
    n_action = n_actions[0]
    env.action_range  = action_range = gen_action_range(env.action_space)
    obs_dims = [env.observation_space[i].shape[0] for i in range(n_agent)] # [26] * n_agent
    obs_dim = obs_dims[0]
    obs_dims.insert(0, 0) #?
    obs_n, info = env.reset()
    sample_action_n = sample_action_n(env.action_space)
    _, obs_graph_info =  scenario.gen_obs_graph(
        batch_obs = obs_n[0], 
        lmax_attr = args.lmax_attr, node_input_type = args.node_input_type,
        world = world, gen_graph_info = True
    )
    _, state_action_n_graph_info = scenario.gen_state_action_n_graph(
        batch_s = info['state'], batch_action_n = sample_action_n, 
        lmax_attr = args.lmax_attr, node_input_type = args.node_input_type,
        world = world, gen_graph_info = True
    )

    # seed
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # E3DDPG agent, eval_agent
    from e3_ddpg_vec import E3DDPG
    agent = E3DDPG(
        gamma = args.gamma, 
        continuous = args.continuous, 
        obs_dim = obs_dim, 
        n_action = n_action, 
        n_agent = n_agent,
        obs_dims = obs_dims, 
        action_range = action_range,
        actor_graph_info = obs_graph_info, critic_graph_info = state_action_n_graph_info,
        actor_type = args.actor_type, critic_type = args.critic_type, 
        actor_hidden_size = args.actor_hidden_size,
        critic_hidden_size = args.critic_hidden_size, 
        lmax_attr = args.lmax_attr, node_input_type = args.node_input_type,
        actor_lr = args.actor_lr, critic_lr=args.critic_lr, 
        fixed_lr = args.fixed_lr, num_episodes = args.num_episodes,
        train_noise = args.train_noise,
        target_update_mode = args.target_update_mode, tau = args.tau,
        device=device,
        scenario = scenario, world = world,
    )

    eval_agent = E3DDPG(
        gamma = args.gamma, 
        continuous = args.continuous, 
        obs_dim = obs_dim, 
        n_action = n_action, 
        n_agent = n_agent,
        obs_dims = obs_dims, 
        action_range = action_range,
        actor_graph_info = obs_graph_info, critic_graph_info = state_action_n_graph_info,
        actor_type = args.actor_type, critic_type = args.critic_type, 
        actor_hidden_size = args.actor_hidden_size,
        critic_hidden_size = args.critic_hidden_size, 
        lmax_attr = args.lmax_attr, node_input_type = args.node_input_type,
        actor_lr = args.actor_lr, critic_lr=args.critic_lr, 
        fixed_lr = args.fixed_lr, num_episodes = args.num_episodes,
        train_noise = args.train_noise,
        target_update_mode = args.target_update_mode, tau = args.tau,
        device='cpu', # eval is always on cpu
        scenario = None, # to be set in eval.py
        world = None,    # to be set in eval.py
    )

    # replay
    from replay_memory import MAReplayMemory
    memory = MAReplayMemory(args.replay_size)

    rewards = []
    total_numsteps = 0
    updates = 0
    exp_save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)
    

    test_q = Queue()
    done_training = Value('i', False)
    p = mp.Process(target=eval_model_q, args=(test_q, done_training, args))
    p.start()
    
    start_time = time.time()

    copy_actor_policy(agent, eval_agent) # eval_agent is always on cpu
    tr_log = {  'exp_save_dir': exp_save_dir, 
                'total_numsteps': total_numsteps,
                'i_episode': 0, 'start_time': start_time,
                'value_loss': None, 'policy_loss': None,}
    test_q.put([eval_agent, tr_log])

    for i_episode in range(args.num_episodes):
        obs_n, info = env.reset()
        episode_reward = 0
        episode_step = 0
        agents_rew = [[] for _ in range(n_agent)]

        # reset OUnoise
        if env.continuous:
            explr_pct_remaining = max(0, args.n_exploration_eps - i_episode) / args.n_exploration_eps
            agent.scale_noise(args.final_noise_scale + (args.init_noise_scale - args.final_noise_scale) * explr_pct_remaining)
            agent.reset_noise()
        
        while True:
            action_n = agent.select_action(
                np.array(obs_n), 
                action_noise=True, param_noise=False
            ).squeeze().cpu().numpy()
            state = info['state']
            next_obs_n, reward_n, done_n, info = env.step(action_n)
            next_state = info['state']
            total_numsteps += 1
            episode_step += 1
            terminal = (episode_step >= args.num_steps)

            memory.push(         
                state,            # 'state'
                np.array(obs_n),  # 'obs_n'
                action_n,         # 'action_n'
                np.array([not done for done in done_n]),  # 'mask_n'
                next_state,            # 'next_state',
                np.array(next_obs_n),  # 'next_obs_n'
                np.array(reward_n),    # 'reward_n'
            )

            for i, r in enumerate(reward_n):
                agents_rew[i].append(r)
            episode_reward += np.sum(reward_n)
            obs_n = next_obs_n
            # n_update_iter = 5
            
            if len(memory) > args.batch_size:
                if total_numsteps % args.steps_per_actor_update == 0:
                    for _ in range(args.actor_updates_per_step):
                        batch = memory.sample(args.batch_size)
                        policy_loss = agent.update_actor_parameters(batch)
                    print('episode {}, p loss {}, p_lr {}'.format(
                        i_episode, policy_loss, agent.actor_optim.param_groups[0]['lr'])
                    )
                if total_numsteps % args.steps_per_critic_update == 0:
                    value_losses = []
                    for _ in range(args.critic_updates_per_step):
                        batch = memory.sample(args.batch_size)
                        value_losses.append(agent.update_critic_parameters(batch, i))
                    value_loss = np.mean(value_losses)
                    print('episode {}, q loss {},  q_lr {}'.format(
                        i_episode, value_loss, agent.critic_optim.param_groups[0]['lr'])
                    )
                    if args.target_update_mode == 'episodic':
                        hard_update(agent.critic_target, agent.critic)

            if done_n[0] or terminal:
                # print('train epidoe reward', episode_reward)
                episode_step = 0
                break

        if not args.fixed_lr:
            agent.adjust_lr(i_episode)
        
        rewards.append(episode_reward)
        
        if (i_episode + 1) % args.eval_freq == 0:
            copy_actor_policy(agent, eval_agent)
            tr_log = {  'exp_save_dir': exp_save_dir, 
                        'total_numsteps': total_numsteps,
                        'i_episode': i_episode, 'start_time': start_time,
                        'value_loss': value_loss, 'policy_loss': policy_loss
                    }
            test_q.put([eval_agent, tr_log])

    env.close()
    time.sleep(5)
    done_training.value = True

