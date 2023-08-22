from utils import make_env, dict2csv
import numpy as np
import contextlib
import torch
from ckpt_plot.plot_curve import plot_result
import os
import time


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def eval_model_q(test_q, done_training, args):
    plot = {'rewards': [], 'steps': [], 'q_loss': [],
            'p_loss': [], 'final': [], 'abs': []}
    best_eval_reward = -100000000
    agent = None
    while True:
        if not test_q.empty():
            print('=================== start eval ===================')
            eval_env = make_env(args.scenario, args.continuous, arglist = args)
            eval_env.seed(args.seed + 10)
            eval_rewards = []
            good_eval_rewards = []
            agent, tr_log = test_q.get() # eval_agent is always on cpu
            agent.world, agent.scenario = eval_env.world, eval_env.scenario
            action_noise = True if not eval_env.continuous else False # action_noise is indeed True when discrete in PIC 
            with temp_seed(args.seed):
                for n_eval in range(args.num_eval_runs):
                    obs_n, info = eval_env.reset()
                    episode_reward = 0
                    episode_step = 0
                    n_agents = eval_env.n
                    agents_rew = [[] for _ in range(n_agents)]
                    while True:
                        action_n = agent.select_action(np.array(obs_n), 
                                                       action_noise=action_noise,
                                                       param_noise=False).squeeze().cpu().numpy()
                        next_obs_n, reward_n, done_n, _ = eval_env.step(action_n)
                        episode_step += 1
                        terminal = (episode_step >= args.num_steps)
                        episode_reward += np.sum(reward_n)
                        for i, r in enumerate(reward_n):
                            agents_rew[i].append(r)
                        obs_n = next_obs_n
                        if done_n[0] or terminal:
                            eval_rewards.append(episode_reward)
                            agents_rew = [np.sum(rew) for rew in agents_rew]
                            good_reward = np.sum(agents_rew)
                            good_eval_rewards.append(good_reward)
                            # if n_eval % 100 == 0:
                            #     print('test reward', episode_reward)
                            break
                
                if np.mean(eval_rewards) > best_eval_reward:
                    best_eval_reward = np.mean(eval_rewards)
                    agent.world, agent.scenario = None, None # not save these
                    torch.save({'agents': agent}, os.path.join(tr_log['exp_save_dir'], 'agents_best.ckpt'))
                    agent.world, agent.scenario = eval_env.world, eval_env.scenario # set these back

                plot['rewards'].append(np.mean(eval_rewards))
                plot['steps'].append(tr_log['total_numsteps'])
                plot['q_loss'].append(tr_log['value_loss'])
                plot['p_loss'].append(tr_log['policy_loss'])
                print("========================================================")
                print(
                    "Episode: {}, total numsteps: {}, {} eval runs, total time: {} s".
                        format(tr_log['i_episode'], tr_log['total_numsteps'], args.num_eval_runs,
                               time.time() - tr_log['start_time']))
                print("reward: avg {} std {}, average reward last 10: {}, best reward {}".format(np.mean(eval_rewards),
                                                                                              np.std(eval_rewards),
                                                                                              np.mean(plot['rewards'][
                                                                                                      -10:]),
                                                                                              best_eval_reward))
                plot['final'].append(np.mean(plot['rewards'][-10:]))
                plot['abs'].append(best_eval_reward)
                dict2csv(plot, os.path.join(tr_log['exp_save_dir'], 'train_curve.csv'))
                agent.world, agent.scenario = None, None # not save these
                torch.save({'agents': agent}, os.path.join(tr_log['exp_save_dir'], 
                                                           'agents_ep{}.ckpt'.format(tr_log['i_episode'] + 1)))
                agent.world, agent.scenario = eval_env.world, eval_env.scenario # set these back
                eval_env.close()
        if done_training.value and test_q.empty():
            if agent is not None:
                agent.world, agent.scenario = None, None # not save these
                torch.save({'agents': agent}, os.path.join(tr_log['exp_save_dir'], 'agents.ckpt'))
                agent.world, agent.scenario = eval_env.world, eval_env.scenario # set these back
            break