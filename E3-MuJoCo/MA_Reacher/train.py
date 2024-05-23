import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import utils.dmc as dmc
import utils.utils as utils
from utils.logger import Logger
from utils.replay_buffer import ReplayBufferStorage, make_replay_loader
from utils.video import TrainVideoRecorder, VideoRecorder
import copy


torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg, obs_graph_info = None, obs_action_n_graph_info = None):
    cfg.agent.obs_shape = obs_spec
    if cfg.discrete_actions:
        cfg.agent.num_actions = action_spec.num_values
    else:
        cfg.agent.action_shape = action_spec.shape
    
    #graph_info for critic and actor
    cfg.agent.obs_graph_info = obs_graph_info
    cfg.agent.obs_action_n_graph_info = obs_action_n_graph_info
    
    return hydra.utils.instantiate(cfg.agent)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        obs_graph_info, obs_action_n_graph_info, obs_shape = self.setup()
        self.agent = make_agent(obs_shape,
                                self.train_env.action_spec(),
                                self.cfg.agent, obs_graph_info, obs_action_n_graph_info)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # some assertions
        utils.assert_agent(self.cfg['agent']['agent_name'], self.cfg['pixel_obs'])

        # create logger
        self.logger = Logger(self.work_dir)
        # create envs
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed,
                                  self.cfg.pixel_obs, self.cfg.discrete_actions)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed,
                                 self.cfg.pixel_obs, self.cfg.discrete_actions)
        
        sample_action = np.random.uniform(-1.0, 1.0, size=self.train_env.action_spec().shape)
        sample_time_step = self.eval_env.reset()
        sample_point_cloud_obs = self.get_point_cloud_obs(self.eval_env)
        
        _, obs_action_n_graph_info = utils.gen_obs_action_n_graph(
        batch_s = sample_point_cloud_obs, batch_action_n = sample_action,
        lmax_attr = 3, node_input_type = '', gen_graph_info = True)
       
        sample_point_cloud_obs_id = utils.obs_with_id(sample_point_cloud_obs)
        _, obs_graph_info =  utils.gen_obs_graph(
        batch_s = sample_point_cloud_obs_id,
        lmax_attr = 3, node_input_type = '', gen_graph_info = True
        )         
        
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None,
            fps=60 // self.cfg.action_repeat
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None,
            fps=60 // self.cfg.action_repeat
        )

        self.plot_dir = self.work_dir / 'plots'
        self.plot_dir.mkdir(exist_ok=True)
        self.model_dir = self.work_dir / 'models'
        self.model_dir.mkdir(exist_ok=True)

        # save cfg
        utils.save_cfg(self.cfg, self.work_dir)
        
        return obs_graph_info, obs_action_n_graph_info, sample_point_cloud_obs_id.shape[1]

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=episode == 0)
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    point_cloud_obs = self.get_point_cloud_obs(self.eval_env)
                    action = self.agent.act(point_cloud_obs,
                                        self.global_step,
                                        eval_mode=False)
                    
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}_{episode}.mp4')
                
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def get_point_cloud_obs(self, env):
        pos_target = env.physics.named.data.geom_xpos['target'][None,:]
        pos_finger = env.physics.named.data.geom_xpos['finger'][None,:]
        pos_hand = env.physics.named.data.geom_xpos['hand'][None,:]
        pos_arm = env.physics.named.data.geom_xpos['arm'][None,:]
        pos_root = env.physics.named.data.geom_xpos['root'][None,:]

        pos = np.concatenate((pos_target, pos_finger, pos_hand, pos_arm, pos_root), axis=0)
        
        vel_arm = env.physics.data.object_velocity("arm", "geom", local_frame=False)[0:1,:]
        vel_hand =  env.physics.data.object_velocity("hand", "geom", local_frame=False)[0:1,:]
        vel_finger =  env.physics.data.object_velocity("finger", "geom", local_frame=False)[0:1,:]
        vel_target =  env.physics.data.object_velocity("target", "geom", local_frame=False)[0:1,:]
        vel = np.concatenate((vel_arm, vel_hand,vel_finger,vel_target), axis=0)
        
        vel_origin = env.physics.data.object_velocity("root", "geom", local_frame=False)[0:1,:]
        
        vel = np.concatenate((vel, vel_origin), axis=0)
        
        point_cloud_obs = np.concatenate((pos, vel), axis=1)
                                                               
        point_cloud_obs = point_cloud_obs.reshape(-1)
        return point_cloud_obs
                                                                          
    def train(self, task_id=1):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames * task_id,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames + self.cfg.num_train_frames * (task_id - 1),
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        plot_every_step = utils.Every(self.cfg.plot_every_frames,
                                      self.cfg.action_repeat)
        save_every_step = utils.Every(self.cfg.save_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        
        point_cloud_obs = self.get_point_cloud_obs(self.train_env)
        
        self.replay_storage.add(time_step, point_cloud_obs)

        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                #self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                
                point_cloud_obs = self.get_point_cloud_obs(self.train_env)
                self.replay_storage.add(time_step, point_cloud_obs)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate   need to uncomment
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
                self.eval()

            if save_every_step(self.global_step):
                self.agent.save(self.model_dir, self.global_frame)

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                point_cloud_obs = self.get_point_cloud_obs(self.train_env)
                action = self.agent.act(point_cloud_obs,
                                        self.global_step,
                                        eval_mode=False)
            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            
            point_cloud_obs = self.get_point_cloud_obs(self.train_env)
                                                                  
            episode_reward += time_step.reward
            self.replay_storage.add(time_step, point_cloud_obs)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
