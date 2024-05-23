import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from utils import *
import os


class ModularEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml):
        self.xml = xml
        self.target = np.zeros(2)
        mujoco_env.MujocoEnv.__init__(self, xml, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        torso_quat = self.sim.data.qpos[3:7]
        torso_rotmat = quat2mat(torso_quat)
        heading = np.arctan2(torso_rotmat[1,0], torso_rotmat[0,0])
        pitch = np.arctan2(-torso_rotmat[2,0], np.sqrt(torso_rotmat[2,1]**2 + torso_rotmat[2,2]**2))
        roll = np.arctan2(torso_rotmat[2,1], torso_rotmat[2,2])
        heading  = [np.cos(heading), np.sin(heading)]
        pos_before = self.data.get_body_xpos("torso")[:2].copy()
        dist_before = np.linalg.norm(self.target-pos_before)
        self.do_simulation(a, self.frame_skip)
        pos_after = self.data.get_body_xpos("torso")[:2].copy()
        dist_after = np.linalg.norm(self.target-pos_after)
        torso_height = self.sim.data.qpos[2]
        alive_bonus = 1.0
        reward = (dist_before - dist_after) / self.dt
        reward += np.dot(pos_after -  pos_before, heading) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (
            torso_height > 0.8
            and torso_height < 2.0
            and (abs(pitch) < 1)
            and (abs(roll) < 1)
        )
        ob = self._get_obs()

        if dist_after < 1.0 and np.linalg.norm(self.target) > 1:
            rad = self.np_random.uniform(low=-np.pi, high=np.pi)
            self.target = np.array([np.cos(rad),np.sin(rad)])*10000
        return ob, reward, done, {"dist":dist_after}

    def _get_obs(self):
        #torso_x_pos = self.data.get_body_xpos("torso")
        #torso_v = np.clip(self.data.get_body_xvelp("torso"), -10, 10)
        #dir = self.target-torso_x_pos[:2]
        #dir = dir / np.linalg.norm(dir)
        torso_obs = np.zeros(6)
        torso_obs[0:3] = np.concatenate([self.target, np.zeros(1)])
        torso_obs[3] = 0.0
        torso_obs[4] = 0.0
        torso_obs[5] = -9.81
        #torso_obs[6:9] = torso_v
        #torso_obs[9:12] = torso_x_pos

        def _get_obs_per_limb(b):
            obs = np.zeros(15)
            obs[0:3] = self.data.get_body_xpos(b)
            obs[3:6] = np.clip(self.data.get_body_xvelp(b), -10, 10)
            if b != "torso":
                obs[6:9] = self.data.get_joint_xaxis(b+'_joint_x')
                obs[9:12] = self.data.get_joint_xaxis(b+'_joint_y')
                obs[12:15] = self.data.get_joint_xaxis(b+'_joint_z')
            return obs

        full_obs = np.concatenate(
            [torso_obs] + [_get_obs_per_limb(b) for b in self.model.body_names[1:]]
        )
        return full_obs.ravel()

    def reset_model(self):
        qpos = self.init_qpos
        rad = self.np_random.uniform(low=-np.pi, high=np.pi)
        rad = rad/2 
        qpos[3] = np.cos(rad)
        qpos[6] = np.sin(rad)
        self.set_state(
            qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        rad = self.np_random.uniform(low=-np.pi, high=np.pi)
        self.target = np.array([np.cos(rad),np.sin(rad)])*10000
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
