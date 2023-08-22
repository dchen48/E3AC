import random
from collections import namedtuple

import numpy as np

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward')
)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def shuffle(self):
        random.shuffle(self.memory)

    def __len__(self):
        return len(self.memory)

MATransition = namedtuple(
    'MATransition', ('state', 'obs_n',  'action_n',  'mask_n', 'next_state', 'next_obs_n', 'reward_n')
)

class MAReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = MATransition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        tansitions = random.sample(self.memory, batch_size)
        batch = MATransition(*zip(*tansitions))
        
        batch_state = {}
        for k in batch.state[0]:
            batch_state[k] = np.array([batch.state[i][k] for i in range(batch_size)])
        batch_obs_n = np.array(batch.obs_n)
        batch_action_n = np.array(batch.action_n)
        batch_mask_n = np.array(batch.mask_n)
        batch_next_state = {}
        for k in batch.next_state[0]:
            batch_next_state[k] = np.array([batch.next_state[i][k] for i in range(batch_size)])
        batch_next_obs_n = np.array(batch.next_obs_n)
        batch_reward_n = np.array(batch.reward_n)

        return MATransition(
            batch_state, batch_obs_n, batch_action_n, batch_mask_n, batch_next_state, batch_next_obs_n, batch_reward_n
        )

    # def shuffle(self):
    #     random.shuffle(self.memory)

    def __len__(self):
        return len(self.memory)

