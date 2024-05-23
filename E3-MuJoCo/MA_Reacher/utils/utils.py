import random
import re
import time
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

from e3nn.o3 import Irreps, spherical_harmonics


_STATE_AGENTS = ['hpg', 'ddpg', 'td3', 'sac']
_PIXEL_AGENTS = ['hpg', 'drqv2', 'sacae', 'dbc', 'deepmdp']


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def assert_agent(agent_name, pixel_obs):
    agent_name = agent_name.partition('_')[0]
    if pixel_obs:
        assert agent_name in _PIXEL_AGENTS, f"{agent_name} does not support pixel observations"
    else:
        assert agent_name in _STATE_AGENTS, f"{agent_name} does not support state observations"


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def save_cfg(cfg, dir):
    with open(os.path.join(dir, 'cfg.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f.name)


def get_last_model(model_dir):
    if not isinstance(model_dir, Path):
        model_dir = Path(model_dir)
    # return the step of the last saved model
    last_saved = sorted(model_dir.glob(f'**/'))[-1]
    last_step = str(last_saved.stem).partition('_')[-1]
    return int(last_step)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)

def obs_with_id(obs):
    if torch.is_tensor(obs):
        o_id = obs.cpu().numpy()
    else:
        o_id = obs
    if len(o_id.shape) == 1:
        pos_target = o_id[0:3][None,:]
        vel_arm = o_id[3:6][None,:]
        pos_finger = o_id[6:9][None,:]
        vel_hand = o_id[9:12][None,:]
        pos_hand = o_id[12:15][None,:]
        vel_finger = o_id[15:18][None,:]
        pos_arm = o_id[18:21][None,:]
        vel_target = o_id[21:24][None,:]
        pos_root = o_id[24:27][None,:]
        vel_root = o_id[27:30][None,:]
        o_id_pos_0 = np.concatenate((pos_target, pos_arm, pos_root), axis=0)
        o_id_vel_0 = np.concatenate((vel_target, vel_arm, vel_root), axis=0)
        o_id_0 = np.concatenate((o_id_pos_0, o_id_vel_0), axis=1)
        o_id_0 = o_id_0.reshape(-1)[None,:]
        o_id_pos_1 = np.concatenate((pos_target, pos_finger, pos_hand), axis=0)
        o_id_vel_1 = np.concatenate((vel_target, vel_finger, vel_hand), axis=0)
        o_id_1 = np.concatenate((o_id_pos_1, o_id_vel_1), axis=1)
        o_id_1 = o_id_1.reshape(-1)[None,:]
        o_id = np.concatenate((o_id_0, o_id_1), axis=0)
        id = np.identity(2)
        o_id = np.concatenate((id, o_id), axis = 1)
    elif len(o_id.shape) == 2:
        #o_id = o_id[:,None,:]
        #o_id = np.tile(o_id, (1,2,1))
        bz = o_id.shape[0]
        pos_target = o_id[:,0:3][:,None,:]
        vel_arm = o_id[:,3:6][:,None,:]
        pos_finger = o_id[:,6:9][:,None,:]
        vel_hand = o_id[:,9:12][:,None,:]
        pos_hand = o_id[:,12:15][:,None,:]
        vel_finger = o_id[:,15:18][:,None,:]
        pos_arm = o_id[:,18:21][:,None,:]
        vel_target = o_id[:,21:24][:,None,:]
        pos_root = o_id[:,24:27][:,None,:]
        vel_root = o_id[:,27:30][:,None,:]
        
        o_id_pos_0 = np.concatenate((pos_target, pos_arm, pos_root), axis=1)
        o_id_vel_0 = np.concatenate((vel_target, vel_arm, vel_root), axis=1)
        o_id_0 = np.concatenate((o_id_pos_0, o_id_vel_0), axis=2)
        o_id_0 = o_id_0.reshape((bz,-1))[:,None,:]
        o_id_pos_1 = np.concatenate((pos_target, pos_finger, pos_hand), axis=1)
        o_id_vel_1 = np.concatenate((vel_target, vel_finger, vel_hand), axis=1)
        o_id_1 = np.concatenate((o_id_pos_1, o_id_vel_1), axis=2)
        o_id_1 = o_id_1.reshape((bz,-1))[:,None,:]
        o_id = np.concatenate((o_id_0, o_id_1), axis=1)
        id = np.identity(2)[None,:,:]
        id = np.tile(id, (o_id.shape[0],1,1))
        o_id = np.concatenate((id, o_id), axis = 2)
    else:
        raise NotImplementedError
    return o_id

def gen_obs_graph(batch_s, lmax_attr,
        node_input_type = '', node_attr_type = '',
        gen_graph_info = False, device = None
    ):

    if len(batch_s.shape) == 1:
        batch_s = batch_s[None,...]
        if batch_action_n is not None:
            batch_action_n = batch_action_n[None,...]
    
    batch_size = batch_s.shape[0]
    
    graph = {}
    
    
    batch_id = batch_s[:, 0:2]
    batch_s = batch_s[:, 2:]
        
    batch_s = batch_s.reshape((batch_size, 3, 6))
    
    # graph['pos']
    
    all_pos = batch_s[:, :, 0:3]
    
    pos = all_pos[:, 0:2, :] #nodes = [target, finger]
        
    n_nodes = pos.shape[1]
    mean_pos = np.mean(pos, axis=1, keepdims=True) # (batch_size, 1, 3)
    pos_minus_mean_pos = pos - mean_pos
    pos = pos.reshape((-1,3))                                 # (batch_size * n_nodes, 3)
    pos_minus_mean_pos = pos_minus_mean_pos.reshape((-1,3))   # (batch_size * n_nodes, 3)
    graph['pos'] = torch.tensor(
        pos,
        dtype=torch.get_default_dtype(),
        device = device
    )
    
    # graph['edge_src'] and graph['edge_dst'] with all-connected no self-loop
    adj_matrix = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)  # adjacency matrix of all-connected without self-loop
    edge_dst_per_graph, edge_src_per_graph = np.where(adj_matrix > 0.5)  # (n_edges,), (n_edges,)
    edge_dst_per_graph = edge_dst_per_graph[None,:]  # (1, n_edges)
    edge_src_per_graph = edge_src_per_graph[None,:]  # (1, n_edges)
    n_edges = edge_dst_per_graph.shape[-1]
    pattern_to_sum = n_nodes * np.repeat(
        np.arange(0, batch_size, dtype=np.int64)[:,None],           # (batch_size, 1)
        n_edges,
        axis= 1
    ) # (batch_size, n_edges)
    edge_dst = edge_dst_per_graph + pattern_to_sum    # (batch_size, n_edges)
    edge_src = edge_src_per_graph + pattern_to_sum    # (batch_size, n_edges)
    edge_dst = edge_dst.reshape((-1,))                # (batch_size * n_edges)
    edge_src = edge_src.reshape((-1,))                # (batch_size * n_edges)
    graph['edge_src'] = torch.tensor(edge_src, dtype = torch.int64, device = device)
    graph['edge_dst'] = torch.tensor(edge_dst, dtype = torch.int64, device = device)

    # graph['x'], node input feature
    x = []; irreps_node_input = ''

    # graph['x']: pos - mean_pos
    if 'pos' in node_input_type:
        x.append(
            torch.tensor(
                pos_minus_mean_pos,
                dtype=torch.get_default_dtype(), device = device
            )
        ); irreps_node_input += ' + 1o'
        
    #for id
    id_finger = batch_id[:,None,:]
    id_target = batch_id[:,None,:]
    id_all = np.concatenate((id_target, id_finger), axis = 1)
    id_all = id_all.reshape((-1,2))
    x.append(torch.tensor(
            id_all, dtype=torch.get_default_dtype(), device = device
        ));
    irreps_node_input += ' + 2x0e'
    
    # graph['x']: vel
    vel = batch_s[:, :, 3:6]
    
    other_pos = all_pos[:, 2:, :]
    other_v = vel
    
    origin_features = np.concatenate((other_pos, other_v), axis=1)[:,None,:,:] #bz, n=1, #feature, feature_dim
    target_features = np.zeros(origin_features.shape)
    features = np.concatenate((target_features, origin_features), axis=1) #bz, n, #feature, feature_dim
    
    num_features = features.shape[2]
    for feature_dim in range(num_features):
        feature = features[:,:,feature_dim, :] #bz, n, feature_dim
        feature = feature.reshape((-1,3))                     # (batch_size * n_nodes, 3)
        x.append(
            torch.tensor(
                feature, dtype=torch.get_default_dtype(), device = device
            )
        ); irreps_node_input += ' + 1o'
        
        # graph['x']: feature_abs
        feature_abs = np.sqrt(np.power(feature, 2.0).sum(-1, keepdims=True))
        x.append(
            torch.tensor(
                feature_abs, dtype=torch.get_default_dtype(), device = device
            )
        ); irreps_node_input += ' + 0e'
    
    graph['x'] = torch.concat(x, dim = -1); irreps_node_input = irreps_node_input[3:]

    # graph['edge_attr']
    irreps_edge_attr = Irreps.spherical_harmonics(lmax_attr)
    rel_pos = pos[edge_src] - pos[edge_dst]
    edge_attr = spherical_harmonics(
        irreps_edge_attr,
        torch.tensor(rel_pos, dtype=torch.get_default_dtype(), device = device),
        normalize=True, normalization='integral'
    )
    graph['edge_attr'] = edge_attr.clone().detach()

    # graph['node_attr']
    node_attr = []; irreps_node_attr = ''
    
    # graph['node_attr']: node_type
    node_type_per_graph = np.concatenate(
        (
            0 + np.zeros((1,), dtype=np.int64),
            1 + np.zeros((1,), dtype=np.int64),
        ),
        axis = 0
    )
    n_node_types = node_type_per_graph.max() + 1
    node_type_onehot_per_graph = np.zeros((node_type_per_graph.size, n_node_types))
    node_type_onehot_per_graph[np.arange(node_type_per_graph.size), node_type_per_graph] = 1
    node_type_onehot = np.repeat(
        node_type_onehot_per_graph[None,...], # (1, n_nodes, n_node_types)
        batch_size,
        axis = 0
    ) # (batch_size, n_nodes, n_node_types)
    node_type_onehot = node_type_onehot.reshape((-1, n_node_types))  # (batch_size * n_nodes, n_node_types)
    #above may have bugs

    node_attr.append(
        torch.tensor(
            node_type_onehot, dtype=torch.get_default_dtype(), device = device
        )
    ); irreps_node_attr += ' + {}x0e'.format(n_node_types)
    
    # graph['node_attr']: vel_embedding
    if 'vel_embedding' in node_attr_type:
        vel_embedding = spherical_harmonics(
            irreps_edge_attr,
            torch.tensor(vel, dtype=torch.get_default_dtype() , device = device),
            normalize=True, normalization='integral'
        )
        node_attr.append(
            vel_embedding
        ); irreps_node_attr += ' + ' + irreps_edge_attr.__str__()

    # graph['node_attr']: act_embedding
    if 'act_embedding' in node_attr_type:
        if batch_action_n is not None:
            act_embedding = spherical_harmonics(
                irreps_edge_attr,
                act,
                normalize=True, normalization='integral'
            )
            node_attr.append(
                act_embedding
            ); irreps_node_attr += ' + ' + irreps_edge_attr.__str__()
    
    graph['node_attr'] = torch.concat(node_attr, dim = -1); irreps_node_attr = irreps_node_attr[3:]
    
    # graph_info['additional_message_features'], used in Steerable-E3-GNN’s nbody_gravity
    edge_dist = np.sqrt(np.power(rel_pos, 2.0).sum(-1, keepdims=True))
    additional_message_features = []; irreps_additional_message = ''
    additional_message_features.append(edge_dist); irreps_additional_message += ' + 1x0e'
    additional_message_features = np.concatenate(additional_message_features, axis=-1)
    graph['additional_message_features'] = torch.tensor(
        additional_message_features,
        dtype=torch.get_default_dtype(),
        device = device,
    ); irreps_additional_message = irreps_additional_message[3:]

    # graph['batch'], assuming every graph has an equal number of nodes
    graph['batch'] = torch.arange(0, batch_size, dtype = torch.int64, device = device).repeat_interleave(n_nodes)
    
    if not gen_graph_info:
        return graph
    else:
        graph_info = {}
        graph_info['irreps_node_input'] = str(Irreps(irreps_node_input).simplify())
        graph_info['irreps_node_attr'] =  str(Irreps(irreps_node_attr).simplify())
        graph_info['irreps_edge_attr'] = str(irreps_edge_attr)
        graph_info['num_neighbors'] = n_edges / n_nodes
        graph_info['num_nodes'] = n_nodes
        graph_info['irreps_additional_message'] = str(Irreps(irreps_additional_message).simplify())

        return graph, graph_info




def gen_obs_action_n_graph(batch_s, batch_action_n, lmax_attr,
    node_input_type = '', node_attr_type = '',
    gen_graph_info = False, device = None
):

    if len(batch_s.shape) == 1:
        batch_s = batch_s[None,...]
        if batch_action_n is not None:
            batch_action_n = batch_action_n[None,...]
    
    batch_size = batch_s.shape[0]
    
    if batch_action_n is not None:
        if torch.is_tensor(batch_action_n):
            device = batch_action_n.device
            dtype = batch_action_n.dtype
        elif isinstance(batch_action_n, np.ndarray):
            dtype = torch.get_default_dtype()
            batch_action_n = torch.tensor(
                batch_action_n,
                dtype=torch.get_default_dtype(),
                device = device,
            )
        else:
            raise NotImplementedError

    graph = {}
    
    if batch_action_n is None:
        batch_id = batch_s[:, 0:2]
        batch_s = batch_s[:, 2:]
        
    batch_s = batch_s.reshape((batch_size, 5, 6))
    
    # graph['pos']
    
    all_pos = batch_s[:, :, 0:3]
    
    pos = all_pos[:, 0:2, :] #nodes = [target, finger]
    
    n_nodes = pos.shape[1]
    mean_pos = np.mean(pos, axis=1, keepdims=True)
    pos_minus_mean_pos = pos - mean_pos
    pos = pos.reshape((-1,3))
    pos_minus_mean_pos = pos_minus_mean_pos.reshape((-1,3))
    graph['pos'] = torch.tensor(
        pos,
        dtype=torch.get_default_dtype(),
        device = device
    )
    
    # graph['edge_src'] and graph['edge_dst'] with all-connected no self-loop
    adj_matrix = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)  # adjacency matrix of all-connected without self-loop
    edge_dst_per_graph, edge_src_per_graph = np.where(adj_matrix > 0.5)  # (n_edges,), (n_edges,)
    edge_dst_per_graph = edge_dst_per_graph[None,:]  # (1, n_edges)
    edge_src_per_graph = edge_src_per_graph[None,:]  # (1, n_edges)
    n_edges = edge_dst_per_graph.shape[-1]
    pattern_to_sum = n_nodes * np.repeat(
        np.arange(0, batch_size, dtype=np.int64)[:,None],           # (batch_size, 1)
        n_edges,
        axis= 1
    ) # (batch_size, n_edges)
    edge_dst = edge_dst_per_graph + pattern_to_sum    # (batch_size, n_edges)
    edge_src = edge_src_per_graph + pattern_to_sum    # (batch_size, n_edges)
    edge_dst = edge_dst.reshape((-1,))                # (batch_size * n_edges)
    edge_src = edge_src.reshape((-1,))                # (batch_size * n_edges)
    graph['edge_src'] = torch.tensor(edge_src, dtype = torch.int64, device = device)
    graph['edge_dst'] = torch.tensor(edge_dst, dtype = torch.int64, device = device)

    # graph['x'], node input feature
    x = []; irreps_node_input = ''

    # graph['x']: pos - mean_pos
    if 'pos' in node_input_type:
        x.append(
            torch.tensor(
                pos_minus_mean_pos,
                dtype=torch.get_default_dtype(), device = device
            )
        ); irreps_node_input += ' + 1o'
        
    #for id
    if batch_action_n is None:
        id_finger = batch_id[:,None,:]
        id_target = batch_id[:,None,:]
        id_all = np.concatenate((id_target, id_finger), axis = 1)
        id_all = id_all.reshape((-1,2))
        x.append(torch.tensor(
                id_all, dtype=torch.get_default_dtype(), device = device
            ));
        irreps_node_input += ' + 2x0e'
    
    # graph['x']: vel
    vel = batch_s[:, :, 3:6]
    
    other_pos = all_pos[:, 2:, :]
    other_v = vel[:, 0:3, :]
    
    origin_features = np.concatenate((other_pos, other_v), axis=1)[:,None,:,:]
    target_features = np.zeros(origin_features.shape)
    features = np.concatenate((target_features, origin_features), axis=1)
    
    num_features = features.shape[2]
    for feature_dim in range(num_features):
        feature = features[:,:,feature_dim, :]
        feature = feature.reshape((-1,3))
        x.append(
            torch.tensor(
                feature, dtype=torch.get_default_dtype(), device = device
            )
        ); irreps_node_input += ' + 1o'
        
        # graph['x']: feature_abs
        feature_abs = np.sqrt(np.power(feature, 2.0).sum(-1, keepdims=True))
        x.append(
            torch.tensor(
                feature_abs, dtype=torch.get_default_dtype(), device = device
            )
        ); irreps_node_input += ' + 0e'
    
    # graph['x']: act
    if batch_action_n is not None:

        act_finger = batch_action_n.view(batch_size, 1, 2)
        act_target = torch.zeros(act_finger.shape, dtype = torch.get_default_dtype(), device = device)
        
        act = torch.cat((act_target, act_finger), dim = 1)
        act = act.view((-1,2))
        x.append(act);
        irreps_node_input += ' + 2x0e'

    # graph['x']: act_abs
    if batch_action_n is not None:
        act_abs = torch.sqrt(torch.pow(act, 2.0).sum(-1, keepdims=True))
        x.append(act_abs); irreps_node_input += ' + 0e'
    
    graph['x'] = torch.concat(x, dim = -1); irreps_node_input = irreps_node_input[3:]

    # graph['edge_attr']
    irreps_edge_attr = Irreps.spherical_harmonics(lmax_attr)
    rel_pos = pos[edge_src] - pos[edge_dst]
    edge_attr = spherical_harmonics(
        irreps_edge_attr,
        torch.tensor(rel_pos, dtype=torch.get_default_dtype(), device = device),
        normalize=True, normalization='integral'
    )
    graph['edge_attr'] = edge_attr.clone().detach()

    # graph['node_attr']
    node_attr = []; irreps_node_attr = ''
    
    # graph['node_attr']: node_type
    node_type_per_graph = np.concatenate(
        (
            0 + np.zeros((1,), dtype=np.int64),
            1 + np.zeros((1,), dtype=np.int64),
        ),
        axis = 0
    )
    n_node_types = node_type_per_graph.max() + 1
    node_type_onehot_per_graph = np.zeros((node_type_per_graph.size, n_node_types))
    node_type_onehot_per_graph[np.arange(node_type_per_graph.size), node_type_per_graph] = 1
    node_type_onehot = np.repeat(
        node_type_onehot_per_graph[None,...], # (1, n_nodes, n_node_types)
        batch_size,
        axis = 0
    ) # (batch_size, n_nodes, n_node_types)
    node_type_onehot = node_type_onehot.reshape((-1, n_node_types))  # (batch_size * n_nodes, n_node_types)
    #above may have bugs

    node_attr.append(
        torch.tensor(
            node_type_onehot, dtype=torch.get_default_dtype(), device = device
        )
    ); irreps_node_attr += ' + {}x0e'.format(n_node_types)
    
    # graph['node_attr']: vel_embedding
    if 'vel_embedding' in node_attr_type:
        vel_embedding = spherical_harmonics(
            irreps_edge_attr,
            torch.tensor(vel, dtype=torch.get_default_dtype() , device = device),
            normalize=True, normalization='integral'
        )
        node_attr.append(
            vel_embedding
        ); irreps_node_attr += ' + ' + irreps_edge_attr.__str__()

    # graph['node_attr']: act_embedding
    if 'act_embedding' in node_attr_type:
        if batch_action_n is not None:
            act_embedding = spherical_harmonics(
                irreps_edge_attr,
                act,
                normalize=True, normalization='integral'
            )
            node_attr.append(
                act_embedding
            ); irreps_node_attr += ' + ' + irreps_edge_attr.__str__()
    
    graph['node_attr'] = torch.concat(node_attr, dim = -1); irreps_node_attr = irreps_node_attr[3:]
    
    # graph_info['additional_message_features'], used in Steerable-E3-GNN’s nbody_gravity
    edge_dist = np.sqrt(np.power(rel_pos, 2.0).sum(-1, keepdims=True))
    additional_message_features = []; irreps_additional_message = ''
    additional_message_features.append(edge_dist); irreps_additional_message += ' + 1x0e'
    additional_message_features = np.concatenate(additional_message_features, axis=-1)
    graph['additional_message_features'] = torch.tensor(
        additional_message_features,
        dtype=torch.get_default_dtype(),
        device = device,
    ); irreps_additional_message = irreps_additional_message[3:]

    # graph['batch'], assuming every graph has an equal number of nodes
    graph['batch'] = torch.arange(0, batch_size, dtype = torch.int64, device = device).repeat_interleave(n_nodes)
    
    if not gen_graph_info:
        return graph
    else:
        graph_info = {}
        graph_info['irreps_node_input'] = str(Irreps(irreps_node_input).simplify())
        graph_info['irreps_node_attr'] =  str(Irreps(irreps_node_attr).simplify())
        graph_info['irreps_edge_attr'] = str(irreps_edge_attr)
        graph_info['num_neighbors'] = n_edges / n_nodes
        graph_info['num_nodes'] = n_nodes
        graph_info['irreps_additional_message'] = str(Irreps(irreps_additional_message).simplify())

        return graph, graph_info
