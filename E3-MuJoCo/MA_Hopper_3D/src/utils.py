from __future__ import print_function
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import xmltodict
import wrappers
import gym
from gym.envs.registration import register
from shutil import copyfile
from config import *
import torch

from e3nn.o3 import Irreps, spherical_harmonics

def obs_with_id(obs):
    if torch.is_tensor(obs):
        o_id = obs.cpu().numpy()
    else:
        o_id = obs
    o_id = o_id[:,None,:]
    o_id = np.tile(o_id, (1,2,1))
    id = np.identity(2)[None,:,:]
    id = np.tile(id, (o_id.shape[0],1,1))
    o_id = np.concatenate((id, o_id), axis = 2)
    o_id = o_id.reshape((-1, o_id.shape[-1]))
    return o_id

def gen_obs_graph(batch_s, lmax_attr, 
        node_input_type = '', node_attr_type = '', 
        gen_graph_info = False, device = None
    ):
    return gen_obs_action_n_graph(
                batch_s,  batch_action_n = None, lmax_attr = lmax_attr,
                node_input_type = node_input_type, node_attr_type = node_attr_type, 
                gen_graph_info = gen_graph_info,
                device = device
            )

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
        
    goal_pos = batch_s[:, 0:3]
    gravity = batch_s[:, 3:6]
    batch_s = batch_s[:, 6:]
    n_joints = 3
    batch_s = batch_s.reshape((batch_size, n_joints, -1))

    joints_pos = batch_s[:, :, 0:3]
    
    batch_s = batch_s[:,:,3:]
    
    goal_pos = goal_pos[:,None,:]
    
    pos = np.concatenate((goal_pos, joints_pos), axis=1)

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
        id_all = batch_id[:,None,:]
        id_all = np.tile(id_all, (1,n_nodes,1))
        id_all = id_all.reshape((-1,2))
        x.append(torch.tensor(
                id_all, dtype=torch.get_default_dtype(), device = device
            ));
        irreps_node_input += ' + 2x0e'
            
    # graph['x']: vel
    joint_equi_features = batch_s.reshape((batch_size, n_joints, -1, 3))
    goal_equi_features = np.zeros((batch_size, 1, joint_equi_features.shape[2], 3))
    features = np.concatenate((goal_equi_features, joint_equi_features), axis=1)

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
    

    # graph['x']: feature_abs
    gravity = np.tile(gravity[:,None,:], [1,n_nodes,1])
    gravity = gravity.reshape((-1,3))
    x.append(
        torch.tensor(
            gravity, dtype=torch.get_default_dtype(), device = device
        )
    ); irreps_node_input += ' + 3x0e'

    # graph['x']: act
    if batch_action_n is not None:

        act_joints = batch_action_n.view(batch_size, n_joints, 3)
        act_target = torch.zeros((batch_size, 1, 3), dtype = torch.get_default_dtype(), device = device)
        
        act = torch.cat((act_target, act_joints), dim = 1)
        act = act.view((-1,3))
        x.append(act);
        irreps_node_input += ' + 3x0e'

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
            2 + np.zeros((1,), dtype=np.int64),
            3 + np.zeros((1,), dtype=np.int64),
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
        graph_info['irreps_node_input'] = Irreps(irreps_node_input).simplify()
        graph_info['irreps_node_attr'] =  Irreps(irreps_node_attr).simplify()
        graph_info['irreps_edge_attr'] = irreps_edge_attr
        graph_info['num_neighbors'] = n_edges / n_nodes
        graph_info['num_nodes'] = n_nodes
        graph_info['irreps_additional_message'] = Irreps(irreps_additional_message).simplify()

        return graph, graph_info



















def makeEnvWrapper(env_name, obs_max_len=None, seed=0):
    """return wrapped gym environment for parallel sample collection (vectorized environments)"""
    def helper():
        e = gym.make("environments:%s-v0" % env_name)
        e.seed(seed)
        e.action_space.seed(seed)
        return wrappers.ModularEnvWrapper(e, obs_max_len)

    return helper


def findMaxChildren(env_names, graphs):
    """return the maximum number of children given a list of env names and their corresponding graph structures"""
    max_children = 0
    for name in env_names:
        most_frequent = max(graphs[name], key=graphs[name].count)
        max_children = max(max_children, graphs[name].count(most_frequent))
    return max_children


def registerEnvs(env_names, max_episode_steps, custom_xml):
    """register the MuJoCo envs with Gym and return the per-limb observation size and max action value (for modular policy training)"""
    # get all paths to xmls (handle the case where the given path is a directory containing multiple xml files)
    paths_to_register = []
    # existing envs
    if not custom_xml:
        for name in env_names:
            paths_to_register.append(os.path.join(XML_DIR, "{}.xml".format(name)))
    # custom envs
    else:
        if os.path.isfile(custom_xml):
            paths_to_register.append(custom_xml)
        elif os.path.isdir(custom_xml):
            for name in sorted(os.listdir(custom_xml)):
                if ".xml" in name:
                    paths_to_register.append(os.path.join(custom_xml, name))
    # register each env
    observation_space={}
    action_space={}
    for xml in paths_to_register:
        env_name = os.path.basename(xml)[:-4]
        env_file = env_name
        # create a copy of modular environment for custom xml model
        if not os.path.exists(os.path.join(ENV_DIR, "{}.py".format(env_name))):
            # create a duplicate of gym environment file for each env (necessary for avoiding bug in gym)
            copyfile(
                BASE_MODULAR_ENV_PATH, "{}.py".format(os.path.join(ENV_DIR, env_name))
            )
        params = {"xml": os.path.abspath(xml)}
        # register with gym
        print(env_name)
        register(
            id=("%s-v0" % env_name),
            max_episode_steps=max_episode_steps,
            entry_point="environments.%s:ModularEnv" % env_file,
            kwargs=params,
        )
        env = wrappers.IdentityWrapper(gym.make("environments:%s-v0" % env_name))
        # the following is the same for each env
        limb_obs_size = env.limb_obs_size
        limb_action_size = env.limb_action_size
        print("limb_obs_action_size",limb_obs_size,limb_action_size)
        max_action = env.max_action
        observation_space[env_name]=env.env.observation_space
        action_space[env_name]=env.env.action_space
    return limb_obs_size, limb_action_size, max_action, observation_space, action_space

def quat2mat(q):
    """
    Converts a quaternion to a rotation matrix
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    Args
    q: 1x4 quaternion
    Returns
    r: 3x3 rotation matrix
    Raises
    ValueError if the l2 norm of the quaternion is not close to 1
    """
    if np.abs(np.linalg.norm(q) - 1) > 1e-3:
        raise (ValueError, "quat2mat: input quaternion is not norm 1")

    w, x, y, z = q
    r = np.array(
        [
            [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2],
        ]
    )
    return r

def quat2expmap(q):
    """
    Converts a quaternion to an exponential map
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1
    Args
    q: 1x4 quaternion
    Returns
    r: 1x3 exponential map
    Raises
    ValueError if the l2 norm of the quaternion is not close to 1
    """
    if np.abs(np.linalg.norm(q) - 1) > 1e-3:
        raise (ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]
    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps))
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)
    if theta > np.pi:
        theta = 2 * np.pi - theta
        r0 = -r0
    r = r0 * theta
    return r

def quat2axisangle(q):
    """
    Converts a quaternion to an exponential map
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/index.htm
    Args
    q: 1x4 quaternion
    Returns
    r: 1x4 angle x y z
    Raises
    ValueError if the l2 norm of the quaternion is not close to 1
    """
    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]
    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps))
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)
    if theta > np.pi:
        theta = 2 * np.pi - theta
        r0 = -r0
    # r = r0 * theta
    
    return np.concatenate([r0,[theta]])


# replay buffer: expects tuples of (state, next_state, action, reward, done)
# modified from https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
class ReplayBuffer(object):
    def __init__(self, max_size=1e6, slicing_size=None):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        # maintains slicing info for [obs, new_obs, action, reward, done]
        if slicing_size:
            self.slicing_size = slicing_size
        else:
            self.slicing_size = None

    def add(self, data):
        if self.slicing_size is None:
            self.slicing_size = [data[0].size, data[1].size, data[2].size, 1, 1]
        data = np.concatenate([data[0], data[1], data[2], [data[3]], [data[4]]])
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            data = self.storage[i]
            X = data[: self.slicing_size[0]]
            Y = data[self.slicing_size[0] : self.slicing_size[0] + self.slicing_size[1]]
            U = data[
                self.slicing_size[0]
                + self.slicing_size[1] : self.slicing_size[0]
                + self.slicing_size[1]
                + self.slicing_size[2]
            ]
            R = data[
                self.slicing_size[0]
                + self.slicing_size[1]
                + self.slicing_size[2] : self.slicing_size[0]
                + self.slicing_size[1]
                + self.slicing_size[2]
                + self.slicing_size[3]
            ]
            D = data[
                self.slicing_size[0]
                + self.slicing_size[1]
                + self.slicing_size[2]
                + self.slicing_size[3] :
            ]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return (
            np.array(x),
            np.array(y),
            np.array(u),
            np.array(r).reshape(-1, 1),
            np.array(d).reshape(-1, 1),
        )


class MLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MLPBase, self).__init__()
        self.l1 = nn.Linear(num_inputs, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, num_outputs)

    def forward(self, inputs):
        x = F.relu(self.l1(inputs))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


def getGraphStructure(xml_file, graph_type="morphology"):
    """Traverse the given xml file as a tree by pre-order and return the graph structure as a parents list"""

    def preorder(b, parent_idx=-1):
        self_idx = len(parents)
        parents.append(parent_idx)
        if "body" not in b:
            return
        if not isinstance(b["body"], list):
            b["body"] = [b["body"]]
        for branch in b["body"]:
            preorder(branch, self_idx)

    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    parents = []
    try:
        root = xml["mujoco"]["worldbody"]["body"]
        assert not isinstance(
            root, list
        ), "worldbody can only contain one body (torso) for the current implementation, but found {}".format(
            root
        )
    except:
        raise Exception(
            "The given xml file does not follow the standard MuJoCo format."
        )
    preorder(root)
    # signal message flipping for flipped walker morphologies
    if "walker" in os.path.basename(xml_file) and "flipped" in os.path.basename(
        xml_file
    ):
        parents[0] = -2

    if graph_type == "tree":
        parents[1:] = [0] * len(parents[1:])
    elif graph_type == "line":
        for i in range(1, len(parents)):
            parents[i] = i - 1

    return parents


def getGraphJoints(xml_file):
    """Traverse the given xml file as a tree by pre-order and return all the joints defined as a list of tuples (body_name, joint_name1, ...) for each body"""
    """Used to match the order of joints defined in worldbody and joints defined in actuators"""

    def preorder(b):
        # print(b["joint"])
        if "joint" in b:
            if isinstance(b["joint"], list) and b["@name"] != "torso":
                # raise Exception(
                #     "The given xml file does not follow the standard MuJoCo format."
                # )
                pass
            elif not isinstance(b["joint"], list):
                b["joint"] = [b["joint"]]
            joints.append([b["@name"]])
            # print([b["@name"]])
            # print(joints)
            for j in b["joint"]:
                joints[-1].append(j["@name"])
            # print(joints)
        if "body" not in b:
            return
        if not isinstance(b["body"], list):
            b["body"] = [b["body"]]
        for branch in b["body"]:
            preorder(branch)

    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    joints = []
    try:
        root = xml["mujoco"]["worldbody"]["body"]
    except:
        raise Exception(
            "The given xml file does not follow the standard MuJoCo format."
        )
    preorder(root)
    return joints


def getMotorJoints(xml_file):
    """Traverse the given xml file as a tree by pre-order and return the joint names in the order of defined actuators"""
    """Used to match the order of joints defined in worldbody and joints defined in actuators"""
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    joints = []
    motors = xml["mujoco"]["actuator"]["motor"]
    if not isinstance(motors, list):
        motors = [motors]
    for m in motors:
        joints.append(m["@joint"])
    return joints


def getDistance(adjacency):
    def bfs(adjacency, root):
        dist = [-1] * adjacency.shape[0]
        dist[root] = 0
        Q = [(root, 0)]
        while len(Q):
            v, d = Q[0]; Q = Q[1:]
            for u, is_adj in enumerate(adjacency[v]):
                if is_adj and dist[u] == -1:
                    dist[u] = d + 1
                    Q.append((u, d+1))
        return dist

    return np.array([bfs(adjacency, i) for i in range(len(adjacency))]) / len(adjacency)

def getChildrens(parents):
    childrens = []
    for cur_node_idx in range(len(parents)):
        childrens.append([])
        for node_idx in range(cur_node_idx, len(parents)):
            if cur_node_idx == parents[node_idx]:
                childrens[cur_node_idx].append(node_idx)
    return childrens

def lcrs(graph):
    new_graph = [[] for _ in graph]
    for node, children in enumerate(graph):
        if len(children) > 0:
            temp = children[0]
            new_graph[node].insert(0, temp)
            for sibling in children[1:]:
                new_graph[temp].append(sibling)
                temp = sibling
    return new_graph

def getTraversal(parents, traversal_types, device=None):
    """Reconstruct tree and return a lists of node position in multiple traversals"""
    
    def postorder(children):
        trav = []
        def visit(node):
            for i in children[node]:
                visit(i)
            trav.append(node)
        visit(0)
        return trav

    def inorder(children):
        # assert binary tree
        trav = []
        def visit(node):
            if children[node]:
                visit(children[node][0])
            trav.append(node)
            if len(children[node]) == 2:
                visit(children[node][1])
        visit(0)
        return trav

    children = getChildrens(parents)
    traversals = []
    for ttype in traversal_types:
        if ttype == 'pre':
            indices = list(range(len(children)))
        else:
            if ttype == 'inlcrs':
                traversal = inorder(lcrs(children))
            elif ttype == 'postlcrs':
                traversal = postorder(lcrs(children))
            # indices = traversal
            indices = []
            for i in list(range(len(children))):
                indices.append(traversal.index(i))
        if device is not None:
            indices = torch.LongTensor(indices).to(device)
        traversals.append(indices)
    return traversals

def getAdjacency(parents):
    """Compute adjacency matrix of given graph"""
    N = len(parents)
    childrens = getChildrens(parents)
    adj = torch.zeros(N, N) # no self-loop
    for i, children in enumerate(childrens):
        for child in children:
            adj[i][child] = 1
            adj[child][i] = 1
    return adj  # (N, N)

def getGraphTransition(adjacency, self_loop=True):
    """Compute random walker transition in the given graph"""
    N = len(adjacency)
    if self_loop:
        adjacency = adjacency + torch.eye(N)
    degree = 1 / adjacency.sum(1).reshape(-1, 1) # for normalization
    transition = (adjacency * degree).T # (N, N)
    return transition

def PPR(transition, start=None, damping=0.9, max_iter=1000):
    """Compute Personalized PageRank vector"""
    N = transition.size(0)
    start = torch.ones(N, 1) / N \
            if start is None \
            else torch.eye(N)[start].reshape(N, 1)
    if damping == 1:
        prev_ppr = torch.ones(N, 1) / N
        for i in range(max_iter):
            ppr = damping * transition @ prev_ppr + (1 - damping) * start
            if ((ppr - prev_ppr).abs() < 1e-8).all():
                break
            prev_ppr = ppr
    else:
        inv = torch.inverse(torch.eye(N) - damping * transition)
        ppr = (1 - damping) * inv @ start
    return ppr  # (N, 1)

def getGraphDict(parents, trav_types=[], rel_types=[], self_loop=True, ppr_damping=0.9, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(parents) == 1:
        return {'parents': parents}

    adjacency = getAdjacency(parents)
    transition = getGraphTransition(adjacency, self_loop)
    mask = adjacency + torch.eye(len(parents))
    mask = torch.zeros_like(mask).masked_fill(mask==0, -np.inf)
    degree = adjacency.sum(1)
    laplacian = torch.diag(degree) - adjacency
    sym_lap = torch.diag(degree**-0.5) @ laplacian @ torch.diag(degree**-0.5)
    distance = torch.from_numpy(getDistance(adjacency)).float()
    graph_dict = {
        'parents': parents,
        'traversals': getTraversal(parents, trav_types, device),
        'ppr': torch.cat([
            PPR(transition, i, ppr_damping)
            for i in range(len(parents))], dim=1).T.to(device),
        'transition': transition.to(device),
        'adjacency': adjacency.to(device),
        'distance': distance.to(device),
        'sym_lap': sym_lap.to(device),
        'mask': mask.to(device),
    }
    # TODO: define relation (PPR, Laplacian, ...)
    graph_dict['relation'] = torch.stack(
                        [
                            graph_dict['ppr'],
                            graph_dict['sym_lap'],
                            graph_dict['distance']
                        ], dim=2)
    # graph_dict['relation'] = graph_dict['ppr'].unsqueeze(-1)
    # graph_dict['ppr'][i] represents PPR(i, j) for all j (N,)
    return graph_dict


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
