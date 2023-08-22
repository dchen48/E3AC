import numpy as np
import torch
from e3nn.o3 import Irreps, spherical_harmonics

import random
from multiagent.core_vec import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from bridson import poisson_disc_samples


class Scenario(BaseScenario):
    def make_world(self, sort_obs=True, use_numba=False):
        world = World(use_numba)
        self.np_rnd = np.random.RandomState(0)
        self.random = random.Random(0)
        self.sort_obs = sort_obs
        # set any world properties first
        world.dim_c = 2
        self.num_agents = 3
        self.num_landmarks = 3
        world.collaborative = True
        self.agent_size = 0.15
        world.radius = self.world_radius = 1.5
        self.n_others = 2
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = self.agent_size
            agent.id = i
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)

        return world

    def reset_world(self, world):

        self.l_locations = poisson_disc_samples(width=self.world_radius * 2, height=self.world_radius * 2,
                                                r=self.agent_size * 4.5, random=self.random.random)
        while len(self.l_locations) < len(world.landmarks):
            self.l_locations = poisson_disc_samples(width=self.world_radius * 2, height=self.world_radius * 2,
                                                    r=self.agent_size * 4.5, random=self.random.random)
            print('regenerate l location')

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = self.np_rnd.uniform(-self.world_radius, +self.world_radius, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        l_locations = np.array(self.random.sample(self.l_locations, len(world.landmarks))) - self.world_radius
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = l_locations[i, :]
            landmark.state.p_vel = np.zeros(world.dim_p)
        self.collide_th = 2 * world.agents[0].size

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        """
        Vectorized reward function
        Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        """

        rew, rew1 = 0, 0

        if agent == world.agents[0]:
            l_pos = np.array([[l.state.p_pos for l in world.landmarks]]).repeat(len(world.agents), axis=0)
            a_pos = np.array([[a.state.p_pos for a in world.agents]])
            a_pos1 = a_pos.repeat(len(world.agents), axis=0)
            a_pos1 = np.transpose(a_pos1, axes=(1, 0, 2))
            a_pos2 = a_pos.repeat(len(world.agents), axis=0)
            dist = np.sqrt(np.sum(np.square(l_pos - a_pos1), axis=2))
            rew = np.min(dist, axis=0)
            rew = -np.sum(rew)
            if agent.collide:
                dist_a = np.sqrt(np.sum(np.square(a_pos1 - a_pos2), axis=2))
                n_collide = (dist_a < self.collide_th).sum() - len(world.agents)
                rew -= n_collide

        return rew

    def observation(self, agent, world):
        """
        :param agent: an agent
        :param world: the current world
        :return: obs: np array,
        [0-1] self_agent velocity
        [2-3] self_agent location
        [4-9] landmarks location
        """
        if agent.id == 0:
            l_pos = np.array([[l.state.p_pos for l in world.landmarks]]).repeat(len(world.agents), axis=0)
            a_pos = np.array([[a.state.p_pos for a in world.agents]])
            a_pos1 = a_pos.repeat(len(world.agents), axis=0)
            a_pos1 = np.transpose(a_pos1, axes=(1, 0, 2))
            a_pos2 = a_pos.repeat(len(world.agents), axis=0)
            entity_pos = l_pos - a_pos1
            other_pos = a_pos2 - a_pos1

            entity_dist = np.sqrt(np.sum(np.square(entity_pos), axis=2))
            entity_dist_idx = np.argsort(entity_dist, axis=1)
            row_idx = np.arange(self.num_agents).repeat(self.num_landmarks)
            self.sorted_entity_pos = entity_pos[row_idx, entity_dist_idx.reshape(-1)].reshape(self.num_agents,
                                                                                              self.num_landmarks, 2)

            other_dist = np.sqrt(np.sum(np.square(other_pos), axis=2))
            other_dist_idx = np.argsort(other_dist, axis=1)
            row_idx = np.arange(self.num_agents).repeat(self.num_agents)
            self.sorted_other_pos = other_pos[row_idx, other_dist_idx.reshape(-1)].reshape(self.num_agents,
                                                                                            self.num_agents, 2)[:, 1:, :]
            self.sorted_other_pos = self.sorted_other_pos[:, :self.n_others, :]
            self.sorted_entity_pos = self.sorted_entity_pos[:, :self.n_others + 1, :]

        obs = np.concatenate((np.array([agent.state.p_vel]), np.array([agent.state.p_pos]),
                              self.sorted_entity_pos[agent.id, :, :],
                              self.sorted_other_pos[agent.id, :, :]), axis=0).reshape(-1)
        return obs

    def seed(self, seed=None):
        self.np_rnd.seed(seed)
        self.random.seed(seed)
    
    def state(self, world):
        s = {}
        
        s['n_agents'] = len(world.agents)
        s['n_landmarks'] = len(world.landmarks)
        
        s['pos_agents'] = []
        for agent in world.agents:
            s['pos_agents'].append(agent.state.p_pos)
        
        s['pos_landmarks'] = []
        for entity in world.landmarks:
            s['pos_landmarks'].append(entity.state.p_pos)
        
        s['vel_agents'] = []
        for agent in world.agents:
            s['vel_agents'].append(agent.state.p_vel)


        for k in s:
            s[k] = np.array(s[k])

        return s
    
    def gen_state_action_n_graph(self, 
        batch_s, batch_action_n, lmax_attr,
        node_input_type = '', node_attr_type = '', 
        world = None, gen_graph_info = False, device = None
    ):            
        if len(batch_s['n_agents'].shape) == 0:                # non-batch
            s = batch_s
            batch_s = {}
            for k in s:
                batch_s[k] = s[k][None,...]              # (batch_size = 1, ...)
            if batch_action_n is not None:
                assert(len(batch_action_n.shape) == 2)         # (n_agents, action_dim)
                action_n =  batch_action_n
                batch_action_n = None 
                batch_action_n = action_n[None,...]      # (batch_size = 1, n_agents, action_dim)   
        else:
            for k in batch_s:
                assert(batch_s[k].shape[0] == batch_s['n_agents'].shape[0]) # batch_size
            if batch_action_n is not None:
                assert(len(batch_action_n.shape)) == 3        # (batch_size, n_agents, action_dim)
        batch_size = batch_s['n_agents'].shape[0]

        if batch_action_n is not None:
            if torch.is_tensor(batch_action_n):
                device = batch_action_n.device
                dtype = batch_action_n.dtype
            elif isinstance(batch_action_n, np.ndarray):
                dtype = torch.get_default_dtype()
                batch_action_n = torch.tensor(
                    batch_action_n,
                    dtype = dtype, device = device,
                )
            else:
                raise NotImplementedError

        graph = {}

        # graph['pos']                     
        pos_agents = np.array(batch_s['pos_agents'])        # (batch_size, n_agents, 2)
        pos_landmarks = np.array(batch_s['pos_landmarks'])  # (batch_size, n_landmarks, 2)
        pos = np.concatenate(
            (pos_agents, pos_landmarks), 
            axis=1
        ) # (batch_size, n_nodes = n_agents + n_landmarks, 2)
        n_agents, n_landmarks = pos_agents.shape[1], pos_landmarks.shape[1]
        n_nodes = pos.shape[1]
        pos_z = np.zeros((batch_size, n_nodes, 1))    # dummy pos_z
        pos = np.concatenate((pos, pos_z), axis=-1)   # (batch_size, n_nodes, 3)
        mean_pos = np.mean(pos, axis=1, keepdims=True) # (batch_size, 1, 3)
        pos_minus_mean_pos = pos - mean_pos
        pos = pos.reshape((-1,3))                                 # (batch_size * n_nodes, 3)
        pos_minus_mean_pos = pos_minus_mean_pos.reshape((-1,3))   # (batch_size * n_nodes, 3)
        graph['pos'] = torch.tensor(
            pos, 
            dtype = dtype, device = device
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
                    dtype = dtype, device = device
                )
            ); irreps_node_input += ' + 1o'  

        
        # graph['x']: vel
        vel_agents = np.array(batch_s['vel_agents'])
        vel_landmarks = np.zeros((batch_size, n_landmarks, 2))
        vel = np.concatenate((vel_agents, vel_landmarks), axis = 1)
        vel_z = np.zeros((batch_size, n_nodes, 1))    # dummy vel_z
        vel = np.concatenate((vel, vel_z), axis=-1)
        vel = vel.reshape((-1,3))                     # (batch_size * n_nodes, 3)
        x.append(
            torch.tensor(
                vel,
                dtype = dtype, device = device
            )
        ); irreps_node_input += ' + 1o'
        
        # graph['x']: vel_abs
        vel_abs = np.sqrt(np.power(vel, 2.0).sum(-1, keepdims=True))
        x.append(
            torch.tensor(
                vel_abs,
                dtype = dtype, device = device
            )
        ); irreps_node_input += ' + 0e'
        
        # graph['x']: act
        if batch_action_n is not None:
            action_landmarks = torch.zeros((batch_size, n_landmarks, 2), dtype = dtype, device = device)
            act = torch.concat((batch_action_n, action_landmarks), dim=1)
            act_z = torch.zeros((batch_size, n_nodes, 1), dtype = dtype, device = device)    # dummy act_z
            act = torch.concat((act, act_z), dim=-1)
            act = act.reshape((-1, 3))                    # (batch_size * n_nodes, 3)
            x.append(act); irreps_node_input += ' + 1o'
        
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
            torch.tensor(rel_pos, dtype = dtype, device = device), 
            normalize=True, normalization='integral'
        )
        graph['edge_attr'] = edge_attr.clone().detach()

        # graph['node_attr']
        node_attr = []; irreps_node_attr = ''
        
        # graph['node_attr']: node_type
        node_type_per_graph = np.concatenate(
            (
                0 + np.zeros((n_agents,), dtype=np.int64),
                1 + np.zeros((n_landmarks,), dtype=np.int64)
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
        node_attr.append(
            torch.tensor(
                node_type_onehot,
                dtype = dtype, device = device
            )
        ); irreps_node_attr += ' + {}x0e'.format(n_node_types)

        # graph['node_attr']: vel_embedding
        if 'vel_embedding' in node_attr_type:
            vel_embedding = spherical_harmonics(
                irreps_edge_attr, 
                torch.tensor(vel, dtype = dtype, device = device), 
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
            dtype = dtype, device = device,
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
            graph_info['max_radius'] = np.sqrt(8.0) * world.radius
            graph_info['num_neighbors'] = n_edges / n_nodes
            graph_info['num_nodes'] = n_nodes
            graph_info['irreps_additional_message'] = Irreps(irreps_additional_message).simplify()

            return graph, graph_info

    def gen_obs_graph(self, 
        batch_obs, 
        lmax_attr, node_input_type = 'pos', node_attr_type = '', 
        world = None, gen_graph_info = False, device = None
    ):
        obs_dim = batch_obs.shape[-1]
        batch_obs = batch_obs.reshape((-1, obs_dim))
        batch_size = batch_obs.shape[0]

        graph = {}
        n_landmarks = len(world.landmarks)

        # graph['pos']
        loc_agent =   batch_obs[:,2:4].reshape((batch_size, -1, 2))    # (batch_size, 1, 2)               
        loc_others =  batch_obs[:,4 + 2*n_landmarks : ].reshape((batch_size, -1, 2))
        loc_others = loc_others + loc_agent # loc_others in obs is relative to loc_agent
        n_others = loc_others.shape[1]
        loc_landmarks = batch_obs[:, 4 : 4+2*n_landmarks].reshape((batch_size, -1, 2))
        loc_landmarks = loc_landmarks + loc_agent # loc_landmarks in obs is relative to loc_agent
        pos = np.concatenate((loc_agent, loc_others, loc_landmarks), axis=1) # (batch_size, n_nodes, 2)
        n_nodes = pos.shape[1]
        pos_z = np.zeros((batch_size, n_nodes, 1))   # dummy pos_z
        pos = np.concatenate((pos, pos_z), axis=-1)   # (batch_size, n_nodes, 3)
        mean_pos = np.mean(pos, axis=1, keepdims=True) # (batch_size, 1, 3)
        pos_minus_mean_pos = pos - mean_pos
        pos = pos.reshape((-1,3))                                 # (batch_size * n_nodes, 3)
        pos_minus_mean_pos = pos_minus_mean_pos.reshape((-1,3))   # (batch_size * n_nodes, 3)
        graph['pos'] = torch.tensor(
            pos,
            dtype=torch.get_default_dtype(), device=device
        )
        
        # graph['edge_src'] and graph['edge_dst'] with star connection
        edge_dst_per_graph = np.zeros((n_nodes-1,), dtype=np.int64) [None,:] # (1, n_edges,)
        edge_src_per_graph = np.arange(1, n_nodes, dtype=np.int64)  [None,:] # (1, n_edges,)
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
        graph['edge_src'] = torch.tensor(edge_src, dtype = torch.int64, device=device)
        graph['edge_dst'] = torch.tensor(edge_dst, dtype = torch.int64, device=device)
        
        # graph['x'], node input feature  
        x = []; irreps_node_input = '' 

        # graph['x'], pos - mean_pos
        if 'pos' in node_input_type:
            x.append(
                pos_minus_mean_pos
            ); irreps_node_input += ' + 1o'
        
        # graph['x'], vel
        vel_agent = batch_obs[:, :2].reshape((batch_size, -1, 2))
        vel_others = np.zeros((batch_size, n_others, 2))     #  vel_others is unobservable
        vel_landmarks = np.zeros((batch_size, n_landmarks, 2))
        vel = np.concatenate((vel_agent, vel_others, vel_landmarks), axis = 1)
        vel_z = np.zeros((batch_size, n_nodes, 1))    # dummy vel_z
        vel = np.concatenate((vel, vel_z), axis=-1)   # (batch_size, n_nodes, 3)
        vel = vel.reshape((-1,3))                     # (batch_size * n_nodes, 3)
        x.append(vel); irreps_node_input += ' + 1o'
        
        # graph['x'], vel_abs
        vel_abs = np.sqrt(np.power(vel, 2.0).sum(-1, keepdims=True))
        x.append(vel_abs); irreps_node_input += ' + 0e'

        x = np.concatenate(x, axis=-1)
        graph['x'] = torch.tensor(
            x, 
            dtype=torch.get_default_dtype(), device=device
        ); irreps_node_input = irreps_node_input[3:] # trim the first ' + '

        # graph['edge_attr']
        irreps_edge_attr = Irreps.spherical_harmonics(lmax_attr)
        rel_pos = pos[edge_src] - pos[edge_dst]
        edge_attr = spherical_harmonics(irreps_edge_attr, 
                                        torch.tensor(rel_pos, dtype=torch.get_default_dtype(), device=device), 
                                        normalize=True, normalization='integral')
        graph['edge_attr'] = edge_attr.clone().detach()

        # graph['node_attr']
        node_attr = []; irreps_node_attr = ''

        # graph['node_attr']: node_type
        node_type_per_graph = np.concatenate(
            (
                0 + np.zeros((1,), dtype=np.int64),
                1 + np.zeros((n_others,), dtype=np.int64),
                2 + np.zeros((n_landmarks,), dtype=np.int64)
            ),
            axis = 0
        ) # (n_nodes,)
        n_node_types = node_type_per_graph.max() + 1
        node_type_onehot_per_graph = np.zeros((node_type_per_graph.size, n_node_types))
        node_type_onehot_per_graph[np.arange(node_type_per_graph.size), node_type_per_graph] = 1
        node_type_onehot = np.repeat(
            node_type_onehot_per_graph[None,...], # (1, n_nodes, n_node_types)
            batch_size, 
            axis = 0
        ) # (batch_size, n_nodes, n_node_types)
        node_type_onehot = node_type_onehot.reshape((-1, n_node_types))  # (batch_size * n_nodes, n_node_types)
        node_attr.append(node_type_onehot); irreps_node_attr += ' + {}x0e'.format(n_node_types)
        
        # graph['node_attr']: vel_embedding
        if 'vel_embedding' in node_attr_type:
            vel_embedding = spherical_harmonics(
                irreps_edge_attr, 
                torch.tensor(vel, dtype = 'cpu', device = device), 
                normalize=True, normalization='integral'
            ).cpu().numpy()
            node_attr.append(
                vel_embedding
            ); irreps_node_attr += ' + ' + irreps_edge_attr.__str__()
        
        node_attr = np.concatenate(node_attr, axis=-1)
        graph['node_attr'] = torch.tensor(
            node_attr, 
            dtype=torch.get_default_dtype(), device=device
        ); irreps_node_attr = irreps_node_attr[3:]


        # graph_info['additional_message_features'], used in Steerable-E3-GNN’s nbody_gravity
        edge_dist = np.sqrt(np.power(rel_pos, 2.0).sum(-1, keepdims=True))
        additional_message_features = []; irreps_additional_message = ''
        additional_message_features.append(edge_dist); irreps_additional_message += ' + 1x0e'
        additional_message_features = np.concatenate(additional_message_features, axis=-1)
        graph['additional_message_features'] = torch.tensor(
            additional_message_features, 
            dtype=torch.get_default_dtype(), device=device
        ); irreps_additional_message = irreps_additional_message[3:]

        # graph['batch'], assuming every graph has an equal number of nodes
        graph['batch'] = torch.arange(0, batch_size, dtype=torch.int64, device = device).repeat_interleave(n_nodes)

        if not gen_graph_info:
            return graph
        else:
            graph_info = {}
            graph_info['irreps_node_input'] = Irreps(irreps_node_input).simplify()
            graph_info['irreps_node_attr'] =  Irreps(irreps_node_attr).simplify()
            graph_info['irreps_edge_attr'] = irreps_edge_attr
            graph_info['max_radius'] = np.sqrt(8.0) * world.radius
            graph_info['num_neighbors'] = n_edges / n_nodes
            graph_info['num_nodes'] = n_nodes
            graph_info['irreps_additional_message'] = Irreps(irreps_additional_message).simplify()

            return graph, graph_info

