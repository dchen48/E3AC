import numpy as np
import torch
from e3nn.o3 import Irreps, spherical_harmonics

from multiagent.core_vec import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import os

from multiagent.common import action_callback


class Scenario(BaseScenario):
    def __init__(self):
        obs_path = os.path.dirname(os.path.abspath(__file__))
        obs_path = os.path.dirname(os.path.dirname(obs_path))
        scripted_agent_ckpt = os.path.join(obs_path, 'scripted_agent_ckpt/simple_tag_n6_train_prey/agents_best.ckpt')
        self.scripted_agents = torch.load(scripted_agent_ckpt)['agents']

    def make_world(self):
        world = World(self.scripted_agents, self.observation)
        self.np_rnd = np.random.RandomState(0)
        # set any world properties first
        world.dim_c = 2
        world.num_good_agents = num_good_agents = 2
        world.num_adversaries = num_adversaries = 6
        num_agents = num_adversaries + num_good_agents
        world.num_landmarks = num_landmarks = 3
        world.radius = self.world_radius = 1.5
        world.collaborative = True
        # add agents
        world.agents = [Agent() for _ in range(num_adversaries)] \
                       + [Agent(action_callback) for _ in range(num_good_agents)]
        #world.agents = [Agent(), Agent(), Agent(), Agent(action_callback)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.collide_th = self.good_agents(world)[0].size + self.adversaries(world)[0].size
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = self.np_rnd.uniform(-self.world_radius, +self.world_radius, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = self.np_rnd.uniform(-(self.world_radius - 0.1), self.world_radius  - 0.1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew, rew1 = 0, 0
        n_col, n_collide = 0, 0
        if agent == world.agents[0]:
            agents = self.good_agents(world)
            adversaries = self.adversaries(world)

            adv_pos = np.array([[adv.state.p_pos for adv in adversaries]]).repeat(len(agents), axis=0)
            a_pos = np.array([[a.state.p_pos for a in agents]])
            a_pos1 = a_pos.repeat(len(adversaries), axis=0)
            a_pos1 = np.transpose(a_pos1, axes=(1, 0, 2))
            dist = np.sqrt(np.sum(np.square(adv_pos - a_pos1), axis=2))
            rew = np.min(dist, axis=0)
            rew = -0.1 * np.sum(rew)
            if agent.collide:
                n_collide = (dist < self.collide_th).sum()
            rew += 10 * n_collide



            """
            if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
                for adv in adversaries:
                    rew1 -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])

            if agent.collide:
                n_col = 0
                for ag in agents:
                    for adv in adversaries:
                        if self.is_collision(ag, adv):
                            n_col += 1
                            rew1 += 10
            """
        return rew

    def observation(self, agent, world):
        if not agent.adversary:
            # get positions of all entities in this agent's reference frame
            entity_pos = []
            for entity in world.landmarks:
                if not entity.boundary:
                    entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            # communication of all other agents
            comm = []
            other_pos = []
            other_vel = []
            for other in world.agents:
                if other is agent: continue
                comm.append(other.state.c)
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                if not other.adversary:
                    other_vel.append(other.state.p_vel)
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
        else:
            # get positions of all entities in this agent's reference frame
            entity_pos = []
            for entity in world.landmarks:
                if not entity.boundary:
                    entity_pos.append(entity.state.p_pos - agent.state.p_pos)

            # communication of all other agents
            comm = []
            other_pos = []
            other_vel = []
            for other in world.agents:
                if other is agent: continue
                comm.append(other.state.c)
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                if not other.adversary:
                    other_vel.append(other.state.p_vel)

            #other_pos = sorted(other_pos, key=lambda k: [k[0], k[1]])
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    def seed(self, seed=None):
        self.np_rnd.seed(seed)

    def state(self, world):
        s = {}

        s['n_agents'] = world.num_adversaries
        s['n_preys'] = world.num_good_agents
        s['n_landmarks'] = world.num_landmarks
        
        s['pos_agents'] = [agent.state.p_pos for agent in world.agents[:world.num_adversaries]]
        s['pos_preys'] = [agent.state.p_pos for agent in world.agents[world.num_adversaries:]]      
        s['pos_landmarks'] = [entity.state.p_pos for entity in world.landmarks]
        
        s['vel_agents'] = [agent.state.p_vel for agent in world.agents[:world.num_adversaries]]
        s['vel_preys'] = [agent.state.p_vel for agent in world.agents[world.num_adversaries:]]
        s['vel_landmarks'] = [entity.state.p_vel for entity in world.landmarks]

        for k in s:
            s[k] = np.array(s[k])

        return s
    
    def gen_state_graph(self, 
        batch_s, 
        lmax_attr = 3, node_input_type = '', node_attr_type = '', 
        world = None, gen_graph_info = False, device = None
    ):
        return self.gen_state_action_n_graph(
                    batch_s,  batch_action_n = None, lmax_attr = lmax_attr,
                    node_input_type = node_input_type, node_attr_type = node_attr_type, 
                    world = world, gen_graph_info = gen_graph_info,
                    device = device
                )
    
    def gen_state_action_n_graph(self, 
        batch_s, batch_action_n, 
        lmax_attr = 3, node_input_type = '', node_attr_type = '', 
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

        dtype = torch.get_default_dtype()
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
        pos_preys = np.array(batch_s['pos_preys'])          # (batch_size, n_preys, 2)
        pos_landmarks = np.array(batch_s['pos_landmarks'])  # (batch_size, n_landmarks, 2)
        pos = np.concatenate(
            (pos_agents, pos_preys, pos_landmarks), 
            axis=1
        ) # (batch_size, n_nodes = n_agents + n_landmarks + n_targets, 2)
        n_agents, n_preys, n_landmarks = pos_agents.shape[1], pos_preys.shape[1], pos_landmarks.shape[1]
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
        vel_preys = np.array(batch_s['vel_preys'])
        vel_landmarks = np.array(batch_s['vel_landmarks'])
        vel = np.concatenate((vel_agents, vel_preys, vel_landmarks), axis = 1)
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
            action_preys = torch.zeros((batch_size, n_preys, 2), dtype = dtype, device = device)
            action_landmarks = torch.zeros((batch_size, n_landmarks, 2), dtype = dtype, device = device)
            act = torch.concat((batch_action_n, action_preys, action_landmarks), dim=1)
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
                1 + np.zeros((n_preys,), dtype=np.int64),
                2 + np.zeros((n_landmarks,), dtype=np.int64),
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
        batch_obs,  # np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
        lmax_attr = 3, node_input_type = '', node_attr_type = '', 
        world = None, gen_graph_info = False, device = None
    ):
        obs_dim = batch_obs.shape[-1]
        batch_obs = batch_obs.reshape((-1, obs_dim))
        batch_size = batch_obs.shape[0]

        graph = {}
        n_agents = world.num_adversaries
        n_preys = world.num_good_agents
        n_landmarks = world.num_landmarks
        n_others = n_agents - 1

        # graph['pos']
        loc_agent =   batch_obs[:,2:4].reshape((batch_size, -1, 2))    # (batch_size, 1, 2)
        
        loc_landmarks = batch_obs[:, 4 : 4+2*n_landmarks].reshape((batch_size, -1, 2))
        loc_landmarks = loc_landmarks + loc_agent # loc_landmarks in obs is relative to loc_agent

        loc_others =  batch_obs[:,4+2*n_landmarks : 4+2*(n_landmarks+n_others)].reshape((batch_size, -1, 2))
        loc_others = loc_others + loc_agent # loc_others in obs is relative to loc_agent

        loc_preys = batch_obs[:,4+2*(n_landmarks+n_others) : 4+2*(n_landmarks+n_others+n_preys)].reshape((batch_size, -1, 2))
        loc_preys = loc_preys + loc_agent # loc_others in obs is relative to loc_agent
        
        pos = np.concatenate((loc_agent, loc_others, loc_preys, loc_landmarks), axis=1) # (batch_size, n_nodes, 2)
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
        vel_others = np.zeros((batch_size, n_others, 2))       #  vel_others is unobservable
        vel_preys = batch_obs[:, -2*n_preys:].reshape((batch_size, -1, 2))
        vel_landmarks = np.zeros((batch_size, n_landmarks, 2)) #  vel_landmarks is unobservable
        vel = np.concatenate((vel_agent, vel_others, vel_preys, vel_landmarks), axis = 1)
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
                2 + np.zeros((n_preys,), dtype=np.int64),
                3 + np.zeros((n_landmarks,), dtype=np.int64),
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