import numpy as np
import random
import copy
from collections import namedtuple, deque
from CACC.cacc_model import Centralized_Actor, Centralized_Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 16  # minibatch size
GAMMA = 0.98  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4 # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CACC_Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size,
                 num_agents=1, random_seed=0):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents


        # # Critic Network (w/ Target Network)
        self.critic_local = Centralized_Critic(state_size, action_size, random_seed, num_agents)
        self.critic_target = Centralized_Critic(state_size, action_size, random_seed, num_agents)

        self.agents=[Central_DDPGAgent(state_size, action_size, self.critic_local, self.critic_target,
                                       num_agents, random_seed, name=i) for i in range(num_agents)]

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=0)

    def step(self,current_obs, actions, rewards, new_obs_dict, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(current_obs, actions, rewards, new_obs_dict, dones)
        # Learn, if enough samples are available in memory
        #print(len(self.memory))

        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA, dones)

    def act(self, states):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        actions={}
        for name, agent in enumerate(self.agents):
            agent.actor_local.eval()
            with torch.no_grad():
                action = agent.actor_local(states.flatten(),states[name,:]).cpu().data.numpy()
            agent.actor_local.train()
            actions[name]=(np.clip(action, 0, 1))
        return actions

    def learn(self, experiences, gamma, dones):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states_concat, actions_concat, rewards_concat, next_states_concat, dones_concat = experiences

        for name, agent in enumerate(self.agents):
            if dones[name] == False:
                # ---------------------------- update critic ---------------------------- #
                agent.critic_optimizer.zero_grad()
                actions_next=torch.cat([my_agent.actor_target(next_states_concat.view(BATCH_SIZE, -1), next_states_concat[:,name,:])
                                        if dones[my_name] == False else torch.zeros(BATCH_SIZE, self.action_size)
                                        for my_name, my_agent in enumerate(self.agents)], dim=1)
                Q_target_next = self.critic_target(torch.cat((next_states_concat.view(BATCH_SIZE,-1), actions_next), dim=1).to(device),
                                            actions_next.view(BATCH_SIZE, self.num_agents, -1)[:, name, :])
                Q_target = rewards_concat[:,name,:] + (gamma * Q_target_next * (1 - dones_concat[:,name,:]))
                Q_expected=self.critic_local(torch.cat((states_concat.view(BATCH_SIZE,-1),actions_concat.view(BATCH_SIZE,-1)),dim=1),actions_concat[:,name,:])
                huber_loss = torch.nn.SmoothL1Loss()
                critic_loss = huber_loss(Q_expected, Q_target.detach())
                critic_loss.backward()
                agent.critic_optimizer.step()

                # ---------------------------- update actor ---------------------------- #
                agent.actor_optimizer.zero_grad()
                actions_pred = torch.cat([my_agent.actor_local(states_concat.view(BATCH_SIZE, -1), states_concat[:,my_name,:])
                                          if dones[my_name] == False else torch.zeros(BATCH_SIZE, self.action_size)
                                          for my_name, my_agent in enumerate(self.agents)], dim=1)
                actor_loss = -self.critic_local(torch.cat((states_concat.view(BATCH_SIZE, -1), actions_pred), dim=1),
                                                actions_pred.view(BATCH_SIZE, self.num_agents, -1)[:, name,:]).mean()
                actor_loss.backward()
                print('grad\n',[param.grad for param in agent.actor_local.parameters()])
                agent.actor_optimizer.step()
                self.soft_update(self.critic_local, self.critic_target, TAU)
                self.soft_update(agent.actor_local, agent.actor_target, TAU)




    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.asarray([np.vstack([e.state[bank_name] for bank_name in range(len(e.state))]) for e in experiences if
                        e is not None],
                       dtype='float64')).float().to(device)
        actions = torch.from_numpy(
            np.asarray([np.vstack([e.action[bank_name] for bank_name in range(len(e.action))]) for e in experiences if
                        e is not None],
                       dtype='float64')).float().to(device)
        rewards = torch.from_numpy(
            np.asarray([np.vstack([e.reward[bank_name] for bank_name in range(len(e.reward))]) for e in experiences if
                        e is not None],
                       dtype='float64')).float().to(device)
        next_states = torch.from_numpy(
            np.asarray(
                [np.vstack([e.next_state[bank_name] for bank_name in range(len(e.next_state))]) for e in experiences if
                 e is not None],
                dtype='float64')).float().to(device)
        dones = torch.from_numpy(
            np.asarray([np.vstack([e.done[bank_name] for bank_name in range(len(e.done))]) for e in experiences if
                        e is not None],
                       dtype='float64').astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Central_DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """

    def __init__(self, state_size, action_size, critic_local, critic_target, num_agents, random_seed=0, name=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Centralized_Actor(state_size, action_size, random_seed, num_agents).to(device)
        self.actor_target = Centralized_Actor(state_size, action_size, random_seed, num_agents).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = critic_local
        self.critic_target = critic_target
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)



