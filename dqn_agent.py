import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from collections import deque
from model import DQNMetricsMLP, DQNCNNAgent
import math


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, action_dim=4, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.02, epsilon_decay = 0.9999,
                 tau=0.01, batch_size=128, buffer_size=100000,
                 device=None, network='cnn'):

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.network_type = network

        if network == 'cnn':
            self.policy_net = DQNCNNAgent(output_dim=action_dim).to(self.device)
            self.target_net = DQNCNNAgent(output_dim=action_dim).to(self.device)
        else:
            self.policy_net = DQNMetricsMLP(input_dim=256, hidden_dim=512,
                                            output_dim=action_dim).to(self.device)
            self.target_net = DQNMetricsMLP(input_dim=256, hidden_dim=512,
                                            output_dim=action_dim).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)

    def preprocess_state(self, state):
        return state.flatten()

    def get_valid_actions(self, env, state):
        """Check which actions actually change the board."""
        valid = []
        for action in range(env.action_space.n):
            env_copy = copy.deepcopy(env)
            try:
                next_obs, _, _, _, _ = env_copy.step(action)
                next_state = self.preprocess_state(next_obs)
                if not np.array_equal(state, next_state):
                    valid.append(action)
            except:
                pass
        return valid

    def select_action(self, state, env, eval_mode=False):
        # get valid actions first
        valid_actions = self.get_valid_actions(env, state)

        # if no valid actions game is over
        if not valid_actions:
            return None

        if not eval_mode and random.random() < self.epsilon:
            return random.choice(valid_actions)  # random but always valid!

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()

            # if best action is invalid, pick best valid action instead
            if action not in valid_actions:
                # get q values only for valid actions
                valid_tensor = torch.tensor(valid_actions)
                best_valid = q_values[0][valid_tensor].argmax().item()
                action = valid_actions[best_valid]

            return action

    def step(self, state, action, reward, next_state, done):
        # store experience in memory
        self.memory.push(state, action, reward, next_state, done)
        loss = 0

        # track total steps
        self.step_count = getattr(self, 'step_count', 0) + 1

        # learn every 2 steps, not every step
        # gives memory time to collect diverse experiences
        if len(self.memory) > self.batch_size and self.step_count % 2 == 0:
            loss = self.learn()
        return loss

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)

        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.SmoothL1Loss()(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        for target_param, policy_param in zip(self.target_net.parameters(),
                                               self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'network_type': self.network_type
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        saved_network = checkpoint.get('network_type', 'mlp')
        if saved_network != self.network_type:
            print(f"Warning: saved model used '{saved_network}' but current is '{self.network_type}'")
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)