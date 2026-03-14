import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from model import DQNMetricsMLP
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
    def __init__(self, input_dim=256, action_dim=4, lr=1e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995,
                 tau=0.005, batch_size=128, buffer_size=100000, device=None):
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
        
        self.policy_net = DQNMetricsMLP(input_dim, 256, action_dim).to(self.device)
        self.target_net = DQNMetricsMLP(input_dim, 256, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
    
    def preprocess_state(self, state):
        # Flatten the (4, 4, 16) array to (256,)
        return state.flatten()
        
    def select_action(self, state, env, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
            
    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
        loss = 0
        if len(self.memory) > self.batch_size:
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
        
        loss = nn.MSELoss()(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
            
        return loss.item()
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def save(self, filepath):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)


class MCTSNode:
    """A single node in the MCTS tree."""
    
    def __init__(self, state, parent=None, action=None):
        self.state = state.copy()  # Game state
        self.parent = parent
        self.action = action
        self.visit_count = 0
        self.value_sum = 0.0
        
        # Children: action -> MCTSNode
        self.children = {}
    
    def average_value(self):
        """Return average value (Q-value) of this node."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
class DQNAgentWithMCTS:
    def __init__(self, dqn_agent, env, mcts_kwargs=None):
        """
        Args:
            dqn_agent: Trained DQNAgent instance
            env: Gym environment
            mcts_kwargs: Dict with MCTS hyperparameters
                - num_simulations (int): Number of tree simulations (default: 100)
                - max_depth (int): Maximum tree depth (default: 20)
                - ucb_constant (float): Exploration constant (default: sqrt(2) ≈ 1.41)
        """
        self.agent = dqn_agent
        self.env = env
        self.device = dqn_agent.device
        mcts_kwargs = mcts_kwargs or {}
        self.num_simulations = mcts_kwargs.get('num_simulations', 100)
        self.max_depth = mcts_kwargs.get('max_depth', 20)
        self.ucb_constant = mcts_kwargs.get('ucb_constant', math.sqrt(2))
        self.discount_factor = mcts_kwargs.get('discount_factor', 0.99)
        self.root = None
        self.action_space_size = env.action_space.n
        self.last_search_stats = {}
    
    def select_action(self, state, use_mcts=True, verbose=False):
        if not use_mcts:
            return self.agent.select_action(state, self.env, eval_mode=True)
        
        self.root = MCTSNode(state)
        for sim in range(self.num_simulations):
            self._mcts_simulate(self.root, depth=0)
        
        # Get best action (highest visit count)
        best_action, stats = self._get_best_action()
        self.last_search_stats = stats
        
        if verbose:
            print(f"  MCTS: {self.num_simulations} sims | Best action: {best_action} | Visits: {stats['visits']}")
        
        return best_action if best_action is not None else self.env.action_space.sample()
    
    def _mcts_simulate(self, node, depth):
        if depth >= self.max_depth:
            return self._evaluate_state(node.state)

        current = self._selection(node)
        node_to_update, value = self._expansion_and_evaluation(current, depth)

        # only backup if _expand didn't already do it
        if node_to_update is not None:
            self._backup(node_to_update, value)

        return value
    
    
    def _selection(self, node):
        """Phase 1: Walk down the tree using UCB until we find a node with untried actions."""
        current = node

        while len(current.children) == self.action_space_size:
            best_child = self._select_best_child(current)
            if best_child is None:
                break
            current = best_child

        return current


    def _expansion_and_evaluation(self, node, depth):
        """Phase 2 + 3: Expand an untried action and evaluate the new state with DQN."""
        untried_actions = [a for a in range(self.action_space_size) if a not in node.children]

        if untried_actions and depth < self.max_depth:
            return self._expand(node, untried_actions, depth)
        else:
            return node, self._evaluate_state(node.state)


    def _expand(self, node, untried_actions, depth):
        action = random.choice(untried_actions)
        next_state, reward = self._take_action(node.state, action)

        child = MCTSNode(next_state, parent=node, action=action)
        node.children[action] = child

        recursive_value = self._mcts_simulate(child, depth + 1)
        value = reward + self.discount_factor * (recursive_value or 0)

        # backup already happened inside _mcts_simulate, so do it here directly
        # and return None so the parent does NOT backup again
        self._backup(child, value)
        return None, None 
        
    
    def _select_best_child(self, node):
        """Select child with highest UCB score."""
        if not node.children:
            return None
        
        best_child = None
        best_score = -float('inf')
        for action, child in node.children.items():
            score = self._ucb_score(child, node.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _ucb_score(self, child, parent_visits):
        """
        Calculate Upper Confidence Bound score.
        
        UCB = (Q/N) + C * sqrt(ln(P)/N)
        
        Q: sum of returns
        N: visit count
        P: parent visit count
        C: exploration constant (ucb_constant)
        """
        if child.visit_count == 0:
            return float('inf')  # Prioritize unvisited nodes
        
        exploitation = child.value_sum / child.visit_count
        exploration = self.ucb_constant * math.sqrt(math.log(parent_visits) / child.visit_count)
        
        return exploitation + exploration
    
    def _take_action(self, state, action):
        """
        Simulate taking action in environment.
        
        Args:
            state: Current state (flattened)
            action: Action to take
        
        Returns:
            next_state: Resulting state
            reward: Immediate reward
        """
        # Unflatten state for environment
        board = state.reshape(4, 4, 16)
        
        # Create environment copy and set state
        import copy
        env_copy = copy.deepcopy(self.env)

        # instead of bare except:
        try:
            next_obs, reward, terminated, truncated, info = env_copy.step(action)
            next_state = next_obs.flatten()
        except Exception as e:
            print(f"Warning: step failed for action {action}: {e}")
            next_state = state.copy()
            reward = 0
            
        return next_state, reward
    
    def _evaluate_state(self, state):   
        """
        Use DQN to estimate value of state.
        
        This is the heuristic that guides MCTS.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.agent.policy_net(state_tensor)
            # Return max Q-value as value estimate
            value = q_values.max(dim=1)[0].item()
        
        return value
    
    def _backup(self, node, value):
        """
        Backpropagate value up to root.
        
        Updates visit counts and cumulative values.
        """
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            current = current.parent
    
    def _get_best_action(self):
        """
        Get best action from root based on visit counts.
        
        After MCTS, choose action with most visits (exploitation).
        """
        if self.root is None or not self.root.children:
            return None, {'visits': 0}
        
        best_action = None
        best_visits = 0
        visits_by_action = {}
        
        for action, child in self.root.children.items():
            visits_by_action[action] = child.visit_count
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_action = action
        
        return best_action, {'visits': best_visits, 'action_visits': visits_by_action}
