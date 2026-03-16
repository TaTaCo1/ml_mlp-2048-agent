import math
import random
import copy
import torch


# ─────────────────────────────────────────
# MCTS Node
# ─────────────────────────────────────────
class MCTSNode:
    """A single node in the MCTS tree."""

    def __init__(self, state, parent=None, action=None):
        self.state = state.copy()
        self.parent = parent
        self.action = action
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    def average_value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


# ─────────────────────────────────────────
# MCTS Agent
# ─────────────────────────────────────────
class DQNAgentWithMCTS:
    def __init__(self, dqn_agent, env, mcts_kwargs=None):
        """
        Args:
            dqn_agent: Trained DQNAgent instance
            env: Gym environment
            mcts_kwargs:
                - num_simulations (int): default 100
                - max_depth (int): default 20
                - ucb_constant (float): default sqrt(2)
                - discount_factor (float): default 0.99
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

    # ─────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────
    def select_action(self, state, use_mcts=True, verbose=False):
        """Select best action using MCTS or fallback to DQN."""
        if not use_mcts:
            return self.agent.select_action(state, self.env, eval_mode=True)

        # FIX 1: only consider valid actions at root
        valid_actions = self.agent.get_valid_actions(self.env, state)
        if not valid_actions:
            return None

        self.root = MCTSNode(state)
        for _ in range(self.num_simulations):
            self._simulate(self.root, depth=0)

        best_action, stats = self._get_best_action(valid_actions)
        self.last_search_stats = stats

        if verbose:
            print(f"  MCTS: {self.num_simulations} sims | best={best_action} | visits={stats['visits']}")

        return best_action if best_action is not None else random.choice(valid_actions)

    # ─────────────────────────────────────
    # Core simulation
    # ─────────────────────────────────────
    def _simulate(self, node, depth):
        """Single MCTS simulation: Selection → Expansion → Evaluation → Backup."""
        # base case — max depth reached
        if depth >= self.max_depth:
            return self._evaluate_state(node.state)

        # phase 1 — selection
        current = self._selection(node)

        # phase 2+3 — expansion and evaluation
        node_to_update, value = self._expansion_and_evaluation(current, depth)

        # phase 4 — backup only if _expand did not already do it
        if node_to_update is not None:
            self._backup(node_to_update, value)

        return value

    # ─────────────────────────────────────
    # Phase 1 — Selection
    # ─────────────────────────────────────
    def _selection(self, node):
        """Walk down tree using UCB until node with untried actions found."""
        current = node
        while len(current.children) == self.action_space_size:
            best_child = self._select_best_child(current)
            if best_child is None:
                break
            current = best_child
        return current

    def _select_best_child(self, node):
        """Pick child with highest UCB score."""
        if not node.children:
            return None
        best_child = None
        best_score = -float('inf')
        for child in node.children.values():
            score = self._ucb_score(child, node.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _ucb_score(self, child, parent_visits):
        """
        UCB = Q/N + C * sqrt(ln(P) / N)
        Q = value sum
        N = visit count
        P = parent visit count
        C = exploration constant
        """
        if child.visit_count == 0:
            return float('inf')  # always try unvisited nodes first
        exploitation = child.value_sum / child.visit_count
        exploration = self.ucb_constant * math.sqrt(
            math.log(parent_visits) / child.visit_count)
        return exploitation + exploration

    # ─────────────────────────────────────
    # Phase 2+3 — Expansion and Evaluation
    # ─────────────────────────────────────
    def _expansion_and_evaluation(self, node, depth):
        """Expand one untried action or evaluate if fully expanded."""
        untried_actions = [
            a for a in range(self.action_space_size)
            if a not in node.children
        ]
        if untried_actions and depth < self.max_depth:
            return self._expand(node, untried_actions, depth)
        else:
            return node, self._evaluate_state(node.state)

    def _expand(self, node, untried_actions, depth):
        """Create child node, recurse deeper, backup internally."""
        # FIX 2: prefer valid actions when expanding
        valid = self.agent.get_valid_actions(self.env, node.state)
        valid_untried = [a for a in untried_actions if a in valid]

        # if no valid untried actions fall back to any untried
        action = random.choice(valid_untried if valid_untried else untried_actions)

        next_state, reward = self._take_action(node.state, action)

        child = MCTSNode(next_state, parent=node, action=action)
        node.children[action] = child

        # recurse deeper
        recursive_value = self._simulate(child, depth + 1)
        value = reward + self.discount_factor * (recursive_value or 0)

        # FIX 3: backup here so parent does not backup again
        self._backup(child, value)
        return None, None

    # ─────────────────────────────────────
    # Phase 4 — Backup
    # ─────────────────────────────────────
    def _backup(self, node, value):
        """Propagate value up to root updating visit counts."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            current = current.parent

    # ─────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────
    def _take_action(self, state, action):
        """Simulate action on environment copy."""
        env_copy = copy.deepcopy(self.env)
        try:
            next_obs, reward, terminated, truncated, info = env_copy.step(action)
            next_state = next_obs.flatten()
        except Exception as e:
            print(f"Warning: step failed for action {action}: {e}")
            next_state = state.copy()
            reward = 0
        return next_state, reward

    def _evaluate_state(self, state):
        """Use DQN as heuristic to estimate state value."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.agent.policy_net(state_tensor)
            return q_values.max(dim=1)[0].item()

    def _get_best_action(self, valid_actions):
        """Pick action with most visits among valid actions only."""
        if self.root is None or not self.root.children:
            return None, {'visits': 0}

        best_action = None
        best_visits = 0
        visits_by_action = {}

        for action, child in self.root.children.items():
            # FIX 4: only consider valid actions for final decision
            if action not in valid_actions:
                continue
            visits_by_action[action] = child.visit_count
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_action = action

        return best_action, {'visits': best_visits, 'action_visits': visits_by_action}