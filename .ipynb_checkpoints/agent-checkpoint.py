import numpy as np
import random

class IHRLAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size          # Number of features
        self.action_size = action_size        # 0 = normal, 1 = anomaly
        self.q_table = {}                     # (discretized_state) → action values
        self.alpha = alpha                    # Learning rate
        self.gamma = gamma                    # Discount factor
        self.epsilon = epsilon                # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def _discretize(self, state):
        # Round features to nearest 10s for simplicity
        return tuple((state // 10).astype(int))

    def get_action(self, state):
        key = self._discretize(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return int(np.argmax(self.q_table[key]))

    def update(self, state, action, reward, next_state):
        state_key = self._discretize(state)
        next_key = self._discretize(next_state) if next_state is not None else state_key
        
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_size)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        target = reward + self.gamma * np.max(self.q_table[next_key])
        self.q_table[state_key][action] += self.alpha * (target - self.q_table[state_key][action])

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
