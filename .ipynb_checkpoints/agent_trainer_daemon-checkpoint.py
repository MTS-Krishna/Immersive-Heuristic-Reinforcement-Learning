import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import psutil
import pickle
import time
import os
from collections import deque
from datetime import datetime

# -----------------------------
# Heuristic Rule (for reference)
# -----------------------------
def heuristic_rule(cpu, mem):
    if cpu > 80 and mem > 75:
        return 1  # urgent alert
    elif cpu > 60:
        return 0  # moderate attention
    else:
        return 2  # normal

# -----------------------------
# Q-Network
# -----------------------------
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------------
# Deep IHRL Agent
# -----------------------------
class DeepIHRLAgent:
    def __init__(self, state_size, action_size, memory_path="memory.pkl", epsilon_path="params.pkl"):
        self.state_size = state_size
        self.action_size = action_size
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.update_target_model()

        self.memory_path = memory_path
        self.epsilon_path = epsilon_path

        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        self.load_memory()
        self.load_params()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target = self.model(state_tensor).clone().detach()

            if done or next_state is None:
                target[0][action] = reward
            else:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                with torch.no_grad():
                    t = self.target_model(next_state_tensor)
                target[0][action] = reward + self.gamma * torch.max(t)

            output = self.model(state_tensor)
            loss = self.loss_fn(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="final_agent.pth"):
        torch.save(self.model.state_dict(), path)
        self.save_memory()
        self.save_params()

    def load(self, path="final_agent.pth"):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            self.update_target_model()

    def save_memory(self):
        with open(self.memory_path, "wb") as f:
            pickle.dump(self.memory, f)

    def load_memory(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "rb") as f:
                self.memory = pickle.load(f)

    def save_params(self):
        with open(self.epsilon_path, "wb") as f:
            pickle.dump({"epsilon": self.epsilon}, f)

    def load_params(self):
        if os.path.exists(self.epsilon_path):
            with open(self.epsilon_path, "rb") as f:
                data = pickle.load(f)
                self.epsilon = data.get("epsilon", 1.0)

# -----------------------------
# Live Learning Loop
# -----------------------------
def live_training_loop(interval=2, save_every=50, batch_size=64):
    agent = DeepIHRLAgent(state_size=3, action_size=3)
    agent.load("final_agent.pth")

    step_count = 0
    total_reward = 0

    while True:
        cpu = round(random.uniform(0, 100), 1)
        mem = round(random.uniform(65, 100), 1)
        procs = random.randint(150, 200)

        state = np.array([cpu, mem, procs])

        action = agent.get_action(state)
        expected = heuristic_rule(cpu, mem)

        # Smarter reward
        cpu_penalty = max(0, cpu - 85) * 0.05
        mem_penalty = max(0, mem - 80) * 0.05
        bonus = 1.0 if cpu < 65 and mem < 70 else 0.0
        reward = bonus - (cpu_penalty + mem_penalty)
        reward += 0.5 if action == expected else -0.2

        done = True
        agent.remember(state, action, reward, state, done)
        agent.replay(batch_size=batch_size)

        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] CPU: {cpu:.1f}% | Mem: {mem:.1f}% | Action: {action} | Expected: {expected} | Reward: {reward:.2f} | Eps: {agent.epsilon:.4f}")

        total_reward += reward
        step_count += 1
        if step_count % save_every == 0:
            agent.save("final_agent.pth")
            print(f"Agent saved after {step_count} steps | Total Reward: {total_reward:.2f}\n")
            total_reward = 0

        time.sleep(interval)


# Run the live agent
if __name__ == "__main__":
    live_training_loop()
