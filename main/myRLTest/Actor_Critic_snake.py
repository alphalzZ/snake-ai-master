import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
# 导入贪吃蛇游戏环境
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snake_game_custom_wrapper_mlp import SnakeEnv


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.network(x)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.network(x)

class ActorCritic:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001, gamma=0.99):
        self.policy_network = PolicyNetwork(state_dim, hidden_dim, action_dim)
        self.value_network = ValueNetwork(state_dim, hidden_dim)
        
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.memory = deque(maxlen=10000)
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy_network(state)
        action = torch.multinomial(probs, 1).item()
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self):
        if len(self.memory) < 32:
            return
        
        batch = random.sample(self.memory, 32)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 计算优势函数
        current_values = self.value_network(states).squeeze()
        next_values = self.value_network(next_states).squeeze()
        advantages = rewards + self.gamma * next_values * (1 - dones) - current_values
        
        # 更新策略网络
        probs = self.policy_network(states)
        action_probs = probs[range(len(actions)), actions]
        policy_loss = -(action_probs * advantages.detach()).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # 更新价值网络
        value_loss = advantages.pow(2).mean()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
    
    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])

def train(model_path):
    env = SnakeEnv()
    state_dim = env.observation_space.shape[0]*env.observation_space.shape[1]
    action_dim = env.action_space.n
    
    agent = ActorCritic(state_dim, action_dim)
    if 0:
        agent.load(model_path)
    episodes = 100000
    max_steps = 1000
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        print(f"Episode {episode + 1}, Reward: {episode_reward}")
        
        if (episode + 1) % 1000 == 0:
            agent.save(f"main/myRLTest/model/actor_critic_model_{episode + 1}.pth")

if __name__ == "__main__":
    model_path =  "main/myRLTest/model/actor_critic_model_1000.pth"  # 根据实际保存的模型文件名修改
    train(model_path) 