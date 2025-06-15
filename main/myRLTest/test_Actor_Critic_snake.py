import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snake_game_custom_wrapper_mlp import SnakeEnv
from Actor_Critic_snake import ActorCritic
import time

def test_model(model_path, num_episodes=10, render=True):
    env = SnakeEnv(silent_mode=False)
    state_dim = env.observation_space.shape[0]*env.observation_space.shape[1]
    action_dim = env.action_space.n
    
    agent = ActorCritic(state_dim, action_dim)
    agent.load(model_path)
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            if render:
                env.render()
                time.sleep(0.1)  # 控制渲染速度
            
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            
            if done:
                print(f"Episode {episode + 1} finished with reward: {episode_reward}")
                print(f"Snake size: {info['snake_size']}")
                total_rewards.append(episode_reward)
                break
    
    env.close()
    avg_reward = np.mean(total_rewards)
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward

if __name__ == "__main__":
    # 测试最新保存的模型
    model_path = "actor_critic_model_700.pth"  # 根据实际保存的模型文件名修改
    test_model(model_path) 