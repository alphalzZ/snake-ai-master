import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snake_game_custom_wrapper_cnn import SnakeEnv
from PPO_snake import PPOAgent


def test_model(model_path: str, num_episodes: int = 5, render: bool = True) -> None:
    """Run a trained PPO agent in the environment."""
    env = SnakeEnv(silent_mode=not render)
    action_dim = env.action_space.n
    agent = PPOAgent(action_dim)
    agent.load(model_path)
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            if render:
                env.render()
                time.sleep(0.05)
            action, _ = agent.select_action(state)
            state, reward, done, info = env.step(action)
            episode_reward += reward
        print(f"Episode {episode + 1} reward: {episode_reward:.2f}, size: {info['snake_size']}")
    env.close()


if __name__ == "__main__":
    MODEL_PATH = "main/myRLTest/model/ppo_snake_final.pth"
    test_model(MODEL_PATH)
