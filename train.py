from agent import Agent
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import ale_py

episodes = 1000
max_episode_steps = 10000
hidden_layer = 128
learning_rate = 0.0001
step_repeat = 4
gamma = 0.99
batch_size = 64
epsilon = 1
min_epsilon = 0.05

env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
env = ResizeObservation(env, (64, 64))
env = GrayscaleObservation(env, keep_dim=True)
env = FrameStackObservation(env, stack_size=4)  # obs shape becomes (4, 64, 64, 1)

agent = Agent(env, hidden_layer=hidden_layer, learning_rate=learning_rate,
              step_repeat=step_repeat, gamma=gamma)

summary_writer_suffix = f'dqn_lr={learning_rate}_hl={hidden_layer}_bs={batch_size}'

agent.train(episodes=episodes,
            max_episode_steps=max_episode_steps,
            summary_writer_suffix=summary_writer_suffix,
            batch_size=batch_size,
            epsilon=epsilon,
            min_epsilon=min_epsilon)