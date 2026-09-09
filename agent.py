from buffer import ReplayBuffer
from model import Model, soft_update
from eps_funcs import norm1, norm2
from collections import deque
import torch
import torch.optim as optim
import torch.nn.functional as F
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import random
import os


class Agent():

    def __init__(self, env, hidden_layer, learning_rate, step_repeat, gamma, stack_size=4):

        self.env = env
        self.step_repeat = step_repeat
        self.gamma = gamma
        self.stack_size = stack_size

        obs, info = self.env.reset()
        obs = self.process_observation(obs)

        stacked_shape = (self.stack_size, *obs.shape)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f'Loaded model on device {self.device}')

        self.memory = ReplayBuffer(max_size=500000, input_shape=stacked_shape, device=self.device)

        self.model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer,
                            observation_shape=stacked_shape).to(self.device)

        self.target_model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer,
                                   observation_shape=stacked_shape).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate

    def process_observation(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).squeeze(-1)
        return obs

    def test(self):

        self.model.load_the_model(map_location=self.device)

        done = False
        truncated = False
        total_steps = 0
        episode_steps = 0
        episode_reward = 0

        obs, info = self.env.reset()
        obs = self.process_observation(obs)

        frame_stack = deque([obs] * self.stack_size, maxlen=self.stack_size)

        episode_start_time = time.time()

        while not (done or truncated):

            stacked_obs = torch.stack(list(frame_stack), dim=0).to(self.device)

            if random.random() < 0.05:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = self.model(stacked_obs.unsqueeze(0))
                    action = torch.argmax(q_values, dim=-1).item()

            cumulative_reward = 0

            for _ in range(self.step_repeat):
                next_obs, reward, done, truncated, info = self.env.step(action)
                cumulative_reward += reward
                total_steps += 1
                episode_steps += 1

                if done or truncated:
                    break

            episode_reward += cumulative_reward
            next_obs = self.process_observation(next_obs)
            frame_stack.append(next_obs)

        episode_time = time.time() - episode_start_time
        print(f"Episode finished | Steps: {episode_steps} | Reward: {episode_reward:.2f} | Time: {episode_time:.2f}s")

        return episode_reward, episode_steps

    def train(self, episodes, max_episode_steps, summary_writer_suffix, batch_size, epsilon, min_epsilon):
        summary_writer_name = (
            f"runs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            f"_{summary_writer_suffix}"
        )
        writer = SummaryWriter(summary_writer_name)

        os.makedirs("models", exist_ok=True)

        total_steps = 0

        last_10_rewards = deque([-21] * 10, maxlen=10)
        last_10_avg = sum(last_10_rewards) / len(last_10_rewards)

        best_avg = float('-inf')

        for episode in range(episodes):

            obs, info = self.env.reset()
            obs = self.process_observation(obs)

            frame_stack = deque([obs] * self.stack_size, maxlen=self.stack_size)

            episode_reward = 0
            episode_steps = 0
            done = False
            truncated = False

            episode_start_time = time.time()

            while not (done or truncated) and episode_steps < max_episode_steps:

                stacked_obs = torch.stack(list(frame_stack), dim=0)

                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = self.model(stacked_obs.unsqueeze(0).to(self.device))[0]
                        action = torch.argmax(q_values).item()

                reward_total = 0
                for _ in range(self.step_repeat):
                    next_obs, r, done, truncated, info = self.env.step(action)
                    reward_total += r
                    if done or truncated:
                        break

                next_obs = self.process_observation(next_obs)
                frame_stack.append(next_obs)

                self.memory.store_transition(next_obs, action, reward_total, done)

                episode_reward += reward_total
                episode_steps += 1
                total_steps += 1

                if self.memory.can_sample(batch_size):

                    states, actions, rewards, next_states, dones = self.memory.sample_buffer(batch_size)

                    dones = dones.unsqueeze(1).float()
                    actions = actions.unsqueeze(1).long()
                    rewards = rewards.unsqueeze(1)

                    q_values = self.model(states)
                    qsa = q_values.gather(1, actions)

                    with torch.no_grad():
                        next_actions = torch.argmax(self.model(next_states), dim=1, keepdim=True)
                        next_q_values = self.target_model(next_states).gather(1, next_actions)
                        target = rewards + (1 - dones) * self.gamma * next_q_values

                    loss = F.mse_loss(qsa, target)

                    writer.add_scalar("Loss", loss.item(), total_steps)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if total_steps % 4 == 0:
                        soft_update(self.target_model, self.model)

            last_10_rewards.append(episode_reward)
            last_10_avg = sum(last_10_rewards) / len(last_10_rewards)

            if last_10_avg > best_avg:
                best_avg = last_10_avg
                self.model.save_the_model()

            writer.add_scalar("Score", episode_reward, episode)
            writer.add_scalar("Epsilon", epsilon, episode)

            POOR_PERFORMANCE_THRESHOLD = -15

            if last_10_avg < POOR_PERFORMANCE_THRESHOLD and epsilon < 0.5:
                boost = min(0.02, (norm2(last_10_avg) - 1) * epsilon)
                epsilon = min(1.0, epsilon + boost)
            else:
                epsilon = max(min_epsilon, epsilon * norm1(last_10_avg))

            episode_time = time.time() - episode_start_time

            print(f"Episode {episode}  |  Score: {episode_reward}  |  Steps: {episode_steps}")
            print(f"Time: {episode_time:.2f}s  |  Epsilon: {epsilon:.4f}\n")

        self.env.close()
