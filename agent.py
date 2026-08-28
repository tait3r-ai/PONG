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

    def __init__(self, env, hidden_layer, learning_rate, step_repeat, gamma):

        self.env = env
        self.step_repeat = step_repeat
        self.gamma = gamma

        obs, info = self.env.reset()
        obs = self.process_observation(obs)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f'Loaded model on device {self.device}')

        self.memory = ReplayBuffer(max_size=500000, input_shape=obs.shape, device=self.device)

        self.model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer,
                            observation_shape=obs.shape).to(self.device)

        self.target_model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer,
                                   observation_shape=obs.shape).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate

    def process_observation(self, obs):
        # With FrameStackObservation(stack_size=4) wrapping GrayscaleObservation(keep_dim=True),
        # obs arrives as (4, H, W, 1). Squeeze the trailing channel dim -> (4, H, W),
        # i.e. 4 stacked grayscale frames as separate input channels.
        obs = torch.tensor(obs, dtype=torch.float32).squeeze(-1)
        return obs

    def test(self):

        # Load weights before evaluation
        self.model.load_the_model(map_location=self.device)

        done = False
        truncated = False
        total_steps = 0
        episode_steps = 0
        episode_reward = 0

        obs, info = self.env.reset()
        obs = self.process_observation(obs).to(self.device)

        episode_start_time = time.time()

        while not (done or truncated):

            # --- Action selection (5% random, 95% greedy) ---
            if random.random() < 0.05:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = self.model(obs.unsqueeze(0))  # shape (1, action_dim)
                    action = torch.argmax(q_values, dim=-1).item()

            # --- Execute action with step-repeat ---
            cumulative_reward = 0

            for _ in range(self.step_repeat):
                next_obs, reward, done, truncated, info = self.env.step(action)
                cumulative_reward += reward
                total_steps += 1
                episode_steps += 1

                if done or truncated:
                    break

            episode_reward += cumulative_reward
            obs = self.process_observation(next_obs).to(self.device)

        episode_time = time.time() - episode_start_time
        print(f"Episode finished | Steps: {episode_steps} | Reward: {episode_reward:.2f} | Time: {episode_time:.2f}s")

        return episode_reward, episode_steps

    def train(self, episodes, max_episode_steps, summary_writer_suffix, batch_size, epsilon, min_epsilon):
        # -------------------------------------------------------
        # Setup TensorBoard writer
        # -------------------------------------------------------
        summary_writer_name = (
            f"runs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            f"_{summary_writer_suffix}"
        )
        writer = SummaryWriter(summary_writer_name)

        os.makedirs("models", exist_ok=True)

        total_steps = 0

        # Seed with pessimistic values so the epsilon schedule doesn't
        # over-react to noise in the first few episodes.
        last_10_rewards = deque([-21] * 10, maxlen=10)
        last_10_avg = sum(last_10_rewards) / len(last_10_rewards)

        best_avg = float('-inf')

        # -------------------------------------------------------
        # Episode Loop
        # -------------------------------------------------------
        for episode in range(episodes):

            obs, info = self.env.reset()
            obs = self.process_observation(obs)

            episode_reward = 0
            episode_steps = 0
            done = False
            truncated = False

            episode_start_time = time.time()

            # -------------------------------------------------------
            # Step Loop
            # -------------------------------------------------------
            while not (done or truncated) and episode_steps < max_episode_steps:

                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = self.model(obs.unsqueeze(0).to(self.device))[0]
                        action = torch.argmax(q_values).item()

                # Execute action (with step_repeat)
                reward_total = 0
                for _ in range(self.step_repeat):
                    next_obs, r, done, truncated, info = self.env.step(action)
                    reward_total += r
                    if done or truncated:
                        break

                next_obs = self.process_observation(next_obs)

                # ---------------------------------------------------
                # Store transition
                # ---------------------------------------------------
                self.memory.store_transition(obs, action, reward_total, next_obs, done)
                obs = next_obs

                episode_reward += reward_total
                episode_steps += 1
                total_steps += 1

                # ---------------------------------------------------
                # Learning step (Double DQN update)
                # ---------------------------------------------------
                if self.memory.can_sample(batch_size):

                    states, actions, rewards, next_states, dones = self.memory.sample_buffer(batch_size)

                    dones = dones.unsqueeze(1).float()
                    actions = actions.unsqueeze(1).long()
                    rewards = rewards.unsqueeze(1)

                    # Q(s, a)
                    q_values = self.model(states)
                    qsa = q_values.gather(1, actions)

                    # Double DQN target: action selection from online net,
                    # action evaluation from target net. No grad needed for
                    # either forward pass here.
                    with torch.no_grad():
                        next_actions = torch.argmax(self.model(next_states), dim=1, keepdim=True)
                        next_q_values = self.target_model(next_states).gather(1, next_actions)
                        target = rewards + (1 - dones) * self.gamma * next_q_values

                    loss = F.mse_loss(qsa, target)

                    writer.add_scalar("Loss", loss.item(), total_steps)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Soft-update target network
                    if total_steps % 4 == 0:
                        soft_update(self.target_model, self.model)

            # -------------------------------------------------------
            # Episode finished
            # -------------------------------------------------------
            last_10_rewards.append(episode_reward)
            last_10_avg = sum(last_10_rewards) / len(last_10_rewards)

            if last_10_avg > best_avg:
                best_avg = last_10_avg
                self.model.save_the_model()

            writer.add_scalar("Score", episode_reward, episode)
            writer.add_scalar("Epsilon", epsilon, episode)

            # Reward-scaled epsilon schedule: boost exploration (norm2, can
            # exceed 1x) when the smoothed average is poor, otherwise decay
            # gently (norm1, always <1x). Both driven by the smoothed
            # average rather than single-episode reward to avoid noise.
            POOR_PERFORMANCE_THRESHOLD = -15

            if last_10_avg < POOR_PERFORMANCE_THRESHOLD:
                epsilon = min(1.0, max(min_epsilon, epsilon * norm2(last_10_avg)))
            else:
                epsilon = max(min_epsilon, epsilon * norm1(last_10_avg))

            episode_time = time.time() - episode_start_time

            print(f"Episode {episode}  |  Score: {episode_reward}  |  Steps: {episode_steps}")
            print(f"Time: {episode_time:.2f}s  |  Epsilon: {epsilon:.4f}\n")

        self.env.close()