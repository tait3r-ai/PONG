import numpy as np
import torch


class ReplayBuffer():

    def __init__(self, max_size, input_shape, device='cpu'):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.stack_size = input_shape[0]          # 4
        H, W = input_shape[1], input_shape[2]

        self.frame_memory = np.zeros((self.mem_size, H, W), dtype=np.uint8)
        self.action_memory = np.zeros(self.mem_size, dtype=np.uint8)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        self.device = device

    def can_sample(self, batch_size):
        return self.mem_ctr > (batch_size * 5)

    def store_transition(self, frame, action, reward, done):
        index = self.mem_ctr % self.mem_size
        self.frame_memory[index] = frame
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_ctr += 1

    def _get_stack(self, index):
        """Build a (stack_size, H, W) stack ending at `index`, without
        reaching back across an episode boundary."""
        frames = [None] * self.stack_size
        current = index

        for slot in range(self.stack_size - 1, -1, -1):
            frames[slot] = self.frame_memory[current]

            prev = (current - 1) % self.mem_size
            if slot > 0 and self.terminal_memory[prev]:
                for pad_slot in range(slot - 1, -1, -1):
                    frames[pad_slot] = frames[slot]
                break

            current = prev

        return np.stack(frames, axis=0)  # (stack_size, H, W)

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)

        newest_index = (self.mem_ctr - 1) % self.mem_size
        valid_indices = np.array([i for i in range(max_mem) if i != newest_index])

        batch = np.random.choice(valid_indices, batch_size)

        states = np.stack([self._get_stack(i) for i in batch])
        next_states = np.stack([self._get_stack((i + 1) % self.mem_size) for i in batch])

        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        return states, actions, rewards, next_states, dones
