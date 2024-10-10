import random
from collections import namedtuple
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
import pandas as pd
import ast

BATCH_SIZE = 64
NUM_WORKERS = 1

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

# some utils functionalities specific for Decision Transformer
def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


class SequenceDataset(IterableDataset):
    def __init__(self, seq_len: int = 10, reward_scale: float = 1.0, dataset=None, state_mean=None, state_std=None):
        # self.dataset, info = load_d4rl_trajectories(env_name, gamma=1.0)
        self.dataset = dataset
        self.reward_scale = reward_scale
        self.seq_len = seq_len

        # self.state_mean = info["obs_mean"]
        # self.state_std = info["obs_std"]
        self.state_mean = state_mean
        self.state_std = state_std

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        states = traj["observations"][start_idx : start_idx + self.seq_len]
        actions = traj["actions"][start_idx : start_idx + self.seq_len]
        returns = traj["returns"][start_idx : start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        if self.state_mean is not None and self.state_std is not None:
            states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale

        # pad up to seq_len if needed, padding is masked during training
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)

        return states, actions, returns, time_steps, mask

    def __iter__(self):
        while True:
            # traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            traj_idx = np.random.choice(len(self.dataset))
            start_idx = random.randint(0, self.dataset[traj_idx]["rewards"].shape[0] - 1)
            yield self.__prepare_sample(traj_idx, start_idx)

def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum

if __name__ == '__main__':
    train_data_path = "./data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"
    training_data = pd.read_csv(train_data_path)
    
    def safe_literal_eval(val):
        if pd.isna(val):
            return val
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            print(ValueError)
            return val

    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)


    dataset = []    
    traj = {"observations": [], "actions": [], "returns": [], "rewards": []}
    for _, row in training_data.iterrows():
        traj["observations"].append(row["state"])
        traj["actions"].append(row["action"])
        traj["rewards"].append(row["reward"])
        # traj["returns"].append(row["reward"])
        if row["done"] == 1:
            traj["observations"] = torch.tensor(traj["observations"])
            traj["actions"] = torch.tensor(traj["actions"])
            traj["rewards"] = torch.tensor(traj["rewards"])
            traj["returns"] = torch.tensor(discounted_cumsum(np.array(traj["rewards"]), 1))
            # traj["returns"] = torch.tensor(traj["returns"])
            dataset.append(traj)
            traj = {"observations": [], "actions": [], "returns": [], "rewards": []}

    dataset = SequenceDataset(dataset=dataset)

    trainloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )
    
    trainloader_iter = iter(trainloader)
    for step in range(10):
        try:
            batch = next(trainloader_iter)
            states, actions, returns, time_steps, mask = [b for b in batch]
            print(states.shape, actions.shape, returns.shape, time_steps.shape, mask.shape)    
        except:
            continue