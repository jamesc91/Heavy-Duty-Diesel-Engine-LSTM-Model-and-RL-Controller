"""
RLController.py (auto-load companion, dtype-safe)
"""
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from pathlib import Path
import math
import random
import gym
from gym import spaces

HERE = Path(__file__).resolve().parent

class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0; self.prev_error = 0.0
    def reset(self):
        self.integral = 0.0; self.prev_error = 0.0
    def update(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.kp*error + self.ki*self.integral + self.kd*derivative

class TorqueControlEnv(gym.Env):
    def __init__(self, lstm_model, scaler_x, scaler_y, speed_profile, torque_targets, sequence_length=20, segment_length=None):
        super().__init__()
        self.lstm_model = lstm_model.eval()
        self.scaler_x = scaler_x; self.scaler_y = scaler_y
        self.sequence_length = sequence_length
        self.speed_profile = np.asarray(speed_profile, dtype=float)
        self.torque_targets = np.asarray(torque_targets, dtype=float)
        self.segment_length = int(segment_length or (len(self.speed_profile) - self.sequence_length))
        self.obs_dim = 4
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.sequence_length*self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.current_step = 0
        self.input_sequence = None
        self.last_pedal_voltage = 1.86

    def reset(self):
        self.current_step = 0
        self.last_pedal_voltage = 1.86
        T = self.sequence_length
        self.input_sequence = np.zeros((T, 4), dtype=np.float32)

        pid = PIDController(0.01, 0.0001, 0.001)
        pedal = 1.86
        first_speed = float(self.speed_profile[0])
        ped_s, spd_s = self.scaler_x.transform([[pedal, first_speed]])[0].astype(np.float32)
        lstm_window = np.repeat(np.array([[ped_s, spd_s]], dtype=np.float32), T, axis=0)

        for i in range(T):
            spd = float(self.speed_profile[i])
            des = float(self.torque_targets[i])
            with torch.no_grad():
                pred_s = float(self.lstm_model(torch.tensor(lstm_window, dtype=torch.float32).unsqueeze(0)).item())
            pred = float(self.scaler_y.inverse_transform([[pred_s]])[0,0])
            err = des - pred
            pedal = float(np.clip(pedal + pid.update(err)/100.0, 1.1, 3.85))
            xs = self.scaler_x.transform([[pedal, spd]])[0].astype(np.float32)
            des_s = float(self.scaler_y.transform([[des]])[0,0])
            self.input_sequence[i] = np.array([xs[0], xs[1], des_s, pred_s], dtype=np.float32)
            lstm_window = np.vstack([lstm_window[1:], xs[:2]])
        return self.input_sequence.flatten().astype(np.float32)

    def step(self, action):
        a = float(np.clip(action[0], 0.0, 1.0))
        pedal = a * (3.85 - 1.1) + 1.1
        idx = self.current_step + self.sequence_length
        if idx >= len(self.torque_targets):
            return self.input_sequence.flatten().astype(np.float32), -1000.0, True, {}

        speed = float(self.speed_profile[idx])
        desired_torque = float(self.torque_targets[idx])
        lstm_seq = self.input_sequence[:, :2].astype(np.float32)
        with torch.no_grad():
            pred_s = float(self.lstm_model(torch.tensor(lstm_seq, dtype=torch.float32).unsqueeze(0)).item())
        pred = float(self.scaler_y.inverse_transform([[pred_s]])[0,0])
        xs = self.scaler_x.transform([[pedal, speed]])[0].astype(np.float32)
        des_s = float(self.scaler_y.transform([[desired_torque]])[0,0])
        new_row = np.array([xs[0], xs[1], des_s, pred_s], dtype=np.float32)
        self.input_sequence = np.vstack([self.input_sequence[1:], new_row])
        error = abs(pred - desired_torque)
        pedal_delta = abs(pedal - self.last_pedal_voltage)
        reward = float(np.exp(-error/100.0) - 0.1 * pedal_delta)
        self.last_pedal_voltage = pedal
        self.current_step += 1
        done = (self.current_step >= self.segment_length - 1)
        info = {"desired": desired_torque, "pred": pred, "pedal": pedal}
        return self.input_sequence.flatten().astype(np.float32), reward, done, info

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Sigmoid()
        )
        self.max_action = max_action
    def forward(self, s): return self.max_action * self.net(s)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, s, a): return self.q(torch.cat([s, a], dim=1))

class ReplayBuffer:
    def __init__(self, max_size=100000): self.buffer = deque(maxlen=max_size)
    def add(self, t): self.buffer.append(t)
    def sample(self, batch):
        b = random.sample(self.buffer, batch)
        s,a,r,ns,d = map(np.array, zip(*b))
        import torch
        return (torch.FloatTensor(s), torch.FloatTensor(a),
                torch.FloatTensor(r).unsqueeze(1),
                torch.FloatTensor(ns), torch.FloatTensor(d).unsqueeze(1))
    def __len__(self): return len(self.buffer)

if __name__ == "__main__":
    print("Env + TD3 nets ready.")
