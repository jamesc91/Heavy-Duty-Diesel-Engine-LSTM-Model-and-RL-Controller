"""
ModelUI.py (auto-load RL actor: td3_actor.pth)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
RL_ACTOR_FILENAME = "td3_actor.pth"

class TorqueLSTM(nn.Module):
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

class RLActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Sigmoid()
        )
        self.max_action = max_action
    def forward(self, state): return self.max_action * self.net(state)

class RLController:
    def __init__(self, model_path, state_dim, action_dim, max_action=1.0):
        self.actor = RLActor(state_dim, action_dim, max_action)
        sd = torch.load(model_path, map_location=torch.device("cpu"))
        self.actor.load_state_dict(sd); self.actor.eval()
        self.max_action = max_action
    def select_action(self, obs_flat_80):
        state = torch.tensor(obs_flat_80, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a01 = self.actor(state).numpy()[0, 0]
        pedal = 1.1 + a01 * (3.85 - 1.1)
        return float(np.clip(pedal, 1.1, 3.85))

class EngineUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diesel Engine Torque Simulator")
        self.setGeometry(100, 100, 1200, 700)

        self.model = self._load_lstm_and_scalers()
        self.sequence_length = 20
        self.lstm_buffer_2ch = None
        self.obs_buffer_4ch  = None

        self.time_step = 0
        self.speed_trace = None
        self.torque_trace = None
        self.pedal_vals = []
        self.predicted_torque = []
        self.target_torque = []
        self.timestamps = []

        self.pid = PIDController(0.01, 0.0001, 0.001)
        self.rl_path = HERE / RL_ACTOR_FILENAME
        self.rl_controller = None

        self._setup_ui()

    def _load_lstm_and_scalers(self):
        model = TorqueLSTM()
        model.load_state_dict(torch.load(HERE / "lstm_torque_model.pth", map_location=torch.device("cpu")))
        model.eval()
        self.scaler_x = joblib.load(HERE / "scaler_x.save")
        self.scaler_y = joblib.load(HERE / "scaler_y.save")
        return model

    def _setup_ui(self):
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(); cw.setLayout(layout)

        top = QtWidgets.QHBoxLayout()
        self.btn_load_speed  = QtWidgets.QPushButton("Load Speed CSV")
        self.btn_load_torque = QtWidgets.QPushButton("Load Desired Torque CSV")
        for b in [self.btn_load_speed, self.btn_load_torque]:
            top.addWidget(b)
        self.btn_load_speed.clicked.connect(self._load_speed_trace)
        self.btn_load_torque.clicked.connect(self._load_torque_trace)
        layout.addLayout(top)

        self.control_selector = QtWidgets.QComboBox()
        self.control_selector.addItems(["Manual", "PID", "RL"])
        layout.addWidget(QtWidgets.QLabel("Control Mode"))
        layout.addWidget(self.control_selector)

        pid_row = QtWidgets.QHBoxLayout()
        self.kp = QtWidgets.QDoubleSpinBox(); self.kp.setRange(0,1); self.kp.setDecimals(4); self.kp.setValue(0.01)
        self.ki = QtWidgets.QDoubleSpinBox(); self.ki.setRange(0,1); self.ki.setDecimals(6); self.ki.setValue(0.0001)
        self.kd = QtWidgets.QDoubleSpinBox(); self.kd.setRange(0,1); self.kd.setDecimals(6); self.kd.setValue(0.001)
        for lbl, w in [("Kp", self.kp), ("Ki", self.ki), ("Kd", self.kd)]:
            pid_row.addWidget(QtWidgets.QLabel(lbl)); pid_row.addWidget(w)
        layout.addLayout(pid_row)

        self.pedal_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pedal_slider.setMinimum(110); self.pedal_slider.setMaximum(385)
        self.pedal_label = QtWidgets.QLabel("Pedal: 1.10 V")
        layout.addWidget(self.pedal_label); layout.addWidget(self.pedal_slider)
        self.pedal_slider.valueChanged.connect(lambda: self.pedal_label.setText(f"Pedal: {self.pedal_slider.value()/100.0:.2f} V"))

        start_btn = QtWidgets.QPushButton("Start Simulation"); layout.addWidget(start_btn)
        start_btn.clicked.connect(self._start_sim)

        self.fig = Figure(figsize=(8,4)); self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111)

        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self._step)

    def _load_speed_trace(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Speed CSV", str(HERE), "CSV Files (*.csv)")
        if path: self.speed_trace = pd.read_csv(path).iloc[:,0].values

    def _load_torque_trace(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Desired Torque CSV", str(HERE), "CSV Files (*.csv)")
        if path: self.torque_trace = pd.read_csv(path).iloc[:,0].values

    def _start_sim(self):
        if self.speed_trace is None and (HERE/"Speed.csv").exists():
            self.speed_trace = pd.read_csv(HERE/"Speed.csv").iloc[:,0].values
        if self.torque_trace is None and (HERE/"Torque.csv").exists():
            self.torque_trace = pd.read_csv(HERE/"Torque.csv").iloc[:,0].values
        if self.speed_trace is None:
            QtWidgets.QMessageBox.warning(self, "Missing input", "Please load a Speed CSV."); return

        self.time_step = 0
        self.pedal_vals.clear(); self.predicted_torque.clear()
        self.target_torque.clear(); self.timestamps.clear()

        self.pid = PIDController(self.kp.value(), self.ki.value(), self.kd.value()); self.pid.reset()

        initial_speed = float(self.speed_trace[0]); initial_pedal = 1.86
        ped_s, spd_s = self.scaler_x.transform([[initial_pedal, initial_speed]])[0]
        ped_s, spd_s = float(ped_s), float(spd_s)

        T = self.sequence_length
        self.lstm_buffer_2ch = np.repeat(np.array([[ped_s, spd_s]], dtype=np.float32), T, axis=0)

        if self.torque_trace is not None:
            des0_s = float(self.scaler_y.transform([[float(self.torque_trace[0])]])[0,0])
        else:
            des0_s = 0.0
        with torch.no_grad():
            pred0_s = float(self.model(torch.tensor(self.lstm_buffer_2ch, dtype=torch.float32).unsqueeze(0)).item())
        self.obs_buffer_4ch = np.repeat(np.array([[ped_s, spd_s, des0_s, pred0_s]], dtype=np.float32), T, axis=0)

        if self.control_selector.currentText() == "RL":
            if not self.rl_path.exists():
                QtWidgets.QMessageBox.warning(self, "Missing RL actor",
                    f"Put '{RL_ACTOR_FILENAME}' in this folder:\n{HERE}"); return
            self.rl_controller = RLController(self.rl_path, state_dim=T*4, action_dim=1)
            if self.torque_trace is None:
                QtWidgets.QMessageBox.warning(self, "Missing Desired Torque CSV", "RL mode requires a Desired Torque CSV."); return

        self.timer.start(100)

    def _step(self):
        if self.time_step >= len(self.speed_trace):
            self.timer.stop(); return

        speed = float(self.speed_trace[self.time_step])
        mode = self.control_selector.currentText()

        with torch.no_grad():
            pred_s_prev = float(self.model(torch.tensor(self.lstm_buffer_2ch, dtype=torch.float32).unsqueeze(0)).item())
        pred_prev = float(self.scaler_y.inverse_transform([[pred_s_prev]])[0,0])

        if self.torque_trace is not None and self.time_step < len(self.torque_trace):
            desired = float(self.torque_trace[self.time_step])
            des_s = float(self.scaler_y.transform([[desired]])[0,0])
        else:
            desired = None; des_s = 0.0

        if mode == "Manual":
            pedal = self.pedal_slider.value()/100.0
        elif mode == "PID" and desired is not None and len(self.predicted_torque) > 0:
            error = desired - self.predicted_torque[-1]
            pedal = np.clip(self.pedal_vals[-1] + self.pid.update(error)/100.0, 1.1, 3.85) if self.pedal_vals else 1.86
        elif mode == "RL" and desired is not None:
            last_ped = self.pedal_vals[-1] if self.pedal_vals else 1.86
            ped_s_row, spd_s_row = self.scaler_x.transform([[last_ped, speed]])[0]
            new_row = np.array([ped_s_row, spd_s_row, des_s, pred_s_prev], dtype=np.float32)
            self.obs_buffer_4ch = np.vstack([self.obs_buffer_4ch[1:], new_row])
            obs_flat = self.obs_buffer_4ch.flatten().astype(np.float32)
            pedal = self.rl_controller.select_action(obs_flat)
        else:
            pedal = 2.0

        ped_s, spd_s = self.scaler_x.transform([[pedal, speed]])[0]
        self.lstm_buffer_2ch = np.vstack([self.lstm_buffer_2ch[1:], [ped_s, spd_s]])

        with torch.no_grad():
            y_pred_s = float(self.model(torch.tensor(self.lstm_buffer_2ch, dtype=torch.float32).unsqueeze(0)).item())
        y_pred = float(self.scaler_y.inverse_transform([[y_pred_s]])[0,0])

        self.predicted_torque.append(y_pred)
        self.pedal_vals.append(pedal)
        self.timestamps.append(self.time_step)
        self.target_torque.append(desired if desired is not None else np.nan)

        self._redraw_plot()
        self.time_step += 1

    def _redraw_plot(self):
        self.fig.clear(); ax = self.fig.add_subplot(111)
        ax.plot(self.timestamps, self.predicted_torque, label="Predicted Torque")
        if any([t is not None and not np.isnan(t) for t in self.target_torque]):
            ax.plot(self.timestamps, self.target_torque, '--', label="Desired Torque")
        pcts = [(p-1.1)/(3.85-1.1)*100.0 for p in self.pedal_vals]
        ax.plot(self.timestamps, pcts, ':', label="Pedal (%)")
        ax.set_xlabel("Time step"); ax.set_ylabel("Torque / Pedal %")
        ax.legend(); ax.grid(True); self.canvas.draw()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = EngineUI(); w.show()
    sys.exit(app.exec_())
