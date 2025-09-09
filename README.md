# Diesel Torque LSTM + RL Demo (Portfolio Sample)

A self-contained demo of a torque **prediction** model (LSTM), a **controller UI** (PyQt5), and a **TD3 RL** agent.

**What runs without proprietary data:** the UI, LSTM inference, and RL control on the included example drive cycles.  
The LSTM **training** script is provided for completeness and requires your private dataset.

## Publications
- *Improved Diesel Engine Load Control for Heavy-Duty Transient Testing Using Gain-Scheduling & Feed-Forward Algorithms.* SAE Technical Paper 03-16-06-0042.  
  [Read on SAE Mobilus](https://saemobilus.sae.org/articles/improved-diesel-engine-load-control-heavy-duty-transient-testing-using-gain-scheduling-feed-forward-algorithms-03-16-06-0042)

- *Simultaneous Control Optimization of Variable-Geometry Turbocharger and High Pressure EGR on a Medium Duty Diesel Engine.* SAE Technical Paper **2021-01-1178**.  
  [Read on SAE](https://www.sae.org/publications/technical-papers/content/2021-01-1178/)



## Contents
- `ModelUI.py` — Qt app that loads **Speed** and **Desired Torque** CSVs and runs a drive cycle in **Manual**, **PID**, or **RL**.  
  - RL actor consumes a **4-feature × 20-step** observation and outputs a pedal command mapped to **[1.1 V, 3.85 V]**.  
  - **RL mode autoloads** `td3_actor.pth` from the same folder.
- `RLController.py` — Environment + TD3 networks used to train the actor; documents the **4-feature** observation layout.
- `EngineModelLSTM2.py` — LSTM training script (expects your dataset). Saves `lstm_torque_model.pth`, `scaler_x.save`, `scaler_y.save`.
- Example cycles (10 Hz, 30 s - ~10m): `SpeedCycle.csv`, `TorqueCycle.csv`, `SpeedRamp.csv`, `TorqueRamp.csv`.

## Required artifacts (place next to `ModelUI.py`)
- `lstm_torque_model.pth` — trained LSTM weights  
- `scaler_x.save`, `scaler_y.save` — MinMaxScaler pickles used at training time  
- `td3_actor.pth` — **trained TD3 actor** with **state_dim = 80** (20 steps × 4 features)

> **RL observation (80-D)**: for each of 20 recent steps we stack  
> `[scaled_pedal, scaled_speed, scaled_desired_torque, scaled_predicted_torque]`.  
> The LSTM itself only sees `[scaled_pedal, scaled_speed]`.

## Install
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install torch numpy pandas scikit-learn joblib pyqt5 matplotlib gym
```

## Run
```bash
python ModelUI.py
```
1) Keep one of these pairs in the folder and press **Start**:  
   - **Preferred:** `SpeedCycle.csv` + `TorqueCycle.csv`  
   - Or: `SpeedRamp.csv` + `TorqueRamp.csv`  
   - Legacy: `Speed.csv` + `Torque.csv`
2) Choose **Manual**, **PID**, or **RL**.  
   - **PID / RL require** a Desired Torque CSV.  
   - **RL** will try to auto-load `td3_actor.pth` from the same folder.

## Data format
- CSVs are **single-column** numeric series (header optional).  
- **10 Hz** sampling (≈ 100 ms/row).  
- **Speed** and **Desired Torque** must be the **same length**.

## Notes
- Reward balances **torque tracking** and **pedal smoothness** (weights not fully tuned in this sample).
- Emissions modeling/constraints are **not included**.

