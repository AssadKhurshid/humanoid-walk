# Humanoid Walk

A full-size humanoid robot walks in a physics simulator on your laptop — and you drive it with the arrow keys.

Unitree's G1, H1 and H1_2 humanoids walk, turn and walk backwards in [MuJoCo](https://github.com/google-deepmind/mujoco), driven by Unitree's trained reinforcement-learning policies. No training and no GPU needed.

## Requirements

| | |
|---|---|
| CPU | Any 4-core |
| RAM | 4 GB |
| GPU | Not needed |
| Disk | ~1.5 GB (mostly PyTorch) |
| Python | 3.12 – 3.14 |
| Git | [download](https://git-scm.com/downloads) |

Tested on Windows 11 with Python 3.14, MuJoCo 3.17.0 and PyTorch 2.15.0 (CPU).

On Python 3.14, PyTorch prints a `torch.jit.load ... may break` warning. It's harmless here: the policies load and run fine.

## Install

**Windows (PowerShell)**

```powershell
cd humanoid-walk
py -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python download_assets.py
```

**macOS / Linux**

```bash
cd humanoid-walk
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python download_assets.py
```

## Run

```bash
python walk.py                  # drive the G1 with the arrow keys
python walk.py --robot h1       # other robots: g1, h1, h1_2
python walk.py --routine        # hands-off demo: walk, turn, walk backwards, stop
```

On macOS use `mjpython` instead of `python`. Close the window to stop.

## Controls

| Key | Does |
|---|---|
| Up / Down | Walk faster / slower (keep pressing Down to walk backwards) |
| Left / Right | Turn left / right |
| Left-drag / scroll | Rotate / zoom the camera |
| Double-click robot, then Ctrl + right-drag | Push the robot |
| Space / Backspace | Pause / reset |

Speed is limited to −0.5 … 1.0 m/s and turning to ±1.0 rad/s, the range the policies handle well.

## How it works

Every 20 ms (50 times a second), `walk.py`:

1. Reads the robot's state: body spin, which way gravity points, leg joint angles and speeds, the last action, and where it is in its step cycle.
2. Adds your command (forward speed, turn speed).
3. Gives these numbers to the trained policy (`motion.pt`), which returns target angles for the leg joints.
4. A PD controller turns those targets into motor torques for 10 physics steps of 2 ms each.

The logic mirrors Unitree's `deploy/deploy_mujoco/deploy_mujoco.py`, without the Isaac Gym dependency.

## Check

```bash
python test_walk.py
```

Runs every robot through the routine without a window and checks that it doesn't fall and actually walks.

## Credits

- Robots, models and trained policies: [Unitree Robotics, unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) (BSD 3-Clause)
- Physics: [MuJoCo](https://github.com/google-deepmind/mujoco), Google DeepMind
- Idea inspired by [microduck-simulation](https://github.com/Chandukasireddy/microduck-simulation)
