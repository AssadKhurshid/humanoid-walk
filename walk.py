"""A Unitree humanoid (G1, H1 or H1_2) walks in MuJoCo, driven by its trained policy.

The walking policies come from Unitree's unitree_rl_gym repo (deploy/pre_train).
Each policy reads the robot's state (body spin, gravity direction, command, leg
joint angles and speeds, last action, gait phase) and returns leg joint targets,
50 times per second. A PD controller turns those targets into motor torques.

Run:
  python walk.py                 # drive the G1 with the arrow keys
  python walk.py --robot h1      # another robot: g1, h1, h1_2
  python walk.py --routine       # hands-off demo: walk, turn, stop, on a loop

Arrow keys: Up/Down = faster/slower (or backwards), Left/Right = turn.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

UNITREE = Path(__file__).parent / "assets" / "unitree_rl_gym"
ROBOTS = ["g1", "h1", "h1_2"]

# Command = [forward m/s, sideways m/s, turn rad/s]. Kept inside what the policies handle well.
CMD_MIN = np.array([-0.5, 0.0, -1.0], dtype=np.float32)
CMD_MAX = np.array([1.0, 0.0, 1.0], dtype=np.float32)
SPEED_STEP = 0.1
TURN_STEP = 0.2
GAIT_PERIOD = 0.8  # seconds per step cycle, the value the policies were trained with

# GLFW key codes the MuJoCo viewer passes to key_callback
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265

# (label, seconds, forward m/s, turn rad/s)
ROUTINE = [
    ("Stand", 3.0, 0.0, 0.0),
    ("Walk forward", 6.0, 0.5, 0.0),
    ("Turn left while walking", 6.0, 0.5, 0.5),
    ("Walk forward", 4.0, 0.5, 0.0),
    ("Turn right in place", 5.0, 0.0, -0.6),
    ("Walk backwards", 4.0, -0.3, 0.0),
    ("Stop", 3.0, 0.0, 0.0),
]


def gravity_in_body(quat):
    """Gravity direction as seen by the robot's pelvis. Same formula as Unitree's deploy_mujoco.py."""
    qw, qx, qy, qz = quat
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz),
    ])


class Walker:
    """One robot + its policy. Mirrors deploy/deploy_mujoco/deploy_mujoco.py from unitree_rl_gym."""

    def __init__(self, robot: str):
        cfg = yaml.safe_load((UNITREE / "deploy" / "deploy_mujoco" / "configs" / f"{robot}.yaml").read_text())
        root = str(UNITREE)
        self.model = mujoco.MjModel.from_xml_path(cfg["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", root))
        self.model.opt.timestep = cfg["simulation_dt"]
        self.data = mujoco.MjData(self.model)
        self.policy = torch.jit.load(cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", root))

        self.decimation = cfg["control_decimation"]
        self.kps = np.array(cfg["kps"], dtype=np.float32)
        self.kds = np.array(cfg["kds"], dtype=np.float32)
        self.default_angles = np.array(cfg["default_angles"], dtype=np.float32)
        self.ang_vel_scale = cfg["ang_vel_scale"]
        self.dof_pos_scale = cfg["dof_pos_scale"]
        self.dof_vel_scale = cfg["dof_vel_scale"]
        self.action_scale = cfg["action_scale"]
        self.cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
        self.n = cfg["num_actions"]
        self.obs = np.zeros(cfg["num_obs"], dtype=np.float32)

        self.action = np.zeros(self.n, dtype=np.float32)
        self.target = self.default_angles.copy()
        self.control_dt = self.model.opt.timestep * self.decimation

    def step(self, cmd):
        """Run one policy step (20 ms): physics with PD control, then a new action from the policy."""
        d = self.data
        for _ in range(self.decimation):
            d.ctrl[:] = (self.target - d.qpos[7:]) * self.kps - d.qvel[6:] * self.kds
            mujoco.mj_step(self.model, d)

        phase = d.time % GAIT_PERIOD / GAIT_PERIOD
        n, o = self.n, self.obs
        o[:3] = d.qvel[3:6] * self.ang_vel_scale
        o[3:6] = gravity_in_body(d.qpos[3:7])
        o[6:9] = cmd * self.cmd_scale
        o[9:9 + n] = (d.qpos[7:] - self.default_angles) * self.dof_pos_scale
        o[9 + n:9 + 2 * n] = d.qvel[6:] * self.dof_vel_scale
        o[9 + 2 * n:9 + 3 * n] = self.action
        o[9 + 3 * n:9 + 3 * n + 2] = np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)
        with torch.no_grad():
            self.action = self.policy(torch.from_numpy(o).unsqueeze(0)).numpy().squeeze()
        self.target = self.action * self.action_scale + self.default_angles


def routine_command(t):
    """The command for time t (seconds) in the looping ROUTINE, plus its label."""
    t %= sum(seconds for _, seconds, _, _ in ROUTINE)
    for label, seconds, forward, turn in ROUTINE:
        if t < seconds:
            return np.array([forward, 0.0, turn], dtype=np.float32), label
        t -= seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", choices=ROBOTS, default="g1")
    parser.add_argument("--routine", action="store_true", help="play the hands-off routine instead of keyboard driving")
    args = parser.parse_args()

    walker = Walker(args.robot)
    cmd = np.zeros(3, dtype=np.float32)

    def on_key(key):
        change = {KEY_UP: (0, SPEED_STEP), KEY_DOWN: (0, -SPEED_STEP),
                  KEY_LEFT: (2, TURN_STEP), KEY_RIGHT: (2, -TURN_STEP)}.get(key)
        if change and not args.routine:
            cmd[change[0]] += change[1]
            np.clip(cmd, CMD_MIN, CMD_MAX, out=cmd)
            print(f"forward {cmd[0]:+.1f} m/s   turn {cmd[2]:+.1f} rad/s")

    print(f"Robot: {args.robot}. " + ("Playing the routine." if args.routine else "Arrow keys to drive."))
    label = None
    with mujoco.viewer.launch_passive(walker.model, walker.data, key_callback=on_key) as viewer:
        while viewer.is_running():
            tick = time.perf_counter()
            if args.routine:
                cmd[:], new_label = routine_command(walker.data.time)
                if new_label != label:
                    label = new_label
                    print(label)
            with viewer.lock():
                walker.step(cmd)
            viewer.sync()
            # ponytail: sleeps once per 20 ms policy step (Windows sleep is too coarse for 2 ms physics steps)
            time.sleep(max(0.0, walker.control_dt - (time.perf_counter() - tick)))


if __name__ == "__main__":
    main()
