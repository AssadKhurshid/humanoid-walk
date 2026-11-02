"""Headless check: every robot plays the full routine without falling and actually walks.

Run:  python test_walk.py
"""

import numpy as np

from walk import ROBOTS, ROUTINE, Walker, routine_command

MIN_PELVIS_HEIGHT = 0.5  # metres; all three robots stand well above this


def check(robot: str) -> None:
    walker = Walker(robot)
    total = sum(seconds for _, seconds, _, _ in ROUTINE)
    lowest, start = 10.0, None
    while walker.data.time < total:
        cmd, label = routine_command(walker.data.time)
        if label == "Walk forward" and start is None:
            start = walker.data.qpos[:2].copy()
        walker.step(cmd)
        lowest = min(lowest, walker.data.qpos[2])
    walked = np.linalg.norm(walker.data.qpos[:2] - start)
    print(f"{robot}: lowest pelvis {lowest:.2f} m, travelled {walked:.1f} m")
    assert lowest > MIN_PELVIS_HEIGHT, f"{robot} fell over"
    assert walked > 1.0, f"{robot} did not walk"


if __name__ == "__main__":
    for robot in ROBOTS:
        check(robot)
    print("All robots OK.")
