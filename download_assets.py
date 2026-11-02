"""Download the Unitree humanoid models and trained walking policies.

Run once from the project folder:  python download_assets.py

Pinned to the exact commit this project was tested with, so a later change in
Unitree's repo won't break anything. Only the folders we need are checked out.
"""

import subprocess
from pathlib import Path

REPO = "https://github.com/unitreerobotics/unitree_rl_gym.git"
COMMIT = "276801e46c5d433564f24658bac64f254b7d2d4b"
TARGET = Path(__file__).parent / "assets" / "unitree_rl_gym"
FOLDERS = [
    "deploy/pre_train",
    "deploy/deploy_mujoco/configs",
    "resources/robots/g1_description",
    "resources/robots/h1",
    "resources/robots/h1_2",
]


def git(*args: str) -> None:
    subprocess.run(["git", *args], check=True)


def main() -> None:
    if TARGET.exists():
        print("unitree_rl_gym: already downloaded")
        return
    print("Downloading unitree_rl_gym (robot models + policies)...")
    git("clone", "--filter=blob:none", "--no-checkout", "--sparse", REPO, str(TARGET))
    git("-C", str(TARGET), "sparse-checkout", "set", *FOLDERS)
    git("-C", str(TARGET), "-c", "advice.detachedHead=false", "checkout", COMMIT)
    print("All assets are ready.")


if __name__ == "__main__":
    main()
