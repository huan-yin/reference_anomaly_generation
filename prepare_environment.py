#!/usr/bin/env python3
"""
app.py - Install uv, create a virtual environment, and install dependencies.
"""

import subprocess
import sys
import os

VENV_DIR = ".venv"
PYTHON_VERSION = "3.10.18"
REQUIREMENTS_FILE = "requirement.txt"
UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple/"


def run(cmd, **kwargs):
    """Execute a command and print output in real time."""
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, **kwargs)
    return result


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    # ── 1. Install uv ──────────────────────────────────────────────────────
    print("[1/3] Installing uv ...")
    run([sys.executable, "-m", "pip", "install", "uv"])

    # ── 2. Set mirror and create virtual environment ───────────────────────
    os.environ["UV_INDEX_URL"] = UV_INDEX_URL
    print(f"[info] UV_INDEX_URL set to {UV_INDEX_URL}")

    venv_path = os.path.join(base_dir, VENV_DIR)

    if not os.path.isdir(venv_path):
        print(f"[2/3] Creating virtual environment ({PYTHON_VERSION}) ...")
        run(["uv", "venv", VENV_DIR, "--python", PYTHON_VERSION])
    else:
        print(f"[2/3] Virtual environment already exists, skipping creation.")

    # ── 3. Install dependencies ────────────────────────────────────────────
    if os.path.isfile(REQUIREMENTS_FILE):
        print(f"[3/3] Installing dependencies from {REQUIREMENTS_FILE} ...")
        run([
            "uv", "pip", "install",
            "-r", REQUIREMENTS_FILE,
            "--python", os.path.join(venv_path, "bin", "python"),
        ])
    else:
        print(f"[3/3] {REQUIREMENTS_FILE} not found, skipping dependency installation.")

    print("\nDone. Virtual environment is ready and dependencies are installed.")


if __name__ == "__main__":
    main()
