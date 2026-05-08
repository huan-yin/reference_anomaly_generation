#!/usr/bin/env python3
"""
app.py - Install uv, create a virtual environment, and install dependencies.
(Windows compatible, with GPU PyTorch)
"""

import subprocess
import sys
import os

VENV_DIR = ".venv"
PYTHON_VERSION = "3.10.18"
REQUIREMENTS_FILE = "requirement.txt"
UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple/"
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu118"


def run(cmd, **kwargs):
    """Execute a command and print output in real time."""
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, **kwargs)
    return result


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    # ── 1. Install uv ──────────────────────────────────────────────────────
    print("[1/4] Installing uv ...")
    run([sys.executable, "-m", "pip", "install", "uv"])

    # ── 2. Create virtual environment ──────────────────────────────────────
    os.environ["UV_INDEX_URL"] = UV_INDEX_URL
    print(f"[info] UV_INDEX_URL set to {UV_INDEX_URL}")

    venv_path = os.path.join(base_dir, VENV_DIR)

    if not os.path.isdir(venv_path):
        print(f"[2/4] Creating virtual environment ({PYTHON_VERSION}) ...")
        run(["uv", "venv", VENV_DIR, "--python", PYTHON_VERSION])
    else:
        print(f"[2/4] Virtual environment already exists, skipping.")

    # Windows vs Linux/macOS
    if sys.platform == "win32":
        python_path = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        python_path = os.path.join(venv_path, "bin", "python")

    # ── 3. Install PyTorch (GPU, CUDA 11.8) ────────────────────────────────
    print(f"[3/4] Installing PyTorch (cu118) from {PYTORCH_INDEX_URL} ...")
    run([
        "uv", "pip", "install",
        "torch==2.7.1+cu118",
        "torchvision==0.22.1+cu118",
        "torchaudio==2.7.1+cu118",
        "--index-url", PYTORCH_INDEX_URL,
        "--python", python_path,
    ])

    # ── 4. Install remaining dependencies ──────────────────────────────────
    if os.path.isfile(REQUIREMENTS_FILE):
        print(f"[4/4] Installing dependencies from {REQUIREMENTS_FILE} ...")
        run([
            "uv", "pip", "install",
            "-r", REQUIREMENTS_FILE,
            "--python", python_path,
        ])
    else:
        print(f"[4/4] {REQUIREMENTS_FILE} not found, skipping.")

    print("\nDone. GPU PyTorch and all dependencies are installed.")


if __name__ == "__main__":
    main()
