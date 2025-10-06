#!/usr/bin/env python3
import subprocess
import os

# Change to root directory
os.chdir('e:\\')

# Run git commands
commands = [
    ['git', 'add', '-A'],
    ['git', 'commit', '-m', 'Fix hybrid mining: Xmrig 10 threads, SRBMiner stderr monitoring, debug output'],
    ['git', 'push']
]

for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")
    print(f"Return code: {result.returncode}")
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        break
    print("---")