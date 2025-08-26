# scripts/train_baseline_B3.py
import subprocess, sys
sys.exit(subprocess.call(["python","train.py","--epochs","3","--amp","1","--identity_weight","0.0"]))