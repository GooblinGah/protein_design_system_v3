# scripts/train_baseline_B2.py
import subprocess, sys
sys.exit(subprocess.call(["python","train.py","--epochs","3","--amp","1","--identity_weight","0.0"]))