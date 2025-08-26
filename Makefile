
.PHONY: smoke train eval generate test
smoke: ; python scripts/smoke_data.py
train: ; python train.py --epochs 1 --amp 0
eval: ; python eval.py
generate: ; python generate.py --prompt "alpha beta hydrolase with motif GXSXG, length 260..320, secreted" --checkpoint checkpoints/model.pt
test: ; pytest -q tests
