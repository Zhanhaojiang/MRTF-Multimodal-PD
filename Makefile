.PHONY: install demo eval repro

install:
	pip install -r requirements.txt
	pip install -e .

demo:
	python scripts/train_dummy.py --epochs 3 --batch_size 8 --save_json results/demo_metrics.json

eval:
	python scripts/eval_dummy.py --batch_size 8 --out results/demo_eval.json

repro:
	bash scripts/reproduce_supplementary.sh
