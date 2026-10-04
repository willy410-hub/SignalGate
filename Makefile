.PHONY: install test figures results demo

install:
	pip install -e ".[dev,rl,viz]"

test:
	python -m pytest -q

demo:
	sgate demo

# regenerate every chart from code (RL figures need results/rl_results.json)
figures:
	cd scripts && python make_banner.py && python make_figs_core.py && python make_figs_rl.py

# reruns all RL experiments: 25 training runs, roughly 45-60 minutes on 2 CPU cores
results:
	python -m sgate_eval.rl.experiments --out results/rl_results.json
