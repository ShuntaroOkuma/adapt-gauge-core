.PHONY: test run install

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

run:
	python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json
