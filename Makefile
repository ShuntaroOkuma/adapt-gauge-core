PYTHON_VERSIONS := python3.11 python3.12 python3.13
VENV_DIR := .venv-test

.PHONY: install test run test-all test-all-clean help

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

run:
	python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json

test-all:
	@failed=""; \
	for pyver in $(PYTHON_VERSIONS); do \
		echo ""; \
		echo "========================================"; \
		echo "  Testing with $$pyver"; \
		echo "========================================"; \
		if ! command -v $$pyver > /dev/null 2>&1; then \
			echo "SKIP: $$pyver not found"; \
			continue; \
		fi; \
		$$pyver --version; \
		$$pyver -m venv $(VENV_DIR)-$$pyver; \
		$(VENV_DIR)-$$pyver/bin/pip install --quiet -e ".[dev]"; \
		if $(VENV_DIR)-$$pyver/bin/python -m pytest tests/ -v; then \
			echo "PASS: $$pyver"; \
		else \
			echo "FAIL: $$pyver"; \
			failed="$$failed $$pyver"; \
		fi; \
	done; \
	echo ""; \
	echo "========================================"; \
	echo "  Summary"; \
	echo "========================================"; \
	if [ -z "$$failed" ]; then \
		echo "All versions passed."; \
	else \
		echo "Failed:$$failed"; \
		exit 1; \
	fi

test-all-clean:
	rm -rf $(VENV_DIR)-python3.*

help:
	@echo "Available targets:"
	@echo "  install         Install package in dev mode"
	@echo "  test            Run tests with current Python"
	@echo "  test-all        Run tests with Python 3.11, 3.12, 3.13"
	@echo "  test-all-clean  Remove test venvs"
	@echo "  run             Run the evaluation runner"
	@echo "  help            Show this help"
