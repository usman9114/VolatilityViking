.PHONY: bootstrap test lint graph

bootstrap:
	bash scripts/bootstrap.sh

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src main.py tests

graph:
	.venv/bin/graphify update .
