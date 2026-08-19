.PHONY: lint format test install

install:
	pip install -e .
	pip install pytest ruff black

lint:
	ruff check w2v_factory tests scripts
	black --check w2v_factory tests scripts

format:
	black w2v_factory tests scripts

test:
	pytest -q