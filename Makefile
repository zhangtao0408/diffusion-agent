.PHONY: install test lint run clean

install:
	pip install -e ".[dev]"

test:
	pytest -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

run:
	python -m diffusion_agent --help

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
