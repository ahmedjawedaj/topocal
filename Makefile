.PHONY: install lint type test check demo

install:
	python -m pip install -e '.[dev]'

lint:
	ruff check .

type:
	mypy src/topocal

test:
	pytest --cov=topocal --cov-report=term-missing

check: lint type test

demo:
	topocal demo
