.PHONY: install format lint typecheck test check pipeline clean

install:
	pip install -r requirements-dev.txt

format:
	black src tests
	ruff check --fix src tests

lint:
	ruff check src tests
	black --check src tests

typecheck:
	mypy --ignore-missing-imports src/*.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=95

check: lint typecheck test

pipeline:
	python src/create_db.py --train data/events_train.csv --candidates data/events_candidates.csv --db productivity.db
	python src/train_regression.py --db productivity.db --sql src/queries.sql --outdir outputs
	python src/score_new_days.py --db productivity.db --sql src/queries.sql --model outputs/model.joblib --outdir outputs

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
