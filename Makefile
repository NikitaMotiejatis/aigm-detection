.PHONY: lint test mlflow-up mlflow-down

lint:
	uv run ruff check .

test:
	uv run pytest -q

mlflow-up:
	mkdir -p mlflow-data
	docker compose up -d mlflow

mlflow-down:
	docker compose down
