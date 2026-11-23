GIT_COMMIT := $(shell git rev-parse HEAD)
GIT_DIRTY := $(shell test -n "$$(git status --porcelain)" && echo true || echo false)
IMAGE_TAG := $(shell git rev-parse --short=12 HEAD)$(if $(filter true,$(GIT_DIRTY)),-dirty)
IMAGE := aigm:$(IMAGE_TAG)

.PHONY: lint test image submit mlflow-up mlflow-down

lint:
	uv run ruff check .

test:
	uv run pytest -q

image:
	docker build --build-arg GIT_COMMIT=$(GIT_COMMIT) --build-arg GIT_DIRTY=$(GIT_DIRTY) \
		--build-arg IMAGE_TAG=$(IMAGE_TAG) -t $(IMAGE) .

submit:
	uv run python scripts/submit.py $(EXP) --image $(IMAGE) $(ARGS)

mlflow-up:
	mkdir -p mlflow-data
	docker compose up -d mlflow

mlflow-down:
	docker compose down
