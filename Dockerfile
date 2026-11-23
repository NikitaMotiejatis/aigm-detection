# One image for training, evaluation and analysis (PLAN.md decision 9). Build with `make image`.
# The PyTorch wheels bring their own CUDA libraries. Data is mounted at run time, never copied in.
FROM python:3.12-slim

ENV UV_PYTHON_PREFERENCE=only-system UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_NO_CACHE=1
RUN pip install --no-cache-dir uv==0.13.8

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
COPY splits ./splits
RUN uv sync --frozen --no-dev --no-editable && useradd --uid 1000 --create-home aigm

# Last, so that a new commit does not invalidate the dependency layers.
ARG GIT_COMMIT=unknown
ARG GIT_DIRTY=true
ARG IMAGE_TAG=none
ENV PATH=/app/.venv/bin:$PATH \
    AIGM_GIT_COMMIT=$GIT_COMMIT \
    AIGM_GIT_DIRTY=$GIT_DIRTY \
    AIGM_IMAGE_TAG=$IMAGE_TAG

USER aigm
ENTRYPOINT ["aigm"]
CMD ["--help"]
