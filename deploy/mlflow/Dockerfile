# MLflow tracking server image for the Proxmox host (Intel Core 2 Duo E6750, x86-64-v1: no SSE4.2/AVX).
# Current numpy and every pyarrow wheel use newer instructions and kill the server at import with
# SIGILL (exit code 132). The tracking server works without pyarrow.
FROM ghcr.io/mlflow/mlflow:v3.18.0
RUN pip install --no-cache-dir "numpy<2.4" && pip uninstall -y pyarrow
