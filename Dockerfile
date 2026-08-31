# FraudMesh serving API — local-only container, no cloud/registry required.
# Build+run entirely on your machine: `docker compose up --build`.
FROM python:3.10-slim

# libgomp1: XGBoost's OpenMP runtime (mirrors the `brew install libomp`
# note in README.md for macOS). curl: used by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch wheel first, from PyTorch's own index — the default PyPI
# wheel pulls in ~2GB of CUDA libraries this service never uses. `pip
# install .` below is satisfied by this version (dependency spec is
# torch>=2.1) and won't reinstall a different one.
RUN pip install --no-cache-dir "torch>=2.1" --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml config.py README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN mkdir -p /app/data /app/models /app/results \
    && useradd --create-home --uid 1000 fraudmesh \
    && chown -R fraudmesh:fraudmesh /app
USER fraudmesh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["fraudmesh-serve"]
