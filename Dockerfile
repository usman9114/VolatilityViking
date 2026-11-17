# syntax=docker/dockerfile:1
# VolatilityViking daily runner and data refresh image. No secrets in the image: keys come from .env at run time.

FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements-live.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --require-hashes -r requirements-live.txt

FROM python:3.12-slim AS runtime
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp
RUN groupadd --system --gid 1000 vv \
 && useradd --system --uid 1000 --gid vv --no-create-home --shell /usr/sbin/nologin vv
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
# Code is root-owned and read-only to the runtime user.
COPY vv/ ./vv/
COPY research/ ./research/
COPY config/runner.example.yaml ./config/runner.example.yaml
RUN mkdir -p /app/state /app/reports /app/data /app/config \
 && chown vv:vv /app/state /app/reports /app/data
USER vv
ENTRYPOINT ["python", "-m", "vv.runner"]
