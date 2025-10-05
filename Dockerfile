# syntax=docker/dockerfile:1.6

FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv

RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

RUN groupadd --system fds && \
    useradd --system --gid fds --create-home --uid 1000 fds && \
    install -d -o fds -g fds /var/lib/fds

WORKDIR /app
COPY fds.py ./
COPY fds_cli.py ./
COPY features.py ./
COPY examples ./examples
COPY example.env ./example.env

ARG ISS_URL="http://localhost:8000"
ENV ISS_URL=${ISS_URL} \
    FDS_DATA=/var/lib/fds

USER fds

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://127.0.0.1:8080/health || exit 1

VOLUME ["/var/lib/fds"]

CMD ["uvicorn", "fds:app", "--host", "0.0.0.0", "--port", "8080"]
