FROM python:3.9-bookworm AS worker

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install agentrl-worker

COPY ./src/server/tasks/toolemu /app/src/server/tasks/toolemu
COPY ./configs/tasks/toolemu.yaml /app/configs/tasks/toolemu.yaml
COPY ./data/toolemu /app/data/toolemu

COPY --chmod=0755 ./extra/worker-entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh", "-c", "configs/tasks/toolemu.yaml"]
