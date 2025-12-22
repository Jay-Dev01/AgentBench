FROM python:3.10

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Docker CLI (for docker-in-docker)
RUN apt-get update && \
    apt-get install -y docker.io && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./src/server/tasks/swebench_rebench/requirements.txt /app/src/server/tasks/swebench_rebench/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install \
      agentrl-worker \
      -r ./src/server/tasks/swebench_rebench/requirements.txt

# Copy task files
COPY ./src/server/tasks/swebench_rebench /app/src/server/tasks/swebench_rebench
COPY ./configs/tasks/swebench_rebench.yaml /app/configs/tasks/swebench_rebench.yaml
COPY ./data/swebench_rebench /app/data/swebench_rebench

COPY --chmod=0755 ./extra/worker-entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh", "-c", "configs/tasks/swebench_rebench.yaml"]
