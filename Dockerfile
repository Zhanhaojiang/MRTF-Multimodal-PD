FROM python:3.10-slim

# System deps (audio + scientific basics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml /workspace/pyproject.toml
COPY src /workspace/src
RUN pip install --no-cache-dir -e .

COPY notebooks /workspace/notebooks
COPY scripts /workspace/scripts
COPY configs /workspace/configs
COPY docs /workspace/docs

CMD ["bash"]
