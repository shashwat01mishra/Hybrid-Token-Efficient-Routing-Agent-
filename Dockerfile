# Submission image for AMD Developer Hackathon Track 1.
# Target: 4GB RAM / 2 vCPU / no GPU grading box. Must build linux/amd64.
FROM python:3.11-slim

WORKDIR /app

# Build tools needed to compile llama-cpp-python from source.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-submission.txt .

# Compile llama-cpp-python from source (CPU-only, no BLAS, no GPU).
RUN pip install --no-cache-dir -r requirements-submission.txt

# Model is pre-downloaded locally (./model/) — COPY is instant vs 10-min wget.
COPY model/qwen2.5-3b-instruct-q4_k_m.gguf /app/model/qwen2.5-3b-instruct-q4_k_m.gguf

# Source files — last layer so code changes don't bust model/pip cache
COPY config.py math_tool.py prompt_templates.py local_model_gguf.py \
     router.py router_submission.py remote_client_submission.py \
     harness.py verify.py agent_loop.py .

ENV LOCAL_MODEL_PATH=/app/model/qwen2.5-3b-instruct-q4_k_m.gguf

ENTRYPOINT ["python", "agent_loop.py"]

