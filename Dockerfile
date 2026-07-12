FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-submission.txt .
RUN pip install --no-cache-dir -r requirements-submission.txt

# Download model during build (runs on GH Actions' fast network,
# not your local connection)
RUN pip install --no-cache-dir huggingface_hub && \
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF', filename='qwen2.5-3b-instruct-q4_k_m.gguf', local_dir='/app/model')"

COPY config.py math_tool.py prompt_templates.py local_model_gguf.py \
    router.py router_submission.py remote_client_submission.py \
    harness.py verify.py agent_loop.py .

ENV LOCAL_MODEL_PATH=/app/model/qwen2.5-3b-instruct-q4_k_m.gguf

ENTRYPOINT ["python", "harness.py"]