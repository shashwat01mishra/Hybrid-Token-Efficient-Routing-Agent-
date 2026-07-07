# Placeholder base image — swap to AMD's ROCm PyTorch image once the
# AMD Developer Cloud instance details (image name/tag) are known, e.g.:
#   FROM rocm/pytorch:rocm6.x-ubuntu22.04-py3.x
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "agent_loop.py"]
