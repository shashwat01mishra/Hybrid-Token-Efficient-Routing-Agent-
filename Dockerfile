FROM python:3.11-slim
# NOTE: on AMD Developer Cloud, swap this base image for a ROCm-enabled
# PyTorch image (e.g. rocm/pytorch:rocm6.x-ubuntu22.04-py3.11) once the
# instance is provisioned, and reinstall torch from the ROCm wheel index
# instead of the default line in requirements.txt.

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "agent_loop.py"]
