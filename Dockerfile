FROM python:3.12.13-slim-bookworm
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt
COPY run_agent.py python_repairs.py csv_plan.py search_evidence.py ./
ENTRYPOINT ["python", "/app/run_agent.py"]

